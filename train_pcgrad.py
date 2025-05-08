# Modded gpt speedrun (GPU poor ver. minus Hopper optimization tricks such as FP8 matmul etc.)
# Heavily borrow code from @KellerJordan
import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import time
import copy
import random
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import torch
from torch import nn, Tensor
import torch.distributed as dist

from src.pcgrad import SimPGrad, YetAnotherMixer, YetAnotherMixer2
from src.utils import RRScheduler
from src.utils import plot_training_losses
from src.arith_exp import create_result_mask, cal_masked_entropy, DigitTokenizer, cal_unmasked_entropy

import argparse

# -----------------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--train_files", type=str, default="data/fineweb10B/fineweb_train_*.bin")
    parser.add_argument("--val_files", type=str, default="data/fineweb10B/fineweb_val_*.bin")
    parser.add_argument("--test_files", type=str, default="data/multiplication_test_ood*.bin")
    parser.add_argument("--train_seq_len", type=int, default=32*1024)
    parser.add_argument("--val_seq_len", type=int, default=16*1024)
    parser.add_argument("--no_reg", action="store_true")
    parser.add_argument("--additive_grad", action="store_true")
    parser.add_argument("--np_weight", type=float, default=1.0)
    parser.add_argument("--positive_scale_factor", type=float, default=1.0)
    parser.add_argument("--negative_scale_factor", type=float, default=1.0)
    parser.add_argument("--proj_product", action="store_true") # use Entropy grad projection times MBE grad magnitude or Entropy grad projection alone for scaling
    parser.add_argument("--switch_phase", action="store_true")
    parser.add_argument("--log_grad_info", action="store_true")
    parser.add_argument("--diff_mbe", action="store_true")
    parser.add_argument("--num_iterations", type=int, default=1750)
    parser.add_argument("--entropy_patience", type=int, default=125)
    parser.add_argument("--entropy_min_delta", type=float, default=0.01)
    parser.add_argument("--mbe_patience", type=int, default=125)
    parser.add_argument("--mbe_min_delta", type=float, default=0.002)
    parser.add_argument("--inverse_ib_target", action="store_true")
    parser.add_argument("--patch_schedule", action="store_true")
    parser.add_argument("--init_patch_size", type=int, default=8)
    parser.add_argument("--use_prior_weights", action="store_true")
    parser.add_argument("--include_inner_cycle", action="store_true")
    parser.add_argument("--inv_mbe_patience", type=int, default=50)
    parser.add_argument("--period", type=int, default=5)
    parser.add_argument("--prior_weight", type=str, default="natural")
    parser.add_argument("--mask_entropy_train", action="store_true")
    parser.add_argument("--mask_entropy_val", action="store_true")
    parser.add_argument("--test_guided_phase_switch", action="store_true")
    parser.add_argument("--test_guided_early_stop", action="store_true")
    
    return parser.parse_args()

# -----------------------------------------------------------------------------
# Muon optimizer
# -----------------------------------------------------------------------------
# Muon optimizer @KellerJordan
import torch
from torch import Tensor

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params: list[Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            b = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
            group = dict(params=[p for p in params if p.numel() == size],
                         update_buffer=b, update_buffer_views=[b[i] for i in range(world_size)])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            update_buffer: Tensor = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            # generate weight updates in distributed fashion
            params: list[Tensor] = group["params"]
            handle = None
            params_world = None
            def update_prev(): # optimized Muon implementation contributed by @YouJiacheng
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.add_(g_world.view_as(p_world),
                                 alpha=-group["lr"] * max(1, p_world.size(-2) / p_world.size(-1))**0.5)
            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"]).flatten()
                else:
                    g = update_buffer_views[self.rank]
                if base_i > 0:
                    update_prev() # async all_gather instead of sync all_reduce by @YouJiacheng
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
            update_prev()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

# from src.rg_model import GPT, GPTConfig # random grouping model
# from src.model import GPT, GPTConfig # original model
from src.reprank import GPT_diff, GPT_mask, GPT, GPTConfig # rank regularized model


# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

from src import distributed_data_generator

# -----------------------------------------------------------------------------
# int main

# torchrun sets these env variables
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
assert torch.cuda.is_available()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
master_process = (rank == 0) # this process will do logging, checkpointing etc.

@dataclass
class Hyperparameters:
    # data
    train_files : str = "data/fineweb10B/fineweb_train_*.bin" # input .bin to train on
    val_files : str = "data/fineweb10B/fineweb_val_*.bin" # input .bin to eval validation loss on
    test_files : str = "data/multiplication_test_ood*.bin" # input .bin to eval test loss on
    val_tokens : int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    train_seq_len : int = 32*1024 # FlexAttention sequence length (per GPU)
    val_seq_len : int = 4*64*1024 # FlexAttention sequence length for validation (per GPU)
    test_seq_len : int = 32*1024
    batch_size : int = 8 # Batch size, across all devices
    # optimization
    num_iterations : int = 1750 # number of iterations to run
    cooldown_frac : float = 0.4 # fraction of training spent cooling down the learning rate
    # architecture
    vocab_size : int = 50257
    # evaluation and logging
    val_loss_every : int = 125 # every how many steps to evaluate val loss? 0 for only at the end
    save_checkpoint : bool = False
    no_reg: bool = False
    additive_grad: bool = False
    np_weight: float = 1.0 # weight on non-conflicting gradient, for observing behavior of mbe without backprop on it
    switch_phase: bool = False
    positive_scale_factor: float = 1.0
    negative_scale_factor: float = 1.0
    proj_product: bool = False
    log_grad_info: bool = False
    diff_mbe: bool = False
    entropy_patience: int = 125
    entropy_min_delta: float = 0.01
    mbe_patience: int = 125
    inv_mbe_patience: int = 50
    mbe_min_delta: float = 0.002
    inverse_ib_target: bool = False # use inverse IB target for regularization
    include_inner_cycle: bool = False
    patch_schedule: bool = False # curriculum for patch size (MBE)
    init_patch_size: int = 8
    use_prior_weights: bool = False
    period: int = 5
    prior_weight: str = "natural"
    mask_entropy_train: bool = False
    mask_entropy_val: bool = False
    test_guided_phase_switch: bool = False
    test_guided_early_stop: bool = False
    
cli_args = parse_args()
args = Hyperparameters()
for k, v in vars(cli_args).items():
    if v is not None:
        setattr(args, k, v)

model_config = GPTConfig(
    flex_kernel_options={
        "BLOCK_M": 64, "BLOCK_N": 64, # forward
        "BLOCK_M1": 32, "BLOCK_N1": 64, "BLOCK_M2": 64, "BLOCK_N2": 32 # backwards 
    }
)
if args.mask_entropy_val: # put in long file paths for arithmetic experiments
    args.val_seq_len = 32*1024
    args.train_files = "data/multiplication_train*.bin"
    if "id" in args.val_files: 
        args.val_files = "data/multiplication_val_id*.bin"
        args.val_tokens = 163840
    elif "ood" in args.val_files: 
        args.val_files = "data/multiplication_val_ood*.bin"
        args.val_tokens = 294912
    if args.test_guided_early_stop or args.test_guided_phase_switch:
        args.test_files = "data/multiplication_test_ood*.bin"
        args.test_seq_len = 32*1024

assert args.batch_size % (world_size) == 0
train_accumulation_steps = args.batch_size // world_size # long seq train is more efficient than big batch
    
# begin logging
logfile = None
no_priority = True
if master_process:
    run_id = uuid.uuid4()
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{run_id}.txt"
    print(logfile)
def print0(s, console=False):
    if master_process:
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)

# begin by printing this file (the Python code)
print0(code)
print0("="*100)
# log information about the hardware/software environment this is running on
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
def nvidia_smi():
    import subprocess  # avoid top level import
    return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
print0(nvidia_smi())
print0("="*100)



########################################
#    Construct model and optimizer     #
########################################
if args.diff_mbe: 
    model: nn.Module = GPT_diff(model_config).cuda()
elif args.mask_entropy_val: 
    model: nn.Module = GPT_mask(model_config).cuda()
else: 
    model: nn.Module = GPT(model_config).cuda()
    
for m in model.modules():
    if isinstance(m, nn.Embedding):
        m.bfloat16()
for param in model.parameters():
    dist.broadcast(param.detach(), 0)

# collect the parameters to optimize
hidden_matrix_params = [p for n,p in model.transformer.h.named_parameters() if p.ndim >= 2 and "embed" not in n]
embed_params = [p for n, p in model.named_parameters() if "wte" in n]
scalar_params = [p for p in model.parameters() if p.ndim < 2]
head_params = [model.lm_head.weight]

adam_params = [dict(params=head_params, lr=0.008),
               dict(params=embed_params, lr=0.6),
               dict(params=scalar_params, lr=0.04)] 
optimizer1 = torch.optim.Adam(adam_params, betas=(0.8, 0.95), eps=1e-10, fused=True)
optimizer2 = Muon(hidden_matrix_params, lr=0.05, momentum=0.95, rank=rank, world_size=world_size)
optimizers = [optimizer1, optimizer2]
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

# learning rate schedule: stable then decay
def get_lr(step: int):
    x = step / args.num_iterations # progress in training
    assert 0 <= x <= 1
    if x < 1 - args.cooldown_frac:
        return 1.0
    else:
        w = (1 - x) / args.cooldown_frac
        return w * 1.0 + (1 - w) * 0.1

# ---------------------------------------------------------

################################################
# Projective Non-conflicting Gradient Composer #
################################################

# grad_composer = SimPGrad(model)
grad_composer = YetAnotherMixer2(model, "entropy", positive_scale_factor=args.positive_scale_factor, negative_scale_factor=args.negative_scale_factor,
                                 proj_product=args.proj_product)

# ---------------------------------------------------------

####################################
# Arithmetic Experiments Tokenizer #
####################################

if args.mask_entropy_val: 
    digit_tokenizer = DigitTokenizer() # specific tokenizer for arithmetic experiments 

# ---------------------------------------------------------


model: nn.Module = torch.compile(model, dynamic=True)

########################################
#            Warmup kernels            #
########################################
import time 

# Warmup the training kernels, then re-initialize the state so we aren't cheating
warmup_steps = 20
initial_state = dict(model=copy.deepcopy(model.state_dict()),
                     optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers]) # save the initial state
attn_blocksize = torch.tensor(64, dtype=torch.int, device="cuda")
        
for i in range(warmup_steps):
    inputs = targets = torch.randint(0, args.vocab_size, size=(1, args.train_seq_len,), device="cuda")
    print(f" :: Forward propagation starts with inputs & targets of length {inputs.shape[1]}")
    forward_start = time.time() 
    if i % 2 == 0: 
        loss_dict = model.forward(inputs.to(torch.int32), targets, attn_blocksize, args.init_patch_size)
        if args.mask_entropy_val: 
            mask = create_result_mask(targets, digit_tokenizer)
            cal_masked_entropy(loss_dict, mask)
        loss_dict = {"entropy": loss_dict["entropy"]}
    else: 
        # Random value created in master_process, broadcasted to all GPUs 
        # ------------------------------------------------------------------------
        layer_idx = i % (model.num_encoder_layers)  
        # ------------------------------------------------------------------------
        loss_dict = model.forward(inputs.to(torch.int32), targets, attn_blocksize, args.init_patch_size)
        
        if args.diff_mbe: 
            diff_mbe_loss = loss_dict[f"diff_mbe_{layer_idx}"]
            loss_dict = {"diff_mbe": diff_mbe_loss}            
        else: 
            mbe_loss = loss_dict[f"mbe_{layer_idx}"]
            loss_dict = {"mbe": mbe_loss}
        print(f" :: Calculating Rank regularization loss for layer {layer_idx+1}")
        
    backward_start = time.time()
    loss_name = ', '.join(loss_dict.keys())
    print(f" :: Forward computation of loss [{loss_name}] takes {backward_start - forward_start} second")
    if args.additive_grad: 
        grad_composer.naive_backward(loss_dict)
    else: 
        grad_composer.backward(loss_dict) # project grad
    backward_end = time.time() 
    print(f" :: Backward gradient calculation for loss [{loss_name}] takes {backward_end - backward_start} second")
    for param in model.parameters():
        if param.grad is not None: 
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
model.load_state_dict(initial_state["model"])
for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
    opt.load_state_dict(opt_state)
del initial_state


########################################
#        Training and validation       #
########################################
print("--------"*10)
print("Train & Evaluation")

train_loader = distributed_data_generator(args.train_files, world_size * args.train_seq_len, rank, world_size)
training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.perf_counter()
# begin training
train_steps = args.num_iterations
loss_record = defaultdict(list)
scheduler = RRScheduler(train_accumulation_steps, 
                        train_steps,
                        start_layer=2,
                        end_layer=model.num_encoder_layers + model.num_decoder_layers - 2,
                        switch_phase=args.switch_phase,
                        period=args.period,
                        use_prior_weights=args.use_prior_weights,
                        prior_weight=args.prior_weight,
                        ib_target= not args.inverse_ib_target,
                        entropy_patience=args.entropy_patience, 
                        entropy_min_delta=args.entropy_min_delta,
                        mbe_patience=args.mbe_patience,
                        inv_mbe_patience=args.inv_mbe_patience,
                        mbe_min_delta=args.mbe_min_delta,
                        include_inner_cycle=args.include_inner_cycle) # scheduler for rank regularization
scheduler.phase = 1 if args.switch_phase else 2
early_stop = False

valid_patch_sizes = [8, 16, 32, 64, 128, 256, 512, 1024] # divide train seq len (32 x 1024) & smaller than maximal attention span (1792)

for step in range(train_steps + 1):
    last_step = (step == train_steps) or early_stop
    attn_blocksize = torch.tensor(64*((step/train_steps * (1792 - 64) + 64)//64), dtype=torch.int, device='cuda')
    
    if args.patch_schedule: # curriculum for patch size (MBE)
        patch_size = args.init_patch_size * ((step/train_steps * (1024 - args.init_patch_size) + args.init_patch_size)//args.init_patch_size)
    else: 
        patch_size = args.init_patch_size
    patch_size = min(valid_patch_sizes, key=lambda x: abs(x - patch_size))
    
    # --------------- VALIDATION SECTION -----------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        val_seq_len = world_size * args.val_seq_len
        # assert args.val_tokens % val_seq_len == 0
        val_steps = args.val_tokens // val_seq_len
        val_loader = distributed_data_generator(args.val_files, val_seq_len, rank, world_size)
        val_loss = defaultdict(float)
        with torch.no_grad():
            for i in range(val_steps):
                inputs, targets = next(val_loader)
                loss_dict = model.forward(inputs, targets, attn_blocksize, args.init_patch_size)
                if args.mask_entropy_val: 
                    mask = create_result_mask(targets, digit_tokenizer)
                    cal_masked_entropy(loss_dict, mask)
                for name, loss in loss_dict.items(): 
                    val_loss[name] += loss                
        for name in val_loss: 
            val_loss[name] /= val_steps
            loss_record[name].append(val_loss[name].item())
        del val_loader
        for key in val_loss: 
            dist.all_reduce(val_loss[key], op=dist.ReduceOp.AVG)            
        val_info = " ".join([f"{item} loss: {value:.4f}" for (item, value) in val_loss.items()])
        print0(f"step:{step}/{train_steps} {val_info} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)
        model.train()
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    if last_step:
        if master_process: 
            os.makedirs(f"logs/{run_id}", exist_ok=True)
            if args.log_grad_info: 
                grad_composer.save_grad_info(f"logs/{run_id}/grad_step{step:06d}.pkl")
            plot_training_losses(loss_record, save_path=f"logs/{run_id}/loss_curve.png")
            
            if args.save_checkpoint:
                log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
                os.makedirs(f"logs/{run_id}", exist_ok=True)
                torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt") 
        # the last step only has the validation loop, so break to avoid training
        break
    
    # --------------- TESTING SECTION -----------------
    if args.test_guided_early_stop or args.test_guided_phase_switch:
        model.eval() 
        test_seq_len = world_size * args.test_seq_len
        test_steps = args.test_tokens // test_seq_len
        test_loader = distributed_data_generator(args.test_files, test_seq_len, rank, world_size)
        test_loss = defaultdict(float)
        with torch.no_grad():
            for i in range(test_steps):
                inputs, targets = next(test_loader)
                loss_dict = model.forward(inputs, targets, attn_blocksize, args.init_patch_size)
                mask = create_result_mask(targets, digit_tokenizer)
                cal_masked_entropy(loss_dict, mask)
                for name, loss in loss_dict.items(): 
                    test_loss[name] += loss
        scheduler.step(test_loss)   
        if args.test_guided_early_stop and test_loss["entropy"] > scheduler.min_entropy * (1 + args.entropy_min_delta):
            early_stop = True
            
    # --------------- TRAINING SECTION -----------------
    for accum_step in range(train_accumulation_steps): 
        inputs, targets = next(train_loader)
        loss_dict = model.forward(inputs, targets, attn_blocksize, patch_size)
        if args.mask_entropy_train: 
            mask = create_result_mask(targets, digit_tokenizer)
            cal_masked_entropy(loss_dict, mask)
        elif args.mask_entropy_val:
            cal_unmasked_entropy(loss_dict)
        
        if accum_step == train_accumulation_steps - 1 and not args.test_guided_early_stop: 
            scheduler.step(loss_dict)
        if args.no_reg or len(scheduler.rr_layer_index) == 0: 
            loss_dict = {"entropy": loss_dict["entropy"]}
            print(f"- backward on entropy loss -")
        else:        
            layer_idx = scheduler.rr_layer_index[0]
            mbe_weight = scheduler.rr_layer_weight
            mbe_loss_name = f"mbe_{layer_idx}" if not args.diff_mbe else f"diff_mbe_{layer_idx}"
            print(f"- backward on {mbe_loss_name} loss -")
            loss_dict = {mbe_loss_name: loss_dict[mbe_loss_name] * mbe_weight}
        if args.additive_grad: 
            if (step % (20 * scheduler.num_reg_layers) < scheduler.num_reg_layers) and args.log_grad_info: 
                print(" :: logging gradient info :: ")
                grad_composer.naive_backward_info(loss_dict)
            else: 
                grad_composer.naive_backward(loss_dict)
        else: 
            if (step % (20 * scheduler.num_reg_layers) < scheduler.num_reg_layers) and args.log_grad_info: 
                print(" :: logging gradient info :: ")
                grad_composer.backward_info(loss_dict) # with gradient info accumulated
            else: 
                grad_composer.backward(loss_dict)
        
        
    for param in model.parameters():
        param.grad /= train_accumulation_steps
        dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
    # set optimization hyperparameters
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * get_lr(step)
    for group in optimizer2.param_groups:
        frac = min(step / 300, 1) # momentum warmup for muon
        group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
    # step the optimizers
    for opt in optimizers:
        opt.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    # ----------------------------------------------------
    # logging
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)

print0(f"loss record:\n{loss_record}", console=True)

dist.destroy_process_group()