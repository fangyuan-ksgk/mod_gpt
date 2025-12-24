# SoRL on TinyStories ... (info gain reward SoRL)
# -------------------------------------------------------------------------------------
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

import argparse

# -----------------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    # Data & Training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--train_files", type=str, default="data/tinystories/tinystory_train_*.bin")
    parser.add_argument("--val_files", type=str, default="data/tinystories/tinystory_val_*.bin")
    parser.add_argument("--train_seq_len", type=int, default=32*1024)
    parser.add_argument("--val_seq_len", type=int, default=32*1024)
    parser.add_argument("--num_iterations", type=int, default=1750)
    parser.add_argument("--save_checkpoint", action="store_true", default=False)
    parser.add_argument("--log_grad_info", action="store_true")

    # Model
    parser.add_argument("--model_size", type=str, default="small") # model size
    
    # SoRL / Search
    parser.add_argument("--num_rollouts", type=int, default=2)
    parser.add_argument("--num_rollouts_val", type=int, default=2)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--max_iterations", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=5.0) # search temperature
    parser.add_argument("--min_temperature", type=float, default=0.0) # prediction temperature
    
    # Architecture / Vocab
    parser.add_argument("--abstract_vocab_size", type=int, default=256)
    parser.add_argument("--use_static_memory_span", action="store_true", default=False)
    parser.add_argument("--min_memory_span", type=int, default=64)
    
    # Loss / Regularization
    parser.add_argument("--alpha_select", type=float, default=0.0)  # selection regularization strength
    parser.add_argument("--alpha_loss", type=float, default=0.0)  # abs loss weight
    parser.add_argument("--use_orthogonal_init", action="store_true", default=False)
    parser.add_argument("--alpha_marg_ent", type=float, default=1.0) # marginal entropy weight
    parser.add_argument("--decay", type=float, default=0.8) # decay for mutual information loss
    parser.add_argument("--target_vocab_util", type=float, default=0.9) # target vocabulary utilization
    parser.add_argument("--alpha_info_gain", type=float, default=1.0) # info gain loss weight
    parser.add_argument("--coarse_to_fine_search", action="store_true", default=False) # coarse to fine search (curriculum on K ratio)
    parser.add_argument("--max_K", type=int, default=512) # initial K ratio (max K ratio)
    parser.add_argument("--num_c2f_phases", type=int, default=4) # number of coarse to fine phases
    parser.add_argument("--no_attn_sweep", action="store_true", default=False) # no attention sweep (use static attention blocksize)
    
    # GAPT
    parser.add_argument("--use_gapt", action="store_true", default=False) # use GAPT to balance objectives
    parser.add_argument("--traj_perplexity_patience", type=int, default=5) # patience for traj perplexity
    parser.add_argument("--abs_perplexity_patience", type=int, default=5) # patience for abstract perplexity
    parser.add_argument("--tau_plateau", type=float, default=0.01) # plateau threshold for traj perplexity
    parser.add_argument("--tau_spike", type=float, default=0.1) # spike threshold for traj perplexity
    
    parser.add_argument("--run_info", type=str, default="")

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
# from src.model import GPT, GPTConfig

# GAT model
from sorl.gat_sim import GAT, GATConfig
from sorl.neo_utils import sorl_evaluate
from sorl.eval import compute_vocab_utilization_rate

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
    train_files : str = "data/fineweb10B/fineweb_train_*.bin"
    val_files : str = "data/fineweb10B/fineweb_val_*.bin"
    val_tokens : int = 10485760 
    train_seq_len : int = 32*1024
    val_seq_len : int = 32*1024
    batch_size : int = 16
    
    # optimization
    num_iterations : int = 1750
    cooldown_frac : float = 0.4
    
    # architecture
    vocab_size : int = 50257
    abstract_vocab_size : int = 256
    
    # evaluation and logging
    val_loss_every : int = 125 
    save_checkpoint : bool = False
    log_grad_info: bool = False
    
    # sorl specific
    num_rollouts: int = 2
    num_rollouts_val: int = 2
    K: int = 8
    max_iterations: int = 2
    temperature: float = 5.0
    min_temperature: float = 0.5
    
    use_static_memory_span: bool = False
    min_memory_span: int = 64
    
    alpha_select: float = 0.0
    alpha_loss: float = 0.0
    alpha_marg_ent: float = 1.0
    decay: float = 0.8
    target_vocab_util: float = 0.8
    coarse_to_fine_search: bool = False
    max_K: int = 512
    num_c2f_phases: int = 4
    no_attn_sweep: bool = False

    use_gapt: bool = False
    traj_perplexity_patience: int = 5
    abs_perplexity_patience: int = 5
    tau_plateau: float = 0.01
    tau_spike: float = 0.1
    
    run_info: str = ""

cli_args = parse_args()
args = Hyperparameters()
for k, v in vars(cli_args).items():
    if v is not None:
        setattr(args, k, v)

# check SRAM
if "40" in torch.cuda.get_device_properties("cuda").name: 
    model_config = GATConfig.gpt_size(
        args.model_size,
        vocab_sizes=[args.vocab_size, args.abstract_vocab_size],
        flex_kernel_options={
            "BLOCK_M": 32, "BLOCK_N": 32,
            "BLOCK_M1": 32, "BLOCK_N1": 64, "BLOCK_M2": 64, "BLOCK_N2": 32
        }
    )
else: 
    model_config = GATConfig.gpt_size(
        args.model_size,
        vocab_sizes=[args.vocab_size, args.abstract_vocab_size],
        flex_kernel_options={
            "BLOCK_M": 64, "BLOCK_N": 64,
            "BLOCK_M1": 32, "BLOCK_N1": 64, "BLOCK_M2": 64, "BLOCK_N2": 32
        }
    )

assert args.batch_size % (world_size) == 0
train_accumulation_steps = args.batch_size // world_size # long seq train is more efficient than big batch
    
# begin logging
logfile = None
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

# -------------------------- data loader -------------------------------------
from src.utils import distributed_data_generator_sorl as distributed_data_generator

# ------------------------- sorl search ---------------------------------------
from sorl.neo_utils import sorl_search_v8 as sorl_search

########################################
#    Construct model and optimizer     #
########################################

model: nn.Module = GAT(model_config).cuda()
    
if args.use_orthogonal_init:
    from sorl.topo import orthogonalize_abs_param
    orthogonalize_abs_param(model, gain=1.0, do_wte=True, do_head=True)

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

# SoRL loss function (info gain loss)
from sorl.info import SoRLLoss_v7
loss_fn = SoRLLoss_v7(model.vocab_sizes[1], decay=args.decay, target_vocab_util=args.target_vocab_util)
loss_fn = loss_fn.to(device)

# --- coarse to fine search ---
def get_current_K(step: int, total_steps: int, current_vocab_util: float, curr_K: int) -> int:
    if not args.coarse_to_fine_search:
        return args.K
    if current_vocab_util >= args.vocab_util_halt_threshold and step >= 200: 
        return curr_K

    steps_per_phase = total_steps // args.num_c2f_phases
    current_phase = min(step // steps_per_phase, args.num_c2f_phases - 1)
    log2_max_K = np.log2(args.max_K) 
    log2_min_K = np.log2(args.K)   
    if args.num_c2f_phases == 1:
        return args.K
    exponent = log2_max_K - current_phase * (log2_max_K - log2_min_K) / (args.num_c2f_phases - 1)
    current_K = int(2 ** round(exponent))
    return current_K

# compile model
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
memory_span = torch.tensor(1792, dtype=torch.int, device="cuda")
temperature_warmup = torch.cat([
    torch.tensor([args.min_temperature], device="cuda"),  # Low temp for first rollout
    torch.full((args.num_rollouts - 1,), args.temperature, device="cuda")  # Low temp for diversity
])

for i in range(warmup_steps):
    tokens = torch.randint(0, args.vocab_size, size=(1, args.train_seq_len,), device="cuda")
    print(f" :: Sorl search propagation starts with tokens of length {tokens.shape[1]}")
    forward_start = time.time() 
    K = args.K  # Use fixed K for warmup (kernel compilation only)
    # GAT specific function 
    # --- sorl search --- 
    search_start = time.time()
    with torch.no_grad():
        search_tokens, search_ppt, search_adv, rew = sorl_search(tokens, model, n=args.num_rollouts, K=K, max_iterations=args.max_iterations, memory_span=memory_span, attn_blocksize=attn_blocksize, 
                                                                          temperature=temperature_warmup)
    search_end = time.time()
    print(f" :: Sorl search takes {search_end - search_start} second")
    # --- compute loss --- 
    base_loss = model.forward(tokens, memory_span, attn_blocksize)[0].mean()
    info_loss, abs_loss, zipf_loss = loss_fn(search_tokens, model, base_loss.detach(), memory_span, attn_blocksize)
    forward_end = time.time()
    print(f" :: Loss computation takes {forward_end - search_end} second")
    # --- backward --- 
    loss = base_loss + args.alpha_info_gain * info_loss + args.alpha_loss * abs_loss + args.alpha_marg_ent * zipf_loss
    loss.backward() 
    backward_end = time.time()
    print(f" :: Backward takes {backward_end - forward_end} second")
   
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

# --- GAPT ---
from sorl.gapt import GatedPhaseTransition
gapt = GatedPhaseTransition(p_m=args.traj_perplexity_patience, p_a=args.abs_perplexity_patience,
                            tau_plateau=args.tau_plateau, tau_spike=args.tau_spike)
# -------------
temperature_val = torch.cat([
    torch.tensor([args.min_temperature], device="cuda"),
    torch.full((args.num_rollouts - 1,), args.temperature, device="cuda")
])
temperature_train = torch.cat([
    torch.tensor([args.min_temperature], device="cuda"),
    torch.full((args.num_rollouts - 1,), args.temperature, device="cuda")
])

training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.perf_counter()
# begin training
train_steps = args.num_iterations
loss_record = defaultdict(list)
test_loss_record = defaultdict(list)
early_stop = False
train_vocab_util = 1.0
curr_K = args.max_K if args.coarse_to_fine_search else args.K

for step in range(train_steps + 1):
    last_step = (step == train_steps) or early_stop
    if args.no_attn_sweep: 
        attn_blocksize = torch.tensor(1792, dtype=torch.int, device='cuda')
    else: 
        attn_blocksize = torch.tensor(64*((step/train_steps * (1792 - 64) + 64)//64), dtype=torch.int, device='cuda')    
    if args.use_static_memory_span:
        memory_span = torch.tensor(1792, dtype=torch.int, device='cuda') # keep static
    else:
        memory_span = torch.tensor(64*(((1 - step/train_steps) * (1792 - args.min_memory_span) + args.min_memory_span)//64), dtype=torch.int, device='cuda')
    
    # --- coarse to fine search ---
    K = get_current_K(step, train_steps, train_vocab_util, curr_K)

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
                tokens = next(val_loader)

                val_tokens, greedy_adv, greedy_ppt, greedy_abs_ppt = sorl_evaluate(tokens, model, n=args.num_rollouts_val, K=K, max_iterations=args.max_iterations, 
                                                                    memory_span=memory_span, attn_blocksize=attn_blocksize, 
                                                                    temperature=temperature_val)

                search_tokens, search_ppt, search_adv, _ = sorl_search(tokens, model, n=args.num_rollouts_val, K=K, max_iterations=args.max_iterations, 
                                                                memory_span=memory_span, attn_blocksize=attn_blocksize, 
                                                                temperature=temperature_val)

                greedy_util_rate = compute_vocab_utilization_rate(val_tokens, model)
                search_util_rate = compute_vocab_utilization_rate(search_tokens, model)

                # --- this base loss unfairly includes many zero ppt on BOS token position ---
                base_loss = model.forward(tokens, memory_span, attn_blocksize)[0].mean()

                greedy_info_loss, greedy_abs_loss, greedy_zipf_loss = loss_fn(val_tokens, model, base_loss.detach(), memory_span, attn_blocksize)
                search_info_loss, search_abs_loss, search_zipf_loss = loss_fn(search_tokens, model, base_loss.detach(), memory_span, attn_blocksize)
                
                greedy_traj_loss = greedy_info_loss + base_loss 
                search_traj_loss = search_info_loss + base_loss 
                greedy_rel_info_gain = -greedy_info_loss / base_loss 
                search_rel_info_gain = -search_info_loss / base_loss 

                val_loss["base_traj_loss"] += base_loss.mean()
                val_loss["cond_traj_loss (greedy)"] += greedy_traj_loss.mean()
                val_loss["cond_traj_loss (search)"] += search_traj_loss.mean()
                val_loss["abs_loss (greedy)"] += greedy_abs_loss.mean()
                val_loss["abs_loss (search)"] += search_abs_loss.mean()
                val_loss["greedy_adv"] += greedy_adv.mean()
                val_loss["info_gain (greedy)"] += greedy_rel_info_gain.mean()
                val_loss["info_gain (search)"] += search_rel_info_gain.mean()
                val_loss["util_rate (greedy)"] += torch.tensor(greedy_util_rate, device=greedy_traj_loss.device)
                val_loss["util_rate (search)"] += torch.tensor(search_util_rate, device=search_traj_loss.device)
                val_loss["K"] += torch.tensor(float(K), device="cuda")  # Set (not +=), K is same for all val steps
            
        for name in val_loss: 
            val_loss[name] /= val_steps
            dist.all_reduce(val_loss[name], op=dist.ReduceOp.AVG)            
            loss_record[name].append(val_loss[name])

        del val_loader       
        val_info = " ".join([f"{item} loss: {value.item():.4f}" for (item, value) in val_loss.items()])    
        print0(f"step:{step}/{train_steps} {val_info} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)
        model.train()
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    if last_step:
        if master_process: 
            os.makedirs(f"logs/{run_id}", exist_ok=True)

            if args.save_checkpoint:
                log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
                os.makedirs(f"logs/{run_id}", exist_ok=True)
                torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt") 
        # the last step only has the validation loop, so break to avoid training
        break            
            
    # --------------- TRAINING SECTION -----------------
    for accum_step in range(train_accumulation_steps): 
        tokens = next(train_loader)
        with torch.no_grad(): 
            search_tokens, search_ppt, search_adv, rew = sorl_search(tokens, model, n=args.num_rollouts, K=K, max_iterations=args.max_iterations, 
                                                                memory_span=memory_span, attn_blocksize=attn_blocksize, 
                                                                temperature=temperature_train,
                                                                )

        # --- compute loss --- 
        base_loss = model.forward(tokens, memory_span, attn_blocksize)[0].mean()
        info_loss, abs_loss, zipf_loss = loss_fn(search_tokens, model, base_loss.detach(), memory_span, attn_blocksize)
        loss = base_loss + args.alpha_info_gain * info_loss + args.alpha_loss * abs_loss + args.alpha_marg_ent * zipf_loss
        
        loss.backward()
        print0(f" - step: {step} | accum step: {accum_step} | base loss: {base_loss.item()} | info loss: {info_loss.item()} | abs loss: {abs_loss.item()} | zipf loss: {zipf_loss.item()}")
        
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

    # log train vocab util (for dynamic K curriculum halting)
    train_vocab_util = compute_vocab_utilization_rate(search_tokens, model)
            
    # ----------------------------------------------------
    # logging
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)

print0(f"Experiment configuration: {args.run_info}\n", console=True)
print0(f"-- batch_size: {args.batch_size}", console=True)
print0(f"-- train_seq_len: {args.train_seq_len}", console=True)
print0(f"-- val_seq_len: {args.val_seq_len}", console=True)
print0(f"-- num_iterations: {args.num_iterations}", console=True)
print0(f"-- num_rollouts: {args.num_rollouts}", console=True)
print0(f"-- K: {args.K}", console=True)
print0(f"-- max_iterations: {args.max_iterations}", console=True)
print0(f"-- temperature: {args.temperature}", console=True)
print0(f"-- min_temperature: {args.min_temperature}", console=True)
print0(f"-- use_static_memory_span: {args.use_static_memory_span}", console=True)
print0(f"-- min_memory_span: {args.min_memory_span}", console=True)
print0(f"-- abstract_vocab_size: {args.abstract_vocab_size}", console=True)
print0(f"-- alpha_select: {args.alpha_select}", console=True)
print0(f"-- alpha_loss: {args.alpha_loss}", console=True)
print0(f"-- alpha_marg_ent: {args.alpha_marg_ent}", console=True)
print0(f"-- decay: {args.decay}", console=True)
print0(f"-- target_vocab_util: {args.target_vocab_util}", console=True)
print0(f"-- alpha_info_gain: {args.alpha_info_gain}", console=True)
print0(f"-- coarse_to_fine_search: {args.coarse_to_fine_search}", console=True)
print0(f"-- max_K: {args.max_K}", console=True)
print0(f"-- num_c2f_phases: {args.num_c2f_phases}", console=True)
print0(f"-- no_attn_sweep: {args.no_attn_sweep}", console=True)
print0(f"loss record:\n{loss_record}", console=True)

dist.destroy_process_group()