# Modded gpt speedrun (GPU poor ver. minus Hopper optimization tricks such as FP8 matmul etc.)
# Heavily borrow code from @KellerJordan
import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import time
import copy
import glob
import itertools
import contextlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional
from pathlib import Path

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
# Use of FlexAttention contributed by @KoszarskyB
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
flex_attention = torch.compile(flex_attention, dynamic=False)
create_block_mask = torch.compile(create_block_mask, dynamic=False)

# -----------------------------------------------------------------------------
# Muon optimizer

from src import Muon

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

from src import GPT

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

from src import distributed_data_generator

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data
    train_files : str = "data/fineweb10B/fineweb_train_*.bin" # input .bin to train on
    val_files : str = "data/fineweb10B/fineweb_val_*.bin" # input .bin to eval validation loss on
    val_tokens : int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    train_seq_len : int = 32*1024 # FlexAttention sequence length (per GPU)
    val_seq_len : int = 4*64*1024 # FlexAttention sequence length for validation (per GPU)
    batch_size : int = 8 # Batch size, across all devices
    # optimization
    num_iterations : int = 1750 # number of iterations to run
    cooldown_frac : float = 0.4 # fraction of training spent cooling down the learning rate
    # architecture
    vocab_size : int = 50257
    # evaluation and logging
    val_loss_every : int = 125 # every how many steps to evaluate val loss? 0 for only at the end
    save_checkpoint : bool = False

if len(sys.argv) > 1 and sys.argv[1] == "poor": 
    args = Hyperparameters(batch_size=16, train_seq_len=32*1024, val_seq_len=16*1024)
    model_config = GPTConfig(
        flex_kernel_options={
            "BLOCK_M": 64, "BLOCK_N": 64, # forward
            "BLOCK_M1": 32, "BLOCK_N1": 64, "BLOCK_M2": 64, "BLOCK_N2": 32 # backwards 
        }
    )
else: 
    args = Hyperparameters() # default 8xH100
    model_config = GPTConfig() 
    assert world_size == 8 # this code is designed for 8xH100


# torchrun sets these env variables
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
assert torch.cuda.is_available()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
master_process = (rank == 0) # this process will do logging, checkpointing etc.
assert args.batch_size % (world_size) == 0
train_accumulation_steps = args.batch_size // world_size


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


########################################
#    Construct model and optimizer     #
########################################

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
    assert 0 <= x < 1
    if x < 1 - args.cooldown_frac:
        return 1.0
    else:
        w = (1 - x) / args.cooldown_frac
        return w * 1.0 + (1 - w) * 0.1

model: nn.Module = torch.compile(model, dynamic=False)


########################################
#            Warmup kernels            #
########################################

# Warmup the training kernels, then re-initialize the state so we aren't cheating
warmup_steps = 10
initial_state = dict(model=copy.deepcopy(model.state_dict()),
                     optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers]) # save the initial state
attn_blocksize = torch.tensor(64, dtype=torch.int, device="cuda")
for _ in range(warmup_steps):
    inputs = targets = torch.randint(0, args.vocab_size, size=(args.train_seq_len,), device="cuda")
    model(inputs.to(torch.int32), targets, attn_blocksize).backward()
    for param in model.parameters():
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

train_loader = distributed_data_generator(args.train_files, world_size * args.train_seq_len, rank, world_size)
training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.perf_counter()
# begin training
train_steps = args.num_iterations

for step in range(train_steps + 1):
    last_step = (step == train_steps)
    attn_blocksize = torch.tensor(64*((step/train_steps * (1792 - 64) + 64)//64), dtype=torch.int, device='cuda')

    # --------------- VALIDATION SECTION -----------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        val_seq_len = world_size * args.val_seq_len
        assert args.val_tokens % val_seq_len == 0
        val_steps = args.val_tokens // val_seq_len
        val_loader = distributed_data_generator(args.val_files, val_seq_len, rank, world_size)
        val_loss = 0
        with torch.no_grad():
            for _ in range(val_steps):
                inputs, targets = next(val_loader)
                val_loss += model(inputs, targets, attn_blocksize)
        val_loss /= val_steps
        del val_loader
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)
        model.train()
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    if last_step:
        if master_process and args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            os.makedirs(f"logs/{run_id}", exist_ok=True)
            torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
        # the last step only has the validation loop, so break to avoid training
        break

    # --------------- TRAINING SECTION -----------------
    for _ in range(train_accumulation_steps): 
        inputs, targets = next(train_loader)
        model(inputs, targets, attn_blocksize).backward()    
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
dist.destroy_process_group()