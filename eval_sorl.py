"""
Fast evaluation script for SORL on TinyStories validation set
Uses compiled model + binary data generator for speed
Supports distributed runs: torchrun --nproc_per_node=N eval_sorl.py ...
"""

import torch
import torch.distributed as dist
import argparse
import os
import time
from collections import defaultdict
from pathlib import Path
import glob
import itertools

from huggingface_hub import login, hf_hub_download

from sorl.gat_sim import GAT, GATConfig, BOS_TOKEN_ID
from src.model import GPTConfig, GPT
from sorl.neo_utils import sorl_search_v8 as sorl_search, sorl_evaluate
from sorl.eval import compute_vocab_utilization_rate
from sorl.info import SoRLLoss_v7
from src.utils import distributed_data_generator_sorl, distributed_data_generator_sorl_v3


# --- Distributed utilities ---
def setup_distributed():
    """Initialize distributed training if available"""
    if 'RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ['LOCAL_RANK'])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        rank, world_size = 0, 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return rank, world_size, device


def cleanup_distributed():
    """Cleanup distributed process group"""
    if dist.is_initialized():
        dist.destroy_process_group()


def print0(*args, **kwargs):
    """Print only from rank 0"""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="Fast SORL Evaluation")
    
    # Model
    parser.add_argument("--model_size", type=str, default="small")
    parser.add_argument("--abstract_vocab_size", type=int, default=128)
    parser.add_argument("--hf_repo_id", type=str, required=True)
    parser.add_argument("--hf_filename", type=str, required=True, help="Hugging Face filename (sorl model)")
    parser.add_argument("--hf_filename_base", type=str, required=True, help="Hugging Face filename (base GPT-2 model)")
    
    # Data
    parser.add_argument("--val_files", type=str, default="data/tinystories/tinystory_val_*.bin")
    parser.add_argument("--val_tokens", type=int, default=10485760, help="Total tokens to evaluate")
    parser.add_argument("--val_seq_len", type=int, default=16*1024, help="Sequence length per batch")
    
    # SORL config
    parser.add_argument("--num_rollouts", type=int, default=2)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--max_iterations", type=int, default=2)
    parser.add_argument("--min_temperature", type=float, default=0.0)
    parser.add_argument("--max_temperature", type=float, default=5.0)
    
    # Speed options
    parser.add_argument("--use_compile", action="store_true", default=True)
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--avoid_prefix_truncation", action="store_true", default=True)
    
    # Output
    parser.add_argument("--save_path", type=str, default=None)
    
    return parser.parse_args()


def load_model(hf_repo_id, hf_filename, model_size, abstract_vocab_size, device, use_compile=True):
    """Load model from checkpoint"""
    gat_config = GATConfig.gpt_size(
        model_size,
        vocab_sizes=[BOS_TOKEN_ID + 1, abstract_vocab_size],
        flex_kernel_options={
            "BLOCK_M": 64, "BLOCK_N": 64,
            "BLOCK_M1": 32, "BLOCK_N1": 64, "BLOCK_M2": 64, "BLOCK_N2": 32
        }
    )
    model = GAT(gat_config).to(device)
    
    ckpt_path = hf_hub_download(repo_id=hf_repo_id, filename=hf_filename)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    
    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    
    if "cuda" in device and use_compile:
        model = torch.compile(model, dynamic=True)
    
    model.eval()
    return model 

def load_base_model(hf_repo_id, hf_filename, model_size, device, use_compile=True):
    gpt_config = GPTConfig.gpt_size(
        model_size, 
        flex_kernel_options={
            "BLOCK_M": 64, "BLOCK_N": 64,
            "BLOCK_M1": 32, "BLOCK_N1": 64, "BLOCK_M2": 64, "BLOCK_N2": 32
        }
    )
    model = GPT(gpt_config).to(device)

    ckpt_path = hf_hub_download(repo_id=hf_repo_id, filename=hf_filename)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    
    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    
    if "cuda" in device and use_compile:
        model = torch.compile(model, dynamic=True)
    
    model.eval()
    return model

def main():
    args = parse_args()
    rank, world_size, device = setup_distributed()
    use_compile = args.use_compile and not args.no_compile
    
    # Login from rank 0 only
    if rank == 0 and "HF_TOKEN" in os.environ:
        login(token=os.environ["HF_TOKEN"])
    if dist.is_initialized():
        dist.barrier()
    
    print0("=" * 70)
    print0(f"Fast SORL Evaluation ({world_size} GPU{'s' if world_size > 1 else ''})")
    print0(f"Model: {args.model_size}, Checkpoint: {args.hf_repo_id}/{args.hf_filename}")
    print0(f"K={args.K}, n={args.num_rollouts}, max_iter={args.max_iterations}")
    print0(f"Val tokens: {args.val_tokens:,}, Seq len: {args.val_seq_len:,}")
    print0(f"Compiled: {use_compile}, Avoid prefix truncation: {args.avoid_prefix_truncation}")
    print0("=" * 70)
    
    # Set precision
    torch.set_float32_matmul_precision('high')
    
    # Load model
    model = load_model(args.hf_repo_id, args.hf_filename, args.model_size, 
                       args.abstract_vocab_size, device, use_compile=use_compile)

    base_model = load_base_model(args.hf_repo_id_base, args.hf_filename_base, 
                                 args.model_size, device, use_compile=use_compile)

    # Loss function (expects tensor for vocab_size to infer device)
    loss_fn = SoRLLoss_v7(torch.tensor(args.abstract_vocab_size, device=device), decay=0.8, target_vocab_util=0.8)
    
    # Setup - memory_span should accommodate sequence length
    memory_span = torch.tensor(2 * args.val_seq_len + 2, dtype=torch.int, device=device)
    attn_blocksize = torch.tensor(1792, dtype=torch.int, device=device)
    temperature = torch.tensor(
        [args.min_temperature] + [args.max_temperature] * (args.num_rollouts - 1),
        device=device
    )
    
    # Data generator - each rank gets different shard
    if args.avoid_prefix_truncation: 
        val_loader = distributed_data_generator_sorl_v3(args.val_files, args.val_seq_len, rank, world_size)
    else: 
        val_loader = distributed_data_generator_sorl(args.val_files, args.val_seq_len, rank, world_size)

    # Each rank processes its share of steps
    total_steps = args.val_tokens // args.val_seq_len
    local_steps = total_steps // world_size
    
    print0(f"\nTotal steps: {total_steps}, per rank: {local_steps}")
    
    # Warmup (for compiled model)
    if use_compile:
        print0("Warming up compiled model...")
        warmup_tokens = torch.randint(0, BOS_TOKEN_ID, (1, args.val_seq_len), device=device, dtype=torch.int32)
        with torch.no_grad():
            for _ in range(3):
                _ = model.forward(warmup_tokens, memory_span, attn_blocksize)
                _ = base_model.forward(warmup_tokens, memory_span, attn_blocksize)
        if dist.is_initialized():
            dist.barrier()
        print0("Warmup complete.")
    
    # Local evaluation
    local_loss = defaultdict(float)
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(local_steps):
            tokens = next(val_loader)
            
            # Greedy evaluation
            val_tokens, greedy_adv, greedy_ppt, greedy_abs_ppt = sorl_evaluate(
                tokens, model, n=args.num_rollouts, K=args.K, 
                max_iterations=args.max_iterations,
                memory_span=memory_span, attn_blocksize=attn_blocksize,
                temperature=temperature
            )
            
            # Search evaluation
            search_tokens, search_ppt, search_adv, _ = sorl_search(
                tokens, model, n=args.num_rollouts, K=args.K,
                max_iterations=args.max_iterations,
                memory_span=memory_span, attn_blocksize=attn_blocksize,
                temperature=temperature
            )
            
            # Utility rates
            greedy_util_rate = compute_vocab_utilization_rate(val_tokens, model)
            search_util_rate = compute_vocab_utilization_rate(search_tokens, model)
            
            # Base loss
            # base_loss = model.forward(tokens, memory_span, attn_blocksize)[0].mean()
            base_loss = base_model.forward(tokens, memory_span, attn_blocksize)[0].mean()
            
            # Info gain losses
            greedy_info_loss, greedy_abs_loss, _ = loss_fn(val_tokens, model, base_loss.detach(), memory_span, attn_blocksize)
            search_info_loss, search_abs_loss, _ = loss_fn(search_tokens, model, base_loss.detach(), memory_span, attn_blocksize)
            
            greedy_traj_loss = greedy_info_loss + base_loss
            search_traj_loss = search_info_loss + base_loss
            greedy_rel_info_gain = -greedy_info_loss / base_loss
            search_rel_info_gain = -search_info_loss / base_loss
            
            # Accumulate locally
            local_loss["base_traj_loss"] += base_loss.item()
            local_loss["cond_traj_loss (greedy)"] += greedy_traj_loss.item()
            local_loss["cond_traj_loss (search)"] += search_traj_loss.item()
            local_loss["abs_loss (greedy)"] += greedy_abs_loss.item()
            local_loss["abs_loss (search)"] += search_abs_loss.item()
            local_loss["greedy_adv"] += greedy_adv.mean().item()
            local_loss["search_adv"] += search_adv.mean().item()
            local_loss["info_gain (greedy)"] += greedy_rel_info_gain.item()
            local_loss["info_gain (search)"] += search_rel_info_gain.item()
            local_loss["util_rate (greedy)"] += greedy_util_rate
            local_loss["util_rate (search)"] += search_util_rate
            
            # Progress (rank 0 only)
            if rank == 0 and ((i + 1) % 10 == 0 or i == 0):
                elapsed = time.time() - start_time
                eta = elapsed / (i + 1) * (local_steps - i - 1)
                print(f"  Step {i+1}/{local_steps} | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
    
    # --- All-reduce results across ranks ---
    if dist.is_initialized():
        dist.barrier()
    
    summary = {}
    for key, val in local_loss.items():
        local_sum = torch.tensor(val, device=device, dtype=torch.float64)
        local_count = torch.tensor(local_steps, device=device, dtype=torch.float64)
        
        if dist.is_initialized():
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
        
        summary[key] = (local_sum / local_count).item()
    
    total_time = time.time() - start_time
    
    # Results (rank 0 only)
    print0("\n" + "=" * 70)
    print0("RESULTS")
    print0("=" * 70)
    
    for key, val in summary.items():
        print0(f"{key}: {val:.6f}")
    
    print0("\n" + "-" * 70)
    print0("KEY INSIGHTS:")
    print0(f"  Base PPL:           {summary['base_traj_loss']:.4f}")
    print0(f"  Greedy PPL:         {summary['cond_traj_loss (greedy)']:.4f}")
    print0(f"  Search PPL:         {summary['cond_traj_loss (search)']:.4f}")
    print0(f"  Greedy Info Gain:   {summary['info_gain (greedy)']*100:.2f}%")
    print0(f"  Search Info Gain:   {summary['info_gain (search)']*100:.2f}%")
    print0(f"  Greedy vs Random:   {summary['greedy_adv']*100:.2f}%")
    print0(f"  Search vs Random:   {summary['search_adv']*100:.2f}%")
    print0("-" * 70)
    print0(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print0(f"Tokens/sec: {args.val_tokens / total_time:,.0f}")
    
    # Save (rank 0 only)
    if rank == 0 and args.save_path:
        import pandas as pd
        df = pd.DataFrame([summary])
        df.to_csv(args.save_path, index=False)
        print0(f"\nResults saved to: {args.save_path}")
    
    cleanup_distributed()
    return summary


if __name__ == "__main__":
    main()
