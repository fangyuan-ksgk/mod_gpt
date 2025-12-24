"""
Evaluation script for SORL on TinyStories validation set
Computes: base_loss, greedy_loss, search_loss, info_gain, search_advantage
GPU-accelerated with distributed support.

Usage:
    Single GPU:  python eval_sorl_slow.py --hf_repo_id ... --hf_filename ...
    Multi-GPU:   torchrun --nproc_per_node=4 eval_sorl_slow.py --hf_repo_id ... --hf_filename ...
"""

import torch
import torch.distributed as dist
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import os
from huggingface_hub import login, hf_hub_download

from sorl.gat_sim import GAT, GATConfig, BOS_TOKEN_ID
from src.model import GPTConfig, GPT
from sorl.neo_utils import sorl_rollout_v3, select_best_info_gain
from sorl.eval import compute_vocab_utilization_rate
from data.tinystory_local import TinyStoriesDataLoader


# --- Distributed utilities ---
def setup_distributed():
    """Initialize distributed training if available"""
    if 'RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = f"cuda:{rank}"
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
    parser = argparse.ArgumentParser(description="Evaluate SORL on TinyStories Validation")
    
    # Model
    parser.add_argument("--model_size", type=str, default="small")
    parser.add_argument("--abstract_vocab_size", type=int, default=128)
    parser.add_argument("--hf_repo_id", type=str, required=True, help="Hugging Face repo ID")
    parser.add_argument("--hf_filename", type=str, required=True, help="Hugging Face filename (sorl model)")
    parser.add_argument("--hf_filename_base", type=str, required=True, help="Hugging Face filename (base model)")
    parser.add_argument("--use_compile", action="store_true", default=True)
    
    # Data
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation"])
    parser.add_argument("--num_stories", type=int, default=None, help="Number of stories to evaluate (None=all)")
    parser.add_argument("--max_len", type=int, default=1792, help="Max tokens per story")
    parser.add_argument("--batch_size", type=int, default=8)
    
    # SORL config
    parser.add_argument("--num_rollouts", type=int, default=5)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--max_iterations", type=int, default=2)
    parser.add_argument("--min_temperature", type=float, default=0.0)
    parser.add_argument("--max_temperature", type=float, default=5.0)
    
    # Output
    parser.add_argument("--save_path", type=str, default=None, help="Save results to CSV")
    
    return parser.parse_args()


def load_model(hf_repo_id, hf_filename, model_size, abstract_vocab_size, device, use_compile):
    """Load model from checkpoint (handles torch.compile prefix)"""
    gat_config = GATConfig.gpt_size(model_size, vocab_sizes=[BOS_TOKEN_ID + 1, abstract_vocab_size])
    model = GAT(gat_config).to(device)
    
    ckpt_path = hf_hub_download(repo_id=hf_repo_id, filename=hf_filename)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    
    # Strip '_orig_mod.' prefix if present (from torch.compile)
    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)

    if use_compile:
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

def compute_abs_stats_v2(tokens, model, base_model, n, K, max_iterations, memory_span, attn_blocksize, temperature, truncate_seq_len):
    """Compute comprehensive abstraction statistics"""
    
    search_data, search_ppt, abs_logits = sorl_rollout_v3(
        tokens, model, n=n, K=K,
        max_iterations=max_iterations,
        memory_span=memory_span,
        attn_blocksize=attn_blocksize,
        temperature=temperature,
        truncate_seq_len=truncate_seq_len
    )
    search_ppt = search_ppt.reshape(search_data.shape[0], -1)

    # --- greedy & random rollout ppt ---
    cond_traj_mask = (search_data[0, 1:] < model.vocab_sizes[0])
    greedy_traj_ppt = search_ppt[0][cond_traj_mask]
    random_traj_ppt = search_ppt[1:].mean(dim=0)[cond_traj_mask]

    # --- base traj loss ---
    base_traj_ppt, _ = base_model.forward(tokens, memory_span, attn_blocksize)
    base_traj_mask = (tokens[:, 1:] != BOS_TOKEN_ID).float()
    base_traj_loss = (base_traj_ppt * base_traj_mask[0]).sum() / base_traj_mask[0].sum().clamp(min=1)

    # --- search (per-doc-best rollout) info gain ---
    levels = (search_data >= model.vocab_sizes[0]).long()
    best_data, best_ppt, _, _ = select_best_info_gain(tokens, base_traj_ppt, search_data, search_ppt, levels)
    best_traj_ppt = best_ppt[cond_traj_mask]

    # --- information gain, search advantage (relative & absolute) ---
    # Align lengths for info gain calculation
    min_len = min(len(base_traj_ppt), len(greedy_traj_ppt), len(best_traj_ppt))
    base_aligned = base_traj_ppt[:min_len]
    greedy_aligned = greedy_traj_ppt[:min_len]
    best_aligned = best_traj_ppt[:min_len]
    random_aligned = random_traj_ppt[:min_len]

    greedy_adv = ((random_aligned - greedy_aligned) / (random_aligned + 1e-8)).mean()
    greedy_abs_adv = (random_aligned - greedy_aligned).mean()
    search_adv = ((random_aligned - best_aligned) / (random_aligned + 1e-8)).mean()
    search_abs_adv = (random_aligned - best_aligned).mean()
    greedy_info_gain = (base_aligned - greedy_aligned).mean()
    search_info_gain = (base_aligned - best_aligned).mean()

    greedy_traj_loss = greedy_traj_ppt.mean()
    search_traj_loss = best_traj_ppt.mean()
    
    # Vocab utilization
    greedy_util = compute_vocab_utilization_rate(search_data[:1], model)
    search_util = compute_vocab_utilization_rate(best_data, model)

    return {
        'base_traj_loss': base_traj_loss,
        'greedy_traj_loss': greedy_traj_loss,
        'search_traj_loss': search_traj_loss,
        'greedy_adv': greedy_adv,
        'greedy_abs_adv': greedy_abs_adv,
        'search_adv': search_adv,
        'search_abs_adv': search_abs_adv,
        'greedy_info_gain': greedy_info_gain,
        'search_info_gain': search_info_gain,
        'greedy_util': greedy_util,
        'search_util': search_util,
    }


def main():
    args = parse_args()
    rank, world_size, device = setup_distributed()
    
    # Login from rank 0 only, then sync
    if rank == 0:
        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            login(token=hf_token)
    if dist.is_initialized():
        dist.barrier()
    
    print0("=" * 70)
    print0(f"SORL Distributed Evaluation ({world_size} GPU{'s' if world_size > 1 else ''})")
    print0(f"Split: {args.split}")
    print0(f"Model: {args.model_size}, Checkpoint: {args.hf_repo_id}/{args.hf_filename}")
    print0(f"K={args.K}, n={args.num_rollouts}, max_iter={args.max_iterations}")
    print0(f"Compile: {args.use_compile}")
    print0("=" * 70)
    
    # Load model on each rank
    model = load_model(args.hf_repo_id, args.hf_filename, args.model_size, 
                       args.abstract_vocab_size, device, args.use_compile)

    base_model = load_base_model(args.hf_repo_id, args.hf_filename_base, 
                                 args.model_size, device, use_compile=args.use_compile)
                                 
    truncate_seq_len = not args.use_compile

    # Load data (each rank loads full dataset, but processes different shards)
    loader = TinyStoriesDataLoader(
        num_stories=args.num_stories,
        max_len=args.max_len,
        chunk_size=args.K,
        device=device,
        split=args.split
    )
    
    # Shard indices across ranks (interleaved for load balance)
    all_indices = list(range(len(loader.stories)))
    local_indices = all_indices[rank::world_size]
    
    # Setup
    memory_span = 2 * args.max_len + 2
    attn_blocksize = 1792
    temperature = torch.tensor(
        [args.min_temperature] + [args.max_temperature] * (args.num_rollouts - 1),
        device=device
    )
    
    # Local results accumulation
    local_results = defaultdict(list)
    num_local_batches = (len(local_indices) + args.batch_size - 1) // args.batch_size
    
    print0(f"\nTotal stories: {len(loader.stories)}, per rank: {len(local_indices)}")
    print0(f"Batches per rank: {num_local_batches}")
    
    # Progress bar only on rank 0
    pbar = tqdm(range(num_local_batches), desc=f"Evaluating", disable=(rank != 0))
    
    with torch.no_grad():
        for batch_idx in pbar:
            start = batch_idx * args.batch_size
            end = min(start + args.batch_size, len(local_indices))
            batch_indices = torch.tensor(local_indices[start:end])
            
            tokens, doc_ids = loader.get_specific(batch_indices)
            
            stats = compute_abs_stats_v2(
                tokens, model, base_model,
                n=args.num_rollouts,
                K=args.K,
                max_iterations=args.max_iterations,
                memory_span=memory_span,
                attn_blocksize=attn_blocksize,
                temperature=temperature,
                truncate_seq_len=truncate_seq_len
            )
            
            for key, val in stats.items():
                v = val.item() if torch.is_tensor(val) else val
                local_results[key].append(v)
    
    # --- All-reduce results across ranks ---
    if dist.is_initialized():
        dist.barrier()
    
    summary = {}
    for key, vals in local_results.items():
        local_sum = torch.tensor(sum(vals), device=device, dtype=torch.float64)
        local_count = torch.tensor(len(vals), device=device, dtype=torch.float64)
        
        if dist.is_initialized():
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
        
        summary[key] = (local_sum / local_count).item()
    
    # Print results from rank 0
    print0("\n" + "=" * 70)
    print0("RESULTS")
    print0("=" * 70)
    
    for key, val in summary.items():
        print0(f"{key}: {val:.6f}")
    
    # Key insights
    print0("\n" + "-" * 70)
    print0("KEY INSIGHTS:")
    print0(f"  Base PPL:           {summary['base_traj_loss']:.4f}")
    print0(f"  Greedy PPL:         {summary['greedy_traj_loss']:.4f} (Δ = {(summary['base_traj_loss'] - summary['greedy_traj_loss']):.4f})")
    print0(f"  Search PPL:         {summary['search_traj_loss']:.4f} (Δ = {(summary['base_traj_loss'] - summary['search_traj_loss']):.4f})")
    print0(f"  Greedy Info Gain:   {summary['greedy_info_gain']*100:.2f}%")
    print0(f"  Search Info Gain:   {summary['search_info_gain']*100:.2f}%")
    print0(f"  Greedy vs Random:   {summary['greedy_adv']*100:.2f}% (relative)")
    print0(f"  Search vs Random:   {summary['search_adv']*100:.2f}% (relative)")
    print0("-" * 70)
    
    # Save results from rank 0 only
    if rank == 0 and args.save_path:
        import pandas as pd
        
        # For distributed, we only have aggregated summary, not per-story results
        # Save summary as CSV
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(args.save_path, index=False)
        print0(f"\nSummary saved to: {args.save_path}")
        
        # Also save as text
        summary_path = args.save_path.replace('.csv', '_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"SORL Evaluation Results\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"Split: {args.split}\n")
            f.write(f"Stories: {len(loader.stories)}\n")
            f.write(f"GPUs: {world_size}\n")
            f.write(f"Model: {args.hf_repo_id}/{args.hf_filename}\n")
            f.write(f"=" * 50 + "\n\n")
            for key, val in summary.items():
                f.write(f"{key}: {val:.6f}\n")
        print0(f"Summary saved to: {summary_path}")
    
    cleanup_distributed()
    return summary


if __name__ == "__main__":
    main()
