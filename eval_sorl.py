"""
Fast evaluation script for SORL on TinyStories validation set
Uses compiled model + binary data generator for speed
"""

import torch
import argparse
import os
import time
from collections import defaultdict
from pathlib import Path
import glob
import itertools

from huggingface_hub import login, hf_hub_download

from sorl.gat_sim import GAT, GATConfig, BOS_TOKEN_ID
from sorl.neo_utils import sorl_search_v8 as sorl_search, sorl_evaluate
from sorl.eval import compute_vocab_utilization_rate
from sorl.info import SoRLLoss_v7

# Login to HF
if "HF_TOKEN" in os.environ:
    login(token=os.environ["HF_TOKEN"])


def parse_args():
    parser = argparse.ArgumentParser(description="Fast SORL Evaluation")
    
    # Model
    parser.add_argument("--model_size", type=str, default="small")
    parser.add_argument("--abstract_vocab_size", type=int, default=128)
    parser.add_argument("--hf_repo_id", type=str, required=True)
    parser.add_argument("--hf_filename", type=str, required=True)
    
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

from src.utils import distributed_data_generator_sorl, distributed_data_generator_sorl_v3

def load_model(hf_repo_id, hf_filename, model_size, abstract_vocab_size, use_compile=True):
    """Load model from checkpoint"""
    gat_config = GATConfig.gpt_size(
        model_size,
        vocab_sizes=[BOS_TOKEN_ID + 1, abstract_vocab_size],
        flex_kernel_options={
            "BLOCK_M": 64, "BLOCK_N": 64,
            "BLOCK_M1": 32, "BLOCK_N1": 64, "BLOCK_M2": 64, "BLOCK_N2": 32
        }
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GAT(gat_config).to(device)
    
    ckpt_path = hf_hub_download(repo_id=hf_repo_id, filename=hf_filename)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    
    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    
    if device == "cuda" and use_compile:
        model = torch.compile(model, dynamic=True)
    
    model.eval()
    return model


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_compile = args.use_compile and not args.no_compile
    
    # Single GPU mode
    rank, world_size = 0, 1
    
    print("=" * 70)
    print(f"Fast SORL Evaluation")
    print(f"Model: {args.model_size}, Checkpoint: {args.hf_repo_id}/{args.hf_filename}")
    print(f"K={args.K}, n={args.num_rollouts}, max_iter={args.max_iterations}")
    print(f"Val tokens: {args.val_tokens:,}, Seq len: {args.val_seq_len:,}")
    print(f"Compiled: {use_compile}, Avoid prefix truncation: {args.avoid_prefix_truncation}")
    print(f"Device: {device}")
    print("=" * 70)
    
    # Set precision
    torch.set_float32_matmul_precision('high')
    
    # Load model
    model = load_model(args.hf_repo_id, args.hf_filename, args.model_size, 
                       args.abstract_vocab_size, use_compile=use_compile)
    
    # Loss function (expects tensor for vocab_size to infer device)
    loss_fn = SoRLLoss_v7(torch.tensor(args.abstract_vocab_size, device=device), decay=0.8, target_vocab_util=0.8)
    
    # Setup - memory_span should accommodate sequence length
    memory_span = torch.tensor(2 * args.val_seq_len + 2, dtype=torch.int, device=device)
    attn_blocksize = torch.tensor(1792, dtype=torch.int, device=device)
    temperature = torch.tensor(
        [args.min_temperature] + [args.max_temperature] * (args.num_rollouts - 1),
        device=device
    )
    
    # Data generator (rank=0, world_size=1 for single GPU)
    if args.avoid_prefix_truncation: 
        val_loader = distributed_data_generator_sorl_v3(args.val_files, args.val_seq_len, rank, world_size)
    else: 
        val_loader = distributed_data_generator_sorl(args.val_files, args.val_seq_len, rank, world_size)

    val_steps = args.val_tokens // args.val_seq_len
    
    print(f"\nEvaluating {val_steps} batches...")
    
    # Warmup (for compiled model)
    if use_compile:
        print("Warming up compiled model...")
        warmup_tokens = torch.randint(0, BOS_TOKEN_ID, (1, args.val_seq_len), device=device, dtype=torch.int32)
        with torch.no_grad():
            for _ in range(3):
                _ = model.forward(warmup_tokens, memory_span, attn_blocksize)
        print("Warmup complete.")
    
    # Evaluation
    val_loss = defaultdict(float)
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(val_steps):
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
            base_loss = model.forward(tokens, memory_span, attn_blocksize)[0].mean()
            
            # Info gain losses
            greedy_info_loss, greedy_abs_loss, _ = loss_fn(val_tokens, model, base_loss.detach(), memory_span, attn_blocksize)
            search_info_loss, search_abs_loss, _ = loss_fn(search_tokens, model, base_loss.detach(), memory_span, attn_blocksize)
            
            greedy_traj_loss = greedy_info_loss + base_loss
            search_traj_loss = search_info_loss + base_loss
            greedy_rel_info_gain = -greedy_info_loss / base_loss
            search_rel_info_gain = -search_info_loss / base_loss
            
            # Accumulate
            val_loss["base_traj_loss"] += base_loss.item()
            val_loss["cond_traj_loss (greedy)"] += greedy_traj_loss.item()
            val_loss["cond_traj_loss (search)"] += search_traj_loss.item()
            val_loss["abs_loss (greedy)"] += greedy_abs_loss.item()
            val_loss["abs_loss (search)"] += search_abs_loss.item()
            val_loss["greedy_adv"] += greedy_adv.mean().item()
            val_loss["search_adv"] += search_adv.mean().item()
            val_loss["info_gain (greedy)"] += greedy_rel_info_gain.item()
            val_loss["info_gain (search)"] += search_rel_info_gain.item()
            val_loss["util_rate (greedy)"] += greedy_util_rate
            val_loss["util_rate (search)"] += search_util_rate
            
            # Progress
            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - start_time
                eta = elapsed / (i + 1) * (val_steps - i - 1)
                print(f"  Step {i+1}/{val_steps} | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
    
    # Average
    for name in val_loss:
        val_loss[name] /= val_steps
    
    total_time = time.time() - start_time
    
    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    for key, val in val_loss.items():
        print(f"{key}: {val:.6f}")
    
    print("\n" + "-" * 70)
    print("KEY INSIGHTS:")
    print(f"  Base PPL:           {val_loss['base_traj_loss']:.4f}")
    print(f"  Greedy PPL:         {val_loss['cond_traj_loss (greedy)']:.4f}")
    print(f"  Search PPL:         {val_loss['cond_traj_loss (search)']:.4f}")
    print(f"  Greedy Info Gain:   {val_loss['info_gain (greedy)']*100:.2f}%")
    print(f"  Search Info Gain:   {val_loss['info_gain (search)']*100:.2f}%")
    print(f"  Greedy vs Random:   {val_loss['greedy_adv']*100:.2f}%")
    print(f"  Search vs Random:   {val_loss['search_adv']*100:.2f}%")
    print("-" * 70)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Tokens/sec: {args.val_tokens / total_time:,.0f}")
    
    # Save
    if args.save_path:
        import pandas as pd
        df = pd.DataFrame([val_loss])
        df.to_csv(args.save_path, index=False)
        print(f"\nResults saved to: {args.save_path}")
    
    return val_loss


if __name__ == "__main__":
    main()

