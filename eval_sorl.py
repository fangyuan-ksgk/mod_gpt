"""
Evaluation script for SORL on TinyStories validation set
Computes: base_loss, greedy_loss, search_loss, info_gain, search_advantage
GPU-accelerated, no torch.compile (to handle variable-length sequences)
"""

import torch
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import os
from huggingface_hub import login, hf_hub_download

from sorl.gat_sim import GAT, GATConfig, BOS_TOKEN_ID
from sorl.neo_utils import sorl_rollout_v3, select_best_info_gain
from sorl.eval import compute_vocab_utilization_rate
from data.tinystory_local import TinyStoriesDataLoader

login(token=os.environ["HF_TOKEN"])

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SORL on TinyStories Validation")
    
    # Model
    parser.add_argument("--model_size", type=str, default="small")
    parser.add_argument("--abstract_vocab_size", type=int, default=128)
    parser.add_argument("--hf_repo_id", type=str, required=True, help="Hugging Face repo ID")
    parser.add_argument("--hf_filename", type=str, required=True, help="Hugging Face filename")
    
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


def load_model(hf_repo_id, hf_filename, model_size, abstract_vocab_size, device):
    """Load model from checkpoint (handles torch.compile prefix)"""
    gat_config = GATConfig.gpt_size(model_size, vocab_sizes=[BOS_TOKEN_ID + 1, abstract_vocab_size])
    model = GAT(gat_config).to(device)
    
    ckpt_path = hf_hub_download(repo_id=hf_repo_id, filename=hf_filename)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    
    # Strip '_orig_mod.' prefix if present (from torch.compile)
    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    
    model.eval()
    return model

def compute_abs_stats_v2(tokens, model, n, K, max_iterations, memory_span, attn_blocksize, temperature, truncate_seq_len):
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
    base_traj_ppt, _ = model.forward(tokens, memory_span, attn_blocksize)
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 70)
    print(f"SORL Evaluation on TinyStories {args.split}")
    print(f"Model: {args.model_size}, Checkpoint: {args.hf_repo_id}/{args.hf_filename}")
    print(f"K={args.K}, n={args.num_rollouts}, max_iter={args.max_iterations}")
    print(f"Device: {device}")
    print("=" * 70)
    
    # Load model (no torch.compile!)
    model = load_model(args.hf_repo_id, args.hf_filename, args.model_size, args.abstract_vocab_size, device)
    
    # Load data
    loader = TinyStoriesDataLoader(
        num_stories=args.num_stories,
        max_len=args.max_len,
        chunk_size=args.K,
        device=device,
        split=args.split
    )
    
    # Setup
    memory_span = 2 * args.max_len + 2
    attn_blocksize = 1792
    temperature = torch.tensor(
        [args.min_temperature] + [args.max_temperature] * (args.num_rollouts - 1),
        device=device
    )
    
    # Accumulate results
    results = defaultdict(list)
    num_batches = (len(loader.stories) + args.batch_size - 1) // args.batch_size
    
    print(f"\nEvaluating {len(loader.stories)} stories in {num_batches} batches...")
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Evaluating"):
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, len(loader.stories))
            batch_indices = torch.arange(start_idx, end_idx)
            
            tokens, doc_ids = loader.get_specific(batch_indices)
            
            stats = compute_abs_stats_v2(
                tokens, model,
                n=args.num_rollouts,
                K=args.K,
                max_iterations=args.max_iterations,
                memory_span=memory_span,
                attn_blocksize=attn_blocksize,
                temperature=temperature,
                truncate_seq_len=False
            )
            
            for key, val in stats.items():
                if torch.is_tensor(val):
                    results[key].append(val.item())
                else:
                    results[key].append(val)
    
    # Compute means
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    summary = {}
    for key, vals in results.items():
        mean_val = sum(vals) / len(vals)
        summary[key] = mean_val
        print(f"{key}: {mean_val:.6f}")
    
    # Key insights
    print("\n" + "-" * 70)
    print("KEY INSIGHTS:")
    print(f"  Base PPL:           {summary['base_traj_loss']:.4f}")
    print(f"  Greedy PPL:         {summary['greedy_traj_loss']:.4f} (Δ = {(summary['base_traj_loss'] - summary['greedy_traj_loss']):.4f})")
    print(f"  Search PPL:         {summary['search_traj_loss']:.4f} (Δ = {(summary['base_traj_loss'] - summary['search_traj_loss']):.4f})")
    print(f"  Greedy Info Gain:   {summary['greedy_info_gain']*100:.2f}%")
    print(f"  Search Info Gain:   {summary['search_info_gain']*100:.2f}%")
    print(f"  Greedy vs Random:   {summary['greedy_adv']*100:.2f}% (relative)")
    print(f"  Search vs Random:   {summary['search_adv']*100:.2f}% (relative)")
    print("-" * 70)
    
    # Save results
    if args.save_path:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(args.save_path, index=False)
        print(f"\nResults saved to: {args.save_path}")
        
        # Also save summary
        summary_path = args.save_path.replace('.csv', '_summary.txt')
        with open(summary_path, 'w') as f:
            for key, val in summary.items():
                f.write(f"{key}: {val:.6f}\n")
        print(f"Summary saved to: {summary_path}")
    
    return summary


if __name__ == "__main__":
    main()