
import os
import sys
import argparse
import torch
from torch import nn
import torch.distributed as dist
from collections import defaultdict
import time

# -----------------------------------------------------------------------------
# Imports from project
# -----------------------------------------------------------------------------
from sorl.gat_sim import GAT, GATConfig
from sorl.neo_utils import sorl_search_v8 as sorl_search
from sorl.eval import compute_vocab_utilization_rate
from src.utils import distributed_data_generator_sorl as distributed_data_generator
from sorl.info import SoRLLoss_v8, SoRLLoss_v7

# -----------------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    
    # Model & Data
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--val_files", type=str, default="data/tinystories/tinystory_val_*.bin")
    parser.add_argument("--model_size", type=str, default="small")
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--abstract_vocab_size", type=int, default=16) # default from run_sorl.sh
    
    # Search probing
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 8, 16], help="List of rollouts to test")
    parser.add_argument("--val_tokens", type=int, default=10485760, help="Total number of validation tokens")
    parser.add_argument("--val_seq_len", type=int, default=2*1024) # smaller seq len for faster eval
    
    # Search params
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--max_iterations", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=5.0)
    parser.add_argument("--min_temperature", type=float, default=0.0)
    parser.add_argument("--use_static_memory_span", action="store_true", default=True)
    parser.add_argument("--min_memory_span", type=int, default=64)

    # Loss params (for metric calculation)
    parser.add_argument("--decay", type=float, default=0.8)
    parser.add_argument("--target_vocab_util", type=float, default=0.8)

    return parser.parse_args()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    
    # Setup distributed (or single device)
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
        master_process = (rank == 0)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        master_process = True
        print("Running in single process mode")

    # Load Model
    if "40" in torch.cuda.get_device_properties(device).name:
        flex_options = {
            "BLOCK_M": 32, "BLOCK_N": 32,
            "BLOCK_M1": 32, "BLOCK_N1": 64, "BLOCK_M2": 64, "BLOCK_N2": 32
        }
    else:
        flex_options = {
            "BLOCK_M": 64, "BLOCK_N": 64,
            "BLOCK_M1": 32, "BLOCK_N1": 64, "BLOCK_M2": 64, "BLOCK_N2": 32
        }
        
    model_config = GATConfig.gpt_size(
        args.model_size,
        vocab_sizes=[args.vocab_size, args.abstract_vocab_size],
        flex_kernel_options=flex_options
    )
    
    model = GAT(model_config).to(device)
    
    # Load Checkpoint
    print(f"Loading checkpoint from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint # handle raw state dict
        
    # Fix keys if needed (e.g. _orig_mod prefix from torch.compile)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.eval()
    # model = torch.compile(model) # Optional: compile for speed

    # Loss function for metrics
    loss_fn = SoRLLoss_v7(model.vocab_sizes[1], decay=args.decay, target_vocab_util=args.target_vocab_util).to(device)

    # Data Loader
    val_seq_len = world_size * args.val_seq_len
    assert args.val_tokens % val_seq_len == 0
    val_steps = args.val_tokens // val_seq_len
    val_loader = distributed_data_generator(args.val_files, val_seq_len, rank, world_size)
    
    # Memory parameters
    attn_blocksize = torch.tensor(64, dtype=torch.int, device=device)
    if args.use_static_memory_span:
        memory_span = torch.tensor(1792, dtype=torch.int, device=device)
    else:
        memory_span = torch.tensor(args.min_memory_span, dtype=torch.int, device=device) # Just use min for testing if not static

    # -------------------------------------------------------------------------
    # Probing Loop
    # -------------------------------------------------------------------------
    results = {}
    
    for n in args.budgets:
        if master_process:
            print(f"\nTesting with budget n={n}...")
            
        temperature_val = torch.cat([
            torch.tensor([args.min_temperature], device=device),
            torch.full((n - 1,), args.temperature, device=device)
        ])
        
        metrics = defaultdict(float)

        t0 = time.time()
        
        with torch.no_grad():
            for i in range(val_steps):
                tokens = next(val_loader)
                
                # Run Search
                val_tokens, val_ppt, val_adv, val_rew = sorl_search(
                    tokens, model, n=n, K=args.K, max_iterations=args.max_iterations, 
                    memory_span=memory_span, attn_blocksize=attn_blocksize, 
                    temperature=temperature_val
                )

                # Compute Metrics
                base_loss = model.forward(tokens, memory_span, attn_blocksize)[0].mean()
                
                info_loss, abs_loss, zipf_loss = loss_fn(val_tokens, model, base_loss.detach(), memory_span, attn_blocksize)

                val_traj_loss = info_loss + base_loss 
                rel_info_gain = -info_loss / base_loss
                util_rate = compute_vocab_utilization_rate(val_tokens, model)

                # Accumulate
                metrics["base_loss"] += base_loss.item()
                metrics["search_loss"] += val_traj_loss.item()
                metrics["search_adv"] += val_adv.mean().item()
                metrics["info_gain"] += rel_info_gain.item()
                metrics["util_rate"] += util_rate
                
                if master_process and i % 10 == 0:
                     print(f"  step {i}/{val_steps}", end="\r")

        # Average and Gather
        for k in metrics:
            metrics[k] /= val_steps
            # Simple average across ranks (approximate for util_rate but fine)
            val_tensor = torch.tensor(metrics[k], device=device)
            if dist.is_initialized():
                dist.all_reduce(val_tensor, op=dist.ReduceOp.AVG)
            metrics[k] = val_tensor.item()
            
        results[n] = metrics
        
        if master_process:
            elapsed = time.time() - t0
            print(f"Budget n={n} | Time: {elapsed:.2f}s")
            print(f"  Base Loss:    {metrics['base_loss']:.4f}")
            print(f"  Search Loss:  {metrics['search_loss']:.4f}")
            print(f"  Search Adv:   {metrics['search_adv']:.4f}")
            print(f"  Info Gain:    {metrics['info_gain']:.4f}")
            print(f"  Util Rate:    {metrics['util_rate']:.4f}")

    if master_process:
        print("\n=== Final Summary ===")
        print(f"{'Budget':<8} {'Search Adv':<12} {'Info Gain':<12} {'Search Loss':<12}")
        for n in args.budgets:
            m = results[n]
            print(f"{n:<8} {m['search_adv']:<12.4f} {m['info_gain']:<12.4f} {m['search_loss']:<12.4f}")

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()

