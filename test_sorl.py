"""
Test script for SORL on Copy & Paste task (CPU-friendly)
Quick testing of various topo_mode, util_dist_mode, and training settings
"""

import torch
import argparse
import os
from pathlib import Path
from collections import defaultdict
from sorl.gat_sim import GAT, GATConfig, BOS_TOKEN_ID
from sorl.neo_utils import (
    sorl_search_v4, compute_sgpo_loss_v2, compute_vocab_utilization_rate,
    sorl_evaluate_v2
)
from sorl.topo import uniformity_loss, contrastive_loss
from sorl.gapt import GatedPhaseTransition
from sorl.stat import save_training_dynamics
from data.copy_paste import CopyPasteDataLoader
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Test SORL on Copy & Paste")
    
    # Model config
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--abstract_vocab_size", type=int, default=6)
    
    # Data config
    parser.add_argument("--vocab_size", type=int, default=16)
    parser.add_argument("--max_token", type=int, default=10)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)
    
    # Training config
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--num_steps", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=2)
    
    # SORL config
    parser.add_argument("--num_rollouts", type=int, default=2)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--max_iterations", type=int, default=2)
    parser.add_argument("--min_temperature", type=float, default=0.0)
    parser.add_argument("--max_temperature", type=float, default=5.0)
    
    # Loss weights
    parser.add_argument("--alpha_loss", type=float, default=0.1)
    parser.add_argument("--alpha_topo", type=float, default=0.0)

    # Contrastive Loss
    parser.add_argument("--use_contrastive_loss", action="store_true")
    parser.add_argument("--use_uniformity_loss", action="store_true")
    parser.add_argument("--contrast_loss_temp", type=float, default=1.0)
    parser.add_argument("--contrast_loss_alpha", type=float, default=1.0)
    
    # Modes to test
    parser.add_argument("--adv_mode", type=int, default=0,
                       help="0:SGPO (favor useless abstraction), 1:all rollout, 2:distillation (favor familiar abstraction), 3:exploitation (favor useful abstraction), 4:exploration (favor un-familiar abstraction)")
    parser.add_argument("--topo_mode", type=int, default=1, 
                       help="0:dot, 1:correlation, 2:covariance")
    parser.add_argument("--util_dist_mode", type=int, default=1,
                       help="0:naive, 1:stop grad on worse rollout")
    
    # GAPT
    parser.add_argument("--use_gapt", action="store_true")
    
    # Visualization
    parser.add_argument("--no_plot", action="store_true", help="Skip plotting at the end")
    parser.add_argument("--save_path", type=str, default="logs/test_sorl.png", help="Save path for the plot")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("="*70)
    print(f"Testing SORL on Copy & Paste Task")
    print(f"adv_mode={args.adv_mode}, topo_mode={args.topo_mode}, util_dist_mode={args.util_dist_mode}")
    print(f"alpha_loss={args.alpha_loss}, alpha_topo={args.alpha_topo}")
    print(f"contrastive_loss={args.use_contrastive_loss}, uniformity_loss={args.use_uniformity_loss}")
    print(f"contrast_loss_alpha={args.contrast_loss_alpha}, contrast_loss_temp={args.contrast_loss_temp}")
    print("="*70)
    
    # Setup model
    device = "cpu"
    gat_config = GATConfig(
        vocab_sizes=[BOS_TOKEN_ID+1, args.abstract_vocab_size],
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        device=device
    )
    model = GAT(gat_config)
    model.train()
    
    # Setup data loader
    loader = CopyPasteDataLoader(
        vocab_size=args.vocab_size,
        max_token=args.max_token,
        seq_len=args.seq_len,
        device=device
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Setup GAPT if needed
    gapt = GatedPhaseTransition() if args.use_gapt else None
    
    # Training params
    memory_span = 2 * args.seq_len + 2
    attn_blocksize = 1792
    temperatures = torch.tensor(
        [args.min_temperature, args.max_temperature],
        device=device
    )
    temperatures_eval = torch.tensor(
        [args.min_temperature, args.max_temperature],
        device=device
    )
    temperatures_topo = torch.tensor(
        [args.max_temperature, args.max_temperature],
        device=device
    )
    
    # Training loop
    record = defaultdict(list)
    
    for step in range(args.num_steps):
        optimizer.zero_grad()
        tokens, _ = loader.get_batch(args.batch_size)
        
        # SORL search
        with torch.no_grad():
            search_tokens, search_ppt, search_adv, abs_dist = sorl_search_v4(
                tokens, model,
                n=args.num_rollouts,
                K=args.K,
                max_iterations=args.max_iterations,
                memory_span=memory_span,
                attn_blocksize=attn_blocksize,
                temperature=temperatures,
                mode=args.adv_mode,
                truncate_seq_len=False
            )
        
        # Compute loss
        traj_loss, grpo_loss, topo_loss = compute_sgpo_loss_v2(
            search_tokens, search_adv, abs_dist, model,
            memory_span, attn_blocksize,
            topo_mode=args.topo_mode,
            util_dist_mode=args.util_dist_mode
        )
        if args.use_contrastive_loss:
            contrast_loss = contrastive_loss(model.transformer.wte.weight[model.vocab_sizes[0]:], temp=args.contrast_loss_temp)
        elif args.use_uniformity_loss:
            contrast_loss = uniformity_loss(model.transformer.wte.weight[model.vocab_sizes[0]:], t=args.contrast_loss_temp)
        else:
            contrast_loss = torch.tensor(0.0, device=device)

        # Combine losses
        if args.use_gapt and gapt is not None:
            loss = gapt.step(traj_loss, args.alpha_loss * grpo_loss + args.alpha_topo * topo_loss + args.contrast_loss_alpha * contrast_loss)
        else:
            loss = traj_loss + args.alpha_loss * grpo_loss + args.alpha_topo * topo_loss + args.contrast_loss_alpha * contrast_loss
        
        # Optimize
        loss.backward()
        optimizer.step()
        
        # Print training loss periodically
        if step % (args.eval_every * 10) == 0:
            contrast_val = contrast_loss.item() if torch.is_tensor(contrast_loss) else contrast_loss
            print(f"train step {step:3d} | "
                  f"loss: {loss.item():.3f} | "
                  f"traj: {traj_loss.item():.3f} | "
                  f"grpo: {grpo_loss.item():.3f} | "
                  f"topo: {topo_loss.item():.3f} | "
                  f"contrast (wte): {contrast_val:.3f}")
        
        # Periodic validation
        if step % args.eval_every == 0:
            with torch.no_grad():
                val_tokens, val_adv, val_traj_loss, val_abs_loss, topo_sim_greedy = sorl_evaluate_v2(
                    tokens, model,
                    n=args.num_rollouts,
                    K=args.K,
                    max_iterations=args.max_iterations,
                    memory_span=memory_span,
                    attn_blocksize=attn_blocksize,
                    temperature=temperatures_eval,
                    truncate_seq_len=False
                )
                _, _, _, _, topo_sim = sorl_evaluate_v2(
                    tokens, model,
                    n=args.num_rollouts,
                    K=args.K,
                    max_iterations=args.max_iterations,
                    memory_span=memory_span,
                    attn_blocksize=attn_blocksize,
                    temperature=temperatures_topo,
                    truncate_seq_len=False
                )
                vocab_util = compute_vocab_utilization_rate(val_tokens, model)
                
                # Record metrics (use field names expected by save_training_dynamics)
                contrast_val = contrast_loss.item() if torch.is_tensor(contrast_loss) else contrast_loss
                record['util_rate'].append(vocab_util)
                record['search_advantage'].append(val_adv)
                record['abs_loss'].append(val_abs_loss)
                record['traj_loss'].append(val_traj_loss)
                record['alpha_loss'].append(args.alpha_loss)
                record['topo_sim'].append(topo_sim)
                record['topo_sim_greedy'].append(topo_sim_greedy)
                record['contrast_loss'].append(contrast_val)

                print(f"validation step {step:3d} | "
                      f"traj_loss: {val_traj_loss.item():.2f} | "
                      f"abs_loss: {val_abs_loss.item():.2f} | "
                      f"search adv: {val_adv.item() * 100:.2f}% | "
                      f"vocab util: {vocab_util * 100:.2f}% | "
                      f"topo_sim: {topo_sim.item():.2f} | "
                      f"topo_sim_greedy: {topo_sim_greedy.item():.2f} | "
                      f"contrast (wte): {contrast_val:.3f}")
    
    print("\n" + "="*70)
    print("Training Complete!")
    print(f"Final vocab util: {record['util_rate'][-1] * 100:.2f}%")
    print(f"Final search advantage: {record['search_advantage'][-1].item() * 100:.2f}%")
    print(f"Final topo_sim: {record['topo_sim'][-1].item():.2f}")
    print(f"Final topo_sim_greedy: {record['topo_sim_greedy'][-1].item():.2f}")
    print("="*70)
    
    # Visualize results
    if not args.no_plot:
        print("\nGenerating plots...")
        save_path = args.save_path
        
        # Ensure directory exists
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save plot and CSV
        loss_type = "contrastive" if args.use_contrastive_loss else ("uniformity" if args.use_uniformity_loss else "none")
        run_info = f"adv={args.adv_mode}, topo={args.topo_mode}, util={args.util_dist_mode}, {loss_type}_loss, α={args.contrast_loss_alpha}, T={args.contrast_loss_temp}"
        save_training_dynamics(record, save_path, run_info)
        
        # Convert tensors to scalars for CSV
        record_dict = {}
        for key, values in record.items():
            record_dict[key] = [v.item() if torch.is_tensor(v) else v for v in values]
        
        df = pd.DataFrame(record_dict)
        csv_path = save_path.replace('.png', '.csv')
        df.to_csv(csv_path, index=False)
        print(f"CSV saved to: {csv_path}")
    
    return record

if __name__ == "__main__":
    main()