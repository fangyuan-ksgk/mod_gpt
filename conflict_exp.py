"""
Large-scale Conflict Experience Experiment
Ablates: datasets × pretrain runs × shift magnitudes
Saves incrementally to avoid data loss.
"""
import os
import time
import copy
import torch
import pandas as pd
from datetime import datetime, timedelta
from src.forget import *

# ============================================================
# Experiment Configuration
# ============================================================
EXP_CFG = {
    # Scale (tune for ~5hr run)
    "num_data": 100,
    "num_runs": 1,
    "n_ft_runs": 5,
    
    # Ablation: shift magnitudes
    "shift_magnitudes": [1.0],
    
    # Data
    "train_size": 500,
    "val_size": 100,
    
    # Paths
    "init_params_path": "initial_mlp_params.pt",
    "output_dir": "bin/sweep_v2",
    "results_csv": "bin/sweep_v2/results.csv",
}

os.makedirs(EXP_CFG["output_dir"], exist_ok=True)

# ============================================================
# Progress Tracking
# ============================================================
class ProgressTracker:
    def __init__(self, total_steps):
        self.total = total_steps
        self.current = 0
        self.start_time = time.time()
        self.step_times = []
    
    def step(self):
        self.current += 1
        self.step_times.append(time.time())
    
    def eta(self):
        if self.current == 0:
            return "calculating..."
        elapsed = time.time() - self.start_time
        rate = elapsed / self.current
        remaining = (self.total - self.current) * rate
        return str(timedelta(seconds=int(remaining)))
    
    def progress_str(self):
        pct = 100 * self.current / self.total
        return f"[{self.current}/{self.total}] {pct:.1f}% | ETA: {self.eta()}"

# ============================================================
# Main Experiment
# ============================================================
def run_experiment():
    # Load base model
    orig_model = SimpleModel(CFG['in_dim'], CFG['hidden_dim'], CFG['out_dim'])
    orig_model.load_state_dict(torch.load(EXP_CFG["init_params_path"]))
    
    # Calculate total steps for progress
    total_steps = (
        len(EXP_CFG["shift_magnitudes"]) * 
        EXP_CFG["num_data"] * 
        EXP_CFG["num_runs"]
    )
    tracker = ProgressTracker(total_steps)
    
    all_results = []
    
    # Resume support: load existing results if any
    if os.path.exists(EXP_CFG["results_csv"]):
        existing_df = pd.read_csv(EXP_CFG["results_csv"])
        all_results = existing_df.to_dict('records')
        print(f"Resuming from {len(all_results)} existing results")
    
    print(f"Starting experiment: {total_steps} total steps")
    print(f"Shift magnitudes: {EXP_CFG['shift_magnitudes']}")
    print("="*60)
    
    for shift_idx, shift_mag in enumerate(EXP_CFG["shift_magnitudes"]):
        print(f"\n{'='*60}")
        print(f">>> Shift Magnitude: {shift_mag} ({shift_idx+1}/{len(EXP_CFG['shift_magnitudes'])})")
        print(f"{'='*60}")
        
        for data_idx in range(EXP_CFG["num_data"]):
            # Generate dataset with this shift magnitude
            torch.manual_seed(1000 + data_idx)
            PARAM_SHIFT_PATH = "param_shift.pkl"
            param_shift = load_param_shift(PARAM_SHIFT_PATH)
            
            # Scale param shift by magnitude
            param_shift = {k: v * shift_mag for k, v in param_shift.items()}
            
            trainset, valset, param_shift = build_dataset(
                EXP_CFG["train_size"], EXP_CFG["val_size"], 
                orig_model, CFG['in_dim'], param_shift
            )
            
            print(f"\n  Dataset {data_idx+1}/{EXP_CFG['num_data']} (shift={shift_mag})")
            
            for run_idx in range(EXP_CFG["num_runs"]):
                # Skip if already done (resume support)
                key = (shift_mag, data_idx, run_idx)
                if any(r['shift_mag'] == shift_mag and 
                       r['data_idx'] == data_idx and 
                       r['run_idx'] == run_idx for r in all_results):
                    tracker.step()
                    print(f"  [SKIP] shift={shift_mag} data={data_idx} run={run_idx} (already done)")
                    continue
                
                step_start = time.time()
                
                # Pretrain
                torch.manual_seed(2000 + data_idx * 100 + run_idx)
                pt_model = copy.deepcopy(orig_model)
                pt_start = time.time()
                pt_loss = train_phase(
                    pt_model, trainset, valset, orig_model, param_shift,
                    method_name="base", epochs=CFG['pretrain_epochs'], train_mode="mix"
                )
                pt_time = time.time() - pt_start
                
                # Save pretrain checkpoint
                pt_path = f"{EXP_CFG['output_dir']}/pt_s{shift_mag}_d{data_idx}_r{run_idx}.pt"
                torch.save(pt_model.state_dict(), pt_path)
                
                # Finetune with all methods
                ft_start = time.time()
                pt_loss_dict, ft_loss = run_ft_experiment(
                    pt_path, trainset, valset, param_shift, orig_model,
                    ft_steps=1000, n_ft_runs=EXP_CFG["n_ft_runs"], patch_size=32
                )
                ft_time = time.time() - ft_start
                step_time = time.time() - step_start

                # Record metrics
                base_mbe_pos = ft_loss['base']['mbe_positive']
                base_l1_neg = ft_loss['base']['l1_negative']
                init_l1_neg = pt_loss_dict["l1_negative"].item() if hasattr(pt_loss_dict["l1_negative"], 'item') else pt_loss_dict["l1_negative"]
                
                allev_results = {}
                for ft_method in ['gapt', 'mbe']:
                    method_l1_neg = ft_loss[ft_method]['l1_negative']
                    
                    # Avoid division by zero
                    denom = base_l1_neg - init_l1_neg
                    if abs(denom) < 1e-6:
                        alleviation = 0.0
                    else:
                        alleviation = (base_l1_neg - method_l1_neg) / denom * 100
                    
                    allev_results[ft_method] = alleviation
                    
                    all_results.append({
                        'shift_mag': shift_mag,
                        'data_idx': data_idx,
                        'run_idx': run_idx,
                        'init_l1_neg': init_l1_neg,
                        'init_mbe_neg': pt_loss_dict["mbe_negative"].item() if hasattr(pt_loss_dict["mbe_negative"], 'item') else pt_loss_dict["mbe_negative"],
                        'ft_method': ft_method,
                        'base_mbe_pos': base_mbe_pos,
                        'method_mbe_pos': ft_loss[ft_method]['mbe_positive'],
                        'base_l1_neg': base_l1_neg,
                        'method_l1_neg': method_l1_neg,
                        'alleviation_pct': alleviation,
                        'compression_ratio': base_mbe_pos / (ft_loss[ft_method]['mbe_positive'] + 1e-8),
                    })
                
                tracker.step()
                
                # Detailed logging
                elapsed = time.time() - tracker.start_time
                elapsed_str = str(timedelta(seconds=int(elapsed)))
                print(f"\n  [{tracker.current}/{tracker.total}] shift={shift_mag} | data={data_idx} | run={run_idx}")
                print(f"    PT: {pt_time:.1f}s | FT: {ft_time:.1f}s | Total step: {step_time:.1f}s")
                print(f"    init_l1_neg={init_l1_neg:.4f} → base_l1_neg={base_l1_neg:.4f}")
                print(f"    GAPT alleviation: {allev_results['gapt']:+.1f}% | MBE alleviation: {allev_results['mbe']:+.1f}%")
                print(f"    Elapsed: {elapsed_str} | ETA: {tracker.eta()}")
                
                # Save incrementally every 5 steps
                if tracker.current % 5 == 0:
                    pd.DataFrame(all_results).to_csv(EXP_CFG["results_csv"], index=False)
                    print(f"    >>> Checkpoint saved ({len(all_results)} results)")
    
    # Final save
    df = pd.DataFrame(all_results)
    df.to_csv(EXP_CFG["results_csv"], index=False)
    print(f"\n{'='*60}")
    print(f"DONE! Saved {len(df)} results to {EXP_CFG['results_csv']}")
    
    return df

# ============================================================
# Analysis (run after experiment completes)
# ============================================================
def analyze_results():
    from scipy import stats
    import matplotlib.pyplot as plt
    
    df = pd.read_csv(EXP_CFG["results_csv"])
    
    print("\n" + "="*60)
    print("ANALYSIS BY SHIFT MAGNITUDE")
    print("="*60)
    
    for shift_mag in EXP_CFG["shift_magnitudes"]:
        subset = df[(df['shift_mag'] == shift_mag) & (df['ft_method'] == 'gapt')]
        if len(subset) == 0:
            continue
        
        corr_mbe, p_mbe = stats.pearsonr(subset['base_mbe_pos'], subset['alleviation_pct'])
        corr_l1, p_l1 = stats.pearsonr(subset['base_l1_neg'], subset['alleviation_pct'])
        mean_allev = subset['alleviation_pct'].mean()
        
        print(f"\nShift Magnitude: {shift_mag}")
        print(f"  N samples: {len(subset)}")
        print(f"  Mean alleviation: {mean_allev:.1f}%")
        print(f"  base_mbe_pos → alleviation: r={corr_mbe:.3f}, p={p_mbe:.4f}")
        print(f"  base_l1_neg → alleviation:  r={corr_l1:.3f}, p={p_l1:.4f}")
    
    # Plot by shift magnitude
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for ax, shift_mag in zip(axes.flat, EXP_CFG["shift_magnitudes"]):
        subset = df[(df['shift_mag'] == shift_mag) & (df['ft_method'] == 'gapt')]
        if len(subset) == 0:
            continue
        ax.scatter(subset['base_l1_neg'], subset['alleviation_pct'], alpha=0.6)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('base_l1_neg')
        ax.set_ylabel('Alleviation (%)')
        ax.set_title(f'Shift Magnitude = {shift_mag}')
    
    plt.tight_layout()
    plt.savefig(f"{EXP_CFG['output_dir']}/analysis_by_shift.png", dpi=150)
    plt.show()
    
    return df

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    df = run_experiment()
    analyze_results()