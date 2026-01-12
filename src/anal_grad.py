# Gradient Analysis Module
# Functions for analyzing token-level gradient magnitude by entropy-probability quadrants
# ============================================

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import io


# ============================================
# Quadrant Definitions
# ============================================
QUADRANT_COLORS = {
    'confident_correct': 'green',
    'uncertain_correct': 'lightgreen', 
    'confident_wrong': 'red',
    'uncertain_wrong': 'orange',
}


def get_quadrant_masks(probs: np.ndarray, entropies: np.ndarray, 
                       prob_thresh: float = 0.5, 
                       entropy_thresh: Optional[float] = None) -> Dict[str, np.ndarray]:
    """
    Partition tokens into 4 quadrants based on probability and entropy.
    
    Args:
        probs: P(correct token) for each token
        entropies: Entropy of logit distribution for each token
        prob_thresh: Threshold for correct vs wrong (default 0.5)
        entropy_thresh: Threshold for confident vs uncertain (default: median)
    
    Returns:
        Dict mapping quadrant name to boolean mask
    """
    if entropy_thresh is None:
        entropy_thresh = np.median(entropies)
    
    return {
        'confident_correct': (probs >= prob_thresh) & (entropies < entropy_thresh),
        'uncertain_correct': (probs >= prob_thresh) & (entropies >= entropy_thresh),
        'confident_wrong':   (probs < prob_thresh) & (entropies < entropy_thresh),
        'uncertain_wrong':   (probs < prob_thresh) & (entropies >= entropy_thresh),
    }


def compute_quadrant_stats(probs: np.ndarray, entropies: np.ndarray, 
                           grad_mags: np.ndarray,
                           prob_thresh: float = 0.5) -> Dict[str, Dict]:
    """
    Compute gradient statistics for each quadrant.
    
    Returns:
        Dict with quadrant names as keys, containing:
        - mean_grad, std_grad, count, mean_prob, mean_entropy
    """
    quadrants = get_quadrant_masks(probs, entropies, prob_thresh)
    
    results = {}
    for name, mask in quadrants.items():
        if mask.sum() > 0:
            results[name] = {
                'mean_grad': grad_mags[mask].mean(),
                'std_grad': grad_mags[mask].std(),
                'count': mask.sum(),
                'pct': 100 * mask.mean(),
                'mean_prob': probs[mask].mean(),
                'mean_entropy': entropies[mask].mean(),
            }
    return results


# ============================================
# Load and Process Token Stats
# ============================================

def load_token_stats(log_dir: str) -> Dict[int, Dict]:
    """
    Load all token_stats_step*.pt files from a directory.
    
    Returns:
        Dict mapping step number to token stats dict
    """
    log_path = Path(log_dir)
    pt_files = sorted(log_path.glob("token_stats_step*.pt"))
    
    all_stats = {}
    for f in pt_files:
        step = int(f.stem.split('step')[1])
        all_stats[step] = torch.load(f)
    
    return all_stats


def extract_arrays_from_stats(stats: Dict) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Extract probs, entropies, grad_mags arrays from stats dict.
    Handles BFloat16 conversion.
    """
    probs = torch.cat(stats['probs']).float().numpy()
    entropies = torch.cat(stats['entropies']).float().numpy()
    
    grad_mags = None
    if 'grad_magnitudes' in stats and stats['grad_magnitudes']:
        grad_mags = torch.cat(stats['grad_magnitudes']).float().numpy()
        # Match lengths
        min_len = min(len(probs), len(grad_mags))
        probs = probs[:min_len]
        entropies = entropies[:min_len]
        grad_mags = grad_mags[:min_len]
    
    return probs, entropies, grad_mags


# ============================================
# Visualization: Single Frame
# ============================================

def create_quadrant_frame(stats: Dict, step: int, 
                          prob_thresh: float = 0.5,
                          max_entropy_ylim: float = 4.0,
                          max_grad_ylim: float = 3.0) -> Image.Image:
    """
    Create a single frame showing quadrant scatter and gradient bar chart.
    
    Args:
        stats: Token stats dict for one step
        step: Training step number
        prob_thresh: Probability threshold for quadrants
        max_entropy_ylim: Fixed y-limit for entropy scatter
        max_grad_ylim: Fixed y-limit for gradient bar chart
    
    Returns:
        PIL Image of the frame
    """
    probs, entropies, grad_mags = extract_arrays_from_stats(stats)
    entropy_thresh = np.median(entropies)
    quadrants = get_quadrant_masks(probs, entropies, prob_thresh, entropy_thresh)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Scatter
    ax1 = axes[0]
    n_sample = min(5000, len(probs))
    idx = np.random.choice(len(probs), n_sample, replace=False)
    
    for name, mask in quadrants.items():
        mask_sampled = mask[idx]
        ax1.scatter(probs[idx][mask_sampled], entropies[idx][mask_sampled], 
                    c=QUADRANT_COLORS[name], alpha=0.4, s=15, label=name)
    
    ax1.axvline(x=prob_thresh, color='gray', linestyle='--', alpha=0.7)
    ax1.axhline(y=entropy_thresh, color='gray', linestyle='--', alpha=0.7)
    ax1.set_xlabel('P(correct)', fontsize=12)
    ax1.set_ylabel('Entropy', fontsize=12)
    ax1.set_title(f'Token Quadrants (Step {step})', fontsize=14)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, max(max_entropy_ylim, entropies.max() * 1.1))
    
    # Plot 2: Bar chart
    ax2 = axes[1]
    if grad_mags is not None:
        quad_stats = compute_quadrant_stats(probs, entropies, grad_mags, prob_thresh)
        
        names = list(quad_stats.keys())
        means = [quad_stats[n]['mean_grad'] for n in names]
        stds = [quad_stats[n]['std_grad'] for n in names]
        bar_colors = [QUADRANT_COLORS[n] for n in names]
        
        bars = ax2.bar(names, means, yerr=stds, capsize=5, color=bar_colors, alpha=0.7)
        ax2.set_ylabel('Mean Gradient Magnitude', fontsize=12)
        ax2.set_title(f'Gradient by Quadrant (Step {step})', fontsize=14)
        ax2.set_ylim(0, max_grad_ylim)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add count annotations
        for bar, name in zip(bars, names):
            count = quad_stats[name]['count']
            ax2.annotate(f'n={count}', xy=(bar.get_x() + bar.get_width()/2, 0.1),
                        ha='center', fontsize=9, color='black')
    else:
        ax2.text(0.5, 0.5, 'No gradient data', ha='center', va='center', fontsize=14)
        ax2.set_title(f'Gradient by Quadrant (Step {step})', fontsize=14)
    
    plt.tight_layout()
    
    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img = Image.open(buf).copy()
    plt.close(fig)
    
    return img


# ============================================
# Visualization: Animation
# ============================================

def create_quadrant_animation(all_stats: Dict[int, Dict], 
                              save_path: str = "quadrant_dynamics.gif",
                              duration_ms: int = 500,
                              prob_thresh: float = 0.5) -> None:
    """
    Create animated GIF of quadrant analysis over training.
    
    Args:
        all_stats: Dict mapping step to token stats
        save_path: Output path for GIF
        duration_ms: Milliseconds per frame
        prob_thresh: Probability threshold for quadrants
    """
    steps = sorted(all_stats.keys())
    frames = []
    
    print("Generating quadrant animation frames...")
    for i, step in enumerate(steps):
        print(f"  Frame {i+1}/{len(steps)} (step {step})", end='\r')
        frame = create_quadrant_frame(all_stats[step], step, prob_thresh)
        frames.append(frame)
    
    print("\nSaving GIF...")
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0
    )
    print(f"Saved: {save_path}")


# ============================================
# Visualization: Dynamics Line Plot
# ============================================

def plot_quadrant_gradient_dynamics(all_stats: Dict[int, Dict],
                                    save_path: str = "quadrant_gradient_dynamics.png",
                                    prob_thresh: float = 0.5) -> Dict[str, List[Tuple[int, float]]]:
    """
    Plot gradient magnitude by quadrant over training steps.
    
    Args:
        all_stats: Dict mapping step to token stats
        save_path: Output path for figure
        prob_thresh: Probability threshold for quadrants
    
    Returns:
        Dict mapping quadrant name to list of (step, mean_grad) tuples
    """
    steps = sorted(all_stats.keys())
    
    quadrant_over_time = {name: [] for name in QUADRANT_COLORS.keys()}
    
    for step in steps:
        probs, entropies, grad_mags = extract_arrays_from_stats(all_stats[step])
        if grad_mags is None:
            continue
        
        quad_stats = compute_quadrant_stats(probs, entropies, grad_mags, prob_thresh)
        
        for name in quadrant_over_time.keys():
            if name in quad_stats:
                quadrant_over_time[name].append((step, quad_stats[name]['mean_grad']))
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Absolute gradient magnitude
    ax1 = axes[0]
    for name, data in quadrant_over_time.items():
        if data:
            steps_q, grads_q = zip(*data)
            ax1.plot(steps_q, grads_q, 'o-', label=name, 
                     color=QUADRANT_COLORS[name], linewidth=2)
    
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Mean Gradient Magnitude', fontsize=12)
    ax1.set_title('Gradient by Quadrant Over Training', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Relative to confident_correct
    ax2 = axes[1]
    cc_data = dict(quadrant_over_time['confident_correct'])
    
    for name, data in quadrant_over_time.items():
        if data and name != 'confident_correct':
            steps_q, grads_q = zip(*data)
            relative = [g / cc_data.get(s, 1) for s, g in zip(steps_q, grads_q)]
            ax2.plot(steps_q, relative, 'o-', label=f'{name} / confident_correct', 
                     color=QUADRANT_COLORS[name], linewidth=2)
    
    ax2.axhline(1, color='green', linestyle='--', alpha=0.7, label='confident_correct (baseline)')
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Relative Gradient Magnitude', fontsize=12)
    ax2.set_title('Gradient Relative to Confident Correct', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")
    
    return quadrant_over_time


def print_quadrant_summary(all_stats: Dict[int, Dict], prob_thresh: float = 0.5) -> None:
    """Print summary of quadrant statistics at key training steps."""
    steps = sorted(all_stats.keys())
    key_steps = [steps[0], steps[len(steps)//2], steps[-1]]
    
    print("=== Quadrant Gradient Summary ===")
    for step in key_steps:
        probs, entropies, grad_mags = extract_arrays_from_stats(all_stats[step])
        if grad_mags is None:
            continue
        
        quad_stats = compute_quadrant_stats(probs, entropies, grad_mags, prob_thresh)
        
        print(f"\nStep {step}:")
        for name in QUADRANT_COLORS.keys():
            if name in quad_stats:
                s = quad_stats[name]
                print(f"  {name:<20}: grad={s['mean_grad']:.4f}, n={s['count']}, {s['pct']:.1f}%")


# ============================================
# Convenience function for full analysis
# ============================================

def analyze_gradient_quadrants(log_dir: str, 
                               output_dir: str = ".",
                               prob_thresh: float = 0.5) -> None:
    """
    Run full quadrant gradient analysis on token stats.
    
    Args:
        log_dir: Directory containing token_stats_step*.pt files
        output_dir: Directory for output files
        prob_thresh: Probability threshold for quadrants
    """
    print(f"Loading token stats from {log_dir}...")
    all_stats = load_token_stats(log_dir)
    print(f"Loaded {len(all_stats)} steps")
    
    # Print summary
    print_quadrant_summary(all_stats, prob_thresh)
    
    # Create dynamics plot
    output_path = Path(output_dir)
    plot_quadrant_gradient_dynamics(
        all_stats, 
        save_path=str(output_path / "quadrant_gradient_dynamics.png"),
        prob_thresh=prob_thresh
    )
    
    # Create animation
    create_quadrant_animation(
        all_stats,
        save_path=str(output_path / "quadrant_dynamics.gif"),
        prob_thresh=prob_thresh
    )


# ============================================
# MBE Batch-Level Analysis
# ============================================

def analyze_batch_level(stats: Dict, step: int) -> Optional[pd.DataFrame]:
    """
    Compute batch-level statistics for MBE correlation analysis.
    
    MBE is computed per-batch (averaged over layers), while other metrics
    (grad, prob, entropy, loss) are per-token and averaged per-batch.
    
    Returns:
        DataFrame with batch-level: avg_mbe, mean_grad, mean_prob, mean_entropy, mean_loss
    """
    if not stats.get('mbe') or not stats.get('grad_magnitudes'):
        return None
    
    batch_data = []
    n_batches = min(len(stats['mbe']), len(stats['grad_magnitudes']))
    
    for i in range(n_batches):
        # MBE for this batch (average across layers)
        mbe_dict = stats['mbe'][i]
        avg_mbe = np.mean(list(mbe_dict.values()))
        
        # Gradient magnitude for this batch
        batch_grads = stats['grad_magnitudes'][i].float().numpy()
        mean_grad = batch_grads.mean()
        
        # Prob and entropy for this batch
        batch_probs = stats['probs'][i].float().numpy()
        batch_entropies = stats['entropies'][i].float().numpy()
        
        row = {
            'batch_idx': i,
            'step': step,
            'avg_mbe': avg_mbe,
            'mean_grad': mean_grad,
            'mean_prob': batch_probs.mean(),
            'mean_entropy': batch_entropies.mean(),
        }
        
        # CE loss if available
        if stats.get('per_token_loss') and len(stats['per_token_loss']) > i:
            batch_loss = stats['per_token_loss'][i].float().numpy()
            row['mean_loss'] = batch_loss.mean()
        
        batch_data.append(row)
    
    return pd.DataFrame(batch_data)


def compute_mbe_correlations(all_stats: Dict[int, Dict]) -> pd.DataFrame:
    """
    Compute MBE correlations with grad, prob, entropy, and loss for each step.
    
    Returns:
        DataFrame with per-step correlations
    """
    results = []
    steps = sorted(all_stats.keys())
    
    for step in steps:
        batch_df = analyze_batch_level(all_stats[step], step)
        if batch_df is None or len(batch_df) < 3:
            continue
        
        row = {'step': step, 'n_batches': len(batch_df)}
        row['mbe_grad_corr'] = batch_df['avg_mbe'].corr(batch_df['mean_grad'])
        row['mbe_prob_corr'] = batch_df['avg_mbe'].corr(batch_df['mean_prob'])
        row['mbe_entropy_corr'] = batch_df['avg_mbe'].corr(batch_df['mean_entropy'])
        
        if 'mean_loss' in batch_df.columns:
            row['mbe_loss_corr'] = batch_df['avg_mbe'].corr(batch_df['mean_loss'])
        
        results.append(row)
    
    return pd.DataFrame(results)


def plot_mbe_correlation_dynamics(all_stats: Dict[int, Dict],
                                  save_path: str = "mbe_correlation_dynamics.png") -> pd.DataFrame:
    """
    Plot MBE correlations over training - the phase transition visualization.
    
    Shows how MBE's relationship with gradient/loss changes over training.
    """
    corr_df = compute_mbe_correlations(all_stats)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: MBE-Grad correlation over time (the phase transition)
    ax1 = axes[0]
    ax1.plot(corr_df['step'], corr_df['mbe_grad_corr'], 'o-', 
             color='#E74C3C', linewidth=2.5, markersize=8, label='MBE-Grad')
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.7)
    ax1.fill_between(corr_df['step'], 0, corr_df['mbe_grad_corr'],
                     where=corr_df['mbe_grad_corr'] > 0, alpha=0.3, color='#3498DB',
                     label='Positive (complexity-driven)')
    ax1.fill_between(corr_df['step'], 0, corr_df['mbe_grad_corr'],
                     where=corr_df['mbe_grad_corr'] < 0, alpha=0.3, color='#E74C3C',
                     label='Negative (compression-driven)')
    
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Correlation', fontsize=12)
    ax1.set_title('MBE-Gradient Correlation Phase Transition', fontsize=14)
    ax1.legend(loc='lower left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1, 1)
    
    # Plot 2: All MBE correlations
    ax2 = axes[1]
    ax2.plot(corr_df['step'], corr_df['mbe_grad_corr'], 'o-', 
             color='#E74C3C', linewidth=2, label='MBE-Grad')
    ax2.plot(corr_df['step'], corr_df['mbe_prob_corr'], 's-', 
             color='#27AE60', linewidth=2, label='MBE-Prob')
    ax2.plot(corr_df['step'], corr_df['mbe_entropy_corr'], '^-', 
             color='#F39C12', linewidth=2, label='MBE-Entropy')
    
    if 'mbe_loss_corr' in corr_df.columns:
        ax2.plot(corr_df['step'], corr_df['mbe_loss_corr'], 'd-', 
                 color='#9B59B6', linewidth=2, label='MBE-Loss')
    
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Correlation', fontsize=12)
    ax2.set_title('MBE Correlations Over Training', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1, 1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")
    
    return corr_df


def print_mbe_phase_summary(corr_df: pd.DataFrame) -> None:
    """Print summary of MBE phase transition."""
    # Find transition point (where correlation flips sign)
    sign_changes = np.where(np.diff(np.sign(corr_df['mbe_grad_corr'])))[0]
    
    print("=== MBE-Gradient Phase Transition ===")
    print(f"{'Step':>6} | {'MBE-Grad':>10} | {'MBE-Loss':>10}")
    print("-" * 35)
    
    for _, row in corr_df.iterrows():
        loss_str = f"{row.get('mbe_loss_corr', float('nan')):.3f}"
        print(f"{row['step']:>6} | {row['mbe_grad_corr']:>10.3f} | {loss_str:>10}")
    
    if len(sign_changes) > 0:
        transition_idx = sign_changes[0]
        transition_step = (corr_df.iloc[transition_idx]['step'] + 
                          corr_df.iloc[transition_idx + 1]['step']) / 2
        print(f"\n⚡ Phase transition around step {transition_step:.0f}")
        print(f"  Early phase (positive corr): High MBE → High gradient")
        print(f"  Late phase (negative corr):  Low MBE → High gradient")


# ============================================
# Patch-Level Analysis (High Statistical Power)
# ============================================

def load_patch_stats(log_dir: str) -> Dict[int, Dict]:
    """
    Load all patch_stats_step*.pt files from a directory.
    
    Returns:
        Dict mapping step number to patch stats dict
    """
    log_path = Path(log_dir)
    pt_files = sorted(log_path.glob("patch_stats_step*.pt"))
    
    all_stats = {}
    for f in pt_files:
        step = int(f.stem.split('step')[1])
        all_stats[step] = torch.load(f)
    
    return all_stats


def compute_patch_correlations(patch_stats: Dict, step: int) -> Dict[str, float]:
    """
    Compute correlations at patch level for a single step.
    
    Each batch contributes B * num_patches data points.
    """
    # Concatenate all batches: (total_patches,)
    patch_mbe = torch.cat([p.flatten() for p in patch_stats['patch_mbe']]).float().numpy()
    patch_loss = torch.cat([p.flatten() for p in patch_stats['patch_loss']]).float().numpy()
    patch_prob = torch.cat([p.flatten() for p in patch_stats['patch_prob']]).float().numpy()
    patch_entropy = torch.cat([p.flatten() for p in patch_stats['patch_entropy']]).float().numpy()
    
    n_patches = len(patch_mbe)
    
    result = {
        'step': step,
        'n_patches': n_patches,
        'mbe_loss_corr': np.corrcoef(patch_mbe, patch_loss)[0, 1],
        'mbe_prob_corr': np.corrcoef(patch_mbe, patch_prob)[0, 1],
        'mbe_entropy_corr': np.corrcoef(patch_mbe, patch_entropy)[0, 1],
        'mean_mbe': patch_mbe.mean(),
        'mean_loss': patch_loss.mean(),
    }
    
    # Add gradient correlation if available
    if 'patch_grad' in patch_stats and patch_stats['patch_grad']:
        patch_grad = torch.cat([p.flatten() for p in patch_stats['patch_grad']]).float().numpy()
        # Match lengths (grad may have fewer batches)
        min_len = min(len(patch_mbe), len(patch_grad))
        result['mbe_grad_corr'] = np.corrcoef(patch_mbe[:min_len], patch_grad[:min_len])[0, 1]
        result['mean_grad'] = patch_grad.mean()
    
    return result


def analyze_patch_mbe_dynamics(log_dir: str, 
                                save_path: str = "patch_mbe_dynamics.png") -> pd.DataFrame:
    """
    Analyze MBE-loss correlation at patch level across training.
    
    This provides ~1000x more data points than batch-level analysis.
    """
    all_patch_stats = load_patch_stats(log_dir)
    
    if not all_patch_stats:
        print(f"No patch stats found in {log_dir}")
        return pd.DataFrame()
    
    results = []
    for step in sorted(all_patch_stats.keys()):
        try:
            corr = compute_patch_correlations(all_patch_stats[step], step)
            results.append(corr)
        except Exception as e:
            print(f"Error at step {step}: {e}")
    
    df = pd.DataFrame(results)
    
    # Print summary
    has_grad = 'mbe_grad_corr' in df.columns
    print("=== Patch-Level MBE Correlations ===")
    if has_grad:
        print(f"{'Step':>6} | {'N Patches':>10} | {'MBE-Grad':>10} | {'MBE-Loss':>10} | {'MBE-Prob':>10}")
        print("-" * 60)
        for _, row in df.iterrows():
            grad_str = f"{row.get('mbe_grad_corr', float('nan')):>10.3f}"
            print(f"{row['step']:>6} | {row['n_patches']:>10} | {grad_str} | {row['mbe_loss_corr']:>10.3f} | {row['mbe_prob_corr']:>10.3f}")
    else:
        print(f"{'Step':>6} | {'N Patches':>10} | {'MBE-Loss':>10} | {'MBE-Prob':>10}")
        print("-" * 45)
        for _, row in df.iterrows():
            print(f"{row['step']:>6} | {row['n_patches']:>10} | {row['mbe_loss_corr']:>10.3f} | {row['mbe_prob_corr']:>10.3f}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MBE-Loss correlation
    ax1 = axes[0]
    ax1.plot(df['step'], df['mbe_loss_corr'], 'o-', color='#9B59B6', linewidth=2.5, markersize=8)
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.7)
    ax1.fill_between(df['step'], 0, df['mbe_loss_corr'],
                     where=df['mbe_loss_corr'] > 0, alpha=0.3, color='#E74C3C',
                     label='High MBE → High Loss')
    ax1.fill_between(df['step'], 0, df['mbe_loss_corr'],
                     where=df['mbe_loss_corr'] < 0, alpha=0.3, color='#27AE60',
                     label='Low MBE → High Loss')
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Correlation', fontsize=12)
    ax1.set_title('Patch-Level MBE-Loss Correlation', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1, 1)
    
    # MBE and Loss trajectories
    ax2 = axes[1]
    ax2.plot(df['step'], df['mean_mbe'], 'o-', color='#3498DB', linewidth=2, label='Mean MBE')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(df['step'], df['mean_loss'], 's-', color='#E74C3C', linewidth=2, label='Mean Loss')
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Mean MBE', fontsize=12, color='#3498DB')
    ax2_twin.set_ylabel('Mean Loss', fontsize=12, color='#E74C3C')
    ax2.set_title('MBE and Loss Over Training', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nSaved: {save_path}")
    
    return df


def plot_patch_scatter(patch_stats: Dict, step: int, 
                       save_path: str = None) -> None:
    """
    Scatter plot of per-patch MBE vs Loss for a single step.
    """
    patch_mbe = torch.cat([p.flatten() for p in patch_stats['patch_mbe']]).float().numpy()
    patch_loss = torch.cat([p.flatten() for p in patch_stats['patch_loss']]).float().numpy()
    patch_prob = torch.cat([p.flatten() for p in patch_stats['patch_prob']]).float().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # MBE vs Loss
    ax1 = axes[0]
    scatter = ax1.scatter(patch_mbe, patch_loss, c=patch_prob, cmap='RdYlGn', 
                          alpha=0.5, s=10, vmin=0, vmax=1)
    plt.colorbar(scatter, ax=ax1, label='P(correct)')
    ax1.set_xlabel('Patch MBE', fontsize=12)
    ax1.set_ylabel('Patch Loss', fontsize=12)
    ax1.set_title(f'Patch MBE vs Loss (Step {step})', fontsize=14)
    
    corr = np.corrcoef(patch_mbe, patch_loss)[0, 1]
    ax1.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax1.transAxes, 
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # MBE vs Prob
    ax2 = axes[1]
    ax2.scatter(patch_mbe, patch_prob, c=patch_loss, cmap='viridis', 
                alpha=0.5, s=10)
    ax2.set_xlabel('Patch MBE', fontsize=12)
    ax2.set_ylabel('Patch P(correct)', fontsize=12)
    ax2.set_title(f'Patch MBE vs Accuracy (Step {step})', fontsize=14)
    
    corr2 = np.corrcoef(patch_mbe, patch_prob)[0, 1]
    ax2.text(0.05, 0.95, f'r = {corr2:.3f}', transform=ax2.transAxes, 
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close()

