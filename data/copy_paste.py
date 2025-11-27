from sympy.series.gruntz import I
import torch
import numpy as np
from dataclasses import dataclass
from sorl.gat_sim import BOS_TOKEN_ID

import torch
import numpy as np

class CopyPasteDataLoader:
    """Generate sequences: [abc...][equal_token_id][abc...]"""
    
    def __init__(self, vocab_size=128, max_token=10, seq_len=256, device='cpu'):
 
        self.vocab_size = min(vocab_size, max_token)
        self.seq_len = seq_len
        self.device = device
    
    def get_batch(self, batch_size):

        """Generate flattened copy-paste sequences."""
        repeat_seqs = torch.randint(0, self.vocab_size, (batch_size, self.seq_len), device=self.device)
        
        bos = torch.full((batch_size, 1), BOS_TOKEN_ID, device=self.device)
        samples = torch.cat([bos, repeat_seqs, repeat_seqs], dim=1)  # [batch_size, 2*seq_len+2]
        
        data = samples.flatten().unsqueeze(0)  # [1, batch_size*(2*seq_len+2)]
        
        mask = torch.ones_like(samples, dtype=torch.float)
        mask[:, :self.seq_len+1] = 0.0 
        loss_mask = mask.flatten().unsqueeze(0)
        
        return data, loss_mask

    def get_specific(self, batch_size, tokens): 
        if len(tokens.shape) == 1: 
            tokens = tokens.unsqueeze(-1)
        assert tokens.shape[0] == batch_size, "batch_size must match the number of tokens"

        repeat_seqs = tokens.clone()
        bos = torch.full((batch_size, 1), BOS_TOKEN_ID, device=self.device)
        samples = torch.cat([bos, repeat_seqs, repeat_seqs], dim=1)  # [batch_size, 2*seq_len+2]
        
        data = samples.flatten().unsqueeze(0)  # [1, batch_size*(2*seq_len+2)]
        
        mask = torch.ones_like(samples, dtype=torch.float)
        mask[:, :self.seq_len+1] = 0.0 
        loss_mask = mask.flatten().unsqueeze(0)

        return data, loss_mask


def extract_doc_components(seq_data, seq_ppt, seq_len, BOS_TOKEN_ID):
    # Collect 'copy value, abs token, abs utility p(s|a), abs predictability p(a|s)'
    assert seq_len == 1, "seq_len must be 1 for this function"
    doc_len = seq_len * 2 + 2

    doc_starts = (seq_data[0] == BOS_TOKEN_ID).nonzero(as_tuple=False).squeeze(1)
    num_docs = len(doc_starts)

    indices = doc_starts.unsqueeze(1) + torch.arange(doc_len, device=seq_data.device)
    docs = seq_data[0, indices]  # [NumDocs, doc_len]

    ppt_indices = indices[:, 1:] - 1  # Adjust for offset, skip BOS column
    ppts = seq_ppt[0, ppt_indices]  # [NumDocs, doc_len-1]

    first_vals = docs[:, 1]
    abs_tokens = docs[:, 1 + seq_len]
    abs_util = ppts[:, 1 + seq_len]
    abs_pred = ppts[:, seq_len]
        
    return first_vals, abs_tokens, abs_util, abs_pred

def collect_rollout_statistics(search_data, search_ppt, best_data, best_ppt, seq_len, BOS_TOKEN_ID):

    # rollout (greedy + random)
    values_greedy, abs_greedy, abs_util_greedy, abs_pred_greedy = extract_doc_components(search_data[:1], search_ppt[:1], seq_len, BOS_TOKEN_ID)
    values_random, abs_random, abs_util_random, abs_pred_random = extract_doc_components(search_data[1:], search_ppt[1:], seq_len, BOS_TOKEN_ID)

    # best rollout
    values_best, abs_best, _, _ = extract_doc_components(best_data, best_ppt.unsqueeze(0), seq_len, BOS_TOKEN_ID)

    # check (value, abs) equality to mark 'is_picked'
    is_picked_greedy = (values_greedy == values_best) & (abs_greedy == abs_best)
    is_picked_random = (values_random == values_best) & (abs_random == abs_best)
    is_picked_random = is_picked_random & ~is_picked_greedy # <- in case both are the same, we 'pick' greedy

    return (values_greedy, abs_greedy, abs_util_greedy, abs_pred_greedy, is_picked_greedy), (values_random, abs_random, abs_util_random, abs_pred_random, is_picked_random)

def check_abs_logits(model, loader, memory_span, attn_blocksize): 
    probs = [] # logit for each value on all abs token (n_val, n_abs)
    for i in range(10): 
        tokens, _ = loader.get_specific(1, torch.tensor([i]))
        idx = tokens[:, :2] # here onwards we need to predict abstract token
        with torch.no_grad(): 
            _, logits = model.forward(idx, memory_span, attn_blocksize)
            next_abs_logits = logits[0, -1, model.vocab_sizes[0]:]
            next_abs_prob = torch.softmax(next_abs_logits, dim=-1)
        probs.append(next_abs_prob)
    val_abs_prob = torch.stack(probs, dim=0)
    return val_abs_prob


import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from PIL import Image
from scipy.stats import entropy


def visualize_abstraction_dynamics(greedy_stat, random_stat, abs_probs, step, vocab_size=10, num_abs=6, abs_offset=21):
    """
    Visualize abstraction selection dynamics and model preferences.
    
    Left plot: Shape = rollout type (greedy/random), Color = abstraction token
    Right plot: Stacked bar of model preferences per value
    
    Args:
        greedy_stat: tuple (values, abs, utility, is_greedy, is_picked)
        random_stat: tuple (values, abs, utility, is_greedy, is_picked)
        abs_probs: [vocab_size, num_abs] - probabilities for each value->abstraction
        step: current training step
        vocab_size: number of values (0-9)
        num_abs: number of abstraction tokens
        abs_offset: offset to normalize abs token IDs to 0-indexed
        
    Returns:
        PIL.Image object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # === Left Plot: Utility vs Value for Rollouts ===
    
    # Extract greedy data
    greedy_values, greedy_abs, greedy_util, _, _ = greedy_stat
    greedy_values = greedy_values.cpu().numpy().flatten()
    greedy_abs = greedy_abs.cpu().numpy().flatten()
    greedy_util = greedy_util.cpu().numpy().flatten()
    
    # Extract random data
    random_values, random_abs, random_util, _, _ = random_stat
    random_values = random_values.cpu().numpy().flatten()
    random_abs = random_abs.cpu().numpy().flatten()
    random_util = random_util.cpu().numpy().flatten()

    # Normalize abstraction tokens to 0-indexed
    greedy_abs = greedy_abs - abs_offset
    random_abs = random_abs - abs_offset
    
    # Color map for abstraction tokens
    abs_colors = plt.cm.tab10(np.linspace(0, 1, num_abs))
    
    # Plot greedy rollout - circles colored by abstraction
    for abs_idx in range(num_abs):
        mask = (greedy_abs == abs_idx)
        if mask.any():
            ax1.scatter(greedy_values[mask], greedy_util[mask], 
                       c=[abs_colors[abs_idx]], marker='*', s=150, alpha=0.7, 
                       edgecolors=[abs_colors[abs_idx]], linewidths=1.5)
    
    # Plot random rollout - triangles colored by abstraction
    for abs_idx in range(num_abs):
        mask = (random_abs == abs_idx)
        if mask.any():
            ax1.scatter(random_values[mask], random_util[mask], 
                       c=[abs_colors[abs_idx]], marker='o', s=150, alpha=0.7, 
                       edgecolors=[abs_colors[abs_idx]], linewidths=1.5)
    
    # Build legend: shape meanings + color meanings
    from matplotlib.lines import Line2D
    
    # Shape legend
    shape_legend = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', 
               markeredgecolor='grey', markersize=13, label='Greedy', markeredgewidth=1.5),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markeredgecolor='grey', markersize=13, label='Random', markeredgewidth=1.5),
    ]
    
    # Color legend for abstractions
    color_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=abs_colors[i], 
               markeredgecolor=abs_colors[i], markersize=10, label=f'Abs {i}', markeredgewidth=1)
        for i in range(num_abs)
    ]
    
    # Combine legends
    legend1 = ax1.legend(handles=shape_legend, loc='upper left', fontsize=11, title='Rollout Type', title_fontsize=12)
    ax1.add_artist(legend1)
    ax1.legend(handles=color_legend, loc='upper right', fontsize=9, title='Abstraction', title_fontsize=10, ncol=2)
    
    ax1.set_xlabel('Value', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Rollout Abstraction Utility', fontsize=13, fontweight='bold')
    ax1.set_title(f'Rollout Utilities by Value (Step {step})', fontsize=15, fontweight='bold')
    ax1.set_xticks(range(vocab_size))
    ax1.set_xlim([-0.5, vocab_size - 0.5])  # Fix (1): Center x-axis labels, prevent cutoff
    ax1.set_ylim([3.5, 0])  # Fix (2): Fixed range (inverted since lower is better)
    ax1.grid(True, alpha=0.3)
    
    # === Right Plot: Abstraction Logits Distribution ===
    
    abs_probs_np = abs_probs.cpu().numpy()  # [vocab_size, num_abs]
    
    # Create stacked bar chart
    x = np.arange(vocab_size)
    width = 0.7
    
    bottom = np.zeros(vocab_size)
    bars = []
    for abs_idx in range(num_abs):
        bar = ax2.bar(x, abs_probs_np[:, abs_idx], width, bottom=bottom, 
               label=f'Abs {abs_idx}', color=abs_colors[abs_idx], 
               edgecolor='black', linewidth=0.8)
        bars.append(bar)
        bottom += abs_probs_np[:, abs_idx]
    
    # Overlay dominant abstraction
    dominant_abs = np.argmax(abs_probs_np, axis=1)
    for val_idx in range(vocab_size):
        dom_abs = dominant_abs[val_idx]
        dom_prob = abs_probs_np[val_idx, dom_abs]
        # if dom_prob > 0.4:  # Only show if reasonably confident
        ax2.text(val_idx, 1.02, f'{dom_abs}', ha='center', va='bottom', 
                fontsize=11, fontweight='bold', color=abs_colors[dom_abs])

    ax2.set_xlabel('Value', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Probability', fontsize=13, fontweight='bold')
    ax2.set_title(f'Model Abstraction Preferences (Step {step})', fontsize=15, fontweight='bold')
    ax2.set_xticks(range(vocab_size))
    ax2.set_ylim([0, 1.15])
    ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add entropy as a measure of uncertainty
    from scipy.stats import entropy
    entropies = [entropy(abs_probs_np[i]) for i in range(vocab_size)]
    avg_entropy = np.mean(entropies)
    max_entropy = np.log(num_abs)
    entropy_text = f'Avg Entropy: {avg_entropy:.2f} / {max_entropy:.2f}'

    norms = np.linalg.norm(abs_probs_np, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    normalized_probs = abs_probs_np / norms
    sim_matrix = np.dot(normalized_probs, normalized_probs.T)
    mask = ~np.eye(vocab_size, dtype=bool)
    avg_similarity = sim_matrix[mask].mean()
    similarity_text = f'Avg Similarity: {avg_similarity:.2f}'
    
    ax2.text(0.02, 0.98, entropy_text + '\n' + similarity_text, transform=ax2.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    
    # Convert to PIL Image
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    
    return img