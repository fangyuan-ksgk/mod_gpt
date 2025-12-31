# Analyzer for abstraction statistics

from collections import defaultdict
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from PIL import Image
from io import BytesIO
from pathlib import Path
import numpy as np
from typing import Optional, Any
from sorl.gat_sim import BOS_TOKEN_ID

# def collect_abstraction_table(model: SorlModelWrapper, eval_loader: MemLoader, sorl_config: SORLConfig, tokenizer: AutoTokenizer):

#     abs_table = defaultdict(list)
#     val_table = defaultdict(list)
#     N_INSPECTION_BATCHES = 100

#     for _ in range(N_INSPECTION_BATCHES): 
#         data, loss_mask = eval_loader.get_batch(sorl_config.val_batch_size)

#         with torch.no_grad():
#             # Using sorl_search to get the *best* abstraction for each sample
#             search_data, _ = sorl_search(data, loss_mask, model, sorl_config)

#         levels = infer_level(search_data, model.vocab_sizes, model.level_mask_tokens[0])
        
#         for i in range(search_data.shape[0]):
#             sample_tokens = search_data[i]
#             sample_levels = levels[i]
            
#             # Find the abstraction token in this sample
#             abs_indices = (sample_levels > 0).nonzero(as_tuple=True)[0]
#             if len(abs_indices) == 0:
#                 continue
#             abstraction_token = sample_tokens[abs_indices[0]].item()

#             # Decode the full sequence to find the value 'n'
#             # This is more robust than splitting, in case of tokenization artifacts
#             decoded_str = tokenizer.decode(sample_tokens, skip_special_tokens=True)
#             try:
#                 # Find the value after the last '='
#                 value = int(decoded_str.split('=')[-1])
                
#                 abs_table[abstraction_token].append(value)
#                 val_table[value].append(abstraction_token)
#             except (ValueError, IndexError):
#                 continue

#     return abs_table, val_table

# def plot_abstraction_preference(val_table, title_str: str = "Abstraction Preference Matrix"):

#     all_abs_tokens = []
#     all_values = []
#     for val, abs_list in val_table.items():
#         for abs_token in abs_list:
#             all_values.append(val)
#             all_abs_tokens.append(abs_token)

#     if all_abs_tokens:
#         df = pd.DataFrame({'Value': all_values, 'Abstraction': all_abs_tokens})
        
#         plt.figure(figsize=(10, 8))
#         confusion_matrix = pd.crosstab(df['Value'], df['Abstraction'])
#         sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap="YlGnBu")
#         plt.title(title_str)
#         plt.xlabel('Abstraction Token ID')
#         plt.ylabel('Input Value (n)')
#         plt.show()
#     else:
#         print("No data to plot.")

# def evaluate_advantage(model: SorlModelWrapper, eval_loader: MemLoader, sorl_config: SORLConfig):
#     g_adv = []
#     b_adv = []
#     g_info_gain = []
#     g_loss = []
#     a_loss = []
#     for _ in range(sorl_config.val_iterations): 
#         data, loss_mask = eval_loader.get_batch(sorl_config.val_batch_size)
#         greedy_advantage, best_advantage, greedy_info_gain, greedy_loss, abstract_free_loss = evaluate(data, loss_mask, sorl_config, model, search_n=1)
#         g_adv.append(greedy_advantage)
#         b_adv.append(best_advantage)
#         g_info_gain.append(greedy_info_gain)
#         g_loss.append(greedy_loss)
#         a_loss.append(abstract_free_loss)

#     g_info_gain = torch.stack(g_info_gain).mean()
#     greedy_adv = torch.stack(g_adv).mean()
#     best_adv = torch.stack(b_adv).mean()
#     greedy_loss = torch.stack(g_loss).mean()
#     abstract_free_loss = torch.stack(a_loss).mean()
#     return greedy_adv, best_adv, g_info_gain, greedy_loss, abstract_free_loss


def materialize_attention_mask(idx, model, memory_span, attn_blocksize):
    levels = (idx >= model.vocab_sizes[0]).long()
    docs = (idx == BOS_TOKEN_ID).cumsum(1)

    batch_size, seq_len = docs.shape
    device = docs.device
    
    q_pos = torch.arange(seq_len, device=device).view(seq_len, 1)
    kv_pos = torch.arange(seq_len, device=device).view(1, seq_len)
    
    b = 0
    causal = q_pos >= kv_pos
    
    doc_q = docs[b].view(seq_len, 1)
    doc_kv = docs[b].view(1, seq_len)
    document = (doc_q == doc_kv)
    
    window = (q_pos - kv_pos) < attn_blocksize
    
    level_kv = levels[b].view(1, seq_len)
    is_higher_level = (level_kv > 0)
    is_recent = (q_pos - kv_pos) <= memory_span
    memory_compression = is_higher_level | is_recent
    
    final_mask = causal & document & window & memory_compression
    return final_mask    

def materialize_attention_mask_dream(idx, model, memory_span_abs, memory_span_traj, attn_blocksize):
    levels = (idx >= model.vocab_sizes[0]).long()
    accum_levels = levels.cumsum(1)
    docs = (idx == BOS_TOKEN_ID).cumsum(1)

    batch_size, seq_len = docs.shape
    device = docs.device
    
    q_pos = torch.arange(seq_len, device=device).view(seq_len, 1)
    kv_pos = torch.arange(seq_len, device=device).view(1, seq_len)
    
    b = 0
    causal = q_pos >= kv_pos
    
    doc_q = docs[b].view(seq_len, 1)
    doc_kv = docs[b].view(1, seq_len)
    document = (doc_q == doc_kv)
    
    window = (q_pos - kv_pos) < attn_blocksize

    levels_kv = levels[b].view(1, seq_len)
    to_abstract = (levels_kv > 0)
    levels_q = levels[b].view(seq_len, 1)
    from_abstract = (levels_q > 0)
    traj_memory_span = (q_pos - kv_pos) <= memory_span_traj
    abs_memory_span = (q_pos - kv_pos) <= memory_span_abs

    accum_levels_kv = accum_levels[b].view(1, seq_len)
    accum_levels_q = accum_levels[b].view(seq_len, 1)
    skip_abs = accum_levels_q > accum_levels_kv
    memory_compression_mask = to_abstract | (from_abstract & abs_memory_span) | (~from_abstract & traj_memory_span & ~skip_abs)
    
    final_mask = causal & document & window & memory_compression_mask
    return final_mask   


class AttentionMaskVisualizer:
    def __init__(self, tokens: torch.Tensor, model, mask: torch.Tensor, tokenizer: Optional[Any] = None):
        # Ensure we are only visualizing a single sample (the first in the batch)
        if tokens.dim() > 1:
            tokens = tokens[0]
        
        levels = (tokens >= model.vocab_sizes[0]).long()
        while mask.dim() > 2:
            mask = mask[0]

        self.tokens = tokens.cpu().numpy()
        self.levels = levels.cpu().numpy()
        self.mask = mask.cpu().numpy()
        self.tokenizer = tokenizer
        
        if self.mask.ndim == 1:
            self.mask = np.expand_dims(self.mask, axis=0)

        self.seq_len = self.mask.shape[-1]
        
        self.labels = []
        for i in range(self.seq_len):
            token_id = self.tokens[i]
                        
            level = self.levels[i]
            # Use a simpler (A1) format for abstraction labels, matching reference
            if level > 0:
                self.labels.append(f"A{token_id - model.vocab_sizes[0]}")
            else:
                self.labels.append(f"T{token_id}")

            if self.tokenizer is not None:
                token_str = self.tokenizer.decode([token_id]).replace(" ", "").replace("Ġ", "")
            else:
                token_str = "".join(self.labels)

    def plot(self, title="Attention Mask") -> Image.Image:

        height = max(5, self.mask.shape[0] * 0.6)
        width = max(8, self.seq_len * 0.8)
        
        fig, ax = plt.subplots(figsize=(width, height))
        
        sns.heatmap(self.mask.astype(int), cmap="YlGnBu", linewidths=.5, cbar=False, annot=False, ax=ax)
        
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0)
        ax.set_xticks(torch.arange(self.seq_len) + 0.5)
        ax.set_xticklabels(self.labels, rotation=90)
        
        # Color xtick labels: green for abstract (A), red for trajectory (T)
        for tick_label in ax.get_xticklabels():
            if tick_label.get_text().startswith('A'):
                tick_label.set_color('green')
            else:
                tick_label.set_color('red')

        if self.mask.shape[0] > 1:
            ax.set_yticks(torch.arange(self.mask.shape[0]) + 0.5)
            # If mask is 2D, y-labels should match x-labels for attention matrices
            ax.set_yticklabels(self.labels, rotation=0)
            # Color ytick labels: green for abstract (A), red for trajectory (T)
            for tick_label in ax.get_yticklabels():
                if tick_label.get_text().startswith('A'):
                    tick_label.set_color('green')
                else:
                    tick_label.set_color('red')
            ax.set_ylabel("Query")
        else:
            ax.set_yticks([])
            ax.set_ylabel("")

        ax.set_xlabel("Key / Sequence Position")
        ax.set_title(title, fontsize=14, pad=20)

        buf = BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
        plt.close(fig)
        
        buf.seek(0)
        return Image.open(buf)

    def plot_arcs(self, title="Attention Arcs", layer=1) -> Image.Image:
        fig, ax = plt.subplots(figsize=(self.seq_len * 0.8, self.seq_len * 0.4))
        
        # Prepare labels at the center
        ax.set_xlim(-1, self.seq_len)
        ax.set_ylim(-self.seq_len / 2.5, self.seq_len / 2.5)
        ax.set_xticks(np.arange(self.seq_len))
        ax.set_xticklabels(self.labels, fontsize=12)
        # Color tick labels: green for abstract (Ax), red for trajectory (Tx)
        for tick_label in ax.get_xticklabels():
            text = tick_label.get_text()
            if text.startswith('A'):
                tick_label.set_color('green')
            elif text.startswith('T'):
                tick_label.set_color('red')
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_position('center')

        for query_pos in range(self.seq_len):
            connections = np.where(self.mask[query_pos, :query_pos+1] > 0)[0]
            for key_pos in connections:
                if query_pos == key_pos:
                    continue
                
                # Green if either endpoint is abstract (A), red for trajectory-to-trajectory only
                involves_abstract = self.labels[key_pos].startswith('A') or self.labels[query_pos].startswith('A')
                edge_color = 'green' if involves_abstract else 'red'
                
                center = (query_pos + key_pos) / 2
                radius = (query_pos - key_pos) / 2
                
                # Layer 1 arcs go above, Layer 2 arcs go below
                theta1, theta2 = (0, 180) if layer == 1 else (180, 360)

                arc = Arc((center, 0), 2 * radius, 2 * radius,
                          theta1=theta1, theta2=theta2,
                          edgecolor=edge_color, lw=2.5, zorder=query_pos)
                ax.add_patch(arc)

        ax.set_title(title, fontsize=14, pad=20)
        
        buf = BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
        plt.close(fig)
        
        buf.seek(0)
        return Image.open(buf)


def plot_attn_arcs(tokens, model, memory_span):
    levels = (tokens >= model.vocab_sizes[0]).long()
    attn_mask = materialize_attention_mask(tokens, levels, memory_span, attn_blocksize=1792)
    visualizer = AttentionMaskVisualizer(tokens, levels, attn_mask)
    return visualizer.plot_arcs()

def plot_dream_attn_arcs(tokens, model, memory_span_abs, memory_span_traj):
    levels = (tokens >= model.vocab_sizes[0]).long()
    attn_mask = materialize_attention_mask_dream(tokens, model, memory_span_abs, memory_span_traj, attn_blocksize=1792)
    visualizer = AttentionMaskVisualizer(tokens, levels, attn_mask)
    return visualizer.plot_arcs()


def save_gif(frames, path, fps=6):
    """frames: list[Image.Image] -> gif at `path`"""
    if not frames:
        raise ValueError("No frames provided.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    duration_ms = int(1000 / fps)
    base = frames[0].convert("P", palette=Image.ADAPTIVE)
    tail = [im.convert("P", palette=Image.ADAPTIVE) for im in frames[1:]]

    base.save(
        path,
        format="GIF",
        save_all=True,
        append_images=tail,
        loop=0,
        duration=duration_ms,
        disposal=2,
    )

import ast, re

def extract_log_data(log_path):
    """
    Extract experiment configuration and loss records from log file.
    
    Handles two formats:
    1. Plain dict with floats: {'key': [1.0, 2.0, ...]}
    2. Tensor format: {'key': [tensor(1.0, device='cuda:0'), ...]}
    
    Returns:
        config (str): Experiment configuration string
        loss_record (dict): Dictionary of loss curves (values as floats)
    """
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    config = None
    loss_record = None
    
    for i in range(len(lines) - 1, -1, -1):
        if 'Experiment configuration:' in lines[i]:
            config = lines[i].split('Experiment configuration:')[-1].strip()
        
        if 'loss record:' in lines[i]:
            dict_str = lines[i+1].strip()
            
            # Remove defaultdict wrapper if present
            dict_str = re.sub(r"defaultdict\(<class 'list'>, ", "", dict_str)
            if dict_str.endswith(')') and 'defaultdict' not in dict_str:
                dict_str = dict_str[:-1]
            
            # Check if it contains tensor() format
            if 'tensor(' in dict_str:
                loss_record = _parse_tensor_dict(dict_str)
            else:
                try:
                    loss_record = ast.literal_eval(dict_str)
                except:
                    print(f"Failed to parse loss record from line {i+1}")
                    print(f"Dict string: {dict_str[:200]}...")
        
        if config is not None and loss_record is not None:
            break
    
    return config, loss_record


def _parse_tensor_dict(dict_str):
    """
    Parse a dictionary string containing tensor(...) values.
    Converts tensor(X.XXXX, device='cuda:0') to float X.XXXX
    
    Example input:
        {'key': [tensor(1.5, device='cuda:0'), tensor(2.3, device='cuda:0')]}
    Returns:
        {'key': [1.5, 2.3]}
    """
    result = {}
    
    # Find all keys and their list contents
    # Pattern: 'key_name': [list_content]
    key_pattern = r"'([^']+)':\s*\[([^\]]+)\]"
    
    for match in re.finditer(key_pattern, dict_str):
        key = match.group(1)
        list_content = match.group(2)
        
        # Extract all tensor values: tensor(VALUE, device='...')
        # Handles: tensor(1.5, device='cuda:0'), tensor(-0.0091, device='cuda:0')
        tensor_pattern = r"tensor\(([-\d.e]+)"
        values = [float(v) for v in re.findall(tensor_pattern, list_content)]
        
        result[key] = values
    
    return result

import io 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def create_sorl_training_frame(step_idx, loss_record, run_info, val_interval=125, axis_limits=None):
    """
    Creates a single frame for SoRL training dynamics animation.
    
    Args:
        step_idx: Current index in the loss arrays
        loss_record: Dictionary with keys like 'base_traj_loss', 'cond_traj_loss (search)', etc.
        val_interval: Training steps between validations
        axis_limits: Dict with precomputed axis limits for stable visualization
    """
    n_total = len(loss_record.get('base_traj_loss', []))
    all_steps = np.arange(n_total) * val_interval
    steps = np.arange(step_idx + 1) * val_interval
    
    # Extract data with correct keys
    base_loss = loss_record.get('base_traj_loss', [])[:step_idx + 1]
    cond_loss_greedy = loss_record.get('cond_traj_loss (greedy)', [])[:step_idx + 1]
    cond_loss_search = loss_record.get('cond_traj_loss (search)', [])[:step_idx + 1]
    greedy_adv = loss_record.get('greedy_adv', [])[:step_idx + 1]
    info_gain_greedy = loss_record.get('info_gain (greedy)', [])[:step_idx + 1]
    info_gain_search = loss_record.get('info_gain (search)', [])[:step_idx + 1]
    util_rate_greedy = loss_record.get('util_rate (greedy)', [])[:step_idx + 1]
    util_rate_search = loss_record.get('util_rate (search)', [])[:step_idx + 1]
    K_values = loss_record.get('K', [])[:step_idx + 1]
    
    # Colors (light mode friendly)
    c_base = '#2E86AB'       # Deep blue
    c_greedy = '#E94F37'     # Red-orange
    c_search = '#9B59B6'     # Purple
    c_adv = '#1B998B'        # Teal
    c_info_greedy = '#7B68EE'  # Medium slate blue
    c_info_search = '#FF6B6B'  # Coral
    c_util_greedy = '#F39C12'  # Orange
    c_util_search = '#27AE60'  # Green
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.patch.set_facecolor('white')
    
    for ax in axes:
        ax.set_facecolor('white')
        ax.grid(True, alpha=0.3, color='#cccccc')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Fixed x-axis limits
    x_min, x_max = 0, all_steps[-1]
    
    # Row 1: Trajectory Losses (base + greedy + search)
    axes[0].plot(steps, base_loss, color=c_base, linewidth=2, label='Base', alpha=0.9)
    axes[0].plot(steps, cond_loss_greedy, color=c_greedy, linewidth=2, label='Cond (Greedy)', alpha=0.9)
    axes[0].plot(steps, cond_loss_search, color=c_search, linewidth=2, label='Cond (Search)', alpha=0.9, linestyle='--')
    axes[0].set_ylabel('Loss', fontsize=10)
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].set_title('Trajectory Losses', fontsize=11, fontweight='bold')
    axes[0].set_xlim(x_min, x_max)
    axes[0].set_ylim(axis_limits['loss_min'], axis_limits['loss_max'])

    # Row 2: Advantage & Info Gain (dual y-axis)
    ax2_twin = axes[1].twinx()
    l1, = axes[1].plot(steps, greedy_adv, color=c_adv, linewidth=2, label='Greedy Adv', alpha=0.9)
    l2, = ax2_twin.plot(steps, info_gain_greedy, color=c_info_greedy, linewidth=2, label='Info Gain (Greedy)', alpha=0.9)
    l3, = ax2_twin.plot(steps, info_gain_search, color=c_info_search, linewidth=2, label='Info Gain (Search)', alpha=0.9, linestyle='--')
    axes[1].set_ylabel('Advantage', color=c_adv, fontsize=10)
    ax2_twin.set_ylabel('Info Gain', fontsize=10)
    ax2_twin.spines['top'].set_visible(False)
    axes[1].tick_params(axis='y', colors=c_adv)
    axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2_twin.axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[1].legend([l1, l2, l3], ['Greedy Adv', 'Info Gain (Greedy)', 'Info Gain (Search)'], loc='upper right', fontsize=8)
    axes[1].set_title('Advantage & Info Gain', fontsize=11, fontweight='bold')
    axes[1].set_xlim(x_min, x_max)
    axes[1].set_ylim(axis_limits['adv_min'], axis_limits['adv_max'])
    ax2_twin.set_ylim(axis_limits['info_min'], axis_limits['info_max'])
    
    # Row 3: Util Rate (greedy + search) + K indicator
    axes[2].plot(steps, util_rate_greedy, color=c_util_greedy, linewidth=2, label='Util Rate (Greedy)', alpha=0.9)
    axes[2].plot(steps, util_rate_search, color=c_util_search, linewidth=2, label='Util Rate (Search)', alpha=0.9, linestyle='--')
    axes[2].set_ylabel('Util Rate', fontsize=10)
    axes[2].set_xlabel('Training Step', fontsize=10)
    axes[2].set_xlim(x_min, x_max)
    axes[2].set_ylim(0, 1.05)
    axes[2].legend(loc='lower right', fontsize=8)
    
    # Add K value as text annotation
    if K_values:
        current_K = K_values[-1]
        axes[2].text(0.98, 0.95, f'K = {int(current_K)}', transform=axes[2].transAxes,
                     fontsize=12, fontweight='bold',
                     ha='right', va='top', bbox=dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='#cccccc'))
    axes[2].set_title('Vocab Utilization', fontsize=11, fontweight='bold')
    
    # Overall title
    fig.suptitle(f'SoRL Training Dynamics (Step {steps[-1]}) {run_info}', fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor='white')
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    
    return img


def generate_sorl_dynamics_gif(loss_record, save_path, run_info, val_interval=125, fps=5):
    """
    Generate a GIF animation of the SoRL training dynamics.
    """
    n_steps = len(loss_record.get('base_traj_loss', []))
    
    if n_steps == 0:
        print("No data found in loss_record")
        return
    
    # Precompute axis limits from ALL data
    all_base = loss_record.get('base_traj_loss', [])
    all_cond_greedy = loss_record.get('cond_traj_loss (greedy)', [])
    all_cond_search = loss_record.get('cond_traj_loss (search)', [])
    all_adv = loss_record.get('greedy_adv', [])
    all_info_greedy = loss_record.get('info_gain (greedy)', [])
    all_info_search = loss_record.get('info_gain (search)', [])
    
    # Compute fixed limits from full data
    loss_all = all_base + all_cond_greedy + all_cond_search
    info_all = all_info_greedy + all_info_search
    axis_limits = {
        'loss_min': min(loss_all) * 0.95,
        'loss_max': max(loss_all) * 1.05,
        'adv_min': min(all_adv) - 0.005,
        'adv_max': max(all_adv) + 0.005,
        'info_min': min(info_all) - 0.01,
        'info_max': max(info_all) + 0.01,
    }
    
    print(f"Fixed axis limits: {axis_limits}")
    
    frames = []
    for i in range(n_steps):
        frame = create_sorl_training_frame(i, loss_record, run_info, val_interval, axis_limits)
        frames.append(frame)
        if (i + 1) % 10 == 0:
            print(f"Generated frame {i + 1}/{n_steps}")
    
    # Save as GIF
    duration = int(1000 / fps)
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
    print(f"GIF saved to: {save_path}")