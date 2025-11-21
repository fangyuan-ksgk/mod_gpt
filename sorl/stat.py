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
                self.labels.append(f"(A({token_id - model.vocab_sizes[0]}))")
            else:
                self.labels.append(f"T_{token_id}")

            if self.tokenizer is not None:
                token_str = self.tokenizer.decode([token_id]).replace(" ", "").replace("Ġ", "")
            else:
                token_str = "".join(self.labels)

    def plot(self, title="Attention Mask") -> Image.Image:

        height = max(5, self.mask.shape[0] * 0.6)
        width = max(8, self.seq_len * 0.8)
        
        fig, ax = plt.subplots(figsize=(width, height))
        
        sns.heatmap(self.mask.astype(int), cmap="YlGnBu", linewidths=.5, cbar=False, annot=True, fmt='d', ax=ax)
        
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0)
        ax.set_xticks(torch.arange(self.seq_len) + 0.5)
        ax.set_xticklabels(self.labels, rotation=90)

        if self.mask.shape[0] > 1:
            ax.set_yticks(torch.arange(self.mask.shape[0]) + 0.5)
            # If mask is 2D, y-labels should match x-labels for attention matrices
            ax.set_yticklabels(self.labels, rotation=0)
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
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_position('center')

        # Use a qualitative colormap for distinct colors
        colors = plt.get_cmap('tab10', self.seq_len)

        for query_pos in range(self.seq_len):
            connections = np.where(self.mask[query_pos, :query_pos+1] > 0)[0]
            for key_pos in connections:
                if query_pos == key_pos:
                    continue
                
                center = (query_pos + key_pos) / 2
                radius = (query_pos - key_pos) / 2
                
                # Layer 1 arcs go above, Layer 2 arcs go below
                theta1, theta2 = (0, 180) if layer == 1 else (180, 360)

                arc = Arc((center, 0), 2 * radius, 2 * radius,
                          theta1=theta1, theta2=theta2,
                          edgecolor=colors(query_pos % 10), lw=2.5, zorder=query_pos)
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


# --------- Adaptive Phase Change Dynamic Visualization -------

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def plot_training_dynamics(data):
    """
    Visualizes training dynamics for search_adv and vocab_util with phase annotations.

    Args:
        data (defaultdict): A defaultdict containing lists for 'vocab_util', 
                            'search_adv', and 'alpha_loss'.
    """
    # Extract data
    vocab_util = data['vocab_util']
    search_adv = data['search_adv']
    alpha_loss = data['alpha_loss']
    steps = range(len(vocab_util))

    # --- Plotting Setup ---
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Create a second y-axis that shares the same x-axis
    ax2 = ax1.twinx()

    # --- Phase Annotation ---
    current_phase = 'Exploitation' if alpha_loss[0] > 0 else 'Exploration'
    phase_start = 0
    
    # Use a dictionary to avoid duplicate labels in the legend
    phase_labels = {}

    for i in range(1, len(alpha_loss)):
        next_phase_positive = alpha_loss[i] > 0
        current_phase_positive = alpha_loss[phase_start] > 0
        
        if next_phase_positive != current_phase_positive:
            phase_name = f'Exploitation (alpha={alpha_loss[phase_start]:.2f})' if current_phase_positive else f'Exploration (alpha={alpha_loss[phase_start]:.2f})'
            color = 'lightcoral' if current_phase_positive else 'lightblue'
            
            if phase_name not in phase_labels:
                phase_labels[phase_name] = ax1.axvspan(phase_start, i, color=color, alpha=0.3, label=phase_name)
            else:
                ax1.axvspan(phase_start, i, color=color, alpha=0.3)
            
            phase_start = i

    # Add the last phase block
    phase_name = f'Exploitation (alpha={alpha_loss[phase_start]:.2f})' if alpha_loss[phase_start] > 0 else f'Exploration (alpha={alpha_loss[phase_start]:.2f})'
    color = 'lightcoral' if alpha_loss[phase_start] > 0 else 'lightblue'
    if phase_name not in phase_labels:
        phase_labels[phase_name] = ax1.axvspan(phase_start, len(steps)-1, color=color, alpha=0.3, label=phase_name)
    else:
        ax1.axvspan(phase_start, len(steps)-1, color=color, alpha=0.3)

    # --- Plot Data ---
    # Plot vocab_util on the primary y-axis (ax1)
    p1, = ax1.plot(steps, vocab_util, 'b-', marker='o', label='Vocab Utilization')
    ax1.set_xlabel('Epochs', fontsize=16)
    ax1.set_ylabel('Vocab Utilization (%)', color='b', fontsize=16)
    ax1.tick_params(axis='y', labelcolor='b', labelsize=14)
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis='x', labelsize=14)

    # Plot search_adv on the secondary y-axis (ax2)
    p2, = ax2.plot(steps, np.array(search_adv), 'g-', marker='x', label='Search Advantage')
    ax2.set_ylabel('Search Advantage (%)', color='g', fontsize=16)
    ax2.tick_params(axis='y', labelcolor='g', labelsize=14)

    # --- Final Touches ---
    plt.title('Vocabulary Utilization vs. Search Advantage Dynamics', fontsize=18)
    fig.tight_layout()
    
    # Create a combined legend for lines and phases
    lines = [p1, p2] + list(phase_labels.values())
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=13)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()



def plot_vocab_abs_dynamics(data):
    """
    Visualizes training dynamics for vocab_util and abs_loss with phase annotations.

    Args:
        data (defaultdict): A defaultdict containing lists for 'vocab_util', 
                            'abs_loss', and 'alpha_loss'.
    """
    # Extract data
    vocab_util = data['vocab_util']
    abs_loss = data['abs_loss']
    alpha_loss = data['alpha_loss']
    steps = range(len(vocab_util))

    # --- Plotting Setup ---
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Create a second y-axis that shares the same x-axis
    ax2 = ax1.twinx()

    # --- Phase Annotation ---
    phase_start = 0
    
    # Use a dictionary to avoid duplicate labels in the legend
    phase_labels = {}

    for i in range(1, len(alpha_loss)):
        next_phase_positive = alpha_loss[i] > 0
        current_phase_positive = alpha_loss[phase_start] > 0
        
        if next_phase_positive != current_phase_positive:
            phase_name = f'Exploitation (alpha={alpha_loss[phase_start]:.2f})' if current_phase_positive else f'Exploration (alpha={alpha_loss[phase_start]:.2f})'
            color = 'lightcoral' if current_phase_positive else 'lightblue'
            
            if phase_name not in phase_labels:
                phase_labels[phase_name] = ax1.axvspan(phase_start, i, color=color, alpha=0.3, label=phase_name)
            else:
                ax1.axvspan(phase_start, i, color=color, alpha=0.3)
            
            phase_start = i

    # Add the last phase block
    phase_name = f'Exploitation (alpha={alpha_loss[phase_start]:.2f})' if alpha_loss[phase_start] > 0 else f'Exploration (alpha={alpha_loss[phase_start]:.2f})'
    color = 'lightcoral' if alpha_loss[phase_start] > 0 else 'lightblue'
    if phase_name not in phase_labels:
        phase_labels[phase_name] = ax1.axvspan(phase_start, len(steps)-1, color=color, alpha=0.3, label=phase_name)
    else:
        ax1.axvspan(phase_start, len(steps)-1, color=color, alpha=0.3)

    # --- Plot Data ---
    # Plot vocab_util on the primary y-axis (ax1)
    p1, = ax1.plot(steps, vocab_util, 'b-', marker='o', markersize=4, label='Vocab Utilization')
    ax1.set_xlabel('Epochs', fontsize=16)
    ax1.set_ylabel('Vocab Utilization (%)', color='b', fontsize=16)
    ax1.tick_params(axis='y', labelcolor='b', labelsize=14)
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis='x', labelsize=14)

    # Plot abs_loss on the secondary y-axis (ax2)
    p2, = ax2.plot(steps, abs_loss, 'r-', marker='x', markersize=4, label='Abstraction Loss')
    ax2.set_ylabel('Abstraction Loss', color='r', fontsize=16)
    ax2.tick_params(axis='y', labelcolor='r', labelsize=14)
    # Add horizontal line at y=0 for reference
    ax2.axhline(0, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)

    # --- Final Touches ---
    plt.title('Vocabulary Utilization vs. Abstraction Loss Dynamics', fontsize=18)
    fig.tight_layout()
    
    # Create a combined legend for lines and phases
    lines = [p1, p2] + list(phase_labels.values())
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=13)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def plot_training_metrics_combined(data):
    """
    Visualizes training dynamics for vocab_util, search_adv, abs_loss, and topo_sim on one plot
    with phase annotations.

    Args:
        data (defaultdict): A defaultdict containing lists for 'vocab_util', 
                            'search_adv', 'abs_loss', 'topo_sim', 'topo_sim_greedy', and 'phase'.
    """
    # Extract data
    vocab_util = data['vocab_util']
    search_adv = data['search_adv']
    abs_loss = data['abs_loss']
    topo_sim = data.get('topo_sim', None)
    topo_sim_greedy = data.get('topo_sim_greedy', None)
    if 'phase' not in data: 
        data['phase'] = ['exploration'] * len(vocab_util)
    phase = data['phase']
    steps = range(len(vocab_util))

    # --- Plotting Setup ---
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    # Create second, third, and fourth y-axes
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax4 = ax1.twinx()
    
    # Offset the third and fourth axes to the right
    ax3.spines['right'].set_position(('outward', 60))
    ax4.spines['right'].set_position(('outward', 130))

    # --- Phase Annotation ---
    phase_start = 0
    phase_labels = {}
    
    for i in range(1, len(phase)):
        if phase[i] != phase[phase_start]:
            # Phase transition detected
            phase_name = phase[phase_start].capitalize()
            color = 'lightcoral' if phase[phase_start] == 'exploitation' else 'lightblue'
            
            if phase_name not in phase_labels:
                phase_labels[phase_name] = ax1.axvspan(phase_start, i, color=color, alpha=0.3, label=phase_name)
            else:
                ax1.axvspan(phase_start, i, color=color, alpha=0.3)
            
            phase_start = i
    
    # Add the last phase block
    phase_name = phase[phase_start].capitalize()
    color = 'lightcoral' if phase[phase_start] == 'exploitation' else 'lightblue'
    if phase_name not in phase_labels:
        phase_labels[phase_name] = ax1.axvspan(phase_start, len(steps)-1, color=color, alpha=0.3, label=phase_name)
    else:
        ax1.axvspan(phase_start, len(steps)-1, color=color, alpha=0.3)

    # --- Plot Data ---
    # Plot vocab_util on the primary y-axis (ax1)
    p1, = ax1.plot(steps, vocab_util, 'b-', marker='o', markersize=3, 
                   linewidth=2, label='Vocab Utilization', alpha=0.8)
    ax1.set_xlabel('Epochs', fontsize=16)
    ax1.set_ylabel('Vocab Utilization (%)', color='b', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='b', labelsize=12)
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis='x', labelsize=12)

    # Plot search_adv on the secondary y-axis (ax2)
    p2, = ax2.plot(steps, search_adv, 'g-', marker='x', markersize=3, 
                   linewidth=2, label='Search Advantage', alpha=0.8)
    ax2.set_ylabel('Search Advantage (%)', color='g', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='g', labelsize=12)
    ax2.axhline(0, color='grey', linestyle='--', linewidth=0.8, alpha=0.3)

    # Plot abs_loss on the third y-axis (ax3)
    p3, = ax3.plot(steps, abs_loss, 'r-', marker='s', markersize=3, 
                   linewidth=2, label='Abstraction Loss', alpha=0.8)
    ax3.set_ylabel('Abstraction Loss', color='r', fontsize=14)
    ax3.tick_params(axis='y', labelcolor='r', labelsize=12)

    # Plot topo_sim metrics on the fourth y-axis (ax4)
    lines = [p1, p2, p3]
    if topo_sim is not None:
        p4, = ax4.plot(steps, topo_sim, 'purple', marker='^', markersize=3, 
                       linewidth=2, label='Topo Sim (random)', alpha=0.8, linestyle='--')
        lines.append(p4)
    if topo_sim_greedy is not None:
        p5, = ax4.plot(steps, topo_sim_greedy, 'magenta', marker='v', markersize=3, 
                       linewidth=2, label='Topo Sim (greedy)', alpha=0.8, linestyle='-.')
        lines.append(p5)
    
    if topo_sim is not None or topo_sim_greedy is not None:
        ax4.set_ylabel('Topological Similarity', color='purple', fontsize=14)
        ax4.tick_params(axis='y', labelcolor='purple', labelsize=12)
        ax4.axhline(0, color='grey', linestyle='--', linewidth=0.8, alpha=0.3)

    # --- Title and Legend ---
    plt.title('Training Dynamics: Vocab, Search Advantage, Abs Loss & Topo Sim', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Create a combined legend with metrics and phases
    lines = lines + list(phase_labels.values())
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=11, framealpha=0.9, ncol=2)
    
    # Grid on primary axis
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    fig.tight_layout()
    plt.show()


def save_training_dynamics(loss_record, save_path, run_info=""):
    """
    Visualizes training dynamics for vocab_util, search_adv, abs_loss, traj_loss, and topo_sim
    with phase annotations and saves to file.
    
    Args:
        loss_record (defaultdict): Contains 'traj_loss', 'abs_loss', 'search_advantage', 
                                   'util_rate', 'topo_sim', 'topo_sim_greedy', 
                                   'phase' (validation-level, optional),
                                   'train_phase', 'train_avg_util_rate' (training-level, optional)
        save_path (str): Path to save the figure (e.g., 'logs/run_id/training_dynamics.png')
        run_info (str): Optional info to add to title
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Convert validation metrics (tensors to numpy)
    traj_loss = [t.cpu().item() if torch.is_tensor(t) else t for t in loss_record['traj_loss']]
    abs_loss = [t.cpu().item() if torch.is_tensor(t) else t for t in loss_record['abs_loss']]
    search_adv = [t.cpu().item() * 100 if torch.is_tensor(t) else t * 100 
                  for t in loss_record['search_advantage']]  # Convert to percentage
    
    # Topo sim metrics (optional)
    topo_sim = None
    topo_sim_greedy = None
    if 'topo_sim' in loss_record and loss_record['topo_sim']:
        topo_sim = [t.cpu().item() if torch.is_tensor(t) else t for t in loss_record['topo_sim']]
    if 'topo_sim_greedy' in loss_record and loss_record['topo_sim_greedy']:
        topo_sim_greedy = [t.cpu().item() if torch.is_tensor(t) else t for t in loss_record['topo_sim_greedy']]
    
    # Use granular train data if available, otherwise fall back to validation data
    if 'train_avg_util_rate' in loss_record and loss_record['train_avg_util_rate']:
        vocab_util = [t * 100 if isinstance(t, (int, float)) else t for t in loss_record['train_avg_util_rate']]
        use_train_granularity = True
    else:
        vocab_util = [t.cpu().item() * 100 if torch.is_tensor(t) else t * 100 
                      for t in loss_record['util_rate']]
        use_train_granularity = False
    
    if 'train_phase' in loss_record and loss_record['train_phase']:
        phase = loss_record['train_phase']
    elif 'phase' in loss_record and loss_record['phase']:
        phase = loss_record['phase']
    else:
        phase = ['exploration'] * len(vocab_util)
    
    # Create step arrays for plotting
    # vocab_util and phase have training-level granularity (all steps)
    vocab_util_steps = list(range(len(vocab_util)))
    
    # Validation metrics are at validation checkpoint intervals
    num_val_checkpoints = len(traj_loss)
    if use_train_granularity and len(vocab_util) > num_val_checkpoints:
        # Map validation checkpoints to training step indices
        val_interval = len(vocab_util) / num_val_checkpoints
        val_steps = [int(i * val_interval) for i in range(num_val_checkpoints)]
    else:
        # No granular data, use simple indices
        val_steps = list(range(num_val_checkpoints))
        vocab_util_steps = val_steps
    
    # --- Plotting Setup ---
    fig, ax1 = plt.subplots(figsize=(18, 9))
    
    # Create second, third, fourth, and fifth y-axes
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax4 = ax1.twinx()
    ax5 = ax1.twinx()
    
    # Offset the axes
    ax3.spines['right'].set_position(('outward', 70))
    ax4.spines['right'].set_position(('outward', 140))
    ax5.spines['right'].set_position(('outward', 210))
    
    # --- Phase Annotation (use all training steps) ---
    phase_start = 0
    phase_labels = {}
    
    if len(phase) > 0:
        for i in range(1, len(phase)):
            if phase[i] != phase[phase_start]:
                phase_name = phase[phase_start].capitalize()
                color = 'lightcoral' if phase[phase_start] == 'exploitation' else 'lightblue'
                
                if phase_name not in phase_labels:
                    phase_labels[phase_name] = ax1.axvspan(phase_start, i, color=color, 
                                                           alpha=0.2, label=phase_name)
                else:
                    ax1.axvspan(phase_start, i, color=color, alpha=0.2)
                
                phase_start = i
        
        # Add the last phase block
        if phase_start < len(phase):
            phase_name = phase[phase_start].capitalize()
            color = 'lightcoral' if phase[phase_start] == 'exploitation' else 'lightblue'
            if phase_name not in phase_labels:
                phase_labels[phase_name] = ax1.axvspan(phase_start, len(vocab_util_steps)-1, color=color, 
                                                       alpha=0.2, label=phase_name)
            else:
                ax1.axvspan(phase_start, len(vocab_util_steps)-1, color=color, alpha=0.2)
    
    # --- Plot Data ---
    # ax1: Vocab Utilization (blue) - ALL training steps (granular)
    p1, = ax1.plot(vocab_util_steps, vocab_util, 'b-', marker='o', markersize=3, 
                   linewidth=2.0, label='Vocab Util (train)', alpha=0.9, markevery=max(1, len(vocab_util_steps)//50))
    ax1.set_xlabel('Training Step', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Vocab Utilization (%)', color='b', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='b', labelsize=12)
    ax1.set_ylim(-5, 105)
    ax1.tick_params(axis='x', labelsize=12)
    
    # ax2: Search Advantage (green) - validation checkpoints only
    p2, = ax2.plot(val_steps, search_adv, 'g-', marker='x', markersize=6, 
                   linewidth=2.5, label='Search Adv (val)', alpha=0.9)
    ax2.set_ylabel('Search Advantage (%)', color='g', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='g', labelsize=12)
    ax2.axhline(0, color='grey', linestyle='--', linewidth=1, alpha=0.5)
    
    # ax3: Trajectory Loss (purple) - validation checkpoints only
    p3, = ax3.plot(val_steps, traj_loss, 'purple', marker='s', markersize=5, 
                   linewidth=2.5, label='Traj Loss (val)', alpha=0.9)
    ax3.set_ylabel('Trajectory Loss', color='purple', fontsize=14, fontweight='bold')
    ax3.tick_params(axis='y', labelcolor='purple', labelsize=12)
    
    # ax4: Abstraction Loss (red) - validation checkpoints only
    p4, = ax4.plot(val_steps, abs_loss, 'r-', marker='d', markersize=5, 
                   linewidth=2.5, label='Abs Loss (val)', alpha=0.9)
    ax4.set_ylabel('Abstraction Loss', color='r', fontsize=14, fontweight='bold')
    ax4.tick_params(axis='y', labelcolor='r', labelsize=12)
    
    # ax5: Topological Similarity (purple/magenta) - validation checkpoints only
    lines = [p1, p2, p3, p4]
    if topo_sim is not None:
        p5, = ax5.plot(val_steps, topo_sim, 'purple', marker='^', markersize=5, 
                       linewidth=2.5, label='Topo Sim (random)', alpha=0.9, linestyle='--')
        lines.append(p5)
    if topo_sim_greedy is not None:
        p6, = ax5.plot(val_steps, topo_sim_greedy, 'magenta', marker='v', markersize=5, 
                       linewidth=2.5, label='Topo Sim (greedy)', alpha=0.9, linestyle='-.')
        lines.append(p6)
    
    if topo_sim is not None or topo_sim_greedy is not None:
        ax5.set_ylabel('Topological Similarity', color='purple', fontsize=14, fontweight='bold')
        ax5.tick_params(axis='y', labelcolor='purple', labelsize=12)
        ax5.axhline(0, color='grey', linestyle='--', linewidth=1, alpha=0.5)
    
    # --- Title and Legend ---
    title = 'Training Dynamics: Vocab, Search Adv, Traj Loss, Abs Loss, Topo Sim'
    if run_info:
        title += f'\n{run_info}'
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Combined legend
    lines = lines + list(phase_labels.values())
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=10, framealpha=0.95, 
               ncol=2 if len(lines) > 5 else 1)
    
    # Grid on primary axis
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    fig.tight_layout()
    
    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training dynamics saved to: {save_path}")
    plt.close(fig)  # Close to free memory