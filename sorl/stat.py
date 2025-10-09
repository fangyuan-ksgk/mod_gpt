# Analyzer for abstraction statistics

from collections import defaultdict
import torch
from src.sorl import infer_level, sorl_search, SORLConfig, evaluate
from dataset.base import MemLoader
from transformers import AutoTokenizer
from model.model_sorl import SorlModelWrapper
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from PIL import Image
from io import BytesIO
from pathlib import Path
import numpy as np

def collect_abstraction_table(model: SorlModelWrapper, eval_loader: MemLoader, sorl_config: SORLConfig, tokenizer: AutoTokenizer):

    abs_table = defaultdict(list)
    val_table = defaultdict(list)
    N_INSPECTION_BATCHES = 100

    for _ in range(N_INSPECTION_BATCHES): 
        data, loss_mask = eval_loader.get_batch(sorl_config.val_batch_size)

        with torch.no_grad():
            # Using sorl_search to get the *best* abstraction for each sample
            search_data, _ = sorl_search(data, loss_mask, model, sorl_config)

        levels = infer_level(search_data, model.vocab_sizes, model.level_mask_tokens[0])
        
        for i in range(search_data.shape[0]):
            sample_tokens = search_data[i]
            sample_levels = levels[i]
            
            # Find the abstraction token in this sample
            abs_indices = (sample_levels > 0).nonzero(as_tuple=True)[0]
            if len(abs_indices) == 0:
                continue
            abstraction_token = sample_tokens[abs_indices[0]].item()

            # Decode the full sequence to find the value 'n'
            # This is more robust than splitting, in case of tokenization artifacts
            decoded_str = tokenizer.decode(sample_tokens, skip_special_tokens=True)
            try:
                # Find the value after the last '='
                value = int(decoded_str.split('=')[-1])
                
                abs_table[abstraction_token].append(value)
                val_table[value].append(abstraction_token)
            except (ValueError, IndexError):
                continue

    return abs_table, val_table

def plot_abstraction_preference(val_table, title_str: str = "Abstraction Preference Matrix"):

    all_abs_tokens = []
    all_values = []
    for val, abs_list in val_table.items():
        for abs_token in abs_list:
            all_values.append(val)
            all_abs_tokens.append(abs_token)

    if all_abs_tokens:
        df = pd.DataFrame({'Value': all_values, 'Abstraction': all_abs_tokens})
        
        plt.figure(figsize=(10, 8))
        confusion_matrix = pd.crosstab(df['Value'], df['Abstraction'])
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap="YlGnBu")
        plt.title(title_str)
        plt.xlabel('Abstraction Token ID')
        plt.ylabel('Input Value (n)')
        plt.show()
    else:
        print("No data to plot.")

def evaluate_advantage(model: SorlModelWrapper, eval_loader: MemLoader, sorl_config: SORLConfig):
    g_adv = []
    b_adv = []
    g_info_gain = []
    g_loss = []
    a_loss = []
    for _ in range(sorl_config.val_iterations): 
        data, loss_mask = eval_loader.get_batch(sorl_config.val_batch_size)
        greedy_advantage, best_advantage, greedy_info_gain, greedy_loss, abstract_free_loss = evaluate(data, loss_mask, sorl_config, model, search_n=1)
        g_adv.append(greedy_advantage)
        b_adv.append(best_advantage)
        g_info_gain.append(greedy_info_gain)
        g_loss.append(greedy_loss)
        a_loss.append(abstract_free_loss)

    g_info_gain = torch.stack(g_info_gain).mean()
    greedy_adv = torch.stack(g_adv).mean()
    best_adv = torch.stack(b_adv).mean()
    greedy_loss = torch.stack(g_loss).mean()
    abstract_free_loss = torch.stack(a_loss).mean()
    return greedy_adv, best_adv, g_info_gain, greedy_loss, abstract_free_loss

class AttentionMaskVisualizer:
    def __init__(self, tokens: torch.Tensor, levels: torch.Tensor, mask: torch.Tensor, tokenizer):
        # Ensure we are only visualizing a single sample (the first in the batch)
        if tokens.dim() > 1:
            tokens = tokens[0]
        if levels.dim() > 1:
            levels = levels[0]
        
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
            # Decode token and clean up extra characters for display
            token_str = self.tokenizer.decode([token_id]).replace(" ", "").replace("Ġ", "")
            
            level = self.levels[i]
            # Use a simpler (A1) format for abstraction labels, matching reference
            if level > 0:
                self.labels.append(f"(A{level})")
            else:
                self.labels.append(token_str if token_str else f"T_{token_id}")

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