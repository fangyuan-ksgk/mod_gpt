import torch
import numpy as np
from datasets import load_dataset
import tiktoken
from collections import Counter
from sorl.gat_sim import BOS_TOKEN_ID
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as PathEffects
from collections import Counter
from PIL import Image
import io
from sklearn.decomposition import PCA
from scipy.interpolate import griddata


class TinyStoriesDataLoader:
    """Lightweight TinyStories loader with n-chunk statistics."""
    # <== this class is actually quite general and doesn't depend on specific dataset
    
    def __init__(self, num_stories=1000, max_len=128, chunk_size=8, device='cpu', split='train'):
        self.max_len = max_len
        self.chunk_size = chunk_size
        self.device = device
        
        # Tokenizer
        self.enc = tiktoken.get_encoding("gpt2")
        self.eot = self.enc._special_tokens['<|endoftext|>'] - 1
        
        # Load and tokenize
        print(f"Loading {num_stories} stories from TinyStories {split}...")
        dataset = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
        
        self.stories = []
        for i, sample in enumerate(dataset):
            if i >= num_stories:
                break
            tokens = self.enc.encode_ordinary(sample['text'])[:max_len]
            self.stories.append(tokens)
        
        total_tokens = sum(len(s) for s in self.stories)
        total_bytes = total_tokens * 4
        total_mb = total_bytes / (1024 ** 2)

        print(f"Loaded {len(self.stories)} stories, {total_tokens} tokens total, {total_mb:.2f} MB")
        
        # Collect chunk statistics
        self.chunk_counts = self._collect_chunk_stats()
        print(f"Collected {len(self.chunk_counts)} unique {chunk_size}-chunks")
    
    def _collect_chunk_stats(self):
        """Vectorized collection of non-overlapping n-chunk statistics."""
        chunk_counts = Counter()
        
        for tokens in self.stories:
            # Truncate to multiple of chunk_size
            n_chunks = len(tokens) // self.chunk_size
            if n_chunks == 0:
                continue
            
            truncated = tokens[:n_chunks * self.chunk_size]
            
            # Reshape to [n_chunks, chunk_size]
            arr = np.array(truncated).reshape(n_chunks, self.chunk_size)
            
            # Convert each chunk to tuple for counting
            for chunk in arr:
                chunk_counts[tuple(chunk)] += 1
        
        return chunk_counts
    
    def get_batch(self, batch_size):
        """Get a random batch of stories, flattened for SoRL format."""
        indices = np.random.choice(len(self.stories), size=batch_size, replace=True)
        
        samples = []
        for idx in indices:
            tokens = self.stories[idx]
            tokens = tokens[:self.max_len - 1]
            # Pad or truncate to max_len - 1 (leave room for BOS)
            
            if len(tokens) < self.max_len - 1:
                tokens = tokens + [self.eot] * (self.max_len - 1 - len(tokens))
            else:
                tokens = tokens[:self.max_len - 1]
            
            sample = [BOS_TOKEN_ID] + tokens
            samples.append(sample)
        
        batch = torch.tensor(samples, dtype=torch.long, device=self.device)
        flat = batch.flatten().unsqueeze(0)
        
        return flat, torch.tensor(indices, dtype=torch.long, device=self.device)
    
    def get_top_chunks(self, n=50):
        """Return top-N most frequent chunks with decoded text."""
        top = self.chunk_counts.most_common(n)
        result = []
        for chunk_tuple, count in top:
            text = self.enc.decode(list(chunk_tuple))
            result.append((text, chunk_tuple, count))
        return result
    
    def decode(self, tokens):
        """Decode token ids to text."""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return self.enc.decode(tokens)

    def get_specific(self, indices):
        samples = []
        for idx in indices:
            tokens = self.stories[idx]
            # Pad or truncate to max_len - 1 (leave room for BOS)
            if len(tokens) < self.max_len - 1:
                tokens = tokens + [self.eot] * (self.max_len - 1 - len(tokens))
            else:
                tokens = tokens[:self.max_len - 1]
            
            sample = [BOS_TOKEN_ID] + tokens
            samples.append(sample)
        
        batch = torch.tensor(samples, dtype=torch.long, device=self.device)
        flat = batch.flatten().unsqueeze(0)
        
        return flat, indices.detach().clone().to(dtype=torch.long, device=flat.device)




# 
def extract_doc_components(seq_data, seq_ppt, doc_len, BOS_TOKEN_ID, model): 
    doc_starts = (seq_data[0] == BOS_TOKEN_ID).nonzero(as_tuple=False).squeeze(1)
    num_docs = len(doc_starts)
    indices = doc_starts.unsqueeze(1) + torch.arange(doc_len, device=seq_data.device)
    docs = seq_data[0, indices]  # [NumDocs, doc_len] <-- doc_ids is for NumDocs dimension

    ppt_indices = indices[:, 1:] - 1  # Adjust for offset, skip BOS column
    ppts = seq_ppt[0, ppt_indices]  # [NumDocs, doc_len-1]

    is_traj = (docs[:, 1:] < model.vocab_sizes[0]).float()
    traj_ppt = ppts * is_traj

    abs_seqs = docs[docs>=model.vocab_sizes[0]].reshape(num_docs, -1)
    abs_seq_utils = traj_ppt.mean(dim=1)
    return abs_seqs, abs_seq_utils


def collect_rollout_statistics(search_data, search_ppt, best_data, best_ppt, doc_len, BOS_TOKEN_ID, model, doc_ids):
    # rollout (greedy + random)
    abs_greedy, abs_util_greedy = extract_doc_components(search_data[:1], search_ppt[:1], doc_len, BOS_TOKEN_ID, model)
    abs_random, abs_util_random = extract_doc_components(search_data[1:], search_ppt[1:], doc_len, BOS_TOKEN_ID, model)

    # best rollout
    abs_best, _ = extract_doc_components(best_data, best_ppt.unsqueeze(0), doc_len, BOS_TOKEN_ID, model)

    # check (value, abs) equality to mark 'is_picked'
    is_picked_greedy = (abs_greedy == abs_best).all(dim=1)
    is_picked_random = (abs_random == abs_best).all(dim=1)
    is_picked_random = is_picked_random & ~is_picked_greedy # <- in case both are the same, we 'pick' greedy

    return (doc_ids, abs_greedy, abs_util_greedy, is_picked_greedy), (doc_ids, abs_random, abs_util_random, is_picked_random)


def compute_logit_similarity(abs_logits, abs_tokens, doc_ids): 
    
    # compute intra & cross doc abs (logits & tokens) similarity matrices
    per_doc_abs_logits = abs_logits.reshape(doc_ids.shape[0], -1, abs_logits.shape[-1])
    per_doc_abs_tokens = abs_tokens.reshape(doc_ids.shape[0], -1)
    normed = torch.nn.functional.normalize(per_doc_abs_logits, p=2, dim=2)

    # intra doc logit similarity
    sim_matrices = torch.bmm(normed, normed.transpose(1, 2))

    # cross doc logit similarity matrix (flatten instead of average)
    flat_doc_logits = per_doc_abs_logits.reshape(per_doc_abs_logits.shape[0], -1)  # [num_docs, n_abs * vocab_size]
    flat_doc_normed = torch.nn.functional.normalize(flat_doc_logits, p=2, dim=1)
    cross_doc_logit_sim = torch.mm(flat_doc_normed, flat_doc_normed.t())

    # cross doc hamming distance matrix
    num_docs = per_doc_abs_tokens.shape[0]
    cross_doc_hamming = torch.zeros(num_docs, num_docs, dtype=torch.float32, device=per_doc_abs_tokens.device)
    for i in range(num_docs):
        mismatches = (per_doc_abs_tokens[i:i+1] != per_doc_abs_tokens).float()
        cross_doc_hamming[i] = mismatches.sum(dim=1)

    return sim_matrices, cross_doc_logit_sim, cross_doc_hamming, per_doc_abs_tokens

@dataclass
class AbstractionStatistics:
    """Store only raw data, compute similarities on-demand."""
    n_doc: int
    n_abs: int
    abs_vocab_size: int
    device: str = 'cpu'
    
    def __post_init__(self):
        # Only store raw data
        self.abs_seqs = torch.zeros(self.n_doc, self.n_abs, dtype=torch.long, device=self.device)
        self.abs_logits = torch.zeros(self.n_doc, self.n_abs, self.abs_vocab_size, dtype=torch.float32, device=self.device)
        self.docs_updated = torch.zeros(self.n_doc, dtype=torch.bool, device=self.device)
        self.traj_perplexity = torch.zeros(self.n_doc, dtype=torch.float32, device=self.device) # - log p(s | a) for each document
        self.info_gain_reward = torch.zeros(self.n_doc, dtype=torch.float32, device=self.device) # info gain reward for each document

    def update(self, abs_logits, traj_loss, info_gain_reward, abs_tokens, doc_ids, ):
        """Store raw data only. Handles duplicate doc_ids by taking the first occurrence."""
        
        batch_size = doc_ids.shape[0]
        reshaped_logits = abs_logits.reshape(batch_size, self.n_abs, self.abs_vocab_size)
        reshaped_tokens = abs_tokens.reshape(batch_size, self.n_abs)
        if info_gain_reward.numel() == 1:
            info_gain_reward = info_gain_reward.repeat(batch_size)

        reshaped_info_gain_reward = info_gain_reward.reshape(batch_size)

        batch_doc_indices, inverse_indices = torch.unique(doc_ids, return_inverse=True)

        perm = torch.arange(inverse_indices.size(0), device=inverse_indices.device)

        unique_indices = torch.empty(batch_doc_indices.size(0), dtype=torch.long, device=self.device)
        unique_indices.scatter_(0, inverse_indices, perm) 

        selected_logits = reshaped_logits[unique_indices]
        selected_tokens = reshaped_tokens[unique_indices]
        selected_info_gain_reward = reshaped_info_gain_reward[unique_indices]
        selected_traj_loss = traj_loss[unique_indices]
        
        self.abs_seqs[batch_doc_indices] = selected_tokens
        self.abs_logits[batch_doc_indices] = selected_logits
        self.traj_perplexity[batch_doc_indices] = selected_traj_loss
        self.docs_updated[batch_doc_indices] = True
        self.info_gain_reward[batch_doc_indices] = selected_info_gain_reward

    @property 
    def valid_docs(self): 
        return self.docs_updated.nonzero().squeeze()

    @property
    def vocab_util(self): 
        return (self.abs_seqs.unique().size(0) / self.abs_vocab_size).item()

    @property 
    def bigram_rep_rate(self):
        repetitions = (self.abs_seqs[:, :-1] == self.abs_seqs[:, 1:]).float()
        rep_rate = repetitions.mean().item()
        return rep_rate

    def compute_cross_doc_logit_sim(self, method='flatten', docs=None):
        """Compute logit similarity on-demand."""
        if docs is None:
            docs = self.valid_docs
        
        logits = self.abs_logits[docs]
        
        if method == 'flatten':
            flat = logits.reshape(len(docs), -1)
            normed = torch.nn.functional.normalize(flat, p=2, dim=1)
        else:  # 'average'
            avg = logits.mean(dim=1)
            normed = torch.nn.functional.normalize(avg, p=2, dim=1)
        
        return torch.mm(normed, normed.t())
    
    def compute_cross_doc_hamming(self, docs=None):
        """Compute hamming distance on-demand."""
        if docs is None:
            docs = self.valid_docs
        
        seqs = self.abs_seqs[docs]
        n = len(docs)
        hamming = torch.zeros(n, n, dtype=torch.float32, device=self.device)
        for i in range(n):
            mismatches = (seqs[i:i+1] != seqs).float()
            hamming[i] = mismatches.sum(dim=1)
        return hamming

    # def save(self, path): 

    # @classmethod 
    # def load(cls, path): 


def visualize_alignment_compact(tokens, abs_tokens, doc_ids, model, enc, K=4, doc_idx=0, max_chunks=None):
    """Compact funnel-shaped visualization: wide text → compressed abstractions.
    
    Args:
        tokens: trajectory tokens
        abs_tokens: abstract tokens
        doc_ids: document IDs
        model: model with vocab_sizes
        enc: tokenizer encoder
        K: chunk size
        doc_idx: int or list of ints - document index/indices to visualize
        max_chunks: int or None - maximum chunks to show (truncate if longer)
    
    Returns:
        PIL.Image
    """
    
    # Handle single or multiple documents
    if isinstance(doc_idx, int):
        doc_indices = [doc_idx]
    else:
        doc_indices = doc_idx
    
    n_docs = len(doc_indices)
    
    abs_start = model.vocab_sizes[0]
    tokens_per_doc = tokens.reshape(doc_ids.shape[0], -1)[:, 1:]
    abs_per_doc = abs_tokens.reshape(doc_ids.shape[0], -1) - abs_start
    
    # Process each document to determine max width needed
    all_n_chunks = []
    for d_idx in doc_indices:
        traj_ids = tokens_per_doc[d_idx].tolist()
        n_chunks = len(traj_ids) // K
        if max_chunks is not None and n_chunks > max_chunks:
            all_n_chunks.append(max_chunks)
        else:
            all_n_chunks.append(n_chunks)
    
    max_n_chunks = max(all_n_chunks)
    
    # Create figure with proper scaling
    fig_width = max(10, max_n_chunks * 1.2)  # Scale width with chunks
    fig_height = 3.5 * n_docs
    fig, axes = plt.subplots(n_docs, 1, figsize=(fig_width, fig_height))
    if n_docs == 1:
        axes = [axes]
    
    for subplot_idx, d_idx in enumerate(doc_indices):
        ax = axes[subplot_idx]
        
        traj_ids = tokens_per_doc[d_idx].tolist()
        abs_seq = abs_per_doc[d_idx].tolist()
        
        n_chunks = len(traj_ids) // K
        
        # Apply max_chunks limit
        if max_chunks is not None and n_chunks > max_chunks:
            n_chunks_display = max_chunks
            is_truncated = True
        else:
            n_chunks_display = n_chunks
            is_truncated = False
        
        # Decode chunks
        chunks_text = []
        for i in range(n_chunks_display):
            chunk_ids = traj_ids[i*K:(i+1)*K]
            chunk_text = enc.decode(chunk_ids)
            chunks_text.append(chunk_text)
        
        # Setup axes
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 3)
        ax.axis('off')
        
        # Colors
        tab20 = plt.cm.tab20.colors
        colors = [tab20[tok % 20] for tok in abs_seq[:n_chunks_display]]
        
        # Title
        title = f"Doc {doc_ids[d_idx].item()}: Text ↔ Abstraction (K={K})"
        if is_truncated:
            title += f" [showing {n_chunks_display}/{n_chunks} chunks]"
        ax.text(5, 2.85, title, fontsize=18, weight='bold', ha='center')
        
        # Top row: Text chunks (WIDE)
        text_y = 2.2
        text_box_width = 10.0 / n_chunks_display
        for i in range(n_chunks_display):
            x_left = i * text_box_width
            x_center = x_left + text_box_width / 2
            text = chunks_text[i][:40] + "..." if len(chunks_text[i]) > 40 else chunks_text[i]
            
            # Text box
            ax.add_patch(FancyBboxPatch((x_left + 0.05, text_y - 0.15), 
                                       text_box_width - 0.1, 0.32,
                                       boxstyle="round,pad=0.04",
                                       facecolor=colors[i], alpha=0.25,
                                       edgecolor=colors[i], linewidth=1))
            ax.text(x_center, text_y, text, 
                   fontsize=8, ha='center', va='center')  # Keep original fontsize
        
        # Connecting lines (funnel shape)
        line_y_top = text_y - 0.2
        line_y_bottom = 0.95
        
        # Calculate compressed positions
        abs_total_width = min(6, n_chunks_display * 0.4)
        abs_start_x = (10 - abs_total_width) / 2
        abs_box_width = abs_total_width / n_chunks_display
        
        for i in range(n_chunks_display):
            x_top = i * text_box_width + text_box_width / 2
            x_bottom = abs_start_x + i * abs_box_width + abs_box_width / 2
            
            # Funnel line
            ax.plot([x_top, x_bottom], [line_y_top, line_y_bottom],
                   color=colors[i], alpha=0.6, linewidth=1.5, linestyle='-')
            
            # Arrow
            ax.annotate('', xy=(x_bottom, line_y_bottom - 0.05), 
                       xytext=(x_bottom, line_y_bottom),
                       arrowprops=dict(arrowstyle='->', lw=1.2, color=colors[i]))
        
        # Bottom row: Abstract tokens (NARROW)
        abs_y = 0.5
        for i in range(n_chunks_display):
            x_center = abs_start_x + i * abs_box_width + abs_box_width / 2
            abs_token = abs_seq[i]
            
            # Smaller box
            box_size = min(0.35, abs_box_width * 0.8)
            ax.add_patch(FancyBboxPatch((x_center - box_size/2, abs_y - 0.15), 
                                       box_size, 0.3,
                                       boxstyle="round,pad=0.05",
                                       facecolor=colors[i], alpha=0.9,
                                       edgecolor='black', linewidth=1.8))
            ax.text(x_center, abs_y, f"a{abs_token}", 
                   fontsize=9, ha='center', va='center', weight='bold')  # Keep original fontsize
        
        # Abstract sequence
        abs_str = ' → '.join([f"a{a}" for a in abs_seq[:n_chunks_display]])
        if is_truncated:
            abs_str += " → ..."
        ax.text(5, 0.1, f"Abstract: {abs_str}", 
               fontsize=8, ha='center', style='italic', color='gray')  # Keep original fontsize
    
    plt.tight_layout()
    
    # Convert to PIL
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    
    return img

def visualize_clustering(sim_matrix, abs_stats, abs_shift=50257, doc_ids=None, step=0, title="Story Abstraction Cluster", seed=42):
    """
    Clustering Map with embedded Distribution HUD (Bottom-Left).
    """
    # --- 1. Physics / PCA Projection ---
    if isinstance(sim_matrix, torch.Tensor):
        X = sim_matrix.cpu().numpy()
    else:
        X = sim_matrix
    
    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(X)
    
    # --- 2. Main Plot Setup ---
    plt.style.use('default')
    bg_color = '#ffffff'
    
    fig, ax = plt.subplots(figsize=(14, 14), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    
    # --- 3. Draw Clustering (Same as before) ---
    # Colors
    rpg_colors = ['#00a8cc', '#e63946', '#2a9d8f', '#e9c46a', '#9b5de5', '#f4a261']
    if doc_ids is not None:
        c_indices = [int(d.item()) % len(rpg_colors) for d in doc_ids]
        node_colors = [rpg_colors[i] for i in c_indices]
        labels = [str(d.item()) for d in doc_ids]
    else:
        n = len(X)
        node_colors = [rpg_colors[i % len(rpg_colors)] for i in range(n)]
        labels = [str(i) for i in range(n)]

    # Fixed Limits
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    
    # Grid & Connections
    ax.grid(True, color='#e0e0e0', linestyle='--', linewidth=0.8, alpha=0.5)
    
    dist_matrix = 1 - torch.tensor(X)
    threshold = torch.quantile(dist_matrix[dist_matrix > 0], 0.15)
    n = len(X)
    for i in range(n):
        for j in range(i+1, n):
            if dist_matrix[i, j] < threshold:
                alpha = 0.4 * (1 - (dist_matrix[i, j] / threshold).item())
                ax.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]], 
                       color='gray', alpha=alpha, linewidth=1, zorder=1)

    # Nodes
    ax.scatter(coords[:, 0], coords[:, 1], c='black', s=150, alpha=0.2, zorder=2)
    ax.scatter(coords[:, 0], coords[:, 1], c=node_colors, s=120, edgecolors='white', linewidth=1.5, zorder=3)

    # Labels (Simplified for brevity)
    # --- 4. Smart Labeling ---
    degrees = (dist_matrix < threshold).sum(dim=1).numpy()
    sorted_indices = np.argsort(-degrees)
    
    labeled_positions = []
    
    # Fixed scale params since axes are fixed
    min_dist = 0.25 # approx 8% of 3.0 range
    
    for idx in sorted_indices:
        x, y = coords[idx]
        
        # Check against fixed bounds to avoid labels flying off map
        if not (-1.4 < x < 1.4 and -1.4 < y < 1.4):
            continue

        is_crowded = False
        for (lx, ly) in labeled_positions:
            if abs(x - lx) < min_dist and abs(y - ly) < min_dist:
                is_crowded = True
                break
        
        if not is_crowded:
            label_text = f"Story {labels[idx]}"
            text = ax.text(x, y + 0.08, 
                          label_text, 
                          color='#333333', 
                          fontsize=15, 
                          fontweight='bold', 
                          ha='center', va='bottom', 
                          zorder=10)
            text.set_path_effects([PathEffects.withStroke(linewidth=4, foreground='white', alpha=0.9)])
            labeled_positions.append((x, y))

            
    # Titles
    ax.set_title(title, fontsize=20, color='#333333', weight='bold', pad=20)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    ax_hist = ax.inset_axes([0.02, 0.02, 0.50, 0.20]) 
    
    # Calculate Distribution
    valid_docs = abs_stats.docs_updated.nonzero().squeeze()
    if len(valid_docs.shape) == 0: valid_docs = valid_docs.unsqueeze(0)
    all_tokens = (abs_stats.abs_seqs[valid_docs] - abs_shift).flatten().cpu().numpy()
    vocab_size = abs_stats.abs_vocab_size 
            
    counts = np.bincount(all_tokens, minlength=vocab_size)
    counts = counts[:vocab_size]
    probs = counts / (counts.sum() + 1e-10)

    # Styling the Inset
    # Semi-transparent background box
    ax_hist.patch.set_facecolor('lightblue')
    ax_hist.patch.set_alpha(0.1)
    ax_hist.patch.set_edgecolor('#cccccc')
    ax_hist.patch.set_linewidth(1)
    
    # Plot Bars
    tab20 = plt.cm.tab20.colors
    bar_colors = [tab20[i % 20] for i in range(vocab_size)]
    ax_hist.bar(range(vocab_size), probs, color=bar_colors, alpha=0.9, width=0.8)
    
    # Minimalist Axis Styling
    ax_hist.set_ylim(0, 0.7)
    ax_hist.set_xticks(range(vocab_size))
    # Smaller fonts for the mini-map
    ax_hist.set_xticklabels([f"a{i}" for i in range(vocab_size)], fontsize=12, fontweight='bold', color='#444')
    ax_hist.set_yticks([]) # Hide Y values
    for spine in ['top', 'right', 'left']:
        ax_hist.spines[spine].set_visible(False)
        
    ax_hist.text(0.05, 0.85, f"P(A)", transform=ax_hist.transAxes, 
                 fontsize=20, fontweight='bold', color='#333')
    
    # --- 5. Stats Text Box (Top-Left) ---
    if isinstance(sim_matrix, torch.Tensor):
        mask = ~torch.eye(len(sim_matrix), dtype=torch.bool, device=sim_matrix.device)
        if len(sim_matrix) > 1:
            avg_sim = sim_matrix[mask].mean().item()
        else:
            avg_sim = 0.0
    else:
        avg_sim = 0.0
        
    # Hamming
    hamming = abs_stats.compute_cross_doc_hamming(docs=abs_stats.valid_docs)
    if len(hamming) > 1:
        mask_h = ~torch.eye(len(hamming), dtype=torch.bool, device=hamming.device)
        avg_ham = hamming[mask_h].mean().item()
    else:
        avg_ham = 0.0
    
    stats_text = f"Avg Logit Sim: {avg_sim:.3f}\nAvg Hamming Dist: {avg_ham:.1f}\nVocab Util: {abs_stats.vocab_util * 100:.1f}%"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=20,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # Update Title with Step
    ax.set_title(f"{title} (Step {step})", fontsize=20, color='#333333', weight='bold', pad=20)
    
    # =========================================================================

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor=bg_color)
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    return img

def stitch_images(img_align, img_cluster, padding=20):
    """
    Stitch Alignment (Left) and Cluster Map (Right) side-by-side.
    Centers them vertically.
    """
    # Calculate dimensions
    total_width = img_align.width + img_cluster.width + padding
    max_height = max(img_align.height, img_cluster.height)
    
    # Create blank canvas
    final_img = Image.new('RGB', (total_width, max_height), (255, 255, 255))
    
    # Paste Alignment (Left) - Center vertically
    y_align = (max_height - img_align.height) // 2
    final_img.paste(img_align, (0, y_align))
    
    # Paste Cluster Map (Right) - Center vertically
    y_cluster = (max_height - img_cluster.height) // 2
    final_img.paste(img_cluster, (img_align.width + padding, y_cluster))
    
    return final_img


def stitch_dashboard(img_align, img_cluster, img_3gram, padding=20):
    from PIL import Image, ImageDraw
    
    # Dimensions
    left_width = max(img_align.width, img_3gram.width)
    left_height = img_align.height + img_3gram.height + padding
    total_height = max(left_height, img_cluster.height)
    total_width = left_width + img_cluster.width + padding * 2 # Extra padding for separator
    
    canvas = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    # Paste Left
    x_align = (left_width - img_align.width) // 2
    canvas.paste(img_align, (x_align, 0))
    
    x_3gram = (left_width - img_3gram.width) // 2
    y_3gram = img_align.height + padding
    canvas.paste(img_3gram, (x_3gram, y_3gram))
    
    # Separator Line
    x_sep = left_width + padding
    draw.line([(x_sep, 20), (x_sep, total_height - 20)], fill='#eeeeee', width=3)
    
    # Paste Right
    x_cluster = left_width + padding * 2
    y_cluster = (total_height - img_cluster.height) // 2
    canvas.paste(img_cluster, (x_cluster, y_cluster))
    
    return canvas
    

def visualize_ngram_statistics(abs_stats, abs_shift=50257, n_gram_n=2, top_k=10, step=0):
    """
    Visualize top N-gram statistics
    """

    valid_docs = abs_stats.docs_updated.nonzero().squeeze()
    if len(valid_docs.shape) == 0: valid_docs = valid_docs.unsqueeze(0)
    seqs_list = (abs_stats.abs_seqs[valid_docs] - abs_shift).cpu().tolist()
    
    ngram_counts = Counter()
    total_ngrams = 0
    for seq in seqs_list:
        if len(seq) < n_gram_n: continue
        for i in range(len(seq) - n_gram_n + 1):
            ngram = tuple(seq[i : i+n_gram_n])
            ngram_counts[ngram] += 1
            total_ngrams += 1
    
    top_ngrams = ngram_counts.most_common(top_k)
    
    # --- VISUALIZATION ---
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    ax.set_facecolor('white')
    ax.axis('off')
    
    tab20 = plt.cm.tab20.colors
    
    ax.text(0.5, 0.95, f"Top {n_gram_n}-gram Patterns (Step {step})", 
            fontsize=20, weight='bold', ha='center', color='#333')
    
    # --- CARD VIEW for Trigrams+ ---
    y_start = 0.8
    y_step = 0.8 / top_k
    
    for i, (ngram, count) in enumerate(top_ngrams):
        pct = (count / total_ngrams) * 100
        y = y_start - i * y_step
        

        total_width = n_gram_n * 0.8 # approx width units
        start_x = 0.5 - (total_width / 20) # Roughly center
        
        bar_width = (pct / top_ngrams[0][1] * total_ngrams) * 0.8 # Scale relative to max
        
        # Draw Tokens
        x_cursor = 0.25
        for token in ngram:
            color = tab20[token % 20]
            # Token Box
            ax.add_patch(FancyBboxPatch((x_cursor, y - 0.04), 0.08, 0.08,
                                        boxstyle="round,pad=0.02",
                                        facecolor=color, alpha=0.9,
                                        edgecolor='black', linewidth=1))
            ax.text(x_cursor + 0.04, y, f"a{token}", ha='center', va='center', fontsize=18, weight='bold')
            
            # Arrow to next
            if token != ngram[-1]:
                ax.text(x_cursor + 0.1, y, "→", ha='center', va='center', color='gray', fontsize=18)
            
            x_cursor += 0.14
        
        # Frequency Label
        ax.text(0.8, y, f"{count} ({pct:.1f}%)", ha='left', va='center', fontsize=18, fontfamily='monospace')
        
        # Progress bar
        ax.add_patch(FancyBboxPatch((0.8, y - 0.04), pct/100 * 2.0, 0.01,
                                    boxstyle="round,pad=0.01", color='gray', alpha=0.3))

    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0); img = Image.open(buf).copy(); buf.close(); plt.close(fig)
    return img

    
def visualize_dynamics(abs_stats, loader, model, enc, K, step=0, doc_ids=torch.arange(0, 100, 34), max_chunks=8):
    tokens, _ = loader.get_specific(doc_ids)
    abs_tokens = abs_stats.abs_seqs[doc_ids]
    abs_tokens[abs_tokens == 0] += BOS_TOKEN_ID
    valid_ids = torch.arange(0, doc_ids.shape[0]).tolist()
    img_align = visualize_alignment_compact(tokens, abs_tokens, doc_ids, model, enc, K=K, doc_idx=valid_ids, max_chunks=8)

    doc_ids = abs_stats.valid_docs
    cs_sim = abs_stats.compute_cross_doc_logit_sim()
    img_sim = visualize_clustering(cs_sim, abs_stats, doc_ids=doc_ids, step=step, title="Story Cluster")

    img_3gram = visualize_ngram_statistics(abs_stats, n_gram_n=3)

    # stitch 2 images together (1 row 2 column)
    # final_img = stitch_images(img_align, img_sim)
    final_img = stitch_dashboard(img_align, img_sim, img_3gram)
    return final_img


def visualize_interleaved_alignment(sequence, model, enc, K=4, max_chunks=None):
    """
    Visualizes the alignment of an interleaved sequence (text + abstract tokens)
    in a compact, continuous stream.
    
    Args:
        sequence: 1D tensor or list of integers.
        model: Model object with vocab_sizes.
        enc: Tokenizer object.
        K: Chunk size (unused for logic, but kept for interface compatibility).
        max_chunks: Maximum number of groups to display.
    """
    if isinstance(sequence, torch.Tensor):
        sequence = sequence.flatten().tolist()
    
    abs_start = model.vocab_sizes[0]
    
    # --- 1. Parse Interleaved Sequence ---
    groups = []
    current_text = []
    current_abs = None
    
    for token in sequence:
        if token >= abs_start:
            if current_abs is not None or current_text:
                groups.append({'abs': current_abs, 'text': current_text})
            current_abs = token - abs_start
            current_text = []
        else:
            current_text.append(token)
            
    if current_abs is not None or current_text:
        groups.append({'abs': current_abs, 'text': current_text})

    # Slice to keep the MOST RECENT chunks (sliding window)
    if max_chunks is not None and len(groups) > max_chunks:
        groups = groups[-max_chunks:]

    n_groups = len(groups)
    if n_groups == 0:
        return Image.new('RGB', (200, 100), color='white')

    # --- 2. Setup Figure (Compact & Fixed Width) ---
    # Tighter spacing: box_width ~= spacing for continuous look
    box_width = 1.6
    spacing = 1.65  
    margin = 0.5
    
    # Use max_chunks for width calculation if available to keep GIF frame size constant
    width_reference = max_chunks if max_chunks is not None else n_groups
    fig_width = max(8, width_reference * spacing + margin * 2)
    
    fig_height = 3.5  # Reduced height
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0, fig_width)
    ax.set_ylim(0, 3.5)
    ax.axis('off')
    
    tab20 = plt.cm.tab20.colors
    
    # --- 3. Draw Groups ---
    start_x = margin + box_width / 2
    text_y_center = 2.2
    abs_y_center = 0.8
    
    for i, group in enumerate(groups):
        abs_id = group['abs']
        text_ids = group['text']
        
        x_center = start_x + i * spacing
        x_left = x_center - box_width / 2
        
        # Color Logic
        if abs_id is not None:
            color = tab20[abs_id % 20]
            face_alpha = 0.2
            edge_color = color
        else:
            color = 'gray'
            face_alpha = 0.05
            edge_color = 'lightgray'
            
        # --- Top: Text Chunk ---
        text_content = enc.decode(text_ids)
        if not text_content: 
            display_text = ""
        else:
            display_text = text_content.replace('\n', '\\n')
            if len(display_text) > 30: # Slightly tighter truncation
                display_text = display_text[:27] + "..."
        
        # Text Box (Continuous Stream)
        ax.add_patch(FancyBboxPatch(
            (x_left, text_y_center - 0.3),
            box_width, 0.6,
            boxstyle="round,pad=0.02,rounding_size=0.1", # Less rounded corners for continuity
            facecolor=color, alpha=face_alpha,
            edgecolor=edge_color, linewidth=1
        ))
        
        ax.text(x_center, text_y_center, display_text, 
                fontsize=10, ha='center', va='center', wrap=True, family='monospace')

        # --- Bottom: Abstract Token (Shrunk & Connected) ---
        if abs_id is not None:
            # Connection Line
            ax.plot([x_center, x_center], [text_y_center - 0.3, abs_y_center + 0.15], 
                    color=color, alpha=0.5, linewidth=1, linestyle='-')
            
            # Tiny Abstract Box
            abs_size = 0.35
            ax.add_patch(FancyBboxPatch(
                (x_center - abs_size/2, abs_y_center - abs_size/2),
                abs_size, abs_size,
                boxstyle="circle,pad=0.1", # Changed to circle/compact style
                facecolor=color, alpha=0.9,
                edgecolor='black', linewidth=1
            ))
            
            ax.text(x_center, abs_y_center, f"{abs_id}", 
                   fontsize=18, ha='center', va='center', weight='bold', color='black')
        else:
            ax.text(x_center, abs_y_center, "Start", 
                   fontsize=17, ha='center', va='center', color='gray')

    # Footer
    abs_seq_str = " → ".join([f"{g['abs']}" for g in groups if g['abs'] is not None])
    ax.text(fig_width/2, 0.1, f"Abstract Path: {abs_seq_str}", 
            fontsize=9, ha='center', style='italic', color='gray')
            
    plt.tight_layout()
    
    # Convert
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    
    return img


def fig_to_pil(fig):
    import io
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, 
                bbox_inches='tight', 
                pad_inches=0.3,  # <-- Add padding here
                facecolor='white')
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    return img


def visualize_perplexity_terrain(coords_2d, perplexity, trained_idx=None, step=0, resolution=50):
    """
    Smooth interpolated surface with VISIBLE trained point marker.
    """
    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    x = coords_2d[:, 0]
    y = coords_2d[:, 1]
    z = perplexity.cpu().numpy() if torch.is_tensor(perplexity) else np.array(perplexity)
    
    # Create grid for interpolation
    xi = np.linspace(x.min() - 0.1, x.max() + 0.1, resolution)
    yi = np.linspace(y.min() - 0.1, y.max() + 0.1, resolution)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolate
    Zi = griddata((x, y), z, (Xi, Yi), method='cubic')
    Zi_nearest = griddata((x, y), z, (Xi, Yi), method='nearest')
    Zi = np.where(np.isnan(Zi), Zi_nearest, Zi)
    
    # Plot surface with lower alpha so marker shows through
    surf = ax.plot_surface(Xi, Yi, Zi, cmap='coolwarm', alpha=0.6,
                           edgecolor='none', antialiased=True)
    
    # Scatter original points
    ax.scatter(x, y, z, c='gray', s=20, alpha=0.4, depthshade=False)
    
    # ===== PROMINENT TRAINED POINT MARKER =====
    if trained_idx is not None:
        tx, ty, tz = x[trained_idx], y[trained_idx], z[trained_idx]
        z_max = z.max()
        
        # Vertical line from surface to above (makes it pop)
        ax.plot([tx, tx], [ty, ty], [tz, z_max + 0.15], 
                color='black', linewidth=2, linestyle='--', alpha=0.8)
        
        # Star marker ABOVE the surface (elevated)
        ax.scatter([tx], [ty], [z_max + 0.15], 
                   c='red', s=500, marker='*', edgecolors='black', linewidth=2,
                   zorder=100, depthshade=False)
        
        # Also mark the actual point on the surface
        ax.scatter([tx], [ty], [tz], 
                   c='yellow', s=150, marker='o', edgecolors='black', linewidth=2,
                   zorder=99, depthshade=False)
        
        # Label
        ax.text(tx, ty, z_max + 0.2, f'Training: {trained_idx}', 
                fontsize=12, ha='center', fontweight='bold', color='red')
    
    ax.set_xlabel('Abstract Dim 1', fontsize=10)
    ax.set_ylabel('Abstract Dim 2', fontsize=10)
    ax.set_zlabel('Perplexity', fontsize=10)
    ax.set_title(f'Perplexity Terrain (Step {step})', fontsize=14)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, label='Perplexity')
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    return fig_to_pil(fig)

def visualize_forget_terrain(coords_2d, perplexity, trained_idx=None, step=0, resolution=50):
    """
    Smooth interpolated surface with VISIBLE trained point marker.
    """
    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.computed_zorder = False
    
    x = coords_2d[:, 0]
    y = coords_2d[:, 1]
    z = perplexity.cpu().numpy() if torch.is_tensor(perplexity) else np.array(perplexity)
    
    # Create grid for interpolation
    xi = np.linspace(x.min() - 0.1, x.max() + 0.1, resolution)
    yi = np.linspace(y.min() - 0.1, y.max() + 0.1, resolution)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolate
    Zi = griddata((x, y), z, (Xi, Yi), method='cubic')
    Zi_nearest = griddata((x, y), z, (Xi, Yi), method='nearest')
    Zi = np.where(np.isnan(Zi), Zi_nearest, Zi)
    
    # Plot surface
    surf = ax.plot_surface(Xi, Yi, Zi, cmap='coolwarm', alpha=0.6,
                           edgecolor='none', antialiased=True)

    # ===== PROMINENT TRAINED POINT MARKER =====
    if trained_idx is not None:
        tx, ty, tz = x[trained_idx], y[trained_idx], z[trained_idx]
        
        # 1. Star marker ABOVE (Larger & Red)
        ax.scatter([tx], [ty], [tz], 
                   c='red', s=600, marker='*', edgecolors='white', linewidth=1.5,
                   zorder=100, depthshade=False)
        
        # 3. Label with Box (Clear Annotation)
        ax.text(tx, ty, tz-1.6, f'Training Sample\n(Epicenter)', 
                fontsize=12, ha='center', fontweight='bold', color='black',
                bbox=dict(facecolor='white', alpha=1.0, edgecolor='red', boxstyle='round,pad=0.3'))
    
    # Intuitive Labels
    ax.set_xlabel('Abstract Rep (PCA 1)', fontsize=11)
    ax.set_ylabel('Abstract Rep (PCA 2)', fontsize=11)
    ax.set_zlabel('Forgetting (Δ Perplexity)', fontsize=11)
    ax.set_title(f'Abstraction Predicts Forgetting Terrain', fontsize=14)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, label='Magnitude of Forgetting')
    ax.view_init(elev=25, azim=45) # Original Angle Preserved
    
    plt.tight_layout()
    return fig_to_pil(fig)

def visualize_forget_terrain(coords_2d, perplexity, trained_idx=None, step=0, resolution=50):
    """
    Smooth interpolated surface with VISIBLE trained point marker.
    """
    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.computed_zorder = False
    
    x = coords_2d[:, 0]
    y = coords_2d[:, 1]
    z = perplexity.cpu().numpy() if torch.is_tensor(perplexity) else np.array(perplexity)
    
    # Create grid for interpolation
    xi = np.linspace(x.min() - 0.1, x.max() + 0.1, resolution)
    yi = np.linspace(y.min() - 0.1, y.max() + 0.1, resolution)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolate
    Zi = griddata((x, y), z, (Xi, Yi), method='cubic')
    Zi_nearest = griddata((x, y), z, (Xi, Yi), method='nearest')
    Zi = np.where(np.isnan(Zi), Zi_nearest, Zi)
    
    # Plot surface
    surf = ax.plot_surface(Xi, Yi, Zi, cmap='coolwarm', alpha=0.6,
                           edgecolor='none', antialiased=True)

    # ===== PROMINENT TRAINED POINT MARKER =====
    if trained_idx is not None:
        tx, ty, tz = x[trained_idx], y[trained_idx], z[trained_idx]
        
        # 1. Star marker ABOVE (Larger & Red)
        ax.scatter([tx], [ty], [tz], 
                   c='red', s=600, marker='*', edgecolors='white', linewidth=1.5,
                   zorder=100, depthshade=False)
        
        # 3. Label with Box (Clear Annotation)
        ax.text(tx, ty, tz-1.6, f'Training Sample\n(Epicenter)', 
                fontsize=12, ha='center', fontweight='bold', color='black',
                bbox=dict(facecolor='white', alpha=1.0, edgecolor='red', boxstyle='round,pad=0.3'))
    
    # Intuitive Labels
    ax.set_xlabel('Abstract Rep (PCA 1)', fontsize=11)
    ax.set_ylabel('Abstract Rep (PCA 2)', fontsize=11)
    ax.set_zlabel('Forgetting (Δ Perplexity)', fontsize=11)
    ax.set_title(f'Abstraction Predicts Forgetting Terrain', fontsize=14)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, label='Magnitude of Forgetting')
    ax.view_init(elev=25, azim=45) # Original Angle Preserved
    
    plt.tight_layout()
    return fig_to_pil(fig)


def visualize_perplexity_comparison(search_data, search_ppt, 
                                     abs_start=None, model=None, enc=None):
    """
    Side-by-side comparison of per-token perplexity.
    Uses 'A1' notation for abstract tokens.
    """
    greedy_ppt = search_ppt[0]
    greedy_seq = search_data[0]
    random_ppt = search_ppt[1]
    random_seq = search_data[1]

    if abs_start is None and model is not None:
        abs_start = model.vocab_sizes[0].item()
    
    # ... conversion to numpy ...
    greedy_seq = greedy_seq.cpu().numpy()
    greedy_ppt = greedy_ppt.cpu().numpy()
    random_seq = random_seq.cpu().numpy()
    random_ppt = random_ppt.cpu().numpy()
    
    greedy_ppl = np.exp(greedy_ppt)
    random_ppl = np.exp(random_ppt)
    
    # Helper to format labels
    def get_token_str(t):
        if t >= abs_start:
            return f"A{t - abs_start}"
        try:
            return enc.decode([t]).strip()
        except:
            return str(t)

    # Assuming sequences are aligned or roughly same length
    # We'll iterate and pair them up
    n = min(len(greedy_seq), len(random_seq)) - 1  # -1 because ppt is for next token
    
    labels = []
    is_abs = []
    
    for i in range(n):
        g_tok = greedy_seq[i+1] # Next token (prediction target)
        r_tok = random_seq[i+1]
        
        g_str = get_token_str(g_tok)
        r_str = get_token_str(r_tok)
        
        # Check if abstract
        g_is_abs = (g_tok >= abs_start)
        r_is_abs = (r_tok >= abs_start)
        
        if g_is_abs or r_is_abs:
            label = f"{g_str}|{r_str}"
            is_abs.append(True)
        else:
            label = g_str # Assuming traj tokens are identical until divergence
            is_abs.append(False)
            
        labels.append(label)

    # Plot
    fig, ax = plt.subplots(figsize=(18, 6))
    
    x = np.arange(n)
    width = 0.35
    
    # Greedy bars
    ax.bar(x - width/2, greedy_ppl[:n], width, label='Greedy', color='steelblue', alpha=0.8)
    
    # Random bars
    ax.bar(x + width/2, random_ppl[:n], width, label='Random', color='salmon', alpha=0.8)
    
    # Highlight Abstract Tokens (Red text for Ax)
    ax.set_xticks(x)
    
    # Color tick labels based on abstract status
    tick_labels = ax.set_xticklabels(labels, rotation=90, fontsize=10)
    for i, label in enumerate(tick_labels):
        if is_abs[i]:
            label.set_color('red')
            label.set_fontweight('bold')
    
    ax.set_yscale('log')
    ax.set_ylabel('Perplexity (Log Scale)')
    ax.set_title('Per-Token Perplexity: Greedy vs Random Rollout', fontsize=14)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    return fig_to_pil(fig)