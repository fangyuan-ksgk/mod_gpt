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
from PIL import Image
import io
from sklearn.decomposition import PCA


class TinyStoriesDataLoader:
    """Lightweight TinyStories loader with n-chunk statistics."""
    # <== this class is actually quite general and doesn't depend on specific dataset
    
    def __init__(self, num_stories=1000, max_len=128, chunk_size=8, device='cpu', split='train'):
        self.max_len = max_len
        self.chunk_size = chunk_size
        self.device = device
        
        # Tokenizer
        self.enc = tiktoken.get_encoding("gpt2")
        self.eot = self.enc._special_tokens['<|endoftext|>']
        
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
    
    def update(self, abs_logits, abs_tokens, doc_ids):
        """Store raw data only. Handles duplicate doc_ids by taking the first occurrence."""
        
        batch_size = doc_ids.shape[0]
        reshaped_logits = abs_logits.reshape(batch_size, self.n_abs, self.abs_vocab_size)
        reshaped_tokens = abs_tokens.reshape(batch_size, self.n_abs)

        batch_doc_indices, inverse_indices = torch.unique(doc_ids, return_inverse=True)

        perm = torch.arange(inverse_indices.size(0), device=inverse_indices.device)

        unique_indices = torch.empty(batch_doc_indices.size(0), dtype=torch.long, device=self.device)
        unique_indices.scatter_(0, inverse_indices, perm) 

        selected_logits = reshaped_logits[unique_indices]
        selected_tokens = reshaped_tokens[unique_indices]
        
        self.abs_seqs[batch_doc_indices] = selected_tokens
        self.abs_logits[batch_doc_indices] = selected_logits
        self.docs_updated[batch_doc_indices] = True

    @property 
    def valid_docs(self): 
        return self.docs_updated.nonzero().squeeze()
    
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
        colors = plt.cm.tab20(np.array(abs_seq[:n_chunks_display]) / model.vocab_sizes[1])
        
        # Title
        title = f"Doc {doc_ids[d_idx].item()}: Text ↔ Abstraction (K={K})"
        if is_truncated:
            title += f" [showing {n_chunks_display}/{n_chunks} chunks]"
        ax.text(5, 2.85, title, fontsize=11, weight='bold', ha='center')
        
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

def visualize_clustering(sim_matrix, doc_ids=None, title="Knowledge Map", seed=42):
    """
    Stable, light-themed clustering visualization using PCA and fixed limits.
    Prevents jitter between frames in animation.
    """

    # --- 1. Stable Projection (PCA) ---
    # We project the similarity matrix itself (which is rotation invariant relative to the features)
    # Cosine similarity is bounded [-1, 1], so PCA projection space is roughly bounded.
    
    if isinstance(sim_matrix, torch.Tensor):
        # Convert to numpy for sklearn
        X = sim_matrix.cpu().numpy()
        device = sim_matrix.device
    else:
        X = sim_matrix
        device = 'cpu'
        
    # PCA is deterministic given the same data (unlike t-SNE)
    # and doesn't arbitrarily rotate like MDS might if eigvals switch order slightly
    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(X)
    
    # --- 2. Styling (Light Theme) ---
    plt.style.use('default')
    bg_color = '#ffffff'
    grid_color = '#e0e0e0'
    
    fig, ax = plt.subplots(figsize=(14, 10), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    
    # "RPG Class" Palette
    rpg_colors = ['#00a8cc', '#e63946', '#2a9d8f', '#e9c46a', '#9b5de5', '#f4a261']
    
    if doc_ids is not None:
        c_indices = [int(d.item()) % len(rpg_colors) for d in doc_ids]
        node_colors = [rpg_colors[i] for i in c_indices]
        labels = [str(d.item()) for d in doc_ids]
    else:
        n = len(sim_matrix)
        node_colors = [rpg_colors[i % len(rpg_colors)] for i in range(n)]
        labels = [str(i) for i in range(n)]

    # --- 3. Draw The Map ---
    
    # CRITICAL FIX FOR JITTER: Fixed Axis Limits
    # PCA on cosine sim matrix typically falls within [-1.5, 1.5] range
    # We fix the camera so the 'world' doesn't shake.
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    
    # A. Subtle Grid
    ax.grid(True, color=grid_color, linestyle='--', linewidth=0.8, alpha=0.5)

    # B. Connections (The "Roads")
    # Use distance derived from similarity for thresholding logic
    dist_matrix = 1 - torch.tensor(X)
    threshold = torch.quantile(dist_matrix[dist_matrix > 0], 0.15)
    
    n = len(X)
    for i in range(n):
        for j in range(i+1, n):
            if dist_matrix[i, j] < threshold:
                alpha = 0.4 * (1 - (dist_matrix[i, j] / threshold).item())
                ax.plot([coords[i, 0], coords[j, 0]], 
                       [coords[i, 1], coords[j, 1]], 
                       color='gray', 
                       alpha=alpha, linewidth=1, zorder=1)

    # C. Nodes
    ax.scatter(coords[:, 0], coords[:, 1], c='black', s=100, alpha=0.2, zorder=2)
    ax.scatter(coords[:, 0], coords[:, 1], c=node_colors, s=80, 
              edgecolors='white', linewidth=1.5, zorder=3)

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
            label_text = f"Doc {labels[idx]}"
            text = ax.text(x, y + 0.08, 
                          label_text, 
                          color='#333333', 
                          fontsize=10, 
                          fontweight='bold', 
                          ha='center', va='bottom', 
                          zorder=10)
            text.set_path_effects([PathEffects.withStroke(linewidth=4, foreground='white', alpha=0.9)])
            labeled_positions.append((x, y))

    # --- 5. Clean UI ---
    ax.set_title(title, fontsize=16, color='#333333', weight='bold', pad=20)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor=bg_color)
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    
    return img

def visualize_distribution(abs_stats, step=0, title="Abstract Token Distribution P(A)"):
    """
    Histogram of abstract token usage.
    """
    valid_docs = abs_stats.docs_updated.nonzero().squeeze()
    if len(valid_docs.shape) == 0: valid_docs = valid_docs.unsqueeze(0)
    
    all_tokens = abs_stats.abs_seqs[valid_docs].flatten().cpu().numpy()
    
    vocab_size = abs_stats.abs_vocab_size 
    
    if len(all_tokens) > 0 and all_tokens.max() >= vocab_size:
        if all_tokens.min() >= 50257: # approximate BOS
             all_tokens = all_tokens - all_tokens.min()
        else:
             all_tokens = all_tokens[all_tokens < vocab_size]

    counts = np.bincount(all_tokens, minlength=vocab_size)
    counts = counts[:vocab_size]
    probs = counts / (counts.sum() + 1e-10) 
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    colors = plt.cm.tab20(np.arange(vocab_size) / vocab_size)
    
    bars = ax.bar(range(vocab_size), probs, color=colors, edgecolor='black', linewidth=1.5)
    
    # X-Axis styling
    ax.set_xticks(range(vocab_size))
    ax.set_xticklabels([f"a{i}" for i in range(vocab_size)], fontsize=18, fontweight='bold')
    
    # Y-Axis styling (Matched size!)
    ax.set_ylabel("P(A)", fontsize=20, fontweight='bold')
    ax.tick_params(axis='y', labelsize=18)  # <--- Added this line
    ax.set_ylim(0, 1.0) 
    
    # Add values on top
    for bar in bars:
        height = bar.get_height()
        if height > 0.01: 
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=16, fontweight='bold')
            
    # Add step indicator
    ax.text(0.98, 0.9, f"STEP: {step}", transform=ax.transAxes, 
            fontsize=20, fontweight='bold', color='red', ha='right')

    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    return img

def stitch_images(img1, img2, padding=20):
    """Stitch two images horizontally with padding."""
    # Create new blank image
    total_width = img1.width + img2.width + padding
    max_height = max(img1.height, img2.height)
    
    new_img = Image.new('RGB', (total_width, max_height), (255, 255, 255))
    
    # Paste images
    # Center vertically if heights differ
    y1_offset = (max_height - img1.height) // 2
    y2_offset = (max_height - img2.height) // 2
    
    new_img.paste(img1, (0, y1_offset))
    new_img.paste(img2, (img1.width + padding, y2_offset))
    
    return new_img

def visualize_dynamics(abs_stats, loader, model, enc, K, doc_ids=torch.arange(0, 100, 25), max_chunks=8):
    tokens, _ = loader.get_specific(doc_ids)
    abs_tokens = abs_stats.abs_seqs[doc_ids]
    abs_tokens[abs_tokens == 0] += BOS_TOKEN_ID
    valid_ids = torch.arange(0, doc_ids.shape[0]).tolist()
    img_align = visualize_alignment_compact(tokens, abs_tokens, doc_ids, model, enc, K=4, doc_idx=valid_ids, max_chunks=8)

    # (c). visualize clustering of all stories' abstract logits
    doc_ids = abs_stats.valid_docs
    cs_sim = abs_stats.compute_cross_doc_logit_sim()
    img_sim = visualize_clustering(cs_sim, doc_ids=doc_ids, title="")

    # stitch 2 images together (1 row 2 column)
    final_img = stitch_images(img_align, img_sim)
    return final_img