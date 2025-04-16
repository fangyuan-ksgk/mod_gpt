import torch 
from pathlib import Path
import glob
import itertools
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# distributed data loader

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, sequence_length: int, rank : int, world_size : int):
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    assert sequence_length % world_size == 0
    local_seq_len = sequence_length // world_size
    file_iter = itertools.cycle(files) # iter(files) instead if you want to do 1-epoch training
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + sequence_length + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + rank * local_seq_len:][:local_seq_len + 1]
        inputs = buf[None, :-1].to(device="cuda", dtype=torch.int32, non_blocking=True) # no sync on host side;
        targets = buf[None, 1:].to(device="cuda", dtype=torch.int64, non_blocking=True) 
        pos += sequence_length
        yield inputs, targets

# -------------------------------------------------------------------------------


def plot_training_losses(loss_record, save_path="loss_curves.png"):
    """
    Plot entropy loss and rank loss curves on the same figure with different y-axes.
    """
    fig, ax1 = plt.figure(figsize=(10, 6)), plt.gca()
    x = np.arange(len(loss_record["entropy"]))
    
    # Plot entropy loss on primary y-axis
    color1 = 'tab:blue'
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Entropy Loss', color=color1)
    ax1.plot(x, loss_record["entropy"], 'o-', color=color1, label='Entropy Loss')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Create secondary y-axis and plot rank loss
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Rank Loss', color=color2)
    ax2.plot(x, loss_record["rank"], 's-', color=color2, label='Rank Loss')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add title and grid
    plt.title("Training Loss Curves")
    ax1.grid(True, alpha=0.3)
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Adjust layout and save
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"- Loss curves saved to {save_path}")
    
    
def compute_gradient_cosine_similarities(param_info):
    grad_arrays = param_info["grad_array"]
    loss_names = param_info["loss_name"]
    pair_similarities = {}    
    for i in range(len(grad_arrays)):
        for j in range(i+1, len(grad_arrays)):
            grad_i = grad_arrays[i].flatten()
            grad_j = grad_arrays[j].flatten()            
            dot_product = np.dot(grad_i, grad_j)
            norm_i = np.linalg.norm(grad_i)
            norm_j = np.linalg.norm(grad_j)
            cosine_sim = dot_product / (norm_i * norm_j)
            loss_pair = tuple(sorted([loss_names[i], loss_names[j]]))            
            if loss_pair not in pair_similarities:
                pair_similarities[loss_pair] = []
            pair_similarities[loss_pair].append(cosine_sim)
    results = []
    for loss_pair, similarities in pair_similarities.items():
        avg_similarity = sum(similarities) / len(similarities)
        results.append({
            "loss_pair": loss_pair,
            "cosine_similarity": avg_similarity
        })
    
    return results