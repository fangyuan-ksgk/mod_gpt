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


def plot_training_losses(loss_record, save_path="loss_curves.png", mbe_alpha=0.7, mbe_linestyle='--'):
    """
    Plot entropy loss and rank loss curves on the same figure with different y-axes.
    Emphasizes entropy loss by making MBE lines dashed and semi-transparent.
    Uses different colors for each layer's MBE loss.
    """
    print("Loss record: ") 
    print(loss_record)
    print("Plotting training loss curve ...")
    
    fig, ax1 = plt.subplots(figsize=(12, 7)) # Use plt.subplots
    
    # Determine the range of iterations based on entropy loss
    if "entropy" not in loss_record or not loss_record["entropy"]:
        print("Error: 'entropy' key missing or empty in loss_record.")
        plt.close(fig) # Close the empty figure
        return
    num_iterations = len(loss_record["entropy"])
    x = np.arange(num_iterations)

    # --- Plot Entropy Loss (Emphasized) ---
    color1 = 'tab:blue'
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Entropy Loss', color=color1, fontweight='bold') # Make label bold
    ax1.plot(x, loss_record["entropy"], 'o-', color=color1, label='Entropy Loss', linewidth=2.5) # Thicker line
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # --- Plot MBE Losses (De-emphasized) ---
    ax2 = ax1.twinx()
    ax2.set_ylabel('MBE Loss')

    mbe_keys = sorted([k for k in loss_record if 'mbe_' in k])
    if not mbe_keys:
         print("No 'mbe_' keys found to plot.")
         # Decide if you want to proceed without MBE plots or stop
    else:
        mbe_colors = plt.cm.viridis(np.linspace(0, 1, len(mbe_keys)))

        for i, loss_name in enumerate(mbe_keys):
             # Ensure MBE data has the same length as entropy data
             if len(loss_record[loss_name]) != num_iterations:
                 print(f"Warning: Length mismatch for {loss_name}. Skipping plot.")
                 continue # Skip this MBE plot

             layer_idx = loss_name.split("mbe_")[-1]
             current_color = mbe_colors[i]
             # Use alpha and linestyle to de-emphasize
             ax2.plot(x, loss_record[loss_name], 
                      marker='s', # Keep marker or remove if too cluttered
                      markersize=4, # Smaller marker
                      linestyle=mbe_linestyle, 
                      color=current_color, 
                      label=f"MBE Layer {layer_idx}", 
                      alpha=mbe_alpha) # Apply transparency

    ax2.tick_params(axis='y')

    # Add title and grid
    plt.title("Training Loss Curves (Entropy Emphasized)")
    ax1.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey', alpha=0.6)
    ax2.grid(True, which='major', linestyle=':', linewidth='0.5', color='grey', alpha=0.3) # More subtle grid for ax2

    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center left', bbox_to_anchor=(1.1, 0.5))

    # Adjust layout and save
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
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
            norm_ij = norm_i * norm_j 
            if norm_ij > 1e-9: 
                cosine_sim = float(dot_product / (norm_i * norm_j))
            else:
                cosine_sim = 1.0 
                
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


# ------- Scheduler for Rank Regularization -------

class RRScheduler: 
    # rotate on layer_idx, compute one layer rank regularization loss per train step
    def __init__(self, num_accumulation_steps, total_iterations, total_layer=12):
        self.num_accumulation_steps = num_accumulation_steps
        self.total_iterations = total_iterations
        self.current_accumulation_step = 0
        self.current_iteration = 0
        
        # Layer rotation setup
        self.layer_indices = list(range(2, total_layer))  # Layers 2-11
        self.current_layer_idx = 0
    
    def step(self):
        self.current_accumulation_step = (self.current_accumulation_step + 1) % self.num_accumulation_steps
        if self.current_accumulation_step == 0:
            self.current_iteration += 1
            self.current_layer_idx = (self.current_layer_idx + 1) % len(self.layer_indices)

    def _do_rr(self):
        return self.current_accumulation_step % 2 == 0
        
    @property
    def rr_layer_index(self): 
        if self._do_rr(): 
            return self.layer_indices[self.current_layer_idx: self.current_layer_idx + 1]
        else: 
            return []