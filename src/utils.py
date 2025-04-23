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
    print("Loss record: ") 
    print(loss_record)
    print("Plotting training loss curve ...")
    
    fig, ax1 = plt.figure(figsize=(10, 6)), plt.gca()
    x = np.arange(len(loss_record["entropy"]))
    
    # Plot entropy loss on primary y-axis
    color1 = 'tab:blue'
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Entropy Loss', color=color1)
    ax1.plot(x, loss_record["entropy"], 'o-', color=color1, label='Entropy Loss')
    ax1.tick_params(axis='y', labelcolor=color1)

    if 'mbe' not in loss_record: 
        layer_mbe = [k for k in loss_record.keys() if 'mbe' in k]
        loss_record["mbe"] = [sum(loss_record[k][i] for k in layer_mbe) / len(layer_mbe) 
                            for i in range(len(loss_record[layer_mbe[0]]))]
    
    # Create secondary y-axis and plot rank loss
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('MBE Loss', color=color2)
    ax2.plot(x, loss_record["mbe"], 's-', color=color2, label='MBE Loss')
    ax2.tick_params(axis='y', labelcolor=color2)

    # plot layer mbe loss too
    # Define a colormap for different layers
    layer_mbe = [k for k in loss_record.keys() if ('mbe' in k and k != 'mbe')]
    cmap = plt.cm.get_cmap('tab10', len(layer_mbe) + 1)  # +1 to avoid last color which might be too light
    
    for i, k in enumerate(layer_mbe):
        layer_idx = k.split('_')[-1]
        layer_color = cmap(i)
        ax2.plot(x, loss_record[k], 's-', color=layer_color, label=f'Layer {layer_idx} MBE Loss', alpha=0.1)

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

import pandas as pd 
import seaborn as sns 
import re
import math


def plot_layer_grad_cosine_similarity(data, layer_idx, steps_per_ckpt=None, ax=None):
    """
    Plots the cosine similarity for a layer onto a given matplotlib Axes object.
    If ax is None, creates a new figure (legacy behavior).

    Args:
        data (dict): Dictionary containing the training data.
        layer_idx (int): The index of the transformer layer to plot.
        steps_per_ckpt (int, optional): Number of training steps per checkpoint.
        ax (matplotlib.axes.Axes, optional): The Axes object to plot onto. If None, creates a new figure.

    Returns:
        bool: True if plotting was successful, False otherwise.
    """
    # If no Axes object is provided, create a new figure and axes for standalone plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 8))
        creating_own_figure = True
    else:
        creating_own_figure = False

    # --- Identify Parameter Keys for the Layer ---
    layer_prefix = f'transformer.h.{layer_idx}.'
    # Exclude 'lambda' parameters if they exist, sort for consistent legend order
    layer_param_keys = sorted([key for key in data.keys() if (key.startswith(layer_prefix) and 'lambda' not in key)])

    if not layer_param_keys:
        if creating_own_figure:
            print(f"Error: No parameter keys found for layer {layer_idx}.")
            plt.close(fig) # Close the empty figure
        # else: Silently return False if part of a composite figure
        return False

    # --- Determine number of checkpoints ---
    n_ckpts = 0
    first_valid_param = None
    for param_key in layer_param_keys:
        # Robust check for valid 'grad_angles' data
        if param_key in data and isinstance(data[param_key], dict) and \
           'grad_angles' in data[param_key] and isinstance(data[param_key]['grad_angles'], list) and \
           len(data[param_key]['grad_angles']) > 0:
            n_ckpts = len(data[param_key]['grad_angles'])
            first_valid_param = param_key
            break

    if n_ckpts == 0 or first_valid_param is None:
        if creating_own_figure:
            print(f"Warning: No valid gradient angle data points found for layer {layer_idx}. Cannot plot.")
            plt.close(fig)
        return False

    # --- X-axis Calculation ---
    x_indices = np.arange(n_ckpts)
    if steps_per_ckpt is not None:
        x_values = x_indices * steps_per_ckpt
        x_label = "Step"
    else:
        x_values = x_indices
        x_label = "Checkpoint Index"

    # --- Plotting ---
    plotted_something = False
    for param_key in layer_param_keys:
        # Additional checks within the loop for robustness
        if param_key not in data or 'grad_angles' not in data.get(param_key, {}):
             continue
        grad_angles_data = data[param_key]['grad_angles']
        if not isinstance(grad_angles_data, list) or len(grad_angles_data) != n_ckpts:
             print(f"Warning: Skipping '{param_key}' for layer {layer_idx}. Inconsistent checkpoints or invalid format.")
             continue

        cosine_similarities = []
        all_valid = True
        for i, item in enumerate(grad_angles_data):
            if isinstance(item, dict) and 'cosine_similarity' in item and isinstance(item['cosine_similarity'], (int, float)) and np.isfinite(item['cosine_similarity']):
                cosine_similarities.append(item['cosine_similarity'])
            else:
                print(f"Warning: Invalid/Non-finite data at index {i} for '{param_key}' layer {layer_idx}. Skipping parameter.")
                all_valid = False
                break

        if all_valid and len(cosine_similarities) == n_ckpts:
            short_label = param_key.replace(layer_prefix, '')
            ax.plot(x_values, cosine_similarities, marker='o', linestyle='-', label=short_label, markersize=4) # Smaller markers
            plotted_something = True

    if not plotted_something:
        if creating_own_figure:
            print(f"Warning: No data could be plotted for layer {layer_idx}.")
            plt.close(fig)
        return False

    # --- Configure the specific subplot (ax) ---
    ax.set_xlabel(x_label)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title(f"Layer {layer_idx}") # Use concise title for subplot
    ax.legend(title="Parameter", fontsize='small', title_fontsize='small') # Smaller legend font
    ax.grid(True)

    # If we created our own figure, adjust layout and show it
    if creating_own_figure:
        fig.tight_layout()
        plt.show()

    return True # Indicate successful plotting on the axes


def plot_all_layers_composite_figure(data, steps_per_ckpt=None, n_cols=3):
    """
    Generates a single composite figure with subplots for each layer's
    gradient consistency.

    Args:
        data (dict): Dictionary containing the training data.
        steps_per_ckpt (int, optional): Number of training steps per checkpoint.
        n_cols (int): Number of columns for the subplot grid. Defaults to 3.
    """
    # Find all unique layer indices
    layer_indices = set()
    for key in data.keys():
        match = re.match(r'transformer\.h\.(\d+)\.', key)
        if match:
            layer_indices.add(int(match.group(1)))

    if not layer_indices:
        print("Error: No layer data found ('transformer.h.{idx}.'). Cannot plot.")
        return

    sorted_indices = sorted(list(layer_indices))
    n_layers = len(sorted_indices)
    n_rows = math.ceil(n_layers / n_cols)

    # Create the figure and the grid of axes
    # Adjust figsize based on grid size - heuristic, might need tuning
    figsize_width = n_cols * 6
    figsize_height = n_rows * 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_width, figsize_height), squeeze=False)

    print(f"Generating composite plot for layers: {sorted_indices}...")

    plot_count = 0
    axes_flat = axes.flatten() # Flatten the 2D array of axes for easy iteration
    for i, layer_idx in enumerate(sorted_indices):
        ax = axes_flat[i] # Get the current subplot axes
        print(f"  Plotting Layer {layer_idx}...")
        success = plot_layer_grad_cosine_similarity(data, layer_idx, steps_per_ckpt=steps_per_ckpt, ax=ax)
        if success:
            plot_count += 1
        else:
            # Optional: Hide unused axes if plotting fails or layer is empty
            ax.set_visible(False)

    # Hide any remaining unused axes in the grid
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    if plot_count == 0:
        print("Error: No layers could be successfully plotted.")
        plt.close(fig) # Close the empty figure
        return

    # Add an overall title
    fig.suptitle("Gradient Cosine Similarity Across Layers", fontsize=16, y=1.02) # Adjust y to prevent overlap

    # Adjust layout to prevent overlapping titles/labels
    fig.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust rect to make space for suptitle

    plt.show()


def calculate_average_consistency(data):
    """
    Calculates the average gradient cosine similarity for each parameter across all checkpoints.

    Args:
        data (dict): Dictionary containing the training data.

    Returns:
        pandas.DataFrame: A DataFrame with columns ['layer', 'param_type', 'avg_consistency', 'param_key']
                          or None if no valid data is found.
    """
    results = []
    param_keys = sorted(data.keys()) # Sort for consistent processing order

    for param_key in param_keys: 
        # Extract layer index and parameter type using regex
        match = re.match(r'transformer\.h\.(\d+)\.(.+)', param_key)
        if not match or 'lambda' in param_key:
            # Skip keys not matching the expected format (e.g., embeddings, final layernorm)
            continue

        layer_idx = int(match.group(1))
        param_type = match.group(2) # e.g., 'attn.c_v.weight', 'mlp.c_proj.weight'

        if 'grad_angles' not in data.get(param_key, {}):
            # print(f"Debug: 'grad_angles' not found for {param_key}")
            continue

        grad_angles_data = data[param_key]['grad_angles']

        if not isinstance(grad_angles_data, list) or not grad_angles_data:
            # print(f"Debug: 'grad_angles' is not a list or is empty for {param_key}")
            continue

        # Extract valid cosine similarities
        similarities = []
        for item in grad_angles_data:
            if isinstance(item, dict) and 'cosine_similarity' in item and isinstance(item['cosine_similarity'], (int, float)):
                 # Basic check for NaN or Inf values which can skew averages
                 if np.isfinite(item['cosine_similarity']):
                    similarities.append(item['cosine_similarity'])
                 else:
                    print(f"Warning: Non-finite cosine similarity value found for {param_key}. Skipping this value.")

        if not similarities:
            # print(f"Debug: No valid similarities found for {param_key}")
            continue # Skip if no valid data points found

        # Calculate average
        avg_consistency = np.mean(similarities)

        results.append({
            'layer': layer_idx,
            'param_type': param_type,
            'avg_consistency': avg_consistency,
            'param_key': param_key # Keep original key for reference if needed
        })

    if not results:
        print("Warning: No average consistency data could be calculated.")
        return None

    # Convert to DataFrame for easier manipulation and visualization
    df = pd.DataFrame(results)
    return df


def visualize_average_consistency_heatmap(avg_consistency_df, title="Average Gradient Cosine Similarity"):
    """
    Visualizes the average gradient consistency using a heatmap.

    Args:
        avg_consistency_df (pandas.DataFrame): DataFrame from calculate_average_consistency.
        title (str): The title for the heatmap plot.
    """
    if avg_consistency_df is None or avg_consistency_df.empty:
        print("Error: Cannot visualize. Average consistency data is missing or empty.")
        return

    try:
        # Pivot the DataFrame to get layers as index and parameter types as columns
        heatmap_data = avg_consistency_df.pivot(index='layer', columns='param_type', values='avg_consistency')

        # Determine figure size based on data dimensions
        n_rows, n_cols = heatmap_data.shape
        # Adjust figsize heuristics as needed
        figsize_width = max(10, n_cols * 1.2)
        figsize_height = max(6, n_rows * 0.6)

        plt.figure(figsize=(figsize_width, figsize_height))
        sns.heatmap(heatmap_data, annot=True, fmt=".3f", linewidths=.5, cmap="coolwarm") # Or choose another cmap like 'coolwarm'
        plt.title(title)
        plt.xlabel("Parameter Type")
        plt.ylabel("Layer Index")
        plt.xticks(rotation=45, ha='right') # Rotate labels if they overlap
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error during heatmap visualization: {e}")
        print("DataFrame sample:\n", avg_consistency_df.head())


def plot_avg_consistency_across_layers(avg_consistency_df, title="Average Gradient Consistency Across Layers"):
    """
    Plots how the average gradient consistency for each parameter type changes across layers.

    Args:
        avg_consistency_df (pandas.DataFrame): DataFrame from calculate_average_consistency,
                                                containing 'layer', 'param_type', 'avg_consistency'.
        title (str): The title for the plot.
    """
    if avg_consistency_df is None or avg_consistency_df.empty:
        print("Error: Cannot visualize. Average consistency data is missing or empty.")
        return

    if not all(col in avg_consistency_df.columns for col in ['layer', 'param_type', 'avg_consistency']):
        print(f"Error: DataFrame must contain 'layer', 'param_type', and 'avg_consistency' columns. Found: {avg_consistency_df.columns}")
        return

    # Determine figure size based on number of param types for legend readability
    num_param_types = avg_consistency_df['param_type'].nunique()
    figsize_height = max(6, num_param_types * 0.4) # Adjust height based on legend items
    plt.figure(figsize=(12, figsize_height))

    # Use seaborn's lineplot for easy grouping by param_type
    sns.lineplot(
        data=avg_consistency_df,
        x='layer',
        y='avg_consistency',
        hue='param_type', # Creates a separate line for each param_type
        marker='o',       # Add markers to each data point
        style='param_type', # Optional: use different line styles too
        markers=True,
        dashes=False
    )

    plt.title(title)
    plt.xlabel("Layer Index")
    plt.ylabel("Average Cosine Similarity")
    plt.xticks(avg_consistency_df['layer'].unique()) # Ensure ticks for all layers present
    plt.grid(True, linestyle='--', alpha=0.7)
    # Place legend outside the plot for better readability
    plt.legend(title="Parameter Type", bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    plt.show()


# ------- Scheduler for Rank Regularization -------

class RRScheduler: 
    # rotate on layer_idx, compute one layer rank regularization loss per train step
    def __init__(self, 
                 num_accumulation_steps, 
                 total_iterations, 
                 total_layer=12, 
                 es_patience=1000, 
                 es_min_delta=0.001):
        self.num_accumulation_steps = num_accumulation_steps
        self.total_iterations = total_iterations
        self.current_accumulation_step = 0
        self.current_iteration = 0

        # Phase management & early stopping
        self.phase = 1
        self.es_patience = es_patience
        self.es_min_delta = es_min_delta
        self.best_val_loss = np.inf 
        self.patience_counter = 0 
        self.validation_checks_done = 0 

        # Layer rotation setup
        self.layer_indices = list(range(2, total_layer))  # layer 2 onwards
        self.current_layer_idx = 0
    
    def step(self):
        self.current_accumulation_step = (self.current_accumulation_step + 1) % self.num_accumulation_steps
        if self.current_accumulation_step == 0:
            self.current_iteration += 1
            self.current_layer_idx = (self.current_layer_idx + 1) % len(self.layer_indices)

    def log_validation_loss(self, val_loss):
        if self.phase == 2: 
            return True 
        self.validation_checks_done += 1
        improvement = self.best_val_loss - val_loss
        if improvement > self.es_min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0 # Reset patience counter
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.es_patience:
                self.phase = 2
                return True
            else:
                return False
        
    def _do_rr(self):
        return self.phase == 2 and self.current_accumulation_step % 2 == 0
        
    @property
    def rr_layer_index(self): 
        if self._do_rr(): 
            return self.layer_indices[self.current_layer_idx: self.current_layer_idx + 1]
        else: 
            return []