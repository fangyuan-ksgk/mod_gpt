import os
import torch 
from pathlib import Path
import glob
import itertools
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from matplotlib.animation import FuncAnimation
import torch
from PIL import Image
from matplotlib.patches import Patch
import io
import torch.distributed as dist

# -----------------------------------------------------------------------------
# Auto-discover validation sets from data/ subdirectories

def discover_eval_suites(data_dir="data"):
    """
    Auto-discover validation sets from data/ subdirectories.
    Returns dict: {folder_name: "data/folder_name/*val*.bin"}
    """
    suites = {}
    if not os.path.isdir(data_dir):
        return suites
    for entry in os.listdir(data_dir):
        subdir = os.path.join(data_dir, entry)
        if os.path.isdir(subdir) and not entry.startswith("__"):
            # Check if val files exist
            val_pattern = os.path.join(subdir, "*val*.bin")
            if glob.glob(val_pattern):
                suites[entry] = val_pattern
    return suites


def discover_forget_suites(data_dir="data"):
    """
    Auto-discover validation sets from data/ subdirectories.
    Returns dict: {folder_name: "data/folder_name/*val*.bin"}
    """
    suites = {}
    if not os.path.isdir(data_dir):
        return suites
    for entry in os.listdir(data_dir):
        subdir = os.path.join(data_dir, entry)
        if os.path.isdir(subdir) and not entry.startswith("__"):
            # Check if val files exist
            val_pattern = os.path.join(subdir, "*.bin")
            if glob.glob(val_pattern):
                suites[entry] = val_pattern
    return suites

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

def distributed_data_generator_sorl(filename_pattern: str, sequence_length: int, rank : int, world_size : int):
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    assert sequence_length % world_size == 0
    local_seq_len = sequence_length // world_size
    file_iter = itertools.cycle(files) # iter(files) instead if you want to do 1-epoch training
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + sequence_length + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + rank * local_seq_len:][:local_seq_len + 1]        
        idx = buf[None, :-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        pos += sequence_length
        yield idx

# v2 is 2x slower, discarded for now
# def distributed_data_generator_sorl_v2(filename_pattern: str, sequence_length: int, rank: int, world_size: int):
#     files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
#     assert sequence_length % world_size == 0
#     local_len = sequence_length // world_size
#     file_iter = itertools.cycle(files) # iter(files) instead if you want to do 1-epoch training
#     tokens, pos = _load_data_shard(next(file_iter)), 0

#     while True:
#         start = pos + rank * local_len

#         matches = (tokens[start:] == 50256).nonzero() if start < len(tokens) else []
#         if len(matches) == 0 or start + matches[0] + local_len + 1 > len(tokens):
#             tokens, pos = _load_data_shard(next(file_iter)), 0
#             continue

#         real_start = start + matches[0].item()
#         buf = tokens[real_start : real_start + local_len + 1]
#         idx = buf[None, :-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
#         pos += sequence_length
#         yield idx


def distributed_data_generator_sorl_v3(filename_pattern: str, sequence_length: int, rank: int, world_size: int):
    files = itertools.cycle([Path(f) for f in sorted(glob.glob(filename_pattern))])
    local_len = sequence_length // world_size
    tokens, pos = _load_data_shard(next(files)), 0
    bos_locs = (tokens == 50256).nonzero(as_tuple=True)[0]

    while True:
        start = pos + rank * local_len
        idx_in_bos = torch.searchsorted(bos_locs, start)
        if idx_in_bos >= len(bos_locs):
            tokens, _ = _load_data_shard(next(files)), 0
            bos_locs = (tokens == 50256).nonzero(as_tuple=True)[0]
            pos = 0
            continue
            
        real_start = bos_locs[idx_in_bos].item()
        
        if real_start + local_len + 1 > len(tokens):
             tokens, _ = _load_data_shard(next(files)), 0
             bos_locs = (tokens == 50256).nonzero(as_tuple=True)[0]
             pos = 0
             continue

        yield tokens[real_start : real_start + local_len + 1][None, :-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        pos += sequence_length

# TBD. optionally include 'loss_mask' in the data generator

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
    

def plot_mbe(mbe_values):
    """
    Plot MBE loss per layer.
    
    Args:
        mbe_values: List of MBE loss values per layer
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Layer indices for x-axis
    layers = np.arange(len(mbe_values))
    
    # Create bar plot
    bars = ax.bar(layers, mbe_values, color='steelblue', alpha=0.8)
    
    # Highlight highest and lowest values
    max_idx = np.argmax(mbe_values)
    min_idx = np.argmin(mbe_values)
    bars[max_idx].set_color('red')
    bars[min_idx].set_color('green')
    
    # Add value labels on top of each bar
    for i, v in enumerate(mbe_values):
        ax.text(i, v + 0.02, f"{v:.4f}", ha='center', fontsize=9)
    
    # Add labels and title
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('MBE Loss')
    ax.set_title('Matrix-based Entropy per layer (Entropy target only)')
    
    # Set x-ticks to layer indices
    ax.set_xticks(layers)
    ax.set_xticklabels([f'Layer {i}' for i in layers])
    plt.xticks(rotation=45)
    
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add legend for highlighted bars
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Highest Loss'),
        Patch(facecolor='green', label='Lowest Loss'),
        Patch(facecolor='steelblue', label='Other Layers')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()


def compute_gradient_cosine_similarities(param_info):
    grad_arrays = param_info["grad_array"]
    n_grad = len(grad_arrays)
    loss_names = param_info["loss_name"][-n_grad:]
    pair_similarities = {}    
    for i in range(n_grad):
        for j in range(i+1, n_grad):
            grad_i = grad_arrays[i].flatten()
            grad_j = grad_arrays[j].flatten()            
            dot_product = np.dot(grad_i, grad_j)
            norm_i = np.linalg.norm(grad_i)
            norm_j = np.linalg.norm(grad_j)
            norm_ij = norm_i * norm_j 
            if norm_ij > 1e-9: 
                cosine_sim = float(dot_product / (norm_i * norm_j))
            else:
                cosine_sim = 0.0 
                
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


def plot_layer_grad_cosine_similarity(data, layer_idx, loss_pair, steps_per_ckpt=None, ax=None, max_pts=None):

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
        steps_per_ckpt = 1750 // n_ckpts
        x_values = x_indices * steps_per_ckpt
        x_label = "Step"

    # --- Collect and prepare data for plotting with average similarities ---
    param_data = []
    for param_key in layer_param_keys:
        # Additional checks for robustness
        if param_key not in data or 'grad_angles' not in data.get(param_key, {}):
            continue
        grad_angles_data = data[param_key]['grad_angles']
        if not isinstance(grad_angles_data, list) or len(grad_angles_data) != n_ckpts:
            print(f"Warning: Skipping '{param_key}' for layer {layer_idx}. Inconsistent checkpoints or invalid format.")
            continue

        cosine_similarities = []
        abs_cosine_similarities = []  # New list for absolute values
        valid_x_values = []
        all_valid = True
        for i, item in enumerate(grad_angles_data):
            if isinstance(item, dict) and 'cosine_similarity' in item and isinstance(item['cosine_similarity'], (int, float)) and np.isfinite(item['cosine_similarity']):
                if loss_pair == item['loss_pair']:
                    # Store both original and absolute values
                    cosine_similarities.append(item['cosine_similarity'])
                    abs_cosine_similarities.append(abs(item['cosine_similarity']))
                    valid_x_values.append(x_values[i])
            else:
                print(f"Warning: Invalid/Non-finite data at index {i} for '{param_key}' layer {layer_idx}. Skipping parameter.")
                all_valid = False
                break
        
        if max_pts is not None and len(valid_x_values) > max_pts: 
            n_interval = len(valid_x_values) // max_pts
            cosine_similarities = cosine_similarities[::n_interval]
            abs_cosine_similarities = abs_cosine_similarities[::n_interval]
            valid_x_values = valid_x_values[::n_interval]

        # Skip leading zeros
        while cosine_similarities and cosine_similarities[0] == 0: 
            cosine_similarities = cosine_similarities[1:]
            abs_cosine_similarities = abs_cosine_similarities[1:]
            valid_x_values = valid_x_values[1:]

        if all_valid and cosine_similarities:
            avg_abs_similarity = np.mean(abs_cosine_similarities)  # Average of absolute values
            similarity_variance = np.var(cosine_similarities)
            short_label = param_key.replace(layer_prefix, '') + f" (avg abs: {avg_abs_similarity:.4f}, var: {similarity_variance:.4f})"
            
            param_data.append({
                'param_key': param_key,
                'short_label': short_label,
                'cosine_similarities': cosine_similarities,  # Original values for plotting
                'valid_x_values': valid_x_values,
                'avg_abs_similarity': avg_abs_similarity  # For sorting
            })

    # --- Sort parameters by average absolute similarity (descending) ---
    param_data.sort(key=lambda x: x['avg_abs_similarity'], reverse=True)
    max_abs_similarity = param_data[0]['avg_abs_similarity']
    min_abs_similarity = max(param_data[-1]['avg_abs_similarity'] - 0.1, 0.0)

    # --- Plotting with varying line thicknesses ---
    plotted_something = False
    for i, data_item in enumerate(param_data):
        # Scale line thickness based on average absolute similarity
        line_thickness = 1.0 + (data_item['avg_abs_similarity'] - min_abs_similarity) / (max_abs_similarity - min_abs_similarity) * 4  # Adjust scaling factor as needed
        line_thickness = min(line_thickness, 5.0)  # Cap maximum thickness
        alpha = 0.5 + (data_item['avg_abs_similarity'] - min_abs_similarity) / (max_abs_similarity - min_abs_similarity) * 0.5  # Adjust scaling factor as needed
        
        ax.plot(
            data_item['valid_x_values'], 
            data_item['cosine_similarities'], 
            marker='o', 
            linestyle='-', 
            label=data_item['short_label'], 
            markersize=4,
            linewidth=line_thickness,
            alpha=alpha
        )
        plotted_something = True

    if not plotted_something:
        if creating_own_figure:
            print(f"Warning: No data could be plotted for layer {layer_idx}.")
            plt.close(fig)
        return False

    # --- Configure the specific subplot (ax) ---
    ax.set_xlabel(x_label)
    ax.set_ylabel("Cosine Similarity")  # Changed from "Absolute Cosine Similarity"
    ax.set_title(f"Layer {layer_idx} | Gradient cosine similarity for {loss_pair}")
    ax.legend(title="Parameter (sorted by avg abs similarity)", fontsize='small', title_fontsize='small')
    ax.grid(True)

    # If we created our own figure, adjust layout and show it
    if creating_own_figure:
        fig.tight_layout()
        plt.show()

    return True # Indicate successful plotting on the axes


def plot_all_entropy_mbe_pair_figures(data, layer_idx=None, steps_per_ckpt=None, n_cols=3, save_dir=None, max_pts=None):

    # Find all available loss pairs in the format ('entropy', 'mbe_x')
    loss_pairs = set()
    for param_key in data.keys():
        if param_key not in data or 'grad_angles' not in data[param_key]:
            continue
            
        for angle_data in data[param_key]['grad_angles']:
            if isinstance(angle_data, dict) and 'loss_pair' in angle_data:
                pair = angle_data['loss_pair']
                if isinstance(pair, tuple) and len(pair) == 2:
                    if layer_idx is not None: 
                        if  pair[0] == 'entropy' and pair[1].startswith(f'mbe_{layer_idx}'):
                            loss_pairs.add(pair)
                    else: 
                        if pair[0] == 'entropy' and pair[1].startswith('mbe_'):
                            loss_pairs.add(pair)
    
    if not loss_pairs:
        print("No valid ('entropy', 'mbe_x') loss pairs found in the data.")
        return {}
    
    # Sort the loss pairs by the mbe layer number for consistent ordering
    sorted_loss_pairs = sorted(loss_pairs, key=lambda p: int(p[1].split('_')[1]) if p[1].split('_')[1].isdigit() else 0)
    
    # Create a figure for each loss pair
    figures = {}
    for loss_pair in sorted_loss_pairs:
        # Pre-check which layers have valid data for this loss pair
        valid_layers = []
        layer_indices = set()
        
        # First identify all layer indices in the data
        for param_key in data.keys():
            match = re.match(r'transformer\.h\.(\d+)\.', param_key)
            if match:
                layer_indices.add(int(match.group(1)))
        
        # Then check which ones have valid data for this loss pair
        for layer_idx in sorted(layer_indices):
            # Create a temporary figure/axis to test plotting
            temp_fig, temp_ax = plt.subplots()
            is_valid = False 
            try: 
                plot_layer_grad_cosine_similarity(
                    data, layer_idx, loss_pair, steps_per_ckpt, ax=temp_ax, max_pts=max_pts
                )
                is_valid = True 
            except Exception as e:
                is_valid = False 
            plt.close(temp_fig)  # Close the temporary figure
            
            if is_valid:
                valid_layers.append(layer_idx)
        
        if not valid_layers:
            print(f"No valid layers found for loss pair {loss_pair}")
            continue
            
        # Calculate grid dimensions for this loss pair
        n_layers = len(valid_layers)
        n_rows = math.ceil(n_layers / n_cols)
        
        # Create figure with appropriate dimensions
        fig_width = n_cols * 7  # Adjust base width per subplot
        fig_height = n_rows * 5  # Adjust base height per subplot
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        
        # Handle different axes shapes based on grid dimensions
        if n_rows > 1 and n_cols > 1:
            axes = axes.flatten()
        elif n_rows == 1 and n_cols > 1:
            axes = axes  # Already 1D array in this case
        elif n_cols == 1 and n_rows > 1:
            axes = axes.flatten()
        else:
            axes = [axes]  # Convert single Axes to list for consistent handling
        
        # Plot each valid layer in its own subplot
        successful_plots = 0
        for i, layer_idx in enumerate(valid_layers):
            if i < len(axes):
                is_plotted = plot_layer_grad_cosine_similarity(
                    data, layer_idx, loss_pair, steps_per_ckpt, ax=axes[i], max_pts=max_pts
                )
                if is_plotted:
                    successful_plots += 1
            else:
                print(f"Warning: Not enough subplots for layer {layer_idx} with loss pair {loss_pair}")
        
        # Hide any unused subplots
        for j in range(successful_plots, len(axes)):
            axes[j].set_visible(False)
        
        # Add overall title and adjust layout
        mbe_layer = loss_pair[1].split('_')[1]
        fig.suptitle(f"Grad Cosine Similarity: entropy & mbe_{mbe_layer} loss pair", 
                     fontsize=12, y=0.98)
        
        # Add common labels for the figure
        fig.text(0.5, 0.01, 'Training Steps', ha='center', fontsize=14)
        fig.text(0.01, 0.5, 'Cosine Similarity', va='center', rotation='vertical', fontsize=14)
        
        plt.tight_layout(rect=[0.02, 0.03, 1, 0.95], h_pad=1.5, w_pad=2.0)  # Adjust to make room for labels and increase horizontal spacing
        
        # Store the figure in our dictionary
        figures[loss_pair] = fig
        
    if save_dir: 
        # Ensure the save directory exists
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Save each figure with a sanitized filename
        for loss_pair, fig in figures.items():
            # Create a safe filename from the loss pair
            filename = f"{loss_pair[0]}_{loss_pair[1].replace('/', '_')}.png"
            save_file = os.path.join(save_dir, filename)
            
            try:
                fig.savefig(save_file)
                print(f"Saved figure to {save_file}")
            except Exception as e:
                print(f"Error saving figure to {save_file}: {e}")
    
    return figures


def plot_all_layer_entropy_grad_consistency(data, steps_per_ckpt=None, n_cols=3, save_dir=None, max_pts=100): 
    """
    Plots the gradient cosine similarity for ('entropy', 'entropy') loss pair across all layers.
    
    Args:
        data (dict): Dictionary containing the training data with gradient angles.
        steps_per_ckpt (int, optional): Number of steps between checkpoints. Default is None.
        n_cols (int, optional): Number of columns in the subplot grid. Default is 3.
        save_dir (str, optional): Directory to save the figures. Default is None.
        
    Returns:
        matplotlib.figure.Figure: The created figure, or None if no valid data found.
    """
    # Find all available layers in the data
    layer_indices = set()
    
    for param_key in data.keys():
        match = re.match(r'transformer\.h\.(\d+)\.', param_key)
        if match:
            layer_indices.add(int(match.group(1)))
    
    if not layer_indices:
        print("No valid layers found in the data.")
        return None
    
    # Sort the layer indices
    valid_layers = sorted(layer_indices)
    
    # Define the loss pair we're interested in
    entropy_pair = ('entropy', 'entropy')
    
    # Calculate grid dimensions
    n_layers = len(valid_layers)
    n_rows = math.ceil(n_layers / n_cols)
    
    # Create figure with appropriate dimensions
    fig_width = n_cols * 7  # Adjust base width per subplot
    fig_height = n_rows * 5  # Adjust base height per subplot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    
    # Handle different axes shapes based on grid dimensions
    if n_rows > 1 and n_cols > 1:
        axes = axes.flatten()
    elif n_rows == 1 and n_cols > 1:
        axes = axes  # Already 1D array in this case
    elif n_cols == 1 and n_rows > 1:
        axes = axes.flatten()
    else:
        axes = [axes]  # Convert single Axes to list for consistent handling
    
    # Plot each valid layer in its own subplot
    successful_plots = 0
    for i, layer_idx in enumerate(valid_layers):
        if i < len(axes):
            is_plotted = plot_layer_grad_cosine_similarity(
                data, layer_idx, entropy_pair, steps_per_ckpt, ax=axes[i], max_pts=max_pts
            )
            if is_plotted:
                successful_plots += 1
        else:
            print(f"Warning: Not enough subplots for layer {layer_idx}")
    
    # Hide any unused subplots
    for j in range(successful_plots, len(axes)):
        axes[j].set_visible(False)
    
    # Add overall title and adjust layout
    fig.suptitle(f"Entropy Loss Gradient Self-Consistency Across Layers", 
                 fontsize=14, y=0.98)
    
    # Add common labels for the figure
    fig.text(0.5, 0.01, 'Training Steps', ha='center', fontsize=14)
    fig.text(0.01, 0.5, 'Cosine Similarity', va='center', rotation='vertical', fontsize=14)
    
    plt.tight_layout(rect=[0.02, 0.03, 1, 0.95], h_pad=1.5, w_pad=2.0)
    
    # Save the figure if save_dir is provided
    if save_dir and successful_plots > 0:
        # Ensure the save directory exists
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Create a filename for the figure
        save_file = os.path.join(save_dir, "entropy_self_consistency.png")
        
        try:
            fig.savefig(save_file)
            print(f"Saved figure to {save_file}")
        except Exception as e:
            print(f"Error saving figure to {save_file}: {e}")
    
    # If no successful plots, close the figure and return None
    if successful_plots == 0:
        plt.close(fig)
        print("No data could be plotted for entropy self-consistency.")
        return None
    
    return fig


def plot_grad_info(data, save_dir): 
    # Plot entropy-entropy gradient consistency for all layers
    figs = plot_all_layer_entropy_grad_consistency(data, max_pts=30, save_dir=save_dir)

    # PLot entroy-mbe gradient similarity for all layers
    for i in range(2, 10): 
        figs = plot_all_entropy_mbe_pair_figures(data, layer_idx=i, n_cols=2, save_dir=save_dir)
    return True 


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
    

def get_layer_colors(n_layer):
    """Returns colors for 3 layer groups (Early, Middle, Late) using a modern palette."""
    # "High-end" Palette (Flat UI / Modern Tech style)
    # Group 1: Slate Blue (Cool, steady)
    # Group 2: Mint/Teal (Fresh, bridge)
    # Group 3: Soft Coral (Warm, output focused)
    group_colors = ['#AED7F4', '#C8F7DC', '#FFF9C4']  # light blue, light green, light yellow
    
    # Calculate group sizes
    group_size = n_layer // 3
    remainder = n_layer % 3
    
    colors = []
    # Assign colors
    for i in range(n_layer):
        if i < group_size + (1 if remainder > 0 else 0):
            colors.append(group_colors[0]) # Early
        elif i < 2 * group_size + (2 if remainder > 0 else 0):
            colors.append(group_colors[1]) # Middle
        else:
            colors.append(group_colors[2]) # Late
    return colors, group_colors

def create_training_frame(step_idx, loss_record, run_info="", n_layer=12, val_interval=125):
    """
    Generates a single frame for the animation at a specific validation step.
    """
    # 1. Setup Data
    mbe_values = [loss_record[f"mbe_{l}"][step_idx] for l in range(n_layer)]
    layers = np.arange(n_layer)
    
    full_mbe_history = np.array([loss_record[f"mbe_{l}"] for l in range(n_layer)])
    avg_mbe_history = full_mbe_history.mean(axis=0)[:step_idx+1]
    entropy_history = loss_record["entropy"][:step_idx+1]
    
    total_points = len(loss_record["entropy"])
    real_steps = np.arange(total_points) * val_interval
    current_step_val = step_idx * val_interval
    
    max_mbe_bar = full_mbe_history.max()
    max_entropy = max(loss_record["entropy"])
    min_entropy = min(loss_record["entropy"])
    max_avg_mbe = full_mbe_history.mean(axis=0).max()
    min_avg_mbe = full_mbe_history.mean(axis=0).min()

    # 2. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=100)
    
    # --- LHS: Per-Layer MBE Loss ---
    colors, palette = get_layer_colors(n_layer)
    # Alpha 0.85 for "translucent but solid" look, no harsh black edges
    ax1.bar(layers, mbe_values, color=colors, alpha=0.95, edgecolor='gray', linewidth=0.5)
    
    legend_elements = [
        Patch(facecolor=palette[0], label='Early Layers', alpha=0.85),
        Patch(facecolor=palette[1], label='Middle Layers', alpha=0.85),
        Patch(facecolor=palette[2], label='Late Layers', alpha=0.85)
    ]
    # Remove frame from legend for cleaner look
    ax1.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=15)

    ax1.set_xlabel("Layer Index", fontsize=15, fontweight='medium', color='#444444')
    ax1.set_ylabel("MBE Loss", fontsize=15, fontweight='medium', color='#444444')
    ax1.set_title(f"Per-Layer MBE Loss @ Step {current_step_val}", fontsize=15, pad=10)
    ax1.set_ylim(0, max_mbe_bar * 1.1)
    ax1.set_xticks(layers)
    # Lighter grid
    ax1.grid(axis='y', alpha=0.15, linestyle='-', color='#000000')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # --- RHS: Dual Axis Line Chart ---
    ax2.set_xlabel("Training Steps", fontsize=15, fontweight='medium', color='#444444')
    
    # Use matching colors for lines: Coral for Entropy (loss), Teal for MBE (structure)
    line1, = ax2.plot(real_steps[:step_idx+1], entropy_history, color='#FB6E52', linewidth=2.5, label='Entropy')
    ax2.set_ylabel("Entropy Loss", color='#FB6E52', fontsize=15, fontweight='medium')
    ax2.tick_params(axis='y', labelcolor='#FB6E52')
    ax2.set_xlim(0, real_steps[-1] if real_steps[-1] > 0 else 1)
    ax2.set_ylim(min_entropy * 0.95, max_entropy * 1.05)
    ax2.spines['top'].set_visible(False)
    
    ax2_r = ax2.twinx()
    # Use the "Early" blue or "Middle" teal for Avg MBE line to contrast with Entropy red
    line2, = ax2_r.plot(real_steps[:step_idx+1], avg_mbe_history, color='#5D9CEC', linewidth=2, linestyle='--', label='Avg MBE')
    ax2_r.set_ylabel("Avg MBE Loss", color='#5D9CEC', fontsize=15, fontweight='medium')
    ax2_r.tick_params(axis='y', labelcolor='#5D9CEC')
    ax2_r.set_ylim(min_avg_mbe * 0.9, max_avg_mbe * 1.1)
    ax2_r.spines['top'].set_visible(False)
    
    ax2.set_title(f"{run_info}:\n Entropy vs MBE", fontsize=15, pad=10)
    ax2.grid(True, alpha=0.15, linestyle='-', color='#000000')
    
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right', frameon=False, fontsize=15)

    ax2.scatter([current_step_val], [entropy_history[-1]], color='#FB6E52', s=50, zorder=5, edgecolor='white')
    ax2_r.scatter([current_step_val], [avg_mbe_history[-1]], color='#5D9CEC', s=50, zorder=5, edgecolor='white')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120) # Slightly higher DPI for crispness
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def fig_to_pil(fig):
    import io
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, 
                bbox_inches='tight', 
                pad_inches=0.2,
                facecolor='white')
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    return img


def visualize_final_losses(loss_record, n_layer=12, run_info=""):
    final_mbe = [loss_record[f"mbe_{l}"][-1] for l in range(n_layer)]
    final_entropy = loss_record["entropy"][-1]
    avg_mbe = np.mean(final_mbe)
    layers = np.arange(n_layer)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    
    # Get modern palette
    colors, palette = get_layer_colors(n_layer)
    
    # Bars with transparency and soft edges
    bars = ax.bar(layers, final_mbe, color=colors, alpha=0.95, edgecolor='gray', linewidth=0.8, width=0.75)
    
    # Clean Legend
    legend_elements = [
        Patch(facecolor=palette[0], label='Early Layers', alpha=0.85),
        Patch(facecolor=palette[1], label='Middle Layers', alpha=0.85),
        Patch(facecolor=palette[2], label='Late Layers', alpha=0.85)
    ]
    
    # Add Mean MBE line
    ax.axhline(y=avg_mbe, color='#888888', linestyle='--', linewidth=1.5, alpha=0.7)
    legend_elements.append(plt.Line2D([0], [0], color='#888888', linestyle='--', linewidth=1.5, label=f'Avg MBE ({avg_mbe:.4f})'))
    
    ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=15)
    
    ax.set_xlabel('Layer Index', fontsize=15, color='#444444')
    ax.set_ylabel('MBE Loss', fontsize=15, color='#444444')
    ax.set_title(f'{run_info}\nFinal Entropy: {final_entropy:.4f}', fontsize=15, pad=15)
    ax.set_xticks(layers)
    
    # Clean up grid and spines
    ax.grid(axis='y', alpha=0.15, linestyle='-', color='#000000')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (max(final_mbe)*0.01),
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=9, color='#555555')

    plt.tight_layout()
    return fig_to_pil(fig)


def plot_dual_metrics(loss_record, names=('l1_positive', 'mbe'), save_path="metrics_curves.png", 
                      y_labels=None, title="Training Metrics", use_markers=True, same_scale=False):
    """
    Plot two metrics with different y-axes.
    
    Args:
        loss_record (dict): Dictionary containing metrics to plot
        names (tuple): Tuple of (left_metric, right_metric) names to plot
        save_path (str): Path to save the figure
        y_labels (tuple): Optional custom labels for y-axes (left, right)
        title (str): Plot title
        use_markers (bool): Whether to use markers on the plot lines
        same_scale (bool): Whether to align the scales of both y-axes
    """
    print(f"Plotting metrics: {names[0]} and {names[1]}...")
    
    # Create figure and primary axis
    fig, ax1 = plt.figure(figsize=(10, 6)), plt.gca()
    
    # Ensure we have enough data points
    if names[0] not in loss_record or not loss_record[names[0]]:
        print(f"Warning: No data for {names[0]}")
        return
    
    x = np.arange(len(loss_record[names[0]]))
    
    # Set up default y-axis labels if not provided
    if y_labels is None:
        y_labels = (f"{names[0].replace('_', ' ').title()}", f"{names[1].replace('_', ' ').title()}")
    
    # Plot first metric on primary y-axis
    color1 = 'tab:blue'
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel(y_labels[0], color=color1)
    marker1 = 'o-' if use_markers else '-'
    ax1.plot(x, loss_record[names[0]], marker1, color=color1, label=y_labels[0])
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Check if we need to compute an aggregate metric
    if names[1] not in loss_record:
        # Find all related metrics (those containing the name as substring)
        related_metrics = [k for k in loss_record.keys() if names[1] in k and k != names[1]]
        
        if related_metrics:
            # Compute average of related metrics
            loss_record[names[1]] = [sum(loss_record[k][i] for k in related_metrics) / len(related_metrics) 
                                    for i in range(len(loss_record[related_metrics[0]]))]
            print(f"Computed aggregate {names[1]} from {len(related_metrics)} related metrics")
    
    # Skip second metric if not available
    if names[1] not in loss_record or not loss_record[names[1]]:
        print(f"Warning: No data for {names[1]}")
        ax1.legend(loc='upper right')
        fig.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"- Plot saved to {save_path}")
        return
    
    # Create secondary y-axis and plot second metric
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel(y_labels[1], color=color2)
    marker2 = 's-' if use_markers else '-'
    ax2.plot(x, loss_record[names[1]], marker2, color=color2, label=y_labels[1])
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Set same scale for both axes if requested
    if same_scale:
        # Get overall min and max values from both metrics
        all_values = loss_record[names[0]] + loss_record[names[1]]
        y_min = min(all_values)
        y_max = max(all_values)
        # Add 5% padding
        padding = (y_max - y_min) * 0.05
        ax1.set_ylim(y_min - padding, y_max + padding)
        ax2.set_ylim(y_min - padding, y_max + padding)
        print(f"Using same scale for both axes: [{y_min:.4f}, {y_max:.4f}]")
    
    # Plot related individual metrics with lower alpha
    related_metrics = [k for k in loss_record.keys() if (names[1] in k and k != names[1])]
    
    if related_metrics:
        # Define a colormap for different individual metrics
        cmap = plt.cm.get_cmap('tab10', len(related_metrics) + 1)
        
        for i, k in enumerate(related_metrics):
            # Try to extract a suffix/index from the key
            parts = k.split('_')
            if len(parts) > 1 and parts[-1].isdigit():
                suffix = parts[-1]
                label = f"{names[1].replace('_', ' ').title()} {suffix}"
            else:
                label = k.replace('_', ' ').title()
                
            layer_color = cmap(i)
            ax2.plot(x, loss_record[k], marker2, color=layer_color, label=label, alpha=0.1)
    
    # Add title and grid
    plt.title(title)
    ax1.grid(True, alpha=0.3)
    
    # Add legend with both axis items
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    # Adjust layout and save
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"- Plot saved to {save_path}")


# Experimental Prior Weights on layer-wise MBE loss 
# ------------------------------------------------------------------------------------------------
prior_weights_natural = {
    0: 0.04309166, 
    1: 0.04309166, 
    2: 0.07181943, 
    3: 0.07181943, 
    4: 0.04787962,
    5: 0.06155951, 
    6: 0.08618331, 
    7: 0.10772914, 
    8: 0.14363885, 
    9: 0.21545828, 
    10: 0.21772914,
    11: 0.10772914, 
}

prior_weights_valley = {
    # Early layers (input processing) - allow higher MBE to preserve input information
    0: 0.3,  # Minimal regularization to preserve rich input representations
    1: 0.4,  # Slight increase in regularization
    # Middle-early layers (feature extraction) - moderate compression
    2: 0.8,  # Begin more substantial compression
    3: 1.0,  # Standard compression weight
    4: 1.2,  # Increased compression
    # Middle-late layers (abstraction) - strongest compression
    5: 1.5,  # Heavy compression to force abstraction
    6: 1.8,  # Maximum compression to create bottleneck
    7: 1.5,  # Slightly reduced compression
    # Late layers (prediction preparation) - moderate compression
    8: 1.0,  # Standard compression
    9: 0.8,  # Reduced compression
    # Final layers (output generation) - lighter touch
    10: 0.5,  # Light compression to allow task-specific representations
    11: 0.3   # Minimal regularization on final layer
}

prior_weights_mountain = {
    # Early layers - moderate regularization
    0: 0.8,  # Substantial compression to filter noise early
    1: 1.2,  # Strong compression to force early abstraction
    # Middle layers - minimal regularization (creating a "mountain")
    2: 0.4,  # Reduce compression to allow representation expansion
    3: 0.3,  # Minimal compression - information expansion zone
    4: 0.2,  # Lowest compression - maximum information preservation
    5: 0.2,  # Lowest compression - maximum information preservation
    6: 0.3,  # Minimal compression - information expansion zone
    # Later layers - strongest compression
    7: 1.2,  # Begin aggressive compression
    8: 1.8,  # Maximum compression to create final bottleneck
    9: 2.0,  # Extreme compression for efficient final representations
    10: 1.5, # Strong but reduced compression
    11: 1.0  # Moderate compression for output layer
}

prior_weights_oscillate = {
    # Alternating high and low compression across all layers
    0: 0.3,   # Low - preserve input information
    1: 1.5,   # High - force early abstraction
    2: 0.4,   # Low - allow expansion after compression
    3: 1.6,   # High - compress again after expansion
    4: 0.5,   # Low - expansion
    5: 1.7,   # High - compression
    6: 0.6,   # Low - expansion
    7: 1.8,   # High - compression
    8: 0.7,   # Low - expansion
    9: 1.9,   # High - maximum compression
    10: 0.8,  # Low - slight expansion
    11: 1.0   # Moderate - balanced final layer
}

# ------------------------------------------------------------------------------------------------

# ------- Scheduler for Rank Regularization || Gated Phase Transition -------


NAIVE_WEIGHTS = {i: 1.0 for i in range(12)}

class RRScheduler: 
    # rotate on layer_idx, compute one layer rank regularization loss per train step
    def __init__(self, 
                 num_accumulation_steps, 
                 total_iterations, 
                 start_layer=2,
                 end_layer=12, 
                 main_loss_name="entropy",
                 full_mbe = False,
                 switch_phase=False,
                 ib_target=True,
                 use_prior_weights=False,
                 prior_weight=None,
                 include_inner_cycle=False,
                 period=5, # how many cycle for each phase
                 entropy_patience=125, 
                 entropy_min_delta=0.01,
                 mbe_patience=100,
                 inv_mbe_patience=25,
                 mbe_min_delta=0.002):
        
        self.num_accumulation_steps = num_accumulation_steps
        self.total_iterations = total_iterations
        self.current_accumulation_step = 0
        self.current_iteration = 0
        self.main_loss_name = main_loss_name
        if use_prior_weights: 
            if prior_weight == "natural": 
                self.prior_weights = prior_weights_natural
            elif prior_weight == "valley": 
                self.prior_weights = prior_weights_valley
            elif prior_weight == "mountain": 
                self.prior_weights = prior_weights_mountain
            elif prior_weight == "oscillate": 
                self.prior_weights = prior_weights_oscillate
        else: 
            self.prior_weights = NAIVE_WEIGHTS
            
        # Layer rotation setup
        self.layer_indices = list(range(start_layer, end_layer))  # layer 2 onwards
        self.num_reg_layers = len(self.layer_indices)
        self.current_layer_idx = 0
        self.full_mbe = full_mbe
        self.include_inner_cycle = include_inner_cycle
        
        # Phase management & early stopping
        self.phase = 1  # - Phase 1. Memorization (minimize CE) || Phase 2. Compression (IB) || Phase 3. Expansion (inverse IB)
        if self.include_inner_cycle or (not ib_target): 
            self._inner_phase_float = 1.0
        else: 
            self._inner_phase_float = 0.0
        self.entropy_patience = entropy_patience
        self.entropy_min_delta = entropy_min_delta
        self.compression_patience = mbe_patience
        self.expansion_patience = inv_mbe_patience
        self.mbe_min_delta = mbe_min_delta
        self.min_entropy = np.inf # global best entropy 
        self.min_mbe_dict = defaultdict(lambda: np.inf)
        self.max_mbe_dict = defaultdict(lambda: -np.inf)
        self.memorization_patience_counter = 0 
        self.compression_patience_counter = 0 
        self.expansion_patience_counter = 0 
        self.switch_phase = switch_phase
        self.period = period
        
    def step(self, loss_dict):
        self.current_accumulation_step = (self.current_accumulation_step + 1) % self.num_accumulation_steps
        if self.current_accumulation_step == 0:
            self.current_iteration += 1
            self.current_layer_idx = (self.current_layer_idx + 1) % len(self.layer_indices)
        if (self.main_loss_name in loss_dict) and self.switch_phase: 
            self._switch_phase(loss_dict)
        elif (self.main_loss_name in loss_dict) and not self.switch_phase: 
            self._update_minimum_entropy_loss(loss_dict)
            
    @property 
    def _inner_phase(self): 
        return int(self._inner_phase_float)
    
    def _update_minimum_entropy_loss(self, loss_dict): 
        for loss_name in loss_dict.keys(): 
            if "entropy" in loss_name: 
                entropy_loss = loss_dict[loss_name].item() if hasattr(loss_dict[loss_name], 'item') else loss_dict[loss_name]
                self.min_entropy = min(self.min_entropy, entropy_loss)

    def _switch_phase(self, loss_dict):
        """
        Phase transition logic should be symmetric: either loss plateau should trigger phase transition
           - Global entropy loss plateau (we need to reduce Entropy, therefore 'plateau' means no improvement compared to current best entropy)
           - Local MBE loss plateau (since we might need to increase MBE in memorization phase, therefore 'plateau' means no consecutive decrease in MBE loss)
           - 'min_mbe' will be reset at the end of each memorization phase
           - Update: we need to assume 'all mbe losses' are in the loss_dict
        """
        # Extract losses - assuming exactly one entropy loss and one MBE loss
        entropy_loss = None
        mbe_loss = None
        mbe_improvement = 0.0
        entropy_improvement = 0.0
        worse_memorization = False 
        
        # Find the entropy and MBE losses in the dictionary
        for loss_name in loss_dict.keys():
            if "entropy" in loss_name:
                entropy_loss = loss_dict[loss_name].item() if hasattr(loss_dict[loss_name], 'item') else loss_dict[loss_name]
                entropy_improvement = max(entropy_improvement, self.min_entropy - entropy_loss)
                worse_memorization = entropy_loss >= self.min_entropy * 1.1  # 10% tolerance for compression phase's spike
                self.min_entropy = min(self.min_entropy, entropy_loss)
            else:
                mbe_loss = loss_dict[loss_name].item() if hasattr(loss_dict[loss_name], 'item') else loss_dict[loss_name]
                if self._inner_phase == 1: 
                    mbe_improvement = max(mbe_improvement, self.min_mbe_dict[loss_name] - mbe_loss) # any layer's mbe improvement counts | assumption is compression stage doesn't increase MBE level
                    self.min_mbe_dict[loss_name] = min(self.min_mbe_dict[loss_name], mbe_loss)
                elif self._inner_phase == 2: 
                    mbe_improvement = max(mbe_improvement, mbe_loss - self.min_mbe_dict[loss_name])
                    self.max_mbe_dict[loss_name] = max(self.max_mbe_dict[loss_name], mbe_loss)
        
        assert mbe_loss is not None, "Missing either MBE or DiffMBE loss in loss_dict"
        
        # Determine if progress was made
        better_memorization = (entropy_improvement > self.entropy_min_delta) and entropy_improvement != np.inf
        better_compression = (mbe_improvement > self.mbe_min_delta) and mbe_improvement != np.inf
        better_expansion = (mbe_improvement > self.mbe_min_delta) and mbe_improvement != np.inf
        
        # Update counters and best values
        if better_memorization:
            self.memorization_patience_counter = 0
        else: 
            self.memorization_patience_counter += 1
            
        if better_compression: 
            self.compression_patience_counter = 0
        else: 
            self.compression_patience_counter += 1
            
        if better_expansion: 
            self.expansion_patience_counter = 0
        else: 
            self.expansion_patience_counter += 1
            
        # Check conditions for phase transitions
        no_patience_for_memorization = self.memorization_patience_counter >= self.entropy_patience
        no_patience_for_compression = self.compression_patience_counter >= self.compression_patience
        no_patience_for_expansion = self.expansion_patience_counter >= self.expansion_patience
        
        print("Conditions:\n", 
              f"better_memorization: {better_memorization}\n", 
              f"better_compression: {better_compression}\n", 
              f"better_expansion: {better_expansion}\n", 
              f"no_patience_for_memorization: {no_patience_for_memorization}\n", 
              f"no_patience_for_compression: {no_patience_for_compression}\n", 
              f"no_patience_for_expansion: {no_patience_for_expansion}\n", 
              f"worse_memorization: {worse_memorization}\n", 
              f"current phase: {'Memorization' if self.phase == 1 else 'Compression' if self.phase == 2 else 'Expansion'}\n")
        
        # Handle phase transitions
        if ((no_patience_for_compression or worse_memorization) and self.phase == 2) or ((no_patience_for_expansion or worse_memorization) and self.phase == 3):
            if worse_memorization: 
                print(f"--> Pulled out of {'Compression' if self.phase == 2 else 'Expansion'} Phase due to worse memorization")
            print("--> Switch to Memorization Phase")
            self.phase = 1 
            self.memorization_patience_counter = 0 
            self.min_mbe_dict = defaultdict(lambda: np.inf)
            self.max_mbe_dict = defaultdict(lambda: -np.inf)
            if self.include_inner_cycle: 
                self._inner_phase_float = (self._inner_phase_float + 1/self.period) % 2
                
        elif no_patience_for_memorization and self.phase == 1:
            print(f"--> Switch to {'Compression' if self._inner_phase == 0 else 'Expansion'} Phase") 
            if self._inner_phase == 0: 
                self.phase = 2 
            elif self._inner_phase == 1: 
                self.phase = 3 
            self.compression_patience_counter = 0 
            self.min_entropy = np.inf
        
    def _do_rr(self): # 1-inner-1-outer schedule could also be changed ...
        return (self.phase == 2 or self.phase == 3) and self.current_accumulation_step % 2 == 1
        
    @property
    def rr_layer_index(self): 
        if self._do_rr(): 
            return self.layer_indices[self.current_layer_idx: self.current_layer_idx + 1]
        else: 
            return []
        
    @property
    def mbe_weight(self): 
        weights = np.array([self.prior_weights[i] for i in self.layer_indices])
        return {i: w for i, w in zip(self.layer_indices, weights.tolist())}
    
    @property
    def rr_layer_weight(self): 
        if self.rr_layer_index: 
            # FY: compression phase minimize MBE (+1 scale on loss), expansion phase maximize MBE (-1 scale on loss)
            return (- 2 * int(self._inner_phase) + 1) * self.mbe_weight[self.rr_layer_index[0]]
        else: 
            return 1.0 # entropy loss weightage

    def process_loss_dict(self, loss_dict): 

        if "diff_mbe" in list(loss_dict.keys()): # weighted sum over all layers 
            print(f"- backward on diff_mbe loss -")
            loss_dict = {"diff_mbe": sum(self.prior_weights[layer_idx] * loss_dict[f"diff_mbe_{layer_idx}"] for layer_idx in self.prior_weights) / len(self.prior_weights)}
            
        elif len(self.rr_layer_index) == 1: 
            layer_idx = self.rr_layer_index[0]
            mbe_loss_name = f"mbe_{layer_idx}"
            print(f"- backward on {mbe_loss_name} loss -")
            loss_dict = {mbe_loss_name: loss_dict[mbe_loss_name]}
            
        elif len(self.rr_layer_index) == 0:
            print(f"- backward on entropy loss -")
            loss_dict = {"entropy": loss_dict["entropy"]}
            
        else:
            avg_mbe_loss = sum([self.prior_weights[layer_idx] * loss_dict[f"mbe_{layer_idx}"] for layer_idx in self.rr_layer_index])
            print(f"- backward on mbe loss -")
            loss_dict = {"mbe": avg_mbe_loss}
            
        return loss_dict 

from typing import Optional

def compute_loss(loss_dict: dict, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is not None:
        loss_dict['entropy'] = (loss_dict['entropy'] * mask).sum() / (mask.sum())
    else:
        loss_dict['entropy'] = loss_dict['entropy'].mean()

import ast 

def extract_log_data(log_path):
    """
    Extract experiment configuration and loss records from IBLM log file.
    
    Returns:
        config (str): Experiment configuration string
        loss_record (dict): Dictionary of loss curves
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
            dict_str = re.sub(r"defaultdict\(<class 'list'>, ", "", dict_str)
            if dict_str.endswith(')'):
                dict_str = dict_str[:-1]  # Remove trailing )
    
            try:
                loss_record = ast.literal_eval(dict_str)
            except:
                print(f"Failed to parse loss record from line {i+1}")
                print(f"Dict string: {dict_str[:200]}...")
        
        if config is not None and loss_record is not None:
            break
    
    return config, loss_record


def plot_mbe_comparison(base_mbe, gapt_mbe): 
    n_layer = len(base_mbe)
    layers = np.arange(n_layer)
    bar_width = 0.38

    # Compute stats
    base_avg, gapt_avg = np.mean(base_mbe), np.mean(gapt_mbe)
    reduction = (base_avg - gapt_avg) / base_avg * 100

    fig, ax = plt.subplots(figsize=(12, 6), dpi=120)

    # Color palette
    base_color = '#5B8DEF'  # Soft blue
    gapt_color = '#FF6B6B'  # Coral red

    # Grouped bars
    bars_base = ax.bar(layers - bar_width/2, base_mbe, bar_width, 
                    label='Baseline', color=base_color, alpha=0.85, edgecolor='#3d5a80', linewidth=0.8)
    bars_gapt = ax.bar(layers + bar_width/2, gapt_mbe, bar_width, 
                    label='GAPT (w=20)', color=gapt_color, alpha=0.85, edgecolor='#9d4452', linewidth=0.8)

    # Average lines
    ax.axhline(y=base_avg, color=base_color, linestyle='--', linewidth=1.8, alpha=0.6)
    ax.axhline(y=gapt_avg, color=gapt_color, linestyle='--', linewidth=1.8, alpha=0.6)

    # Legend
    legend_elements = [
        Patch(facecolor=base_color, label=f'Baseline (avg: {base_avg:.3f})', alpha=0.85, edgecolor='#3d5a80'),
        Patch(facecolor=gapt_color, label=f'GAPT (avg: {gapt_avg:.3f})', alpha=0.85, edgecolor='#9d4452'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=12)

    # Labels & Title
    ax.set_xlabel('Layer Index', fontsize=14, color='#444444')
    ax.set_ylabel('MBE Loss', fontsize=14, color='#444444')
    ax.set_title(f'GPT2-small: Baseline vs GAPT — Layer-wise MBE\n'
                f'Average MBE Reduction: {reduction:.1f}%', fontsize=14, pad=12, fontweight='medium')
    ax.set_xticks(layers)
    ax.set_xticklabels([str(i) for i in layers])

    # Styling
    ax.grid(axis='y', alpha=0.2, linestyle='-', color='#000000')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')
    ax.set_ylim(0, max(max(base_mbe), max(gapt_mbe)) * 1.15)

    # Value labels (optional — can be verbose, uncomment if needed)
    for bar in bars_base:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=7, color='#555555')
    for bar in bars_gapt:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=7, color='#555555')

    plt.tight_layout()
    plt.savefig('gpt2-small-base-vs-gapt-mbe.png', dpi=150, bbox_inches='tight')
    plt.show()

class EvalManager:
    def __init__(self, val_file_dict, val_seq_len, val_tokens, rank, world_size):
        """
        Args:
            val_file_dict: Dict mapping dataset name to file pattern 
                           e.g. {'fineweb': 'data/fw*.bin', 'code': 'data/code*.bin'}
        """
        self.val_file_dict = val_file_dict
        self.val_seq_len = val_seq_len
        self.val_tokens = val_tokens
        self.rank = rank
        self.world_size = world_size
        
        # Pre-calculate steps once
        self.total_val_len = world_size * val_seq_len
        self.val_steps = val_tokens // self.total_val_len

    def evaluate(self, model, attn_blocksize, patch_size):
        """Run evaluation on ALL datasets in the dictionary"""
        model.eval()
        results = {}

        for name, file_pattern in self.val_file_dict.items():
            val_loss = defaultdict(float)
            val_mbe = defaultdict(float)
            # Create a fresh loader for this dataset
            loader = distributed_data_generator(file_pattern, self.total_val_len, self.rank, self.world_size)
            
            with torch.no_grad():
                for i in range(self.val_steps):
                    try:
                        inputs, targets = next(loader)
                    except StopIteration:
                        break

                    loss_dict = model.forward(inputs, targets, attn_blocksize, patch_size)
                    compute_loss(loss_dict)

                    # Record all losses
                    for k, v in loss_dict.items():
                        if isinstance(v, torch.Tensor):
                            val_loss[k] += v.item()
                        else:
                            val_loss[k] += v
                    # Also gather all MBE_{layer} stats
                    # MBE keys are like "mbe_0", "mbe_1", etc.
                    for k, v in loss_dict.items():
                        if k.startswith("mbe_"):
                            if isinstance(v, torch.Tensor):
                                val_mbe[k] += v.item()
                            else:
                                val_mbe[k] += v
                            
            # Average and Reduce
            for k in val_loss:
                val_loss[k] /= self.val_steps
            for k in val_mbe:
                val_mbe[k] /= self.val_steps
            
            # Reduce across GPUs
            for k in val_loss:
                t = torch.tensor(val_loss[k], device="cuda")
                dist.all_reduce(t, op=dist.ReduceOp.AVG)
                val_loss[k] = t.item()
            
            for k in val_mbe:
                t = torch.tensor(val_mbe[k], device="cuda")
                dist.all_reduce(t, op=dist.ReduceOp.AVG)
                val_mbe[k] = t.item()

            # Main metric + attach mbe to result
            results[name] = {
                "entropy": val_loss.get('entropy', None),
                "mbe": {k: val_mbe[k] for k in sorted(val_mbe)}
            }
            
        model.train()
        return results