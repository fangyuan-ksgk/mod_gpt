import torch 
from pathlib import Path
import glob
import itertools
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from matplotlib.animation import FuncAnimation

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
    

def plot_mbe(mbe_values):
    """
    Plot Model Binding Energy (MBE) loss per layer.
    
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
    

# Extract MBE values from experiment data
def extract_mbe_values1(exp_data):
    mbe_values = []
    for iteration_num in sorted(exp_data['progress'].keys()):
        iteration_data = exp_data['progress'][iteration_num]
        iteration_mbes = []
        for key, value in iteration_data.items():
            if key.startswith('mbe_') and key[-1].isdigit():
                layer_idx = int(key.split('_')[1])
                # Ensure the list is long enough
                while len(iteration_mbes) <= layer_idx:
                    iteration_mbes.append(None)
                iteration_mbes[layer_idx] = value
        mbe_values.append(iteration_mbes)
    return mbe_values[1:], exp_data["label"]

def extract_mbe_values2(exp_data): 
    n_ckpt = len(exp_data["record"]["entropy"])
    mbe_record = [] 
    for i in range(1, n_ckpt):
        ckpt_record = []
        for layer_idx in range(12): 
            ckpt_record.append(exp_data["record"][f"mbe_{layer_idx}"][i])
        mbe_record.append(ckpt_record)
    return mbe_record, exp_data["label"]

def extract_mbe_values(exp_data): 
    if "progress" in exp_data: 
        return extract_mbe_values1(exp_data)
    else: 
        return extract_mbe_values2(exp_data)
    
def create_mbe_animation(mbe_data_list, labels_list, output_file='mbe_animation.gif', iteration_multiplier=125):
    
    # Determine number of layers and iterations
    num_layers = max(max(len(mbe) for mbe in data) for data in mbe_data_list if data)
    num_iterations = max(len(data) for data in mbe_data_list if data)
    
    # Calculate global min and max for consistent y-axis limits
    all_values = []
    for mbe_data in mbe_data_list:
        for mbe_list in mbe_data:
            for mbe in mbe_list:
                if mbe is not None:
                    if isinstance(mbe, (list, tuple)):  # Check if mbe is an iterable
                        all_values.extend([v for v in mbe if v is not None])
                    else:  # Handle the case where mbe is a single value (float)
                        all_values.append(mbe)
    
    global_y_min = min(all_values) if all_values else 0
    global_y_max = max(all_values) if all_values else 1
    margin = (global_y_max - global_y_min) * 0.1
    
    # Color cycle for different experiments
    colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Create animation
    fig, ax = plt.subplots(figsize=(12, 6))
    
    def animate(i):
        # Adjust i to handle the extended final frame
        actual_i = min(i, num_iterations - 1)
        
        ax.clear()
        
        # Plot each experiment's data
        for idx, (mbe_data, label) in enumerate(zip(mbe_data_list, labels_list)):
            if len(mbe_data) > 0:  # Make sure we have data
                # If actual_i exceeds available data, use the last available data point
                data_i = min(actual_i, len(mbe_data) - 1)
                vals = mbe_data[data_i]
                x_vals = list(range(len(vals)))
                color = colors[idx % len(colors)]
                ax.plot(x_vals, vals, f'{color}-o', label=label)
        
        ax.set_xlabel('Layer')
        ax.set_ylabel('MBE Value')
        ax.set_title(f'MBE Values Across Layers (Iteration {iteration_multiplier*actual_i})')
        ax.legend()
        ax.grid(True)
        
        # Use the global y-axis limits for all frames
        ax.set_ylim(global_y_min - margin, global_y_max + margin)
        
        # Set x-axis limits
        ax.set_xlim(-0.5, num_layers - 0.5)
    
    # Add extra frames for the last frame (5 seconds at 2fps = 10 extra frames)
    extended_frames = num_iterations + 10
    
    ani = FuncAnimation(fig, animate, frames=extended_frames, interval=500, repeat=True)
    
    # save to gif
    ani.save(output_file, writer='pillow', fps=2)
    
    return ani


# ------- Scheduler for Rank Regularization -------

PRIOR_WEIGHTS = {
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
                 entropy_patience=125, 
                 entropy_min_delta=0.01,
                 mbe_patience=125,
                 mbe_min_delta=0.002):
        
        self.num_accumulation_steps = num_accumulation_steps
        self.total_iterations = total_iterations
        self.current_accumulation_step = 0
        self.current_iteration = 0
        self.main_loss_name = main_loss_name
        self.prior_weights = PRIOR_WEIGHTS
        
        # Layer rotation setup
        self.layer_indices = list(range(start_layer, end_layer))  # layer 2 onwards
        self.num_reg_layers = len(self.layer_indices)
        self.current_layer_idx = 0
        self.full_mbe = full_mbe
        
        # Phase management & early stopping
        self.phase = 1  # - Phase 1. Memorization || Phase 2. Compression
        self.entropy_patience = entropy_patience
        self.entropy_min_delta = entropy_min_delta
        self.mbe_patience = mbe_patience
        self.mbe_min_delta = mbe_min_delta
        self.min_entropy = np.inf # global best entropy 
        self.min_mbe_dict = defaultdict(lambda: np.inf)
        self.memorization_patience_counter = 0 
        self.compression_patience_counter = 0 
        self.switch_phase = switch_phase
        
    def step(self, loss_dict):
        self.current_accumulation_step = (self.current_accumulation_step + 1) % self.num_accumulation_steps
        if self.current_accumulation_step == 0:
            self.current_iteration += 1
            self.current_layer_idx = (self.current_layer_idx + 1) % len(self.layer_indices)
        if (self.main_loss_name in loss_dict) and self.switch_phase: 
            self._switch_phase(loss_dict)

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
                entropy_loss = loss_dict[loss_name].item()
                entropy_improvement = max(entropy_improvement, self.min_entropy - entropy_loss)
                worse_memorization = entropy_loss >= self.min_entropy * 1.1  # 10% tolerance for compression phase's spike
                self.min_entropy = min(self.min_entropy, entropy_loss)
            else:
                mbe_loss = loss_dict[loss_name].item()
                mbe_improvement = max(mbe_improvement, self.min_mbe_dict[loss_name] - mbe_loss) # any layer's mbe improvement counts | assumption is compression stage doesn't increase MBE level
                self.min_mbe_dict[loss_name] = min(self.min_mbe_dict[loss_name], mbe_loss)
        
        assert mbe_loss is not None, "Missing either MBE or DiffMBE loss in loss_dict"
        
        # Determine if progress was made
        better_memorization = (entropy_improvement > self.entropy_min_delta) and entropy_improvement != np.inf
        better_compression = (mbe_improvement > self.mbe_min_delta) and mbe_improvement != np.inf
        
        # Update counters and best values
        if better_memorization:
            self.memorization_patience_counter = 0
        else: 
            self.memorization_patience_counter += 1
            
        if better_compression: 
            self.compression_patience_counter = 0
        else: 
            self.compression_patience_counter += 1
            
        # Check conditions for phase transitions
        no_patience_for_memorization = self.memorization_patience_counter >= self.entropy_patience
        no_patience_for_compression = self.compression_patience_counter >= self.mbe_patience
        
        print("Conditions:\n", 
              f"better_memorization: {better_memorization}\n", 
              f"better_compression: {better_compression}\n", 
              f"no_patience_for_memorization: {no_patience_for_memorization}\n", 
              f"no_patience_for_compression: {no_patience_for_compression}\n", 
              f"worse_memorization: {worse_memorization}\n", 
              f"current phase: {'Memorization' if self.phase == 1 else 'Compression'}\n")
        
        # Handle phase transitions
        if (no_patience_for_compression or worse_memorization) and self.phase == 2: 
            if worse_memorization: 
                print("--> Pulled out of Compression Phase due to worse memorization")
            print("--> Switch to Memorization Phase")
            self.phase = 1 
            self.memorization_patience_counter = 0 
            self.min_mbe_dict = defaultdict(lambda: np.inf)
        elif no_patience_for_memorization and self.phase == 1:
            print("--> Switch to Compression Phase") 
            self.phase = 2 
            self.compression_patience_counter = 0 
            self.min_entropy = np.inf
        
    def _do_rr(self):
        return self.phase == 2 and self.current_accumulation_step % 2 == 1
        
    @property
    def rr_layer_index(self): 
        if self._do_rr(): 
            if not self.full_mbe: 
                return self.layer_indices[self.current_layer_idx: self.current_layer_idx + 1]
            else: 
                return self.layer_indices
        else: 
            return []
        
    @property
    def mbe_weight(self): 
        # weights = {k: v * int(int(k.split("mbe_")[-1]) in self.rr_layer_index) for k,v in self.prior_weights.items()}
        weights = np.array([self.prior_weights[i] for i in self.rr_layer_index])
        weights = weights / weights.sum()
        return weights.tolist()
    
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