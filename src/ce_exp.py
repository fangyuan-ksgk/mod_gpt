import torch
import random 
import copy 
import pickle
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset

from src.gapt import patch_mbe as patch_mbe3
from src.model import GPT, GPTConfig

# utils function
# ------------------------------------------------------------

def sample_sequences(n_samples, seq_len=32, vocab_size=50257, subset="first_half"):
    """
    Generate random sequences from a subset of the vocabulary.
    This simulates 'disjoint distributions' similar to sample_p1/p2.
    """
    half_vocab = vocab_size // 2
    if subset == "first_half":
        # Tokens from 0 to half_vocab
        return torch.randint(0, half_vocab, (n_samples, seq_len))
    else:
        # Tokens from half_vocab to vocab_size
        return torch.randint(half_vocab, vocab_size, (n_samples, seq_len))

def sample_p1(n_samples, in_dim=10, std=0.25):
    if isinstance(in_dim, tuple): # Handle image/sequence shapes if passed blindly
         return torch.randn(n_samples, *in_dim)
    
    half_dim = in_dim // 2
    first_half = torch.randn(n_samples, half_dim) * std + 1.0
    second_half = torch.randn(n_samples, in_dim - half_dim) * std
    return torch.cat([first_half, second_half], dim=1)

def sample_p2(n_samples, in_dim=10, std=0.25):
    half_dim = in_dim // 2
    first_half = torch.randn(n_samples, half_dim) * std
    second_half = torch.randn(n_samples, in_dim - half_dim) * std + 1.0
    return torch.cat([first_half, second_half], dim=1)

def plot_x1_x2(x1, x2):
    plt.figure(figsize=(10, 6))
    plt.scatter(x1[:, 0].numpy(), x1[:, 6].numpy(), color='blue', alpha=0.5, label='x1 distribution')
    plt.scatter(x2[:, 0].numpy(), x2[:, 6].numpy(), color='red', alpha=0.5, label='x2 distribution')
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 500')
    plt.title('Distribution of x1 and x2 samples (0th & 5th features)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
    
def generate_param_shift(model, direction_vector=None):
    direction_vector = {}
    for name, param in model.named_parameters():
        if len(param.shape) > 1:
            rank = min(3, min(param.shape))
            u = torch.randn(param.shape[0], rank, device=param.device)
            v = torch.randn(rank, param.shape[1], device=param.device)
            dir_tensor = torch.matmul(u, v)
            dir_tensor = dir_tensor / dir_tensor.norm() * param.norm() * random.uniform(0.5, 5.0)
        else:
            dir_tensor = torch.randn_like(param)
            dir_tensor = dir_tensor / dir_tensor.norm() * param.norm() * random.uniform(0.5, 5.0)
        direction_vector[name] = dir_tensor.detach()
    return direction_vector

def apply_param_shift(model, direction_vector, magnitude=0.1): # in-place operation
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in direction_vector:
                param.add_(direction_vector[name] * magnitude)
    return model 

def calculate_param_shift(model, model_original): 
    param_shift = {} 
    for name, param in model.named_parameters(): 
        if name in model_original.state_dict(): 
            param_shift[name] = (param - model_original.state_dict()[name]).detach()
    return param_shift

def save_param_shift(param_shift, path="param_shift.pkl"): 
    with open(path, "wb") as f:
        pickle.dump(param_shift, f)

def load_param_shift(path="param_shift.pkl"): 
    with open(path, "rb") as f: 
        param_shift = pickle.load(f)
    return param_shift

def build_dataset(train_size, val_size, model, in_dim, param_shift=None, shift_magnitude=1.0, data_type="vector"):
    if param_shift is None: 
        param_shift = generate_param_shift(model)
    data_size = train_size + val_size 

    # sample inputs (p1 & p2)    
    if data_type == "gpt":
        # in_dim here is interpreted as (seq_len, vocab_size) or just seq_len
        seq_len = in_dim if isinstance(in_dim, int) else 32
        vocab_size = model.config.vocab_size if hasattr(model, 'config') else 50257
        x1 = sample_sequences(data_size, seq_len, vocab_size, "first_half")
        x2 = sample_sequences(data_size, seq_len, vocab_size, "second_half")
    else:
        x1 = sample_p1(data_size, in_dim)
        x2 = sample_p2(data_size, in_dim)
    
    # Ensure model is on the right device for generation
    device = next(model.parameters()).device
    x1, x2 = x1.to(device), x2.to(device)

    model_positive_shift = copy.deepcopy(model)
    mlp_positive_shift = apply_param_shift(model_positive_shift, param_shift, shift_magnitude)
    y_positive, h_positive = mlp_positive_shift(x1)
    
    model_negative_shift = copy.deepcopy(model)
    mlp_negative_shift = apply_param_shift(model_negative_shift, param_shift, -shift_magnitude)
    y_negative, h_negative = mlp_negative_shift(x2)
    
    print(f"- Dataset constructed with {data_size} positive & negative samples")
    trainset = {"positive": (x1[:train_size], y_positive.detach()[:train_size]), "negative": (x2[:train_size], y_negative.detach()[:train_size])}
    valset = {"positive": (x1[train_size:], y_positive.detach()[train_size:]), "negative": (x2[train_size:], y_negative.detach()[train_size:])}
    return trainset, valset, param_shift

def _get_batch(dataset, group_name, batch_size=32): 
    x, y = dataset[group_name]
    indices = torch.randperm(len(x))
    batch_indices = indices[:batch_size]
    return x[batch_indices], y[batch_indices]

def get_batch(dataset, group_name, batch_size=32): 
    if group_name == "mix": 
        x_positive, y_positive = _get_batch(dataset, "positive", batch_size // 2)
        x_negative, y_negative = _get_batch(dataset, "negative", batch_size // 2)
        x = torch.cat([x_positive, x_negative], dim=0)
        y = torch.cat([y_positive, y_negative], dim=0)
        return x, y
    else: 
        return _get_batch(dataset, group_name, batch_size)
    
    
def decide_group_name(epoch, epochs, mod="positive"):
    if mod == "positive": 
        return "positive"
    if mod == "negative": 
        return "negative"
    if mod == "positive->negative":
        return "positive" if epoch < epochs / 2 else "negative"
    if mod == "negative->positive": 
        return "negative" if epoch < epochs / 2 else "positive"
    if mod == "interleaved": 
        return "positive" if epoch % 2 == 0 else "negative"
    if mod == "interleaved_reverse": 
        return "negative" if epoch % 2 == 0 else "positive"
    if mod == "mix": 
        return "mix" 
    if mod == "positive_mix_small_negative": 
        return "positive" if epoch % 100 != 0 else "mix"
    if mod == "negative_mix_small_positive": 
        return "negative" if epoch % 100 != 0 else "mix"

def _build_log_str(group_name, val_loss, similarity_metrics):
    assert group_name in ["positive", "negative"], "group_name must be either 'positive' or 'negative'"
    return f"l1 {group_name} loss: {val_loss[f'l1_{group_name}']:.4f} | mbe {group_name} loss: {val_loss[f'mbe_{group_name}']:.4f} | param shift similarity {group_name} : {similarity_metrics[f'param_shift_cosine_similarity_{group_name}']:.4f} | rep similarity {group_name} : {similarity_metrics[f'rep_cosine_similarity_{group_name}']:.4f}"


def build_log_str(epoch, epochs, group_name, val_loss, similarity_metrics):
    if group_name in ["positive", "negative"]: 
        return f" Epoch {epoch+1}/{epochs}, " + _build_log_str(group_name, val_loss, similarity_metrics)
    else:         
        s = f" Epoch {epoch+1}/{epochs}, "
        for group_name in ["positive", "negative"]: 
            s += f" | {_build_log_str(group_name, val_loss, similarity_metrics)}"
        return s       

def calculate_param_similarity(param_shift1, param_shift2):
    assert set(param_shift1.keys()) == set(param_shift2.keys()), "Parameter shifts have different keys"
    
    flat_vec1 = torch.cat([p.flatten() for p in param_shift1.values()])
    flat_vec2 = torch.cat([p.flatten() for p in param_shift2.values()])
        
    cos_sim = torch.nn.functional.cosine_similarity(flat_vec1.unsqueeze(0), 
                                                   flat_vec2.unsqueeze(0)).item()

    layer_similarities = {}
    for key in param_shift1.keys():
        p1, p2 = param_shift1[key], param_shift2[key]
        flat_p1, flat_p2 = p1.flatten(), p2.flatten()
        layer_cos_sim = torch.nn.functional.cosine_similarity(flat_p1.unsqueeze(0), 
                                                             flat_p2.unsqueeze(0)).item()
        layer_similarities[key] = {
            'cosine_similarity': layer_cos_sim
        }
    
    return {
        'param_shift_cosine_similarity': cos_sim
    }
    
def invert_param_shift(param_shift): 
    inverted_shift = {}
    for name, param in param_shift.items(): 
        inverted_shift[name] = -param.clone()
    return inverted_shift
    
def proc_param_shift(param_shift, group_name): 
    if group_name == "positive": 
        return copy.deepcopy(param_shift)
    else: 
        return invert_param_shift(param_shift)
    
def compute_representation_similarity(mlp, mlp_original, param_shift, inputs):
    _, h_pred = mlp(inputs)
    mlp_shift = copy.deepcopy(mlp_original)
    mlp_shift = apply_param_shift(mlp_shift, param_shift, 1.0)
    _, h_shift = mlp_shift(inputs)
    cosine_similarity = torch.nn.functional.cosine_similarity(h_shift.detach(), h_pred.detach()).mean().item()
    euclidean_deviation = (torch.norm(h_shift.detach() - h_pred.detach()) / torch.norm(h_shift.detach())).mean().item()
    return {"rep_cosine_similarity": cosine_similarity}

def _validate(mlp, mlp_original, group_name, param_shift, valset, val_steps):
    """ 
    Found no improvement in representation alignment --> only observe parameter shift alignment
    - Interestingly, learning mimics parameter shift but not representation alignment ... (why?) 
    - memory conflict in parameter is only reflected in learned parameter shift
    """ 
    mlp.eval() 
    val_loss = defaultdict(float)
    similarity_metrics = defaultdict(float)
    param_shift = proc_param_shift(param_shift, group_name) # again this in-place operation leads to 'oscillation' of original 'param_shift' object

    with torch.no_grad():
        
        for i in range(val_steps):
            inputs, targets = get_batch(valset, group_name)
            loss_dict = mlp.compute_loss(inputs, targets)
            for name, loss in loss_dict.items(): 
                val_loss[name + f"_{group_name}"] += loss 
            
            rep_metrics = compute_representation_similarity(mlp, mlp_original, param_shift, inputs)
            for name, similarity in rep_metrics.items():
                similarity_metrics[name + f"_{group_name}"] += similarity
                
        for name in val_loss: 
            val_loss[name] /= val_steps

        for name in similarity_metrics: 
            similarity_metrics[name] /= val_steps
            
        learned_param_shift = calculate_param_shift(mlp, mlp_original)
        param_shift_metrics = calculate_param_similarity(param_shift, learned_param_shift)
        similarity_metrics.update({key+f"_{group_name}": value for key, value in param_shift_metrics.items()})  
    mlp.train() 
    return val_loss, similarity_metrics

def validate(mlp, mlp_original, param_shift, valset, val_steps):
    val_loss_positive, similarity_metrics_positive = _validate(mlp, mlp_original, "positive", param_shift, valset, val_steps)
    val_loss_negative, similarity_metrics_negative = _validate(mlp, mlp_original, "negative", param_shift, valset, val_steps)
    val_loss = {**val_loss_positive, **val_loss_negative}
    similarity_metrics = {**similarity_metrics_positive, **similarity_metrics_negative}
    return val_loss, similarity_metrics

# MBE & L1 loss
# ------------------------------------------------------------
def mbe_loss(x, patch_size=8):
    return patch_mbe3(x.unsqueeze(0), patch_size)

def l1_loss(y_pred, y): 
    return torch.nn.functional.l1_loss(y_pred, y)

class GPTAdapter(nn.Module):
    """
    Wraps the GPT model from src.model to conform to the CE experiment interface:
    forward(x) -> logits, hidden
    compute_loss(x, y) -> dict
    """
    def __init__(self, config):
        super().__init__()
        self.model = GPT(config)
        self.config = config
        
    def forward(self, x):
        # We need to extract logits and hidden state.
        # Since src.model.GPT.forward returns loss directly, we'll manually access components
        # or rely on a modified forward.
        # Here we replicate the forward pass logic briefly to get internal states:
        
        # 1. Embed
        idx = x
        b, t = idx.size()
        
        # We need block mask if we use the exact forward, but for simplicity in CE exp
        # we might skip complex masking or generate a simple one.
        # However, GPT.forward creates it internally.
        
        # To avoid duplicating code, let's look at how GPT is implemented.
        # It computes x = wte(idx), then layers, then head.
        
        x = self.model.transformer.wte(idx)
        # x = norm(x) # In src/model.py, norm is applied after wte? 
        # Checking src/model.py: x = self.transformer.wte(idx); x = norm(x)
        # We need the 'norm' function from src.model
        from src.model import norm, create_block_mask
        x = norm(x)
        x0 = x
        v1 = None
        
        # Create a dummy block mask for causal attention
        # (Simplified: assume standard causal for this exp)
        # For strict compatibility, we should call the internal create_block_mask
        # or just pass None if the model handles it (it doesn't seem to handle None for block_mask in Block.forward).
        
        # Hack: The src.model.GPT expects 'attn_blocksize' in forward.
        # We'll re-implement a simple forward here that extracts what we need.
        
        device = x.device
        S = idx.shape[1]
        
        # Helper to create standard causal mask if needed, but FlexAttention usually needs one.
        # If we are using the 'CustomGPT' we just ported to HF, it works differently.
        # Assuming we wrap the src.model.GPT:
        
        # We'll use the model's own forward to get the loss, but we need logits/hidden for the experiment.
        # Let's just modify the forward to run the layers manually.
        
        # Generate simple causal block mask if possible, or just pass None 
        # (depends on if CausalSelfAttention handles None block_mask). 
        # Looking at code: flex_attention requires block_mask usually.
        
        # For the sake of this experiment, let's assume we can pass a simple causal mask or the model handles it.
        # If using FlexAttention, we strictly need a block mask.
        
        docs = (idx == 50256).cumsum(1)
        def document_causal_mask(b, h, q_idx, kv_idx):
             return q_idx >= kv_idx

        block_mask = create_block_mask(document_causal_mask, None, None, S, S, device=device, _compile=False)

        for block in self.model.transformer.h:
            x, v1 = block(x, v1, x0, block_mask)
            
        h = x # Hidden representation
        
        x = norm(x)
        logits = self.model.lm_head(x)
        # activation scaling from model
        logits = 30 * torch.tanh(logits / 30)
        
        return logits, h

    def compute_loss(self, x, y, patch_size=8):
        logits, h = self(x)
        # y here is the TARGET LOGITS from the teacher (distillation)
        # So we use MSE or KL Div, not CrossEntropy against tokens.
        # ce_exp uses l1_loss by default.
        
        loss_dict = {
            "l1": l1_loss(logits, y), 
            "mbe": mbe_loss(h, patch_size) # MBE on hidden states
        }
        return loss_dict

# Simple MLP model
# ------------------------------------------------------------
class SimpleModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=50, output_dim=1):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h = self.activation(self.layer1(x))
        return self.layer2(h), h
    
    def compute_loss(self, x, y, patch_size=8): 
        y_pred, h = self(x)
        loss_dict = {"l1": l1_loss(y_pred, y), "mbe": mbe_loss(h, patch_size)}
        return loss_dict
        
    def get_hidden_representation(self, x):
        return self.activation(self.layer1(x))


class SimpleModelV2(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=50, output_dim=1):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h = self.activation(self.layer1(x))
        return self.layer2(h), h
    
    def compute_loss(self, x, y, patch_size=8): 
        y_pred, h = self(x)
        loss_dict = {"l1": l1_loss(y_pred, y), "mbe": mbe_loss(h, patch_size)}
        return loss_dict
        
    def get_hidden_representation(self, x):
        return self.activation(self.layer1(x))
    
    
    
# Activation analysis 
# ------------------------------------------------------------
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import umap

def visualize_representations(trained_model, trainset, batch_size=32):
    # Get batches for each distribution using get_batch
    x_pos, _ = get_batch(trainset, 'positive', batch_size=batch_size)
    x_neg, _ = get_batch(trainset, 'negative', batch_size=batch_size)
    
    # Get hidden representations using the trained model
    with torch.no_grad():
        _, h_pos = trained_model(x_pos)  # Assuming model returns (output, hidden_rep)
        _, h_neg = trained_model(x_neg)
    
    # Combine and apply PCA
    combined = torch.cat([h_pos, h_neg]).cpu().numpy()
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(combined)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced[:len(h_pos), 0], reduced[:len(h_pos), 1], alpha=0.5, label='Positive Representation (PCA)')
    plt.scatter(reduced[len(h_pos):, 0], reduced[len(h_pos):, 1], alpha=0.5, label='Negative Representation (PCA)')
    plt.title('Hidden Representation Space (PCA)')
    plt.legend()
    plt.show()
    
def visualize_representations_umap(trained_model, trainset, batch_size=32, n_neighbors=15, min_dist=0.1):
    x_pos, _ = get_batch(trainset, 'positive', batch_size=batch_size)
    x_neg, _ = get_batch(trainset, 'negative', batch_size=batch_size)
    
    with torch.no_grad():
        _, h_pos = trained_model(x_pos)
        _, h_neg = trained_model(x_neg)
    
    combined = torch.cat([h_pos, h_neg]).cpu().numpy()
    
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=min_dist)
    reduced = reducer.fit_transform(combined)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced[:len(h_pos), 0], reduced[:len(h_pos), 1], alpha=0.5, label='Positive Representation (UMAP)')
    plt.scatter(reduced[len(h_pos):, 0], reduced[len(h_pos):, 1], alpha=0.5, label='Negative Representation (UMAP)')
    plt.title('Hidden Representation Space (UMAP)')
    plt.legend()
    plt.show()
    

def visualize_representations_tsne(trained_model, trainset, batch_size=32, perplexity=30, n_iter=300):
    x_pos, _ = get_batch(trainset, 'positive', batch_size=batch_size)
    x_neg, _ = get_batch(trainset, 'negative', batch_size=batch_size)
    
    with torch.no_grad():
        _, h_pos = trained_model(x_pos)
        _, h_neg = trained_model(x_neg)
    
    combined = torch.cat([h_pos, h_neg]).cpu().numpy()
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=n_iter, init='pca', learning_rate='auto')
    reduced = tsne.fit_transform(combined)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced[:len(h_pos), 0], reduced[:len(h_pos), 1], alpha=0.5, label='Positive Representation (t-SNE)')
    plt.scatter(reduced[len(h_pos):, 0], reduced[len(h_pos):, 1], alpha=0.5, label='Negative Representation (t-SNE)')
    plt.title('Hidden Representation Space (t-SNE)')
    plt.legend()
    plt.show()
    
    
def measure_separation(trained_model, trainset, batch_size=32):
    x_pos, _ = get_batch(trainset, 'positive', batch_size=batch_size)
    x_neg, _ = get_batch(trainset, 'negative', batch_size=batch_size)
    
    with torch.no_grad():
        _, h_pos = trained_model(x_pos)
        _, h_neg = trained_model(x_neg)
    
    # Calculate mean vectors
    mean1 = h_pos.mean(dim=0)
    mean2 = h_neg.mean(dim=0)
    # Between-class distance
    between_distance = torch.norm(mean1 - mean2)
    
    # Within-class variance
    var1 = ((h_pos - mean1)**2).sum(dim=1).mean()
    var2 = ((h_neg - mean2)**2).sum(dim=1).mean()
    within_variance = (var1 + var2) / 2
    
    # Separation ratio (higher is better)
    separation_ratio = between_distance / torch.sqrt(within_variance)
    
    return {
        "between_distance": between_distance.item(),
        "within_variance": within_variance.item(),
        "separation_ratio": separation_ratio.item()
    }
    

def neuron_specialization(trained_model, trainset, batch_size=32, save_path=None):
    x_pos, _ = get_batch(trainset, 'positive', batch_size=batch_size)
    x_neg, _ = get_batch(trainset, 'negative', batch_size=batch_size)
    
    with torch.no_grad():
        _, h_pos = trained_model(x_pos)
        _, h_neg = trained_model(x_neg)
    
    # Calculate mean activation per neuron for each distribution
    mean1 = h_pos.mean(dim=0)
    mean2 = h_neg.mean(dim=0)
    
    # Calculate selectivity index for each neuron
    # (1 = responds only to x1, -1 = responds only to x2, 0 = responds equally)
    selectivity = (mean1 - mean2) / (mean1 + mean2 + 1e-6)
    specialized = ((selectivity.abs() > 0.5).sum() / len(selectivity)).item()

    # Plot histogram of selectivity
    plt.figure(figsize=(10, 6))
    plt.hist(selectivity.numpy(), bins=20, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('Neuron Selectivity Distribution')
    plt.xlabel(f'Selectivity (positive = x1, negative = x2) | Specialized neurons: {specialized:.2%}')
    plt.ylabel('Number of neurons')
    plt.show()
    if save_path:
        plt.savefig(save_path)
    
    # Return percentage of specialized neurons
    return specialized


from matplotlib.backends.backend_agg import FigureCanvas
from PIL import Image
import os

def neuron_specialization_frame(trained_model, trainset, batch_size=32, group_name="positive", epoch=0):
    x_pos, _ = get_batch(trainset, 'positive', batch_size=batch_size)
    x_neg, _ = get_batch(trainset, 'negative', batch_size=batch_size)
    
    with torch.no_grad():
        _, h_pos = trained_model(x_pos)
        _, h_neg = trained_model(x_neg)
    
    # Calculate mean activation per neuron for each distribution
    mean1 = h_pos.mean(dim=0)
    mean2 = h_neg.mean(dim=0)
    
    # Calculate selectivity index for each neuron
    # (1 = responds only to x1, -1 = responds only to x2, 0 = responds equally)
    selectivity = (mean1 - mean2) / (mean1 + mean2 + 1e-6)
    specialized = ((selectivity.abs() > 0.5).sum() / len(selectivity)).item()

    # Plot histogram of selectivity
    fig = plt.figure(figsize=(10, 6))
    plt.hist(selectivity.numpy(), bins=20, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title(f'Neuron Selectivity Distribution - {group_name} (Epoch {epoch})')
    plt.xlabel(f'Selectivity (positive = x1, negative = x2) | Specialized neurons: {specialized:.2%}')
    plt.ylabel('Number of neurons')
    plt.ylim(0, 12)  # Fixed y-limit to [0, 12]
    
    # Instead of showing, capture the figure
    canvas = FigureCanvas(fig)
    canvas.draw()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    # Convert numpy array to PIL image
    pil_image = Image.fromarray(image)
    
    return pil_image, specialized


