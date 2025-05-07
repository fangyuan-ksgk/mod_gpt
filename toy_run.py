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

from src.rank_regularizer import patch_mbe3
from src.pcgrad import YetAnotherMixer2


# utils function
# ------------------------------------------------------------

def sample_p1(n_samples, in_dim=10, std=0.25):
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

def build_dataset(train_size, val_size, model, in_dim, param_shift=None):
    if param_shift is None: 
        param_shift = generate_param_shift(model)
    data_size = train_size + val_size 

    # sample inputs (p1 & p2)    
    x1 = sample_p1(data_size, in_dim)
    x2 = sample_p2(data_size, in_dim)
    
    model_positive_shift = copy.deepcopy(model)
    mlp_positive_shift = apply_param_shift(model_positive_shift, param_shift, 1.0)
    y_positive, h_positive = mlp_positive_shift(x1)
    model_negative_shift = copy.deepcopy(model)
    mlp_negative_shift = apply_param_shift(model_negative_shift, param_shift, -1.0)
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
    
    def compute_loss(self, x, y): 
        y_pred, h = self(x)
        loss_dict = {"l1": l1_loss(y_pred, y), "mbe": mbe_loss(h)}
        return loss_dict
        
    def get_hidden_representation(self, x):
        return self.activation(self.layer1(x))
