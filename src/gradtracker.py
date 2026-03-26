# Gradient Statistics Tracker
# Tracks cosine similarity and norms between gradients from different losses
# -----------------------------------------------------------------------------
import torch 
from collections import defaultdict
import pickle
import numpy as np
import pandas as pd

class GradientTracker:
    """
    Lightweight class to track gradient statistics during training.
    Tracks only norms and cosine similarities (no full gradient storage).
    """
    
    def __init__(self, model):
        self.model = model
        self._init_info()
    
    def _init_info(self):
        """Initialize tracking dictionaries for each parameter"""
        self.grad_info = {
            name: defaultdict(list) 
            for name, p in self.model.named_parameters() 
            if p.requires_grad and p.numel() > 1
        }
    
    def _compute_grad_stats(self, g1, g2):
        """
        Compute gradient statistics: norms and cosine similarity.
        
        Args:
            g1: Previous accumulated gradient
            g2: Current gradient from backward pass
            
        Returns:
            combined_grad, g1_norm, g2_norm, cosine_similarity
        """
        g1_norm = torch.norm(g1)
        g2_norm = torch.norm(g2)
        
        if g1_norm > 1e-8 and g2_norm > 1e-8:
            cosim = torch.sum((g1 / g1_norm) * (g2 / g2_norm))
        else:
            cosim = torch.tensor(0.)
            
        return g1 + g2, g1_norm.item(), g2_norm.item(), cosim.item()
    
    def _update_info(self, param_name, prev_g_norm, curr_g_norm, cosim, loss_name, is_reset):
        """Store gradient statistics for a parameter"""
        if param_name in self.grad_info:
            self.grad_info[param_name]["prev_grad_norm"].append(prev_g_norm)
            self.grad_info[param_name]["curr_grad_norm"].append(curr_g_norm)
            self.grad_info[param_name]["cosine_similarity"].append(cosim)
            self.grad_info[param_name]["loss_name"].append(loss_name)
            self.grad_info[param_name]["reset"].append(is_reset)
    
    def backward_with_tracking(self, loss_dict, retain_graph = False):
        """
        Perform standard backward pass while tracking gradient statistics.
        
        Args:
            loss_dict: Dictionary with single loss {loss_name: loss_value}
        """
        param_names = [p[0] for p in self.model.named_parameters() if p[1].requires_grad]
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        assert len(loss_dict) == 1, "Requires exactly one loss per backward call"
        loss_name = list(loss_dict.keys())[0]
        
        # Track previous gradients and reset flags
        reset_flags = []
        prev_grads = []
        for p in params:
            if p.grad is not None:
                reset_flags.append(False)
                prev_grads.append(p.grad.clone())
                p.grad.zero_()
            else:
                reset_flags.append(True)
                prev_grads.append(torch.zeros_like(p))
                p.grad = torch.zeros_like(p)
        
        # Standard backward pass
        loss_dict[loss_name].backward(retain_graph=retain_graph)
        
        # Track gradient statistics
        for i, p in enumerate(params):
            p.grad, prev_g_norm, curr_g_norm, cosim = \
                self._compute_grad_stats(prev_grads[i], p.grad)
            self._update_info(
                param_names[i], prev_g_norm, curr_g_norm, cosim, 
                loss_name, reset_flags[i]
            )

    def backward(self, loss_dict, retain_graph=False): 
        sum(v for k, v in loss_dict.items()).backward(retain_graph=retain_graph)
    
    def save_grad_info(self, path):
        """Save gradient tracking information to disk"""
        serializable_grad_info = {}
        for param_name, info in self.grad_info.items():
            serializable_grad_info[param_name] = dict(info)
        
        with open(path, "wb") as f:
            pickle.dump(serializable_grad_info, f)
        
        print(f"Gradient statistics saved to {path}")



def track_gradient_similarity(model, ce_loss, mbe_loss):
    """
    Track CE ↔ MBE gradient stats per parameter.
    Returns both global and per-param statistics.
    """
    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named_params]
    names = [n for n, _ in named_params]
    
    # Compute gradients
    ce_grads = torch.autograd.grad(
        ce_loss, params, retain_graph=True, create_graph=False, allow_unused=True
    )
    mbe_grads = torch.autograd.grad(
        mbe_loss, params, retain_graph=True, create_graph=False, allow_unused=True
    )
    
    # Per-parameter stats
    per_param = {}
    ce_flat_all = []
    mbe_flat_all = []
    
    for name, g_ce, g_mbe in zip(names, ce_grads, mbe_grads):
        if g_ce is None or g_mbe is None:
            continue
        
        g_ce_flat = g_ce.flatten()
        g_mbe_flat = g_mbe.flatten()
        
        norm_ce = torch.norm(g_ce_flat).item()
        norm_mbe = torch.norm(g_mbe_flat).item()
        
        if norm_ce > 1e-8 and norm_mbe > 1e-8:
            cos = (torch.dot(g_ce_flat, g_mbe_flat) / (norm_ce * norm_mbe)).item()
        else:
            cos = 0.0
        
        per_param[name] = {
            'ce_norm': norm_ce,
            'mbe_norm': norm_mbe,
            'cosine': cos,
            'numel': g_ce.numel(),
        }
        
        ce_flat_all.append(g_ce_flat)
        mbe_flat_all.append(g_mbe_flat)
    
    # Global stats
    if len(ce_flat_all) > 0:
        all_ce = torch.cat(ce_flat_all)
        all_mbe = torch.cat(mbe_flat_all)
        global_cos = (torch.dot(all_ce, all_mbe) / 
                      (torch.norm(all_ce) * torch.norm(all_mbe))).item()
    else:
        global_cos = 0.0
    
    return {
        'global_cosine': global_cos,
        'per_param': per_param,
    }


def track_multi_gradient_similarity(model, ce_loss, aux_losses):
    """
    Track CE gradient alignment against multiple auxiliary losses.
    
    Args:
        model: The model whose parameters to track.
        ce_loss: Cross-entropy loss tensor (with grad graph).
        aux_losses: Dict of {name: loss_tensor}, e.g. {"mbe": mbe_loss, "frob": frob_loss}.
    
    Returns:
        Dict of {aux_name: {global_cosine, per_param}} — same structure as track_gradient_similarity per aux.
    """
    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named_params]
    names = [n for n, _ in named_params]
    
    # Compute CE gradients once
    ce_grads = torch.autograd.grad(
        ce_loss, params, retain_graph=True, create_graph=False, allow_unused=True
    )
    
    results = {}
    for aux_name, aux_loss in aux_losses.items():
        aux_grads = torch.autograd.grad(
            aux_loss, params, retain_graph=True, create_graph=False, allow_unused=True
        )
        
        per_param = {}
        ce_flat_all = []
        aux_flat_all = []
        
        for name, g_ce, g_aux in zip(names, ce_grads, aux_grads):
            if g_ce is None or g_aux is None:
                continue
            
            g_ce_flat = g_ce.flatten()
            g_aux_flat = g_aux.flatten()
            
            norm_ce = torch.norm(g_ce_flat).item()
            norm_aux = torch.norm(g_aux_flat).item()
            
            if norm_ce > 1e-8 and norm_aux > 1e-8:
                cos = (torch.dot(g_ce_flat, g_aux_flat) / (norm_ce * norm_aux)).item()
            else:
                cos = 0.0
            
            per_param[name] = {
                'ce_norm': norm_ce,
                'aux_norm': norm_aux,
                'cosine': cos,
                'numel': g_ce.numel(),
            }
            
            ce_flat_all.append(g_ce_flat)
            aux_flat_all.append(g_aux_flat)
        
        if len(ce_flat_all) > 0:
            all_ce = torch.cat(ce_flat_all)
            all_aux = torch.cat(aux_flat_all)
            global_cos = (torch.dot(all_ce, all_aux) / 
                          (torch.norm(all_ce) * torch.norm(all_aux))).item()
        else:
            global_cos = 0.0
        
        results[aux_name] = {
            'global_cosine': global_cos,
            'per_param': per_param,
        }
    
    return results


class MultiGradStatsRecorder:
    """Records gradient similarity stats for multiple auxiliary losses over training."""
    
    def __init__(self, aux_names):
        self.aux_names = list(aux_names)
        self.global_history = {name: [] for name in self.aux_names}
        self.per_param_history = {name: {} for name in self.aux_names}
    
    def record(self, multi_stats, step):
        """Record stats from track_multi_gradient_similarity()."""
        for aux_name, stats in multi_stats.items():
            if aux_name not in self.aux_names:
                self.aux_names.append(aux_name)
                self.global_history[aux_name] = []
                self.per_param_history[aux_name] = {}
            
            self.global_history[aux_name].append({
                'step': step,
                'global_cosine': stats['global_cosine'],
            })
            
            for pname, pstats in stats['per_param'].items():
                if pname not in self.per_param_history[aux_name]:
                    self.per_param_history[aux_name][pname] = []
                self.per_param_history[aux_name][pname].append({
                    'step': step,
                    'cosine': pstats['cosine'],
                    'ce_norm': pstats['ce_norm'],
                    'aux_norm': pstats['aux_norm'],
                })
    
    def global_df(self, aux_name=None):
        """Get global cosine over training. If aux_name=None, returns all in one df."""
        if aux_name is not None:
            return pd.DataFrame(self.global_history[aux_name])
        rows = []
        for name, history in self.global_history.items():
            for h in history:
                rows.append({'aux_loss': name, **h})
        return pd.DataFrame(rows)
    
    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'aux_names': self.aux_names,
                'global': self.global_history,
                'per_param': self.per_param_history,
            }, f)
        print(f"Saved multi gradient stats to {path}")
    
    @classmethod
    def load(cls, path):
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        recorder = cls(data['aux_names'])
        recorder.global_history = data['global']
        recorder.per_param_history = data['per_param']
        return recorder


class GradStatsRecorder:
    """Records gradient similarity stats over training."""
    
    def __init__(self):
        self.global_history = []  # [{step, global_cosine, ce_norm, mbe_norm}, ...]
        self.per_param_history = {}  # {param_name: [{step, cosine, ce_norm, mbe_norm}, ...]}
    
    def record(self, stats, step):
        """Record stats from track_gradient_similarity()."""
        # Global stats
        self.global_history.append({
            'step': step,
            'global_cosine': stats['global_cosine'],
        })
        
        # Per-param stats
        for name, pstats in stats['per_param'].items():
            if name not in self.per_param_history:
                self.per_param_history[name] = []
            self.per_param_history[name].append({
                'step': step,
                'cosine': pstats['cosine'],
                'ce_norm': pstats['ce_norm'],
                'mbe_norm': pstats['mbe_norm'],
            })
    
    def to_dataframe(self):
        """Convert to DataFrame for analysis."""
        rows = []
        for name, history in self.per_param_history.items():
            for h in history:
                rows.append({'param': name, **h})
        return pd.DataFrame(rows)
    
    def global_df(self):
        """Get global cosine over training."""
        return pd.DataFrame(self.global_history)
    
    def save(self, path):
        """Save to pickle."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'global': self.global_history,
                'per_param': self.per_param_history
            }, f)
        print(f"Saved gradient stats to {path}")
    
    @classmethod
    def load(cls, path):
        """Load from pickle."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        recorder = cls()
        recorder.global_history = data['global']
        recorder.per_param_history = data['per_param']
        return recorder


