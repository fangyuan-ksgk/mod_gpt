# Gradient Statistics Tracker
# Tracks cosine similarity and norms between gradients from different losses
# -----------------------------------------------------------------------------
import torch 
from collections import defaultdict
import pickle
import numpy as np

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
    
    def backward_with_tracking(self, loss_dict):
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
        loss_dict[loss_name].backward(retain_graph=False)
        
        # Track gradient statistics
        for i, p in enumerate(params):
            p.grad, prev_g_norm, curr_g_norm, cosim = \
                self._compute_grad_stats(prev_grads[i], p.grad)
            self._update_info(
                param_names[i], prev_g_norm, curr_g_norm, cosim, 
                loss_name, reset_flags[i]
            )
    
    def save_grad_info(self, path):
        """Save gradient tracking information to disk"""
        serializable_grad_info = {}
        for param_name, info in self.grad_info.items():
            serializable_grad_info[param_name] = dict(info)
        
        with open(path, "wb") as f:
            pickle.dump(serializable_grad_info, f)
        
        print(f"Gradient statistics saved to {path}")