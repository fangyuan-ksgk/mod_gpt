# Simplified ver. of PCGrad (Yu. 2020) which requires O(N) memory for N tasks to requiring O(1) memory
# Viewing loss in each batch as a different task, conduct non-conflicting projection on their gradient
# -----------------------------------------------------------------------------------------------------
import torch 
from collections import defaultdict
import pickle
import numpy as np
from .utils import compute_gradient_cosine_similarities

class SimPGrad:
    def __init__(self, model):
        self.model = model
        self._init_info() # for info logging
    
    def _project_non_conflict(self, g1, g2):
        """Remove conflicting component from secondary gradient"""
        g_dot = torch.sum(g1 * g2)
        if g_dot < 0: 
            g1_norm = torch.sum(g1 * g1)
            if g1_norm > 1e-8: 
                projection = (g_dot / g1_norm) * g1
                g2 = g2 - projection
        return g2
    
    def backward(self, loss_dict, no_priority=False):
        # Basically replacing additive grad accumulation with non-conflicting accumulation
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        loss_names = list(loss_dict.keys())        
        for loss_name in loss_names:
            prev_grads = []
            for p in params:
                if p.grad is not None:
                    prev_grads.append(p.grad.clone())
                    p.grad.zero_() # zero-out to avoid additive accumulation
                else: 
                    prev_grads.append(torch.zeros_like(p))
            loss_dict[loss_name].backward(retain_graph=True)            
            for p, prev_grad in zip(params, prev_grads):
                if p.grad is not None:
                    if no_priority: # all batch & loss are viewed as equaly important
                        p.grad = self._project_non_conflict(p.grad, prev_grad) + self._project_non_conflict(prev_grad, p.grad)
                    else: # loss order in descending importance, previous batch more important than current one
                        p.grad = prev_grad + self._project_non_conflict(prev_grad, p.grad)  
                        
    def naive_backward(self, loss_dict): # for experiment purpose
        params = [p for p in self.model.parameters() if p.requires_grad]
        loss = sum(loss_dict.values())
        loss.backward() 

    # Extra gadget for experiment logging
    # ------------------------------------------------------------------------------------------
     
    def _init_info(self): 
        self.grad_info = {name: defaultdict(list) for name, p in self.model.named_parameters() if p.requires_grad and p.numel() > 1}

    def _update_info(self, param_name, prev_g_norm, curr_g_norm, cosim, loss_name, is_reset, grad_array): 
        if param_name in self.grad_info: 
            if is_reset: 
                self.grad_info[param_name]["grad_angles"] += compute_gradient_cosine_similarities(self.grad_info[param_name])
                self.grad_info[param_name]["grad_array"] = [] # clear up grad caches
                
            self.grad_info[param_name]["prev_grad_norm"].append(prev_g_norm)
            self.grad_info[param_name]["curr_grad_norm"].append(curr_g_norm)
            self.grad_info[param_name]["cosine_similarity"].append(cosim)
            self.grad_info[param_name]["loss_name"].append(loss_name)
            self.grad_info[param_name]["reset"].append(is_reset)
            self.grad_info[param_name]["grad_array"].append(grad_array)

    def _project_non_conflict_info(self, g1, g2):
        """
        For non-priority projection of non-conflicting gradients
        Provide extra information on prev grad norm, current grad norm, cosine similarity between the two
        """
        g1_norm = torch.norm(g1)
        g2_norm = torch.norm(g2)
        if g1_norm > 0 and g2_norm > 0: 
            cosim = torch.sum((g1/g1_norm) * (g2/g2_norm))
        else: 
            cosim = torch.tensor(0.)
        if cosim < 0: 
            if g1_norm > 1e-4:
                projection = g2_norm * cosim
                g2 -= projection
            if g2_norm > 1e-4: 
                projection = g1_norm * cosim
                g1 -= projection
        return g1 + g2, g1_norm.item(), g2_norm.item(), cosim.item()
    
    def _add_grad_info(self, g1, g2): 
        g1_norm = torch.norm(g1)
        g2_norm = torch.norm(g2)
        if g1_norm > 0 and g2_norm > 0: 
            cosim = torch.sum((g1/g1_norm) * (g2/g2_norm))
        else: 
            cosim = torch.tensor(0.)
        return g1 + g2, g1_norm.item(), g2_norm.item(), cosim.item()

    def backward_info(self, loss_dict, no_priority=False): 
        names = [name for name, p in self.model.named_parameters() if p.requires_grad]
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        loss_names = list(loss_dict.keys())        
        for loss_name in loss_names:
            prev_grads = []
            reset_flags = []
            for p in params:
                if p.grad is not None:
                    reset_flags.append(False)
                    prev_grads.append(p.grad.clone())
                    p.grad.zero_() # zero-out to avoid additive accumulation
                else: 
                    reset_flags.append(True)
                    prev_grads.append(torch.zeros_like(p))
            loss_dict[loss_name].backward(retain_graph=True)            
            for name, p, prev_grad, is_reset in zip(names, params, prev_grads, reset_flags):
                if p.grad is not None:            
                    grad_array = p.grad.detach().cpu().to(torch.float16).numpy()
                    p.grad, curr_g_norm, prev_g_norm, cosim = self._project_non_conflict_info(p.grad, prev_grad)
                    self._update_info(name, prev_g_norm, curr_g_norm, cosim, loss_name, is_reset, grad_array)
                    
    def naive_backward_info(self, loss_dict): 
        names = [name for name, p in self.model.named_parameters() if p.requires_grad]
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        loss_names = list(loss_dict.keys())        
        for loss_name in loss_names:
            prev_grads = []
            reset_flags = []
            for p in params:
                if p.grad is not None:
                    reset_flags.append(False)
                    prev_grads.append(p.grad.clone())
                    p.grad.zero_() # zero-out to avoid additive accumulation
                else: 
                    reset_flags.append(True)
                    prev_grads.append(torch.zeros_like(p))
            loss_dict[loss_name].backward(retain_graph=True)            
            for name, p, prev_grad, is_reset in zip(names, params, prev_grads, reset_flags):
                if p.grad is not None:            
                    grad_array = p.grad.detach().cpu().to(torch.float16).numpy()
                    p.grad, curr_g_norm, prev_g_norm, cosim = self._add_grad_info(p.grad, prev_grad)
                    self._update_info(name, prev_g_norm, curr_g_norm, cosim, loss_name, is_reset, grad_array)
                    
    def save_grad_info(self, path):         
        serializable_grad_info = {}
        for param_name, info in self.grad_info.items():
            serializable_grad_info[param_name] = {k: np.array(v) if k == "curr_grad" else v 
                                                for k, v in dict(info).items()}
        
        with open(path, "wb") as f:  # Note: changed to binary mode
            pickle.dump(serializable_grad_info, f)

# Original PCGrad implementation, for reference
class PCGrad:
    def __init__(self, model):
        self.model = model
        
    def pc_backward(self, loss_dict, priority_loss=None):
        params = [p for p in self.model.parameters() if p.requires_grad]
        grads = {}
        for loss_name, loss_value in loss_dict.items():
            for p in params:
                if p.grad is not None:
                    p.grad.zero_()
            loss_value.backward(retain_graph=True)
            grads[loss_name] = []
            for p in params:
                if p.grad is not None:
                    # grads[loss_name].append(p.grad.clone())
                    grads[loss_name].append(p.grad.clone()/p.grad.norm()) # @ksgk modified to use directional gradient only
                else:
                    grads[loss_name].append(torch.zeros_like(p))
                    
        for i, loss_name_i in enumerate(loss_dict.keys()):
            for j, loss_name_j in enumerate(loss_dict.keys()):
                if i == j:
                    continue
                if priority_loss and i == priority_loss: # priority loss will be kept intact
                    continue 
                for g_i, g_j in zip(grads[loss_name_i], grads[loss_name_j]):
                    g_i_g_j = torch.sum(g_i * g_j)
                    if g_i_g_j < 0:
                        g_j_norm_sq = torch.sum(g_j * g_j)
                        if g_j_norm_sq > 1e-8:
                            projection = (g_i_g_j / g_j_norm_sq) * g_j
                            g_i.sub_(projection) # remove conflicting gradient component
                
        for i, p in enumerate(params):
            if p.requires_grad:
                p.grad = sum([grads[loss_name][i] for loss_name in loss_dict.keys()]) / len(loss_dict)