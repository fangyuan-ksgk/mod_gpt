# Simplified ver. of PCGrad (Yu. 2020) which requires O(N) memory for N tasks to requiring O(1) memory
# Viewing loss in each batch as a different task, conduct non-conflicting projection on their gradient
# -----------------------------------------------------------------------------------------------------
import torch 
from collections import defaultdict
import pickle
import numpy as np
from .utils import compute_gradient_cosine_similarities, plot_grad_info

class SimPGrad:
    def __init__(self, model):
        self.model = model
        self._init_info() # for info logging
        self._project_non_conflict = torch.compile(self._project_non_conflict_noncompiled)
    
    def _project_non_conflict_noncompiled(self, g1, g2):
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
        assert len(loss_names) == 1, "retain_graph conflict with donated_buffer in pytorch, require one backward per forward calls"
        for loss_name in loss_names:
            prev_grads = []
            for p in params:
                if p.grad is not None:
                    prev_grads.append(p.grad.clone())
                    p.grad.zero_() # zero-out to avoid additive accumulation
                else: 
                    prev_grads.append(torch.zeros_like(p))
            loss_dict[loss_name].backward(retain_graph=False)            
            for p, prev_grad in zip(params, prev_grads):
                if p.grad is not None:
                    if no_priority: 
                        p.grad = self._project_non_conflict(p.grad, prev_grad) + self._project_non_conflict(prev_grad, p.grad)
                    else: 
                        p.grad = prev_grad + self._project_non_conflict(prev_grad, p.grad)
                
    def naive_backward(self, loss_dict): # for experiment purpose
        params = [p for p in self.model.parameters() if p.requires_grad]
        loss = sum(loss_dict.values())
        loss.backward() 


    def backward_2phase(self, loss_dict): # prioritize entropy loss while projecting MBE loss
        assert len(loss_dict) == 1, "backward_2phase requires exactly one loss"
        if 'entropy' in loss_dict: 
            self.naive_backward(loss_dict)
        else: 
            self.backward(loss_dict, no_priority=False)


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
    
    def _project_non_conflict_info_priority(self, g1, g2): 
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
        assert len(loss_names) == 1, "retain_graph conflict with donated_buffer in pytorch, require one backward per forward calls"
        
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
            loss_dict[loss_name].backward(retain_graph=False)            
            for name, p, prev_grad, is_reset in zip(names, params, prev_grads, reset_flags):
                if p.grad is not None:            
                    grad_array = p.grad.detach().cpu().to(torch.float16).numpy()
                    if no_priority: 
                        p.grad, curr_g_norm, prev_g_norm, cosim = self._project_non_conflict_info_priority(p.grad, prev_grad)
                    else: 
                        p.grad, curr_g_norm, prev_g_norm, cosim = self._project_non_conflict_info(p.grad, prev_grad)
                    self._update_info(name, prev_g_norm, curr_g_norm, cosim, loss_name, is_reset, grad_array)
                    
    def naive_backward_info(self, loss_dict): 
        names = [name for name, p in self.model.named_parameters() if p.requires_grad]
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        loss_names = list(loss_dict.keys())        
        assert len(loss_names) == 1, "retain_graph conflict with donated_buffer in pytorch, require one backward per forward calls"
        
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
            loss_dict[loss_name].backward(retain_graph=False)            
            for name, p, prev_grad, is_reset in zip(names, params, prev_grads, reset_flags):
                if p.grad is not None:            
                    grad_array = p.grad.detach().cpu().to(torch.float16).numpy()
                    p.grad, curr_g_norm, prev_g_norm, cosim = self._add_grad_info(p.grad, prev_grad)
                    self._update_info(name, prev_g_norm, curr_g_norm, cosim, loss_name, is_reset, grad_array)
                    
    def backward_2phase_info(self, loss_dict): 
        assert len(loss_dict) == 1, "backward_2phase_info requires exactly one loss"
        if 'entropy' in loss_dict: 
            self.naive_backward_info(loss_dict)
        else: 
            self.backward_info(loss_dict, no_priority=False)


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


# Another ver. 
# Remark 1. accumulation step scaling should use priority loss steps instead of full gradient steps
class YetAnotherMixer: 

    def __init__(self, model, priority_loss_name):
        self.model = model
        self.priority_loss_name = priority_loss_name
        self._init_info() # for info logging
        self._project_non_conflict = torch.compile(self._project_non_conflict_noncompiled)
        self.init_priority_grad_cache()
    
    def init_priority_grad_cache(self): 
        self.priority_grad_cache = []
        for p in self.model.parameters(): 
            if p.requires_grad: 
                self.priority_grad_cache.append(torch.zeros_like(p))
                
    def _project_non_conflict_noncompiled(self, g1, g2):
        """Remove conflicting component from secondary gradient"""
        g_dot = torch.sum(g1 * g2)
        if g_dot < 0: 
            g1_norm = torch.sum(g1 * g1)
            if g1_norm > 1e-8: 
                projection = (g_dot / g1_norm) * g1
                g2 = g2 - projection
        return g2
    
    def backward(self, loss_dict, np_weight=1.0):
        params = [p for p in self.model.parameters() if p.requires_grad]
        loss_name = list(loss_dict.keys())[0]

        if loss_name == self.priority_loss_name: 
            loss_dict[loss_name].backward(retain_graph=False)
            for i, p in enumerate(params):
                self.priority_grad_cache[i] = self.priority_grad_cache[i] + p.grad.detach()
        else: 
            prev_grads = [] 
            for p in params: 
                if p.grad is not None: 
                    prev_grads.append(p.grad.clone())
                    p.grad.zero_()
                else: 
                    prev_grads.append(torch.zeros_like(p))
            loss_dict[loss_name].backward(retain_graph=False)
            for p, cache, prev_grad in zip(params, self.priority_grad_cache, prev_grads): 
                # accumulate gradient aligned to priority gradient
                p.grad = prev_grad + self._project_non_conflict(cache, p.grad) * np_weight

    def naive_backward(self, loss_dict): 
        params = [p for p in self.model.parameters() if p.requires_grad]
        loss = sum(loss_dict.values())
        loss.backward() 

    def _project_non_conflict_info(self, g1, g2):
        g2_array = g2.detach().cpu().to(torch.float16).numpy()
        g1_norm = torch.norm(g1)
        g2_norm = torch.norm(g2)
        g_dot = torch.sum(g1 * g2)
        cosim = g_dot / (g1_norm * g2_norm)
        if g_dot < 0: 
            if g1_norm > 1e-8: 
                projection = (g_dot / g1_norm.pow(2)) * g1
                g2 = g2 - projection
        return g2, g1_norm.item(), g2_norm.item(), cosim.item(), g2_array
    
    def _add_grad_info(self, g1, g2): 
        g2_array = g2.detach().cpu().to(torch.float16).numpy()
        g1_norm = torch.norm(g1)
        g2_norm = torch.norm(g2)
        if g1_norm > 0 and g2_norm > 0: 
            cosim = torch.sum((g1/g1_norm) * (g2/g2_norm))
        else: 
            cosim = torch.tensor(0.)
        return g1 + g2, g1_norm.item(), g2_norm.item(), cosim.item(), g2_array
    
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


    def backward_info(self, loss_dict, np_weight=1.0):
        param_names = [p[0] for p in self.model.named_parameters() if p[1].requires_grad]
        params = [p for p in self.model.parameters() if p.requires_grad]
        assert len(loss_dict) == 1, "torch compile requires exactly one loss for explicit backward pass"
        loss_name = list(loss_dict.keys())[0]

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

        if loss_name == self.priority_loss_name: 
            print(" - priority loss | additive accumulation & update priority gradient cache")
            loss_dict[loss_name].backward(retain_graph=False)
            for i, p in enumerate(params): 
                self.priority_grad_cache[i] = self.priority_grad_cache[i] + p.grad.detach()
                p.grad, prev_g_norm, curr_g_norm, cosim, g1_array = self._add_grad_info(prev_grads[i], p.grad)
                self._update_info(param_names[i], prev_g_norm, curr_g_norm, cosim, loss_name, reset_flags[i], g1_array)
        else: 
            print(" - non-priority loss | project to non-conflicting direction to priority gradient")
            loss_dict[loss_name].backward(retain_graph=False)
            for p, name, priority_grad, prev_grad, is_reset in zip(params, param_names, self.priority_grad_cache, prev_grads, reset_flags): 
                p.grad, priority_g_norm, curr_g_norm, cosim, g1_array = self._project_non_conflict_info(priority_grad, p.grad)
                p.grad = p.grad * np_weight + prev_grad
                self._update_info(name, priority_g_norm, curr_g_norm, cosim, loss_name, is_reset, g1_array)


    def naive_backward_info(self, loss_dict): 
        param_names = [p[0] for p in self.model.named_parameters() if p[1].requires_grad]
        params = [p for p in self.model.parameters() if p.requires_grad]
        assert len(loss_dict) == 1, "torch compile requires exactly one loss for explicit backward pass"
        loss_name = list(loss_dict.keys())[0]

        reset_flags = []
        prev_grads = [] 
        for p in params:
            if p.grad is not None:
                reset_flags.append(False)
                prev_grads.append(p.grad.clone())
                p.grad.zero_() 
            else: 
                print(" None Gradient encountered")
                reset_flags.append(True)
                prev_grads.append(torch.zeros_like(p))

        loss_dict[loss_name].backward(retain_graph=False)
        for i, p in enumerate(params): 
            p.grad, prev_g_norm, curr_g_norm, cosim, curr_g_array = self._add_grad_info(prev_grads[i], p.grad)
            self._update_info(param_names[i], prev_g_norm, curr_g_norm, cosim, loss_name, reset_flags[i], curr_g_array)


    def save_grad_info(self, path):         
        serializable_grad_info = {}
        for param_name, info in self.grad_info.items():
            serializable_grad_info[param_name] = {k: np.array(v) if k == "curr_grad" else v 
                                                for k, v in dict(info).items()}
            
        # plot grad info
        plot_grad_info(serializable_grad_info, save_dir=path)
        
        with open(path, "wb") as f:  # Note: changed to binary mode
            pickle.dump(serializable_grad_info, f)
            
            
            
            
# Another Mixer Class: 
# - projective loss name | this is the loss whose gradient we'll project to different components and compose together

class YetAnotherMixer2: 

    def __init__(self, model, projective_loss_name):
        self.model = model
        self.projective_loss_name = projective_loss_name
        self._init_info() # for info logging
        self._scale_projective_component = torch.compile(self._scale_projective_component_noncompiled)
        self.init_priority_grad_cache()
    
    def init_priority_grad_cache(self): 
        self.priority_grad_cache = []
        for p in self.model.parameters(): 
            if p.requires_grad: 
                self.priority_grad_cache.append(torch.zeros_like(p))
                
    def _scale_projective_component_noncompiled(self, g, g_calib, scale_factor=1.0):
        """Scale the projection component of g to g_calib by scale_factor"""
        g_dot = torch.sum(g * g_calib)
        g_calib_norm = torch.sum(g_calib * g_calib)
        if g_calib_norm > 1e-8: 
            g_proj = (g_dot / g_calib_norm) * g_calib
            g = g + g_proj * (scale_factor - 1)
        return g
    
    def backward(self, loss_dict, scale_factor=1.0):
        params = [p for p in self.model.parameters() if p.requires_grad]
        loss_name = list(loss_dict.keys())[0]

        if loss_name == self.projective_loss_name: 
            loss_dict[loss_name].backward(retain_graph=False)
        else: 
            prev_grads = [] 
            for p in params: 
                if p.grad is not None: 
                    prev_grads.append(p.grad.clone())
                    p.grad.zero_()
                else: 
                    prev_grads.append(torch.zeros_like(p))
            loss_dict[loss_name].backward(retain_graph=False)
            for p, prev_grad in zip(params, prev_grads):
                p.grad = self._scale_projective_component(prev_grad, p.grad, scale_factor=scale_factor)

    def naive_backward(self, loss_dict): 
        params = [p for p in self.model.parameters() if p.requires_grad]
        loss = sum(loss_dict.values())
        loss.backward() 
        
    def _scale_projective_component_info(self, g, g_calib, scale_factor=1.0):
        # Issue (I). g_calib does not have significant magnitude to scale g. 
        
        g_dot = torch.sum(g * g_calib)
        g_norm = torch.norm(g)
        g_calib_norm = torch.norm(g_calib)
        cosim = torch.tensor(0.)
        
        if g_calib_norm * g_norm > 1e-8:
            cosim = g_dot / (g_norm * g_calib_norm)
            g_proj = cosim * g_calib
            g = g + g_proj * (scale_factor - 1)
            
        return g, g_norm.item(), g_calib_norm.item(), cosim.item(), g
    
    def _add_grad_info(self, g1, g2): 
        g2_array = g2.detach().cpu().to(torch.float16).numpy()
        g1_norm = torch.norm(g1)
        g2_norm = torch.norm(g2)
        if g1_norm > 1e-8 and g2_norm > 1e-8: 
            cosim = torch.sum((g1/g1_norm) * (g2/g2_norm))
        else: 
            cosim = torch.tensor(0.)
        return g1 + g2, g1_norm.item(), g2_norm.item(), cosim.item(), g2_array
    
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

    def backward_info(self, loss_dict, scale_factor=1.0):
        param_names = [p[0] for p in self.model.named_parameters() if p[1].requires_grad]
        params = [p for p in self.model.parameters() if p.requires_grad]
        assert len(loss_dict) == 1, "torch compile requires exactly one loss for explicit backward pass"
        loss_name = list(loss_dict.keys())[0]

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

        if loss_name == self.projective_loss_name: 
            loss_dict[loss_name].backward(retain_graph=False)
            for i, p in enumerate(params): 
                
                p.grad, prev_g_norm, curr_g_norm, cosim, g1_array = self._add_grad_info(prev_grads[i], p.grad)
                self._update_info(param_names[i], prev_g_norm, curr_g_norm, cosim, loss_name, reset_flags[i], g1_array)
        else: 
            loss_dict[loss_name].backward(retain_graph=False)
            for i, p in enumerate(params): 
                p.grad, prev_g_norm, curr_g_norm, cosim, g1_array = self._scale_projective_component_info(prev_grads[i], p.grad, scale_factor=scale_factor)
                self._update_info(param_names[i], prev_g_norm, curr_g_norm, cosim, loss_name, reset_flags[i], g1_array)
                

    def save_grad_info(self, path):         
        serializable_grad_info = {}
        for param_name, info in self.grad_info.items():
            serializable_grad_info[param_name] = {k: np.array(v) if k == "curr_grad" else v 
                                                for k, v in dict(info).items()}
            
        # plot grad info
        plot_grad_info(serializable_grad_info, save_dir=path)
        
        with open(path, "wb") as f:  # Note: changed to binary mode
            pickle.dump(serializable_grad_info, f)