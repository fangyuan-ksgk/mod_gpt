# Simplified ver. of PCGrad (Yu. 2020) which requires O(N) memory for N tasks to requiring O(1) memory
# Viewing loss in each batch as a different task, conduct non-conflicting projection on their gradient
# -----------------------------------------------------------------------------------------------------
import torch 

class SimPGrad:
    def __init__(self, model):
        self.model = model
    
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
                        p.grad = self._project_non_confict(p.grad, prev_grad) + self._project_non_conflict(prev_grad, p.grad)
                    else: # loss order in descending importance, previous batch more important than current one
                        p.grad = prev_grad + self._project_non_conflict(prev_grad, p.grad)  
                        


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