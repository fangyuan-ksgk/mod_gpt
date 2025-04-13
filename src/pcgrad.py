# Modified Project Conflicting Gradient 
# --------------------------------------------------------------------

class PCGrad:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        
    def zero_grad(self):
        return self.optimizer.zero_grad(set_to_none=True)
        
    def step(self):
        return self.optimizer.step()
    
    def pc_backward(self, loss_dict):
       
        grads = {}
        
        params = []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    params.append(p)

        # collect gradient
        for loss_name, loss_value in loss_dict.items():
            self.optimizer.zero_grad(set_to_none=True)
            loss_value.backward(retain_graph=True)
            grads[loss_name] = []
            for p in params:
                if p.grad is not None:
                    # grads[loss_name].append(p.grad.clone())
                    grads[loss_name].append(p.grad.clone()/p.grad.norm()) # @ksgk modified to use directional gradient only
                else:
                    grads[loss_name].append(torch.zeros_like(p))
        
        # remove conflicting projection between gradients 
        for i, loss_name_i in enumerate(loss_dict.keys()):
            for j, loss_name_j in enumerate(loss_dict.keys()):
                if i == j:
                    continue
                    
                for g_i, g_j in zip(grads[loss_name_i], grads[loss_name_j]):
                    g_i_g_j = torch.sum(g_i * g_j)
                    if g_i_g_j < 0:  # If gradients conflict
                        g_j_norm_sq = torch.sum(g_j * g_j)
                        if g_j_norm_sq > 1e-8:  # Avoid division by zero
                            # Project g_i onto the normal plane of g_j
                            projection = (g_i_g_j / g_j_norm_sq) * g_j
                            g_i.sub_(projection)
        
        # Combine all gradients (average)
        self.optimizer.zero_grad(set_to_none=True)
        for i, p in enumerate(params):
            if p.requires_grad:
                p.grad = sum([grads[loss_name][i] for loss_name in loss_dict.keys()]) / len(loss_dict)