# Experimental Model Components
# ---------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

class DynamicTanh(nn.Module):
    def __init__(self, dim, alpha_init=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init)
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        
    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias

class CastedLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))

        
# Rotary embedding could be applied to randomized SO(2) group
class RandomizedRotary(torch.nn.Module):
    def __init__(self, dim, base=10000, device='cuda'):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = None
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        self.indices1 = torch.arange(0, self.dim//2).to(device)
        self.indices2 = torch.arange(self.dim//2, self.dim).to(device)

    def _randomize_grouping(self, device): 
        generator = torch.Generator(device=device)
        self.permutation = torch.randperm(self.dim, device=device)
        self.indices1 = self.permutation[:self.dim//2]
        self.indices2 = self.permutation[self.dim//2:]

    def forward(self, x, fixed_grouping=True):
        seq_len = x.shape[1]        
        if not fixed_grouping:
            self._randomize_grouping(x.device) 
            
        if seq_len != self.seq_len_cached:
            self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim//2, device=x.device).float() / (self.dim//2)))
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            self.cos_cached = freqs.cos().float()
            self.sin_cached = freqs.sin().float()
        cos, sin = self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]
        # Apply rotary embeddings with randomized grouping
        assert x.ndim == 4  # multihead attention        
        x1 = torch.index_select(x, 3, self.indices1)
        x2 = torch.index_select(x, 3, self.indices2)        
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos        
        return torch.cat([y1, y2], 3).type_as(x) 


# Causal Self Attention with Randomized Head groups
class RandomizedCausalSelfAttention(nn.Module):
    def __init__(self, dim, n_head, flex_kernel_options=None):
        super().__init__()
        assert dim % n_head == 0, "Embedding dimension must be divisible by number of heads"
        self.dim = dim
        self.n_head = n_head        
        self.c_q = nn.Linear(dim, dim)
        self.c_k = nn.Linear(dim, dim)
        self.c_v = nn.Linear(dim, dim)
        # value residual lambda 
        self.lamb = nn.Parameter(torch.tensor(0.5))  # @Grad62304977
        # rotary embeddings
        self.rotary = RandomizedRotary(dim // n_head)
        # output projection
        self.c_proj = nn.Linear(dim, dim)
        self.c_proj.weight.data.zero_()  # zero init suggested by @Grad62304977
        # flex attention kernel options
        self.flex_kernel_options = flex_kernel_options

    def _randomize_feature_dimensions(self, x, device):
        permutation = torch.randperm(self.dim, device=device)
        x = x[..., permutation]
        return x, permutation

    def _reverse_permutation(self, y, permutation, device):
        inverse_permutation = torch.argsort(permutation)
        y = y[..., inverse_permutation]
        return y

    def forward(self, x, v1=None, block_mask=None, randomize=True):
        B, T = x.size(0), x.size(1)  # Batch size and sequence length
        # assert B == 1, "Must use batch size = 1 for FlexAttention"
        
        # Randomly permute feature dimensions
        if randomize: 
            x, permutation = self._randomize_feature_dimensions(x, x.device)
        
        # Compute Q, K, V
        q = self.c_q(x).view(B, T, self.n_head, -1)
        k = self.c_k(x).view(B, T, self.n_head, -1)
        v = self.c_v(x).view(B, T, self.n_head, -1)        
        if v1 is None:
            v1 = v  # If this is the first block, set v1 to v
        v = (1 - self.lamb) * v + self.lamb * v1.view_as(v)  # @Grad62304977
        q, k = norm(q), norm(k) # QK norm suggested by @Grad62304977
        q, k = self.rotary(q, randomize), self.rotary(k, randomize)        
        y = flex_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            block_mask=block_mask,
            kernel_options=self.flex_kernel_options
        )
        y = y.transpose(1, 2).contiguous().view_as(x)
        if randomize:         
            y = self._reverse_permutation(y, permutation, x.device)        
        y = self.c_proj(y)
        return y, v1


class MLP(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.c_fc   = CastedLinear(dim, 4 * dim)
        self.c_proj = CastedLinear(4 * dim, dim)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config.n_embd, config.n_head, config.flex_kernel_options)
        self.mlp = MLP(config.n_embd)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x, v1, x0, block_mask):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        x1, v1 = self.attn(norm(x), v1, block_mask)
        x = x + x1
        x = x + self.mlp(norm(x))
        return x, v1


# TBD: extend 'GPT' model class that has batch forward functional