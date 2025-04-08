# Experimental Model Components
# ---------------------------------------------------------

# Rotary embedding could be applied to randomized SO(2) group
class RandomizedRotary(torch.nn.Module):
    def __init__(self, dim, base=10000, fixed_grouping=True):
        super().__init__()
        self.dim = dim
        self.base = base
        self.fixed_grouping = fixed_grouping
        self.inv_freq = None
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

        if fixed_grouping: 
            self.indices1 = torch.arange(0, self.dim//2)
            self.indices2 = torch.arange(self.dim//2, self.dim)

    def _randomize_grouping(self, device): 
        generator = torch.Generator(device=device)
        self.permutation = torch.randperm(self.dim, device=device)
        self.indices1 = self.permutation[:self.dim//2]
        self.indices2 = self.permutation[self.dim//2:]

    def forward(self, x):
        seq_len = x.shape[1]
        batch_size = x.shape[0]
        heads = x.shape[2]
        
        # Generate new permutation for each forward pass if not fixed
        if not self.fixed_grouping:
            self._randomize_grouping(x.device) 
        
        # Cache position embeddings
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
        
        # Split using the random indices
        x1 = torch.index_select(x, 3, self.indices1)
        x2 = torch.index_select(x, 3, self.indices2)
        
        # Apply rotation (we need to match dimensions between x1/x2 and cos/sin)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        
        # Combine back using original permutation order
        result = torch.zeros_like(x)
        result[..., self.indices1] = y1
        result[..., self.indices2] = y2
        
        return result.type_as(x)