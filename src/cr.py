import torch 
import random

# This version doesn't increase memory consumption, too 
# So what is the issue that leads to memory increment? randomization gadget? 
class ChunkedRotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = None
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim))
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            self.cos_cached = freqs.cos().float()
            self.sin_cached = freqs.sin().float()
        cos, sin = self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]
        # apply_rotary_emb(x, cos, sin)
        assert x.ndim == 4 # multihead attention
        d = x.shape[3]//4
        x1 = x[..., :d]
        x2 = x[..., d:2*d]
        x3 = x[..., 2*d:3*d]
        x4 = x[..., 3*d:]
        y1 = x1 * cos[...,:d] + x2 * sin[...,:d]
        y2 = x1 * (-sin[...,:d]) + x2 * cos[...,:d]
        y3 = x3 * cos[...,d:] + x4 * sin[...,d:]
        y4 = x3 * (-sin[...,d:]) + x4 * cos[...,d:]
        return torch.cat([y1, y2, y3, y4], 3).type_as(x)


class NChunkRotary(torch.nn.Module):

    def __init__(self, dim, base=10000, n_chunk=4):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = None
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        assert n_chunk % 2 == 0
        self.n_chunk = n_chunk
        

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim))
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            self.cos_cached = freqs.cos().float()
            self.sin_cached = freqs.sin().float()
        cos, sin = self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]
        # apply_rotary_emb(x, cos, sin)
        assert x.ndim == 4 # multihead attention
        d = x.shape[3]//self.n_chunk
        assert d >= 1

        rope_emb = None
        for i in range(self.n_chunk//2): 
            x1 = x[..., 2*i*d:(2*i+1)*d]
            x2 = x[..., (2*i+1)*d:(2*i+2)*d]
            y1 = x1 * cos[..., i*d:(i+1)*d] + x2 * sin[...,i*d:(i+1)*d]
            y2 = x1 * (-sin[..., i*d:(i+1)*d]) + x2 * cos[..., i*d:(i+1)*d]
            if rope_emb is None: 
                rope_emb = torch.cat([y1, y2], 3).type_as(x)
            else:
                rope_emb = torch.cat([rope_emb, y1, y2], 3).type_as(x)
        return rope_emb



class NChunkRandomRotary(torch.nn.Module):

    def __init__(self, dim, base=10000, n_chunk=4):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = None
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        assert n_chunk % 2 == 0
        self.n_chunk = n_chunk
        

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim))
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            self.cos_cached = freqs.cos().float()
            self.sin_cached = freqs.sin().float()
        cos, sin = self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]
        # apply_rotary_emb(x, cos, sin)
        assert x.ndim == 4 # multihead attention
        d = x.shape[3]//self.n_chunk
        assert d >= 1
        
        do_permute = torch.rand(self.n_chunk//2)>0.5

        rope_emb = None
        for i in range(self.n_chunk//2): 
            if do_permute[i]:
                x2, x1 = x[..., 2*i*d:(2*i+1)*d], x[..., (2*i+1)*d:(2*i+2)*d]
            else: 
                x1, x2 = x[..., 2*i*d:(2*i+1)*d], x[..., (2*i+1)*d:(2*i+2)*d]
            y1 = x1 * cos[..., i*d:(i+1)*d] + x2 * sin[...,i*d:(i+1)*d]
            y2 = x1 * (-sin[..., i*d:(i+1)*d]) + x2 * cos[..., i*d:(i+1)*d]
            if rope_emb is None: 
                rope_emb = torch.cat([y1, y2], 3).type_as(x)
            else:
                rope_emb = torch.cat([rope_emb, y1, y2], 3).type_as(x)
        return rope_emb


class NChunkRandomRotary2(torch.nn.Module):

    def __init__(self, dim, base=10000, n_chunk=4):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = None
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        assert n_chunk % 2 == 0
        self.n_chunk = n_chunk        

    @torch.compiler.disable
    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim))
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            self.cos_cached = freqs.cos().float()
            self.sin_cached = freqs.sin().float()
        cos, sin = self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]
        # apply_rotary_emb(x, cos, sin)
        assert x.ndim == 4 # multihead attention
        d = x.shape[3]//self.n_chunk
        assert d >= 1

        indices = [i for i in range(self.n_chunk)]
        random.shuffle(indices)
        pairs = [(indices[i], indices[i+1]) for i in range(0, len(indices), 2)]
        
        rope_emb = None
        for i, (j1, j2) in enumerate(pairs): 
            x1, x2 = x[..., j1*d:(j1+1)*d], x[..., j2*d:(j2+1)*d]
            y1 = x1 * cos[..., i*d:(i+1)*d] + x2 * sin[...,i*d:(i+1)*d]
            y2 = x1 * (-sin[..., i*d:(i+1)*d]) + x2 * cos[..., i*d:(i+1)*d]
            if rope_emb is None: 
                rope_emb = torch.cat([y1, y2], 3).type_as(x)
            else:
                rope_emb = torch.cat([rope_emb, y1, y2], 3).type_as(x)
        return rope_emb





# Below ver. gives 4.0 val loss at iter250, n_chunks=2, a pretty nice result compared to Rotary
# however, it requires 2x memory, which is crazy .... 
class ChunkedRandomizedRotary(torch.nn.Module):
    def __init__(self, dim, base=10000, n_chunks=2, device='cuda', eval_randomize=False):
        super().__init__()
        self.dim = dim
        self.base = base
        self.n_chunks = n_chunks
        self.eval_randomize = eval_randomize
        
        assert dim % n_chunks == 0, f"Dimension {dim} must be divisible by n_chunks {n_chunks}"
        self.chunk_size = dim // n_chunks
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
    
    def forward(self, x):
        seq_len = x.shape[1]
        device = x.device
        
        if seq_len != self.seq_len_cached:
            self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.chunk_size, device=device).float() / (self.chunk_size)))
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            self.cos_cached = freqs.cos().float()
            self.sin_cached = freqs.sin().float()
        
        cos, sin = self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]
        
        # Generate random chunk pairings
        if self.training or self.eval_randomize:
            perm = torch.randperm(self.n_chunks, device=device)
        else:
            perm = torch.arange(self.n_chunks, device=device)        
        result_chunks = []
        
        for i in range(0, self.n_chunks, 2):
            if i + 1 < self.n_chunks:
                idx1, idx2 = perm[i], perm[i + 1]
                x1 = x[..., idx1 * self.chunk_size:(idx1 + 1) * self.chunk_size]
                x2 = x[..., idx2 * self.chunk_size:(idx2 + 1) * self.chunk_size]
                y1 = x1 * cos + x2 * sin
                y2 = x1 * (-sin) + x2 * cos
                if idx1 < idx2:
                    result_chunks.append(y1)
                    result_chunks.append(y2)
                else:
                    result_chunks.append(y2)
                    result_chunks.append(y1)
            else:
                idx = perm[i]
                chunk = x[..., idx * self.chunk_size:(idx + 1) * self.chunk_size]
                result_chunks.append(chunk)
        
        sorted_chunks = [None] * self.n_chunks
        for i, chunk in enumerate(result_chunks):
            sorted_chunks[perm[i]] = chunk
        
        return torch.cat(sorted_chunks, dim=-1).type_as(x)