# GAT with recursion + adaptive stop
# - no continuous recursion ver. | no act ver.

from sorl.model import Block, CastedLinear, create_block_mask
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import torch
import torch.nn.functional as F
from sorl.model import norm
import torch.nn as nn

# BOS_TOKEN_ID = 50256
BOS_TOKEN_ID = 15

@dataclass
class GATConfig:
    vocab_sizes : list
    n_layer : int = 12
    n_head : int = 6
    n_embd : int = 768
    flex_kernel_options: Optional[dict] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    _compile: bool = True if device == "cuda" else False

get_level_mask_tokens = lambda vocab_sizes: torch.cat(
    (
        torch.tensor([0]),
        torch.cumsum(vocab_sizes, dim=0)
    )
)

class GAT(nn.Module): 

    def __init__(self, config):
        super().__init__()

        # U-net design by @brendanh0gan
        self.num_encoder_layers = config.n_layer // 2 # Half of the layers for encoder
        self.num_decoder_layers = config.n_layer - self.num_encoder_layers # Remaining for decoder
        # Add learnable skip connection weights for decoder layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

        self.register_buffer('vocab_sizes', torch.tensor(config.vocab_sizes))
        self.vocab_size = sum(self.vocab_sizes)

        
        level_starts = torch.cat([torch.tensor([0]), 
                                  torch.cumsum(self.vocab_sizes, dim=0)[:-1] + 1])
        level_ends = torch.cumsum(self.vocab_sizes, dim=0)
        
        self.register_buffer('level_starts', level_starts)
        self.register_buffer('level_ends', level_ends)
        

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = CastedLinear(config.n_embd, self.vocab_size)
        self.lm_head.weight.data.zero_() # @Grad62304977

        # ACT should be done by argmax criteria (on trajectory chunk)
        self.n_embd = config.n_embd
      
        self.device = config.device
        self._compile = config._compile


    def _forward_pass(self, idx: torch.Tensor, memory_span: int, attn_blocksize: int):

        docs = (idx == BOS_TOKEN_ID).cumsum(1)
        
        levels = (idx >= self.vocab_sizes[0]).long()

        def causal_mask(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[b, q_idx] == docs[b, kv_idx]
            window_mask = q_idx - kv_idx < attn_blocksize
            is_higher_level = levels[b, kv_idx] > 0
            is_recent = (q_idx - kv_idx) <= memory_span
            memory_compression_mask = is_higher_level | is_recent 
            return causal_mask & document_mask & window_mask & memory_compression_mask

        S = idx.shape[1]
        block_mask = create_block_mask(causal_mask, None, None, S, S, device=self.device, _compile=self._compile)

        x = self.transformer.wte(idx)

        x = norm(x)
        x0 = x
        v1 = None

        skip_connections = []
        for i in range(self.num_encoder_layers):
            x, v1 = self.transformer.h[i](x, v1, x0, block_mask)
            skip_connections.append(x)
        
        for i in range(self.num_decoder_layers):
            skip_idx = self.num_encoder_layers - 1 - i
            x = x + self.skip_weights[i] * skip_connections[skip_idx]
            x, v1 = self.transformer.h[self.num_encoder_layers + i](x, v1, x0, block_mask)

        x = norm(x)
        return x

    def forward(self, idx, memory_span, attn_blocksize):

        x = self._forward_pass(idx, memory_span, attn_blocksize)
        logits = self.lm_head(x)
        logits = 30 * torch.tanh(logits / 30)
        logits = logits.float()

        # --- loss --- 
        loss = F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, logits.size(-1)), 
            idx[:, 1:].contiguous().view(-1).long(), 
            reduction="none"
        )
        
        # Don't predict: (1) what comes after BOS, (2) BOS itself
        bos_pos_mask = torch.logical_and(idx[:, :-1] != BOS_TOKEN_ID, idx[:, 1:] != BOS_TOKEN_ID).view(-1).float()        
        loss = loss * bos_pos_mask
        return loss, logits.detach()

def get_next_token_level(seq_length, abstraction_interval):
    # assumes L=2 (to be extended)
    assert seq_length > 0, "Sequence length must be greater than 0"
    return 1 if (seq_length % abstraction_interval == 0) else 0

def get_logits_mask(level, vocab_sizes):
    """
    Mask logits to only allow tokens from a specific level.
    """
    level_starts = torch.cat([torch.tensor([0], device=vocab_sizes.device), torch.cumsum(vocab_sizes, dim=0)[:-1] + 1])
    level_ends = torch.cumsum(vocab_sizes, dim=0)

    start_logits = level_starts[level]
    end_logits = level_ends[level]
    
    vocab_indices = torch.arange(sum(vocab_sizes), device=vocab_sizes.device)
    mask = (vocab_indices.unsqueeze(0) >= start_logits.unsqueeze(-1)) & (vocab_indices.unsqueeze(0) < end_logits.unsqueeze(-1))
    
    return mask

@torch.compile
def extract_and_sample(logits, idx, recursion_mask, vocab_sizes, temperature):
    """Compiled helper: extract masked logits and sample."""
    
    predict_mask = torch.roll(recursion_mask, -1, dims=1)
    predict_mask[:, -1] = False
    recursion_logits = logits[predict_mask]
    
    abstract_start = vocab_sizes[0]
    recursion_logits[:, :abstract_start + 1] = float('-inf')
    
    if temperature == 0.0:
        new_tokens = torch.argmax(recursion_logits, dim=-1)
    else:
        probs = F.softmax(recursion_logits / temperature, dim=-1)
        new_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

    idx[recursion_mask] = new_tokens.to(idx.dtype)
    return idx
    
def recursion(model, idx, max_iterations=5, memory_span=1792, attn_blocksize=1792, temperature=0.0):

    recursion_mask = (idx >= model.vocab_sizes[0])
    recursion_mask[:, 0] = False

    for _ in range(max_iterations): 
        _, logits = model.forward(idx, memory_span, attn_blocksize)
        idx = extract_and_sample(
            logits, idx, recursion_mask, model.vocab_sizes, temperature
        )
    
    # -- evaluation --
    loss, _ = model.forward(idx, memory_span, attn_blocksize)

    return idx, loss

def generate(model, idx, K, max_iterations=0, memory_span=1792, attn_blocksize=1792, temperature=0.0):
    
    # --- prefix recursion ---
    recursion_mask = (idx >= model.vocab_sizes[0])
    recursion_mask[:, 0] = False
    for _ in range(max_iterations): 
        _, logits = model.forward(idx, memory_span, attn_blocksize)
        idx = extract_and_sample(
            logits, idx, recursion_mask, model.vocab_sizes, temperature
        )
    
    # --- rhythmic generation ---
    _, logits = model.forward(idx, memory_span, attn_blocksize)
    
    # --- decide next token level --- 
    next_abstract_mask = (recursion_mask[:, -K:].sum(dim=1) == 0)
    next_token_logits = logits[:, -1]
    next_token_logits[:, :model.vocab_sizes[0]][next_abstract_mask] = float('-inf')  
    next_token_logits[:, model.vocab_sizes[0]:][~next_abstract_mask] = float('-inf')

    if temperature == 0.0:
        new_tokens = torch.argmax(next_token_logits, dim=-1)
    else:
        probs = F.softmax(next_token_logits / temperature, dim=-1)
        new_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

    idx = torch.cat((idx, new_tokens.unsqueeze(1)), dim=1)
    return idx