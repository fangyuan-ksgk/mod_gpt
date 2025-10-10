# GAT with recursion + adaptive stop

from sorl.model import Block, CastedLinear, create_block_mask
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import torch
import torch.nn.functional as F
from sorl.model import norm
import torch.nn as nn

BOS_TOKEN_ID = 50526

@dataclass
class GATConfig:
    vocab_sizes : list
    n_layer : int = 12
    n_head : int = 6
    n_embd : int = 768
    flex_kernel_options: Optional[dict] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    _compile: bool = True if device == "cuda" else False


def infer_level(indices: torch.Tensor, vocab_sizes: torch.Tensor):
    vocab_sizes = vocab_sizes.to(indices.device)
    indices_expanded = indices.unsqueeze(-1)
    levels = (indices_expanded < vocab_sizes.cumsum(dim=0)).int().argmax(dim=-1)
    return levels

get_level_mask_tokens = lambda vocab_sizes: torch.cat(
    (
        torch.tensor([0]),
        torch.cumsum(vocab_sizes, dim=0)
    )
)


# TBD. Include 'recursion' into GAT 
# - (1). add prev representation into abstract embedding
# - (2). optionally update abstract token index (or only update its embedding)
# - (3). KV-cache eviction (secondary to pre-train experiment etc.)

class GAT(nn.Module): 

    def __init__(self, config):
        super().__init__()

        # U-net design by @brendanh0gan
        self.num_encoder_layers = config.n_layer // 2 # Half of the layers for encoder
        self.num_decoder_layers = config.n_layer - self.num_encoder_layers # Remaining for decoder
        # Add learnable skip connection weights for decoder layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

        self.vocab_sizes = torch.tensor(config.vocab_sizes)
        self.vocab_size = sum(self.vocab_sizes)

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = CastedLinear(config.n_embd, self.vocab_size)
        self.lm_head.weight.data.zero_() # @Grad62304977

        self.device = config.device
        self._compile = config._compile


    def _forward_pass(self, idx: torch.Tensor, abstract_repr: torch.Tensor, abstract_mask: torch.Tensor, memory_span: int):

        docs = (idx == BOS_TOKEN_ID).cumsum(1)
        
        levels = infer_level(idx, self.vocab_sizes)

        def causal_mask(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[b, q_idx] == docs[b, kv_idx]
            is_higher_level = levels[b, kv_idx] > 0
            is_recent = (q_idx - kv_idx) <= memory_span
            memory_compression_mask = is_higher_level | is_recent 
            return causal_mask & document_mask & memory_compression_mask

        S = idx.shape[1]
        block_mask = create_block_mask(causal_mask, None, None, S, S, device=self.device, _compile=self._compile)

        x = self.transformer.wte(idx)
        x[abstract_mask] += abstract_repr # recursion
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

    def forward(self, idx, target, abstract_repr, abstract_mask, memory_span):

        x = self._forward_pass(idx, abstract_repr, abstract_mask, memory_span)
        logits = self.lm_head(x)
        logits = 30 * torch.tanh(logits / 30)
        logits = logits.float()
        
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), reduction="none")
        return loss

    def denoise(self, idx: torch.Tensor, abstract_mask: torch.Tensor, abstract_repr: torch.Tensor, memory_span: int):

        x = self._forward_pass(idx, abstract_repr, abstract_mask, memory_span)
        logits = self.lm_head(x)
        logits = 30 * torch.tanh(logits / 30)
        logits = logits.float()
        
        predict_mask = torch.roll(abstract_mask, -1, dims=1) # prev token embedding predict next token
        predict_mask[:, -1] = False

        abstract_repr = x[abstract_mask]
        
        return logits[predict_mask], abstract_repr


# Generation & Denoising Gadget (Detached from gradient graph)
# ------------------------------------------------------------------------------------------------

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

def parallel_denoise(model, idx, num_iterations=5, memory_span=128, temperature=0.0):
    """Simple denoising loop using the model's denoise method."""
    idx = idx.clone()
    levels = infer_level(idx, model.vocab_sizes)
    
    abstract_mask = levels > 0 # to-be-extended (t_search etc.)
    abstract_mask[:, 0] = False # can't denoise first token, no context
    if not abstract_mask.any():
        return idx

    abstract_levels = levels[abstract_mask]
    mask = get_logits_mask(abstract_levels, model.vocab_sizes)
    # (To be Checked) - use 'level-mask-token' wte as the initial abstract representation
    abstract_repr = model.transformer.wte(idx[abstract_mask]) # or just use zero-out ver.

    for _ in range(num_iterations-1):
        logits, abstract_repr = model.denoise(idx, abstract_mask, abstract_repr, memory_span)
        logits = torch.where(mask, logits, torch.tensor(float('-inf'), device="cpu")) # process on logits via masking

        if temperature == 0.0:
            new_tokens = torch.argmax(logits, dim=-1)
        else:
            new_tokens = torch.multinomial(F.softmax(logits / temperature, dim=-1), num_samples=1).squeeze(-1)

        idx[abstract_mask] = new_tokens
    
    return idx


# 1-step-backprop training gadget
# -------------------------------------------------------
# def forward_with_recursion
# with torch.no_grad(): tokens = parallel_denoise(...)
# idx, target =  tokens[..., :-1], tokens[..., 1:]
# model.forward(idx, target)