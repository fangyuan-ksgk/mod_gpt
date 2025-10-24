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

        # ACT should be done by argmax criteria (on trajectory chunk)
        self.n_embd = config.n_embd
      
        self.device = config.device
        self._compile = config._compile


    def _forward_pass(self, idx: torch.Tensor, abstract_repr: torch.Tensor, recursion_mask: torch.Tensor, memory_span: int):

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
        x[recursion_mask] += abstract_repr # recursion

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


    def _compute_act(logits, idx, abstract_mask, act_threshold: float = 0.9):
        
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)

            pred_matches_target = (predictions[:, :-1] == idx[:, 1:])
            trajectory_mask_shifted = ~abstract_mask[:, 1:]

            trajectory_correct = (pred_matches_target & trajectory_mask_shifted).float()
            trajectory_total = trajectory_mask_shifted.float()

            # ACT: deterministic, threshold based
            accuracy = trajectory_correct.sum(dim=1) / (trajectory_total.sum(dim=1) + 1e-8)
            has_abstract = (abstract_mask.sum(dim=1) > 0)
            should_stop = (accuracy > act_threshold) | ~has_abstract

        return should_stop


    def forward(self, idx, abstract_repr, abstract_mask, memory_span):
        """idx is the full token sequence"""

        x = self._forward_pass(idx, abstract_repr, abstract_mask, memory_span) # continuous recursion
        logits = self.lm_head(x)
        logits = 30 * torch.tanh(logits / 30)
        logits = logits.float()
        act = self._compute_act(logits, idx, abstract_mask)

        # --- recursion ----
        abstract_repr = x[abstract_mask] # continuous recursion

        # --- loss --- 
        loss = F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, logits.size(-1)), 
            idx[:, 1:].contiguous().view(-1), 
            reduction="none"
        )

        return loss, logits.detach(), abstract_repr.detach(), act


    def denoise(self, idx, abstract_repr, recursion_mask, memory_span): 
 
        x = self._forward_pass(idx, abstract_repr, recursion_mask, memory_span)
        logits = self.lm_head(x)
        logits = 30 * torch.tanh(logits / 30)
        logits = logits.float()
        
        predict_mask = torch.roll(recursion_mask, -1, dims=1) # prev token embedding predict next token
        predict_mask[:, -1] = False
        
        return logits[predict_mask]



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

def parallel_denoise(model, idx, num_iterations=5, memory_span=1024, temperature=0.0):
    """Simple denoising loop using the model's denoise method."""
    # (TBD)
    # - ACT logits determine early stop of recursion
    # - "should_stop" target computed with argmax criteria (on trajectory tokens)
    # - act stop loss should be computed and added to the main loss

    idx = idx.clone()
    levels = infer_level(idx, model.vocab_sizes)
    
    recursion_mask = levels > 0 # to-be-extended (t_search etc.)
    recursion_mask[:, 0] = False # can't denoise first token, no context
    abstract_repr = torch.zeros(recursion_mask.sum(), model.n_embd, device=idx.device)    

    if not abstract_mask.any():
        return idx, abstract_mask, abstract_repr

    abstract_levels = levels[abstract_mask]
    mask = get_logits_mask(abstract_levels, model.vocab_sizes)

    for _ in range(num_iterations):
        logits, abstract_repr, act = model.denoise(idx, abstract_mask, abstract_repr, memory_span)

        logits = torch.where(mask, logits, torch.tensor(float('-inf'), device=model.device)) # process on logits via masking

        if temperature == 0.0:
            new_tokens = torch.argmax(logits, dim=-1)
        else:
            new_tokens = torch.multinomial(F.softmax(logits / temperature, dim=-1), num_samples=1).squeeze(-1)

        idx[abstract_mask] = new_tokens

        if q[0] > q[1]: # ACT stop 
            break
    
    return idx, abstract_mask, abstract_repr, q


# 1-step-backprop training gadget (fixed compute-time ver.)
# -------------------------------------------------------
# def deep_supervision(model, token_ids, num_iterations=5, memory_span=1024, temperature=0.0): 
#     with torch.no_grad(): 



def forward_with_recursion(model, token_ids, num_iterations=5, memory_span=1024, temperature=0.0): 
    with torch.no_grad(): 
        token_ids, abstract_mask, abstract_repr, q = parallel_denoise(model, token_ids, num_iterations, memory_span, temperature)
    
    idx, target =  token_ids[..., :-1].contiguous(), token_ids[..., 1:].contiguous()
    ppt = model.forward(idx, target, abstract_repr, abstract_mask[:, :-1], memory_span)

    return ppt