# GAT with recursion + adaptive stop

from sorl.model import Block, CastedLinear, create_block_mask
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import torch
import torch.nn.functional as F
from sorl.model import norm
import torch.nn as nn

BOS_TOKEN_ID = 50256

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
    if indices.dtype in [torch.uint8, torch.uint16, torch.uint32, torch.uint64]:
        indices = indices.long()
    
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
        x[abstract_mask] += abstract_repr # recursion (additive)

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

    def _compute_logits(self, x):
        logits = self.lm_head(x)
        logits = 30 * torch.tanh(logits / 30)
        return logits.float()

    def forward(self, idx, abstract_repr, abstract_mask, memory_span):
        """idx is the full token sequence"""

        x = self._forward_pass(idx, abstract_repr, abstract_mask, memory_span) # continuous recursion
        logits = self._compute_logits(x)
        act = compute_act(logits, idx, abstract_mask)

        # --- recursion ----
        abstract_repr = x[abstract_mask] # continuous recursion

        # --- loss --- 
        loss = F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, logits.size(-1)), 
            idx[:, 1:].contiguous().view(-1), 
            reduction="none"
        )
        
        # Don't predict: (1) what comes after BOS, (2) BOS itself
        bos_pos_mask = torch.logical_and(idx[:, :-1] != BOS_TOKEN_ID, idx[:, 1:] != BOS_TOKEN_ID).view(-1).float()        
        loss = loss * bos_pos_mask
        return loss, logits.detach(), abstract_repr.detach(), act


# Generation & Denoising Gadget (Detached from gradient graph)
# ------------------------------------------------------------------------------------------------
def compute_act(logits, idx, abstract_mask, act_threshold: float = 0.9):
        
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

def _continuous_recursion_step(model, idx, abstract_repr, abs_mask, recursion_mask, memory_span):
    """Perform one step of continuous recursion, updating abstract_repr."""
    x = model._forward_pass(idx, abstract_repr, abs_mask, memory_span)
    new_abs_repr = x[abs_mask]
    update_mask = recursion_mask[abs_mask].unsqueeze(-1)
    abstract_repr = torch.where(update_mask, new_abs_repr, abstract_repr)
    return abstract_repr, x

def _discrete_recursion_step(model, idx, logits, recursion_mask, temperature=0.0):
    """Perform one step of discrete recursion, updating idx in-place."""
    levels = infer_level(idx, model.vocab_sizes)
    
    predict_mask = torch.roll(recursion_mask, -1, dims=1)
    predict_mask[:, -1] = False
    recursion_logits = logits[predict_mask]
    
    recursion_levels = levels[recursion_mask]
    logits_mask = get_logits_mask(recursion_levels, model.vocab_sizes)
    recursion_logits = torch.where(logits_mask, recursion_logits, 
                                    torch.tensor(float('-inf'), device=model.device))
    
    if temperature == 0.0:
        new_tokens = torch.argmax(recursion_logits, dim=-1)
    else:
        new_tokens = torch.multinomial(F.softmax(recursion_logits / temperature, dim=-1), 
                                        num_samples=1).squeeze(-1)
    
    idx[recursion_mask] = new_tokens
    return idx

# Recursion with per-iteration loss & detaching
# ------------------------------------------------------------------------------------------------
def recursion(model, idx, max_iterations=5, n_continuous=1, do_discrete=True, memory_span=1024, temperature=0.0):

    idx = idx.clone() 

    levels = infer_level(idx, model.vocab_sizes)
    abs_mask = levels > 0 
    abs_mask[:, 0] = False # don't need to evict tokens during generation, we have sparse attention
    recursion_mask = abs_mask.clone()

    abstract_repr = torch.zeros(abs_mask.sum(), model.n_embd, device=idx.device)

    losses = [] 
    for iteration in range(max_iterations): 

        # --- recursion (no grad) --- 
        with torch.no_grad(): 
            for _ in range(n_continuous): 
                abstract_repr, x = _continuous_recursion_step(
                    model, idx, abstract_repr, abs_mask, recursion_mask, memory_span
                )

            x = model._forward_pass(idx, abstract_repr, abs_mask, memory_span)
            logits = model._compute_logits(x)   
            if do_discrete:
                idx = _discrete_recursion_step(model, idx, logits, recursion_mask, temperature)
            
        # --- loss (with grad) ---
        loss, logits, _, act = model.forward(idx, abstract_repr, abs_mask, memory_span)
        losses.append(loss)

        # --- ACT early stop --- 
        recursion_mask = recursion_mask & ~act.unsqueeze(1)
        if not recursion_mask.any():
            break

    total_loss = sum(losses) / len(losses)

    return idx, total_loss

@torch.no_grad()
def rythmic_decode(logits, model, K, temperature=0.0): 

    nt_logits = logits[:, -1]
    level = get_next_token_level(logits.shape[1], K)
    logit_mask = get_logits_mask(level, model.vocab_sizes)

    nt_logits = torch.where(
        logit_mask.expand_as(nt_logits), 
        nt_logits, 
        torch.tensor(float('-inf'), device=nt_logits.device)
    )

    if temperature == 0.0: 
        next_token_id = nt_logits.argmax(dim=-1, keepdim=True)
    else: 
        probs = F.softmax(nt_logits / temperature, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)

    return next_token_id

@torch.no_grad()
def search(model, idx, max_iterations, n_continuous, memory_span, temperature, K):
    """Recursion without loss computation"""
    idx = idx.clone() 
        
    levels = infer_level(idx, model.vocab_sizes)
    abs_mask = levels > 0 
    abs_mask[:, 0] = False
    recursion_mask = abs_mask.clone()

    abstract_repr = torch.zeros(abs_mask.sum(), model.n_embd, device=idx.device)
    for iteration in range(max_iterations): 
        for _ in range(n_continuous): 
            abstract_repr, x = _continuous_recursion_step(
                model, idx, abstract_repr, abs_mask, recursion_mask, memory_span
            )

        x = model._forward_pass(idx, abstract_repr, abs_mask, memory_span)
        logits = model._compute_logits(x)   
        idx = _discrete_recursion_step(model, idx, logits, recursion_mask, temperature)
        
        act = compute_act(logits, idx, abs_mask)

        recursion_mask = recursion_mask & ~act.unsqueeze(1)
        if not recursion_mask.any():
            break
    
    return idx, logits

@torch.no_grad()
def generate(model, idx, max_new_tokens=50, 
             max_iterations=5, n_continuous=0, memory_span=1024, temperature=0.0, K=4):
  
    idx = idx.clone()

    for _ in range(max_new_tokens): 
        # --- search --- 
        idx, logits = search(model, idx, max_iterations, n_continuous, memory_span, temperature, K)

        # --- decode --- 
        next_token_id = rythmic_decode(logits, model, K, temperature)
        idx = torch.cat((idx, next_token_id), dim=1)

    return idx