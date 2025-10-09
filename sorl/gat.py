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


# New version, aligned with GPT architecture, without level-embedding
class GAT(nn.Module): 

    def __init__(self, config):
        super().__init__()

        # U-net design by @brendanh0gan
        self.num_encoder_layers = config.n_layer // 2 # Half of the layers for encoder
        self.num_decoder_layers = config.n_layer - self.num_encoder_layers # Remaining for decoder
        # Add learnable skip connection weights for decoder layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

        self.vocab_sizes = torch.tensor(config.vocab_sizes)
        self.level_mask_tokens = torch.cumsum(self.vocab_sizes, dim=0)
        self.vocab_size = sum(self.vocab_sizes)

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = CastedLinear(config.n_embd, self.vocab_size)
        self.lm_head.weight.data.zero_() # @Grad62304977

        self.device = config.device
        self._compile = config._compile


    def _forward_pass(self, idx: torch.Tensor, memory_span: int):

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


    def forward(self, idx, target, memory_span):
        x = self._forward_pass(idx, memory_span)
        logits = self.lm_head(x)
        logits = 30 * torch.tanh(logits / 30)
        logits = logits.float()
        
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), reduction="none")
        return loss


    def denoise(self, idx: torch.Tensor, denoise_mask: torch.Tensor, memory_span: int):

        x = self._forward_pass(idx, memory_span)
        logits = self.lm_head(x)
        logits = 30 * torch.tanh(logits / 30)
        logits = logits.float()
        
        predict_mask = torch.roll(denoise_mask, 1, dims=1) # prev token embedding predict next token
        predict_mask[:, 0] = False
        
        return logits[predict_mask]



def decode_logits(logits: torch.Tensor, levels: torch.Tensor, 
                  level_mask_tokens: torch.Tensor, temperature: float = 0.0):
    """
    Decode logits constrained to valid vocabulary for each level.
    
    Args:
        logits: [num_tokens, vocab_size]
        levels: [num_tokens] level for each token
        level_mask_tokens: Cumulative vocabulary boundaries
        temperature: Sampling temperature
    """
    # Mask out level boundaries
    logits[:, level_mask_tokens[1:]] = float('-inf')
    
    # Constrain to valid vocab range per level
    start_logits = level_mask_tokens[levels]
    end_logits = level_mask_tokens[levels + 1]
    vocab_indices = torch.arange(logits.size(-1), device="cpu")
    mask = (vocab_indices >= start_logits.unsqueeze(-1)) & (vocab_indices < end_logits.unsqueeze(-1))
    logits = torch.where(mask, logits, torch.tensor(float('-inf'), device="cpu"))
    
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1)
    else:
        return torch.multinomial(F.softmax(logits / temperature, dim=-1), num_samples=1).squeeze(-1)


def parallel_denoise(model, idx, num_iterations=5, memory_span=128, temperature=0.0):
    """Simple denoising loop using the model's denoise method."""
    idx = idx.clone()
    
    for _ in range(num_iterations):
        current_levels = infer_level(idx, model.vocab_sizes)
        denoise_mask = current_levels > 0  # All abstract tokens
        
        if not denoise_mask.any():
            break
        
        logits = model.denoise(idx, denoise_mask, memory_span)
        target_levels = current_levels[denoise_mask]
        new_tokens = decode_logits(logits, target_levels, model.level_mask_tokens, temperature)
        idx[denoise_mask] = new_tokens
    
    return idx


# Generate function can also be external to the trained GAT model
# Key is to test the optimization for GAT for now