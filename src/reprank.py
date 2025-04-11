# Representation Rank Regularization
# -------------------------------------------------------------------
import torch 

def rep_norm(x): 
    """token-level representation rank norm"""
    return (x**2).mean(-1)

# Customized GPT model with low-rank regularization loss 
# -------------------------------------------------------------------
from torch import nn
from typing import Optional 
import torch.nn.functional as F
from dataclasses import dataclass
from .model import CastedLinear, Block, create_block_mask, norm

@dataclass
class GPTConfig:
    vocab_size : int = 50304
    n_layer : int = 12
    n_head : int = 6
    n_embd : int = 768
    flex_kernel_options: Optional[dict] = None
    alpha: float = 0.1 # weight of rep rank regularizaiton
    window_size: int = 64

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.num_encoder_layers = config.n_layer // 2
        self.num_decoder_layers = config.n_layer - self.num_encoder_layers 
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = CastedLinear(config.n_embd, config.vocab_size)
        self.lm_head.weight.data.zero_()

        self.alpha = config.alpha 
        self.window_size = config.window_size

    def forward(self, idx, target, attn_blocksize):

        docs = (idx == 50256).cumsum(1)
        def document_causal_mask(b, h, q_idx, kv_idx):
          causal_mask = q_idx >= kv_idx
          document_mask = docs[b, q_idx] == docs[b, kv_idx]
          window_mask = q_idx - kv_idx < attn_blocksize
          return causal_mask & document_mask & window_mask

        S = idx.shape[1]
        block_mask = create_block_mask(document_causal_mask, None, None, S, S, device="cuda", _compile=True)

        x = self.transformer.wte(idx)
        reg_loss = rep_norm(x)
        x = norm(x)
        
        x0 = x
        v1 = None

        skip_connections = []
        for i in range(self.num_encoder_layers):
            x, v1 = self.transformer.h[i](x, v1, x0, block_mask)
            reg_loss += rep_norm(x)
            skip_connections.append(x)
        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            x, v1 = self.transformer.h[self.num_encoder_layers + i](x, v1, x0, block_mask)
            reg_loss += rep_norm(x)
        x = norm(x)
        logits = self.lm_head(x)
        logits = 30 * torch.tanh(logits / 30) # @Grad62304977
        logits = logits.float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        print(f"Entropy loss: {loss} | Regularization loss: {reg_loss.mean()}")
        return loss + self.alpha * reg_loss.mean()