# Representation Rank Regularization
# -------------------------------------------------------------------
import torch
torch.set_float32_matmul_precision('high')
from .rank_regularizer import patch_mbe2 as patch_mbe
RANK_REG_LOSS = "regularized_rank"

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

# GPT without encode & decode structure enables compiled regularization without 
# strange spikes in overhead (likely due to simplified computation graph)
# --------------------------------------------
class GPTnoconn(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.num_layers = config.n_layer
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = CastedLinear(config.n_embd, config.vocab_size)
        self.lm_head.weight.data.zero_()

        self.alpha = config.alpha 
        self.window_size = config.window_size

    def forward(self, idx, target, attn_blocksize, reg_layer_indices):
        """Localized Rank Regularization for Each Block"""

        docs = (idx == 50256).cumsum(1)
        def document_causal_mask(b, h, q_idx, kv_idx):
          causal_mask = q_idx >= kv_idx
          document_mask = docs[b, q_idx] == docs[b, kv_idx]
          window_mask = q_idx - kv_idx < attn_blocksize
          return causal_mask & document_mask & window_mask

        S = idx.shape[1]
        block_mask = create_block_mask(document_causal_mask, None, None, S, S, device="cuda", _compile=True)

        x = self.transformer.wte(idx)
        x = norm(x)
        reg_loss = {}
        
        x0 = x
        v1 = None

        skip_connections = []
        for i in range(self.num_layers): 
            x, v1 = self.transformer.h[i](x, v1, x0, block_mask)
            if i in reg_layer_indices: 
                reg_loss[f"{RANK_REG_LOSS}_layer{i+1}"] = patch_mbe(x)

        x = norm(x)
        logits = self.lm_head(x)
        logits = 30 * torch.tanh(logits / 30) # @Grad62304977
        logits = logits.float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        return {"entropy": loss, "rank_reg": sum(reg_loss.values()) / len(reg_loss) if reg_loss else torch.tensor(0., device=x.device)}
        

# SpeedRun GPT module: Extra skip connection
# ---------------------------------------------------------------------------------------
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

    def forward(self, idx, target, attn_blocksize, reg_layer_indices):
        """Localized Rank Regularization for Each Block"""

        docs = (idx == 50256).cumsum(1)
        def document_causal_mask(b, h, q_idx, kv_idx):
          causal_mask = q_idx >= kv_idx
          document_mask = docs[b, q_idx] == docs[b, kv_idx]
          window_mask = q_idx - kv_idx < attn_blocksize
          return causal_mask & document_mask & window_mask

        S = idx.shape[1]
        block_mask = create_block_mask(document_causal_mask, None, None, S, S, device="cuda", _compile=True)

        x = self.transformer.wte(idx)
        x = norm(x)
        reg_loss = {}
        
        x0 = x
        v1 = None

        skip_connections = []
        for i in range(self.num_encoder_layers):
            x, v1 = self.transformer.h[i](x, v1, x0, block_mask)
            if i in reg_layer_indices: 
                reg_loss[f"{RANK_REG_LOSS}_layer{i+1}"] = patch_mbe(x)
            skip_connections.append(x)
        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            x, v1 = self.transformer.h[self.num_encoder_layers + i](x, v1, x0, block_mask)
            if (self.num_encoder_layers + i) in reg_layer_indices: 
                reg_loss[f"{RANK_REG_LOSS}_layer{self.num_encoder_layers + i+1}"] = patch_mbe(x)
            
        x = norm(x)
        logits = self.lm_head(x)
        logits = 30 * torch.tanh(logits / 30) # @Grad62304977
        logits = logits.float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        
        loss_dict = {"entropy": loss}
        if reg_loss: 
            loss_dict["mbe"] = sum(reg_loss.values()) / len(reg_loss)
        return loss_dict 