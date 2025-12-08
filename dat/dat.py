
from sorl.model import Block, CastedLinear, create_block_mask
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union
import torch
import torch.nn.functional as F
from sorl.model import norm
import torch.nn as nn
from sorl.model import GPT


# Decision Transformer (DiT)

@dataclass
class DiTConfig:
    state_dim : int
    action_dim : int
    n_layer : int = 12
    n_head : int = 6
    n_embd : int = 768
    flex_kernel_options: Optional[dict] = None
    device: str = "cuda"
    _compile: bool = True


class DiT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.num_layers = config.n_layer
        self.transformer = nn.ModuleDict(dict(
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.state_encoder = CastedLinear(config.state_dim, config.n_embd)
        self.action_encoder = CastedLinear(config.action_dim, config.n_embd)
        self.state_decoder = CastedLinear(config.n_embd, config.state_dim)
        self.action_decoder = CastedLinear(config.n_embd, config.action_dim)
        self.device = config.device
        self._compile = config._compile

    def encode(self, states, actions): 
        s, a = self.state_encoder(states), self.action_encoder(actions)
        x = torch.zeros(s.size(0), s.size(1) + a.size(1), s.size(2), device=s.device, dtype=s.dtype)
        x[:, 0::2], x[:, 1::2] = s, a
        return x

    def decode(self, x):
        return self.state_decoder(x[:, 1::2]), self.action_decoder(x[:, 0::2])

    def _forward_pass(self, states, actions):

        x = self.encode(states, actions)

        def causal_mask(b, h, q_idx, kv_idx):
          causal_mask = q_idx >= kv_idx
          return causal_mask

        S = x.shape[1]
        block_mask = create_block_mask(causal_mask, None, None, S, S, device=self.device, _compile=self._compile)

        x = norm(x)
        x0 = x
        v1 = None

        for i in range(self.num_layers):
            x, v1 = self.transformer.h[i](x, v1, x0, block_mask)

        x = norm(x)
        pred_states, pred_actions = self.decode(x)
        return pred_states, pred_actions

    def forward(self, states, actions): 
        
        pred_states, pred_actions = self._forward_pass(states, actions)

        # obs loss
        obs_loss = F.mse_loss(pred_states[:, :(states.shape[1]-1)], states[:, 1:])

        # acts loss 
        act_loss = F.cross_entropy(
            pred_actions[:, :actions.shape[1]].reshape(-1, actions.shape[-1]),
            actions.reshape(-1, actions.shape[-1])
        )

        return obs_loss, act_loss


    # world modeling loss
    # policy modeling loss (to be scaled by reward)
    # both are just cross-entropy loss right? 



# BOS_TOKEN_ID = 50256
# # BOS_TOKEN_ID = 20 # for arithmetic dataset

# @dataclass
# class DATConfig:
#     vocab_sizes : list
#     n_layer : int = 12
#     n_head : int = 6
#     n_embd : int = 768
#     flex_kernel_options: Optional[dict] = None
#     device: str = "cuda" if torch.cuda.is_available() else "cpu"
#     _compile: bool = True if device == "cuda" else False

# get_level_mask_tokens = lambda vocab_sizes: torch.cat(
#     (
#         torch.tensor([0]),
#         torch.cumsum(vocab_sizes, dim=0)
#     )
# )

# class GAT(nn.Module): 

#     def __init__(self, config):
#         super().__init__()

#         # U-net design by @brendanh0gan
#         self.num_encoder_layers = config.n_layer // 2 # Half of the layers for encoder
#         self.num_decoder_layers = config.n_layer - self.num_encoder_layers # Remaining for decoder
#         # Add learnable skip connection weights for decoder layers
#         self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

#         self.register_buffer('vocab_sizes', torch.tensor(config.vocab_sizes))
#         self.vocab_size = sum(self.vocab_sizes)

        
#         level_starts = torch.cat([torch.tensor([0]), 
#                                   torch.cumsum(self.vocab_sizes, dim=0)[:-1] + 1])
#         level_ends = torch.cumsum(self.vocab_sizes, dim=0)
        
#         self.register_buffer('level_starts', level_starts)
#         self.register_buffer('level_ends', level_ends)
        

#         self.transformer = nn.ModuleDict(dict(
#             wte = nn.Embedding(self.vocab_size, config.n_embd),
#             h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
#         ))
#         self.lm_head = CastedLinear(config.n_embd, self.vocab_size)
#         self.lm_head.weight.data.zero_() # @Grad62304977

#         # ACT should be done by argmax criteria (on trajectory chunk)
#         self.n_embd = config.n_embd
      
#         self.device = config.device
#         self._compile = config._compile


#     def _forward_pass(self, idx: torch.Tensor, memory_span: int, attn_blocksize: int):

#         docs = (idx == BOS_TOKEN_ID).cumsum(1)
        
#         levels = (idx >= self.vocab_sizes[0]).long()

#         def causal_mask(b, h, q_idx, kv_idx):
#             causal_mask = q_idx >= kv_idx
#             document_mask = docs[b, q_idx] == docs[b, kv_idx]
#             window_mask = q_idx - kv_idx < attn_blocksize
#             is_higher_level = levels[b, kv_idx] > 0
#             is_recent = (q_idx - kv_idx) <= memory_span
#             memory_compression_mask = is_higher_level | is_recent
#             return causal_mask & document_mask & window_mask & memory_compression_mask

#         S = idx.shape[1]
#         block_mask = create_block_mask(causal_mask, None, None, S, S, device=self.device, _compile=self._compile)

#         x = self.transformer.wte(idx)

#         x = norm(x)
#         x0 = x
#         v1 = None

#         skip_connections = []
#         for i in range(self.num_encoder_layers):
#             x, v1 = self.transformer.h[i](x, v1, x0, block_mask)
#             skip_connections.append(x)
        
#         for i in range(self.num_decoder_layers):
#             skip_idx = self.num_encoder_layers - 1 - i
#             x = x + self.skip_weights[i] * skip_connections[skip_idx]
#             x, v1 = self.transformer.h[self.num_encoder_layers + i](x, v1, x0, block_mask)

#         x = norm(x)
#         return x

#     def forward(self, idx, memory_span, attn_blocksize):

#         x = self._forward_pass(idx, memory_span, attn_blocksize)
#         logits = self.lm_head(x)
#         logits = 30 * torch.tanh(logits / 30)
#         logits = logits.float()

#         # --- loss --- 
#         loss = F.cross_entropy(
#             logits[:, :-1].contiguous().view(-1, logits.size(-1)), 
#             idx[:, 1:].contiguous().view(-1).long(), 
#             reduction="none"
#         )
        
#         # Don't predict: (1) what comes after BOS, (2) BOS itself
#         bos_pos_mask = torch.logical_and(idx[:, :-1] != BOS_TOKEN_ID, idx[:, 1:] != BOS_TOKEN_ID).view(-1).float()        
#         loss = loss * bos_pos_mask
        
#         return loss, logits