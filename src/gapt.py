# Gated Phase Transition & GPT with MBE regularization
# -------------------------------------------------------------------
import torch
torch.set_float32_matmul_precision('high')
from src.mbe import (
    patch_mbe, patch_mbe_variance, patch_mbe_range,
    patch_mbe_log_barrier, patch_mbe_softmin, patch_mbe_floor,
    patch_mbe_softmin_compress
)
RANK_REG_LOSS = "mbe"

# Customized GPT model with low-rank regularization loss 
# -------------------------------------------------------------------
from torch import nn
from typing import Optional 
import torch.nn.functional as F
from dataclasses import dataclass, field
from .model import CastedLinear, Block, create_block_mask, norm
import time
from .eaft import _eaft_cross_entropy

@dataclass
class GPTConfig:
    vocab_size : int = 50304
    n_layer : int = 12
    n_head : int = 6
    n_embd : int = 768
    flex_kernel_options: Optional[dict] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    _compile: bool = True if device == "cuda" else False
    reg_mode: str = "mbe"

    @classmethod
    def prior(cls, name: str, vocab_size: int = 50304, flex_kernel_options: Optional[dict] = None, reg_mode: str = "mbe"):
        if name == "small": 
            return cls(n_layer=12, n_head=6, n_embd=768, vocab_size=vocab_size, flex_kernel_options=flex_kernel_options, reg_mode=reg_mode)
        elif name == "medium":
            return cls(n_layer=24, n_head=16, n_embd=1024, vocab_size=vocab_size, flex_kernel_options=flex_kernel_options, reg_mode=reg_mode)
        elif name == "large":
            return cls(n_layer=36, n_head=20, n_embd=1280, vocab_size=vocab_size, flex_kernel_options=flex_kernel_options, reg_mode=reg_mode)
        elif name == "xl":
            return cls(n_layer=48, n_head=25, n_embd=1600, vocab_size=vocab_size, flex_kernel_options=flex_kernel_options, reg_mode=reg_mode)
        else:
            raise ValueError(f"Invalid GPT size: {name}")


# GPT with MBE regularization
# ---------------------------------------------------------------------------------------
# class GPT(nn.Module):

#     def __init__(self, config):
#         super().__init__()

#         self.num_encoder_layers = config.n_layer // 2
#         self.num_decoder_layers = config.n_layer - self.num_encoder_layers 
#         self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

#         self.transformer = nn.ModuleDict(dict(
#             wte = nn.Embedding(config.vocab_size, config.n_embd),
#             h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
#         ))
#         self.lm_head = CastedLinear(config.n_embd, config.vocab_size)
#         self.lm_head.weight.data.zero_()

#         self.device = config.device
#         self._compile = config._compile

#     def forward(self, idx, target, attn_blocksize, patch_size):
#         """Localized Rank Regularization for Each Block"""

#         docs = (idx == 50256).cumsum(1)
#         def document_causal_mask(b, h, q_idx, kv_idx):
#           causal_mask = q_idx >= kv_idx
#           document_mask = docs[b, q_idx] == docs[b, kv_idx]
#           window_mask = q_idx - kv_idx < attn_blocksize
#           return causal_mask & document_mask & window_mask

#         S = idx.shape[1]
#         block_mask = create_block_mask(document_causal_mask, None, None, S, S, device=self.device, _compile=self._compile)

#         x = self.transformer.wte(idx)
#         x = norm(x)
#         loss_dict = {}
        
#         x0 = x
#         v1 = None

#         skip_connections = []
#         for i in range(self.num_encoder_layers):
#             x, v1 = self.transformer.h[i](x, v1, x0, block_mask)
#             loss_dict[f"{RANK_REG_LOSS}_{i}"] = patch_mbe(x, patch_size)
#             skip_connections.append(x)
#         for i in range(self.num_decoder_layers):
#             x = x + self.skip_weights[i] * skip_connections.pop()
#             x, v1 = self.transformer.h[self.num_encoder_layers + i](x, v1, x0, block_mask)
#             loss_dict[f"{RANK_REG_LOSS}_{self.num_encoder_layers + i}"] = patch_mbe(x, patch_size)
            
#         x = norm(x)
#         logits = self.lm_head(x)
#         logits = 30 * torch.tanh(logits / 30) # @Grad62304977
#         logits = logits.float()
#         loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
#         loss_dict["entropy"] = loss
#         return loss_dict



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

        self.device = config.device
        self._compile = config._compile
        self.enable_timing = False  # Toggle for timing
        self.reg_mode = config.reg_mode
        if self.reg_mode == "mbe":
            self.reg_func = patch_mbe
        elif self.reg_mode == "mbe_variance":
            self.reg_func = patch_mbe_variance
        elif self.reg_mode == "mbe_range":
            self.reg_func = patch_mbe_range
        elif self.reg_mode == "mbe_log_barrier":
            self.reg_func = patch_mbe_log_barrier
        elif self.reg_mode == "mbe_softmin":
            self.reg_func = patch_mbe_softmin
        elif self.reg_mode == "mbe_floor":
            self.reg_func = patch_mbe_floor
        elif self.reg_mode == "mbe_softmin_compress":
            self.reg_func = patch_mbe_softmin_compress
        else:
            self.reg_func = patch_mbe  # default

    def forward(self, idx, target, attn_blocksize, patch_size, use_eaft: bool = False):
        """Localized Rank Regularization for Each Block"""
        timings = {} if self.enable_timing else None
        
        # ===== Block Mask Creation =====
        if self.enable_timing:
            t0 = time.perf_counter()
        
        docs = (idx == 50256).cumsum(1)
        def document_causal_mask(b, h, q_idx, kv_idx):
          causal_mask = q_idx >= kv_idx
          document_mask = docs[b, q_idx] == docs[b, kv_idx]
          window_mask = q_idx - kv_idx < attn_blocksize
          return causal_mask & document_mask & window_mask

        S = idx.shape[1]
        block_mask = create_block_mask(document_causal_mask, None, None, S, S, device=self.device, _compile=self._compile)
        
        if self.enable_timing:
            timings['block_mask_creation'] = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()

        # ===== Embedding + Norm =====
        x = self.transformer.wte(idx)
        x = norm(x)
        loss_dict = {}
        
        if self.enable_timing:
            timings['embedding_norm'] = (time.perf_counter() - t0) * 1000
        
        x0 = x
        v1 = None

        # ===== Encoder Layers =====
        skip_connections = []
        for i in range(self.num_encoder_layers):
            if self.enable_timing:
                t0 = time.perf_counter()
            
            x, v1 = self.transformer.h[i](x, v1, x0, block_mask)
            
            if self.enable_timing:
                timings[f'encoder_layer_{i}_forward'] = (time.perf_counter() - t0) * 1000
                t0 = time.perf_counter()
            
            loss_dict[f"{RANK_REG_LOSS}_{i}"] = self.reg_func(x, patch_size)
            
            if self.enable_timing:
                timings[f'encoder_layer_{i}_mbe'] = (time.perf_counter() - t0) * 1000
            
            skip_connections.append(x)
        
        # ===== Decoder Layers =====
        for i in range(self.num_decoder_layers):
            if self.enable_timing:
                t0 = time.perf_counter()
            
            x = x + self.skip_weights[i] * skip_connections.pop() # break skip connections
            x, v1 = self.transformer.h[self.num_encoder_layers + i](x, v1, x0, block_mask)
            
            if self.enable_timing:
                timings[f'decoder_layer_{i}_forward'] = (time.perf_counter() - t0) * 1000
                t0 = time.perf_counter()
            
            loss_dict[f"{RANK_REG_LOSS}_{self.num_encoder_layers + i}"] = self.reg_func(x, patch_size)
            
            if self.enable_timing:
                timings[f'decoder_layer_{i}_mbe'] = (time.perf_counter() - t0) * 1000
        
        # ===== Output Head + Loss =====
        if self.enable_timing:
            t0 = time.perf_counter()
        
        x = norm(x)
        logits = self.lm_head(x)
        logits = 30 * torch.tanh(logits / 30)
        logits = logits.float()

        # loss_dict["logits"] = logits # ToBeRemoved | logging purpose only
        if use_eaft:
            loss = _eaft_cross_entropy(logits, target)
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        loss_dict["entropy"] = loss
        
        if self.enable_timing:
            timings['output_head_loss'] = (time.perf_counter() - t0) * 1000
            loss_dict["_timings"] = timings
            self._print_timing_summary(timings)
        
        return loss_dict
    
    def _print_timing_summary(self, timings):
        """Print a formatted timing summary"""
        print("\n" + "="*60)
        print("⏱️  Forward Pass Timing Breakdown (ms)")
        print("="*60)
        
        # Group timings
        setup_time = timings.get('block_mask_creation', 0) + timings.get('embedding_norm', 0)
        encoder_forward = sum(v for k, v in timings.items() if 'encoder' in k and 'forward' in k)
        encoder_mbe = sum(v for k, v in timings.items() if 'encoder' in k and 'mbe' in k)
        decoder_forward = sum(v for k, v in timings.items() if 'decoder' in k and 'forward' in k)
        decoder_mbe = sum(v for k, v in timings.items() if 'decoder' in k and 'mbe' in k)
        output_time = timings.get('output_head_loss', 0)
        
        total_time = setup_time + encoder_forward + encoder_mbe + decoder_forward + decoder_mbe + output_time
        
        print(f"Setup (mask + embed):     {setup_time:8.2f} ms  ({setup_time/total_time*100:5.1f}%)")
        print(f"Encoder Forward:          {encoder_forward:8.2f} ms  ({encoder_forward/total_time*100:5.1f}%)")
        print(f"Encoder MBE:              {encoder_mbe:8.2f} ms  ({encoder_mbe/total_time*100:5.1f}%)")
        print(f"Decoder Forward:          {decoder_forward:8.2f} ms  ({decoder_forward/total_time*100:5.1f}%)")
        print(f"Decoder MBE:              {decoder_mbe:8.2f} ms  ({decoder_mbe/total_time*100:5.1f}%)")
        print(f"Output (head + loss):     {output_time:8.2f} ms  ({output_time/total_time*100:5.1f}%)")
        print("-" * 60)
        print(f"TOTAL:                    {total_time:8.2f} ms")
        print("=" * 60 + "\n")


class GPT_log(nn.Module):

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

        self.device = config.device
        self._compile = config._compile
        self.enable_timing = False  # Toggle for timing
        
        # MBE regularization mode
        self.reg_mode = config.reg_mode
        if self.reg_mode == "mbe":
            self.reg_func = patch_mbe
        elif self.reg_mode == "mbe_variance":
            self.reg_func = patch_mbe_variance
        elif self.reg_mode == "mbe_range":
            self.reg_func = patch_mbe_range
        elif self.reg_mode == "mbe_log_barrier":
            self.reg_func = patch_mbe_log_barrier
        elif self.reg_mode == "mbe_softmin":
            self.reg_func = patch_mbe_softmin
        elif self.reg_mode == "mbe_floor":
            self.reg_func = patch_mbe_floor
        elif self.reg_mode == "mbe_softmin_compress":
            self.reg_func = patch_mbe_softmin_compress
        else:
            self.reg_func = patch_mbe  # default

    def forward(self, idx, target, attn_blocksize, patch_size, use_eaft: bool = False):
        """Localized Rank Regularization for Each Block"""
        timings = {} if self.enable_timing else None
        
        # ===== Block Mask Creation =====
        if self.enable_timing:
            t0 = time.perf_counter()
        
        docs = (idx == 50256).cumsum(1)
        def document_causal_mask(b, h, q_idx, kv_idx):
          causal_mask = q_idx >= kv_idx
          document_mask = docs[b, q_idx] == docs[b, kv_idx]
          window_mask = q_idx - kv_idx < attn_blocksize
          return causal_mask & document_mask & window_mask

        S = idx.shape[1]
        block_mask = create_block_mask(document_causal_mask, None, None, S, S, device=self.device, _compile=self._compile)
        
        if self.enable_timing:
            timings['block_mask_creation'] = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()

        # ===== Embedding + Norm =====
        x = self.transformer.wte(idx)
        x = norm(x)
        loss_dict = {}
        
        if self.enable_timing:
            timings['embedding_norm'] = (time.perf_counter() - t0) * 1000
        
        x0 = x
        v1 = None

        # ===== Encoder Layers =====
        skip_connections = []
        for i in range(self.num_encoder_layers):
            if self.enable_timing:
                t0 = time.perf_counter()
            
            x, v1 = self.transformer.h[i](x, v1, x0, block_mask)
            
            if self.enable_timing:
                timings[f'encoder_layer_{i}_forward'] = (time.perf_counter() - t0) * 1000
                t0 = time.perf_counter()
            
            loss_dict[f"{RANK_REG_LOSS}_{i}"] = self.reg_func(x, patch_size)
            
            if self.enable_timing:
                timings[f'encoder_layer_{i}_mbe'] = (time.perf_counter() - t0) * 1000
            
            skip_connections.append(x)
        
        # ===== Decoder Layers =====
        for i in range(self.num_decoder_layers):
            if self.enable_timing:
                t0 = time.perf_counter()
            
            x = x + self.skip_weights[i] * skip_connections.pop() # break skip connections
            x, v1 = self.transformer.h[self.num_encoder_layers + i](x, v1, x0, block_mask)
            
            if self.enable_timing:
                timings[f'decoder_layer_{i}_forward'] = (time.perf_counter() - t0) * 1000
                t0 = time.perf_counter()
            
            loss_dict[f"{RANK_REG_LOSS}_{self.num_encoder_layers + i}"] = self.reg_func(x, patch_size)
            
            if self.enable_timing:
                timings[f'decoder_layer_{i}_mbe'] = (time.perf_counter() - t0) * 1000
        
        # ===== Output Head + Loss =====
        if self.enable_timing:
            t0 = time.perf_counter()
        
        x = norm(x)
        logits = self.lm_head(x)
        logits = 30 * torch.tanh(logits / 30)
        logits = logits.float()

        loss_dict["logits"] = logits # ToBeRemoved | logging purpose only
        if use_eaft:
            loss = _eaft_cross_entropy(logits, target)
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        loss_dict["entropy"] = loss
        
        if self.enable_timing:
            timings['output_head_loss'] = (time.perf_counter() - t0) * 1000
            loss_dict["_timings"] = timings
            self._print_timing_summary(timings)
        
        return loss_dict
    
    def _print_timing_summary(self, timings):
        """Print a formatted timing summary"""
        print("\n" + "="*60)
        print("⏱️  Forward Pass Timing Breakdown (ms)")
        print("="*60)
        
        # Group timings
        setup_time = timings.get('block_mask_creation', 0) + timings.get('embedding_norm', 0)
        encoder_forward = sum(v for k, v in timings.items() if 'encoder' in k and 'forward' in k)
        encoder_mbe = sum(v for k, v in timings.items() if 'encoder' in k and 'mbe' in k)
        decoder_forward = sum(v for k, v in timings.items() if 'decoder' in k and 'forward' in k)
        decoder_mbe = sum(v for k, v in timings.items() if 'decoder' in k and 'mbe' in k)
        output_time = timings.get('output_head_loss', 0)
        
        total_time = setup_time + encoder_forward + encoder_mbe + decoder_forward + decoder_mbe + output_time
        
        print(f"Setup (mask + embed):     {setup_time:8.2f} ms  ({setup_time/total_time*100:5.1f}%)")
        print(f"Encoder Forward:          {encoder_forward:8.2f} ms  ({encoder_forward/total_time*100:5.1f}%)")
        print(f"Encoder MBE:              {encoder_mbe:8.2f} ms  ({encoder_mbe/total_time*100:5.1f}%)")
        print(f"Decoder Forward:          {decoder_forward:8.2f} ms  ({decoder_forward/total_time*100:5.1f}%)")
        print(f"Decoder MBE:              {decoder_mbe:8.2f} ms  ({decoder_mbe/total_time*100:5.1f}%)")
        print(f"Output (head + loss):     {output_time:8.2f} ms  ({output_time/total_time*100:5.1f}%)")
        print("-" * 60)
        print(f"TOTAL:                    {total_time:8.2f} ms")
        print("=" * 60 + "\n")

    def forward_with_patch_stats(self, idx, target, attn_blocksize, patch_size):
        """
        Forward pass that collects per-patch statistics for detailed MBE analysis.
        
        Returns:
            loss_dict: standard losses
            patch_stats: dict with per-patch metrics for each layer
        """
        from src.mbe import patch_mbe_detailed
        
        # Block mask
        docs = (idx == 50256).cumsum(1)
        def document_causal_mask(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[b, q_idx] == docs[b, kv_idx]
            window_mask = q_idx - kv_idx < attn_blocksize
            return causal_mask & document_mask & window_mask

        B, S = idx.shape
        num_patches = S // patch_size
        block_mask = create_block_mask(document_causal_mask, None, None, S, S, 
                                       device=self.device, _compile=self._compile)
        
        x = self.transformer.wte(idx)
        x = norm(x)
        
        loss_dict = {}
        # Per-layer, per-patch MBE: {layer_idx: (B, num_patches)}
        layer_patch_mbe = {}
        
        x0 = x
        v1 = None

        # Encoder layers
        skip_connections = []
        for i in range(self.num_encoder_layers):
            x, v1 = self.transformer.h[i](x, v1, x0, block_mask)
            
            # Per-patch MBE for this layer
            patch_mbe_vals = patch_mbe_detailed(x, patch_size)  # (B, num_patches)
            layer_patch_mbe[i] = patch_mbe_vals.detach()
            loss_dict[f"{RANK_REG_LOSS}_{i}"] = patch_mbe_vals.mean()
            
            skip_connections.append(x)
        
        # Decoder layers
        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            x, v1 = self.transformer.h[self.num_encoder_layers + i](x, v1, x0, block_mask)
            
            layer_idx = self.num_encoder_layers + i
            patch_mbe_vals = patch_mbe_detailed(x, patch_size)
            layer_patch_mbe[layer_idx] = patch_mbe_vals.detach()
            loss_dict[f"{RANK_REG_LOSS}_{layer_idx}"] = patch_mbe_vals.mean()
        
        # Output head
        x = norm(x)
        logits = self.lm_head(x)
        logits = 30 * torch.tanh(logits / 30)
        logits = logits.float()
        
        # Per-token CE loss
        loss_flat = F.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            target.view(-1), 
            reduction='none'
        )
        per_token_loss = loss_flat.view(B, S)
        
        # Per-token probability and entropy
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            correct_probs = probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # (B, S)
            entropy = -(probs * probs.log().clamp(min=-100)).sum(dim=-1)  # (B, S)
        
        # Aggregate to per-patch
        per_patch_loss = per_token_loss.view(B, num_patches, patch_size).mean(dim=-1)
        per_patch_prob = correct_probs.view(B, num_patches, patch_size).mean(dim=-1)
        per_patch_entropy = entropy.view(B, num_patches, patch_size).mean(dim=-1)
        
        # Average MBE across layers for each patch
        stacked_mbe = torch.stack(list(layer_patch_mbe.values()), dim=0)  # (n_layers, B, num_patches)
        avg_patch_mbe = stacked_mbe.mean(dim=0)  # (B, num_patches)
        
        loss_dict["entropy"] = per_token_loss.mean()
        loss_dict["logits"] = logits
        
        patch_stats = {
            'patch_mbe': avg_patch_mbe.detach().cpu(),           # (B, num_patches)
            'patch_loss': per_patch_loss.detach().cpu(),         # (B, num_patches)  
            'patch_prob': per_patch_prob.detach().cpu(),         # (B, num_patches)
            'patch_entropy': per_patch_entropy.detach().cpu(),   # (B, num_patches)
            'layer_patch_mbe': {k: v.cpu() for k, v in layer_patch_mbe.items()},  # per-layer detail
        }
        
        return loss_dict, patch_stats

    def forward_with_patch_stats_and_grads(self, idx, target, attn_blocksize, patch_size):
        """
        Forward + backward pass to collect per-patch stats INCLUDING gradient magnitudes.
        
        This is expensive (requires backward pass) but gives complete data for analysis.
        
        Returns:
            loss_dict: standard losses
            patch_stats: dict with per-patch MBE, loss, prob, entropy, AND gradient magnitude
        """
        from src.mbe import patch_mbe_detailed
        
        B, S = idx.shape
        num_patches = S // patch_size
        
        # Enable gradients for embedding
        embed_weight = self.transformer.wte.weight
        if embed_weight.grad is not None:
            embed_weight.grad.zero_()
        
        # Block mask
        docs = (idx == 50256).cumsum(1)
        def document_causal_mask(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[b, q_idx] == docs[b, kv_idx]
            window_mask = q_idx - kv_idx < attn_blocksize
            return causal_mask & document_mask & window_mask

        block_mask = create_block_mask(document_causal_mask, None, None, S, S, 
                                       device=self.device, _compile=self._compile)
        
        x = self.transformer.wte(idx)
        x = norm(x)
        
        loss_dict = {}
        layer_patch_mbe = {}
        
        x0 = x
        v1 = None

        # Encoder layers
        skip_connections = []
        for i in range(self.num_encoder_layers):
            x, v1 = self.transformer.h[i](x, v1, x0, block_mask)
            patch_mbe_vals = patch_mbe_detailed(x, patch_size)
            layer_patch_mbe[i] = patch_mbe_vals.detach()
            loss_dict[f"{RANK_REG_LOSS}_{i}"] = patch_mbe_vals.mean()
            skip_connections.append(x)
        
        # Decoder layers
        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            x, v1 = self.transformer.h[self.num_encoder_layers + i](x, v1, x0, block_mask)
            layer_idx = self.num_encoder_layers + i
            patch_mbe_vals = patch_mbe_detailed(x, patch_size)
            layer_patch_mbe[layer_idx] = patch_mbe_vals.detach()
            loss_dict[f"{RANK_REG_LOSS}_{layer_idx}"] = patch_mbe_vals.mean()
        
        # Output head
        x = norm(x)
        logits = self.lm_head(x)
        logits = 30 * torch.tanh(logits / 30)
        logits = logits.float()
        
        # Per-token CE loss (keep graph for backward)
        loss_flat = F.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            target.view(-1), 
            reduction='none'
        )
        per_token_loss = loss_flat.view(B, S)
        
        # Backward pass to get gradients
        total_loss = per_token_loss.sum()
        total_loss.backward()
        
        # Get per-token gradient magnitude from embedding
        # Gradient shape: (vocab_size, embed_dim)
        # We need gradient for each token position
        embed_grad = embed_weight.grad  # (vocab_size, embed_dim)
        
        # Per-token gradient: look up gradient for each token ID
        token_grads = embed_grad[idx]  # (B, S, embed_dim)
        per_token_grad_mag = token_grads.norm(dim=-1)  # (B, S)
        
        # Aggregate to per-patch
        per_patch_grad = per_token_grad_mag.view(B, num_patches, patch_size).mean(dim=-1)
        per_patch_loss_val = per_token_loss.view(B, num_patches, patch_size).mean(dim=-1)
        
        # Per-token probability and entropy (no grad needed)
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            correct_probs = probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
            entropy = -(probs * probs.log().clamp(min=-100)).sum(dim=-1)
        
        per_patch_prob = correct_probs.view(B, num_patches, patch_size).mean(dim=-1)
        per_patch_entropy = entropy.view(B, num_patches, patch_size).mean(dim=-1)
        
        # Average MBE across layers
        stacked_mbe = torch.stack(list(layer_patch_mbe.values()), dim=0)
        avg_patch_mbe = stacked_mbe.mean(dim=0)
        
        loss_dict["entropy"] = per_token_loss.mean()
        loss_dict["logits"] = logits
        
        patch_stats = {
            'patch_mbe': avg_patch_mbe.detach().cpu(),
            'patch_loss': per_patch_loss_val.detach().cpu(),
            'patch_prob': per_patch_prob.detach().cpu(),
            'patch_entropy': per_patch_entropy.detach().cpu(),
            'patch_grad': per_patch_grad.detach().cpu(),  # NEW: per-patch gradient magnitude
            'layer_patch_mbe': {k: v.cpu() for k, v in layer_patch_mbe.items()},
        }
        
        # Clean up gradients
        self.zero_grad(set_to_none=True)
        
        return loss_dict, patch_stats


class GPT_pure(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.num_encoder_layers = config.n_layer // 2
        self.num_decoder_layers = config.n_layer - self.num_encoder_layers
        self.num_layers = config.n_layer
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = CastedLinear(config.n_embd, config.vocab_size)
        self.lm_head.weight.data.zero_()

        self.device = config.device
        self._compile = config._compile
        self.enable_timing = False  # Toggle for timing

    def forward(self, idx, target, attn_blocksize, patch_size):
        """Localized Rank Regularization for Each Block"""
        timings = {} if self.enable_timing else None
        
        # ===== Block Mask Creation =====
        if self.enable_timing:
            t0 = time.perf_counter()
        
        docs = (idx == 50256).cumsum(1)
        def document_causal_mask(b, h, q_idx, kv_idx):
          causal_mask = q_idx >= kv_idx
          document_mask = docs[b, q_idx] == docs[b, kv_idx]
          window_mask = q_idx - kv_idx < attn_blocksize
          return causal_mask & document_mask & window_mask

        S = idx.shape[1]
        block_mask = create_block_mask(document_causal_mask, None, None, S, S, device=self.device, _compile=self._compile)
        
        if self.enable_timing:
            timings['block_mask_creation'] = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()

        # ===== Embedding + Norm =====
        x = self.transformer.wte(idx)
        x = norm(x)
        loss_dict = {}
        
        if self.enable_timing:
            timings['embedding_norm'] = (time.perf_counter() - t0) * 1000
        
        x0 = x
        v1 = None

        for i in range(self.num_layers): 
            x, v1 = self.transformer.h[i](x, v1, x0, block_mask)
            loss_dict[f"{RANK_REG_LOSS}_{i}"] = patch_mbe(x, patch_size)
        
        x = norm(x)
        logits = self.lm_head(x)
        logits = 30 * torch.tanh(logits / 30)
        logits = logits.float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        loss_dict["entropy"] = loss
        
        if self.enable_timing:
            timings['output_head_loss'] = (time.perf_counter() - t0) * 1000
            loss_dict["_timings"] = timings
            self._print_timing_summary(timings)
        
        return loss_dict

# Gated Phase Transition 
# ---------------------------------------------------------------------------------------
class GatedPhaseTransition:
    """
    Gated Phase Transition (GAPT) : https://arxiv.org/pdf/2505.08727
    with percentage-based thresholds.
    """
    def __init__(self, tau_plateau_m: float = 0.01, tau_plateau_a: float = 0.01, tau_spike: float = 0.1, 
                 p_m: int = 5, p_a: int = 5, clamp_a: float = 1e-5, use_softplus: bool = False, initial_phase: int = 1, static_phase: bool = False):
        """
        Args:
            tau_plateau: Relative threshold for detecting plateau (e.g., 0.01 = 1% improvement)
            tau_spike: Relative threshold for detecting spike (e.g., 0.1 = 10% degradation)
            p_m: Patience for main objective (steps without improvement)
            p_c: Patience for auxiliary objective (steps without improvement)
            clamp_a: Clamp value for auxiliary loss (to avoid division by zero)
        """
        self.tau_plateau_m = tau_plateau_m  # % improvement needed to avoid plateau
        self.tau_plateau_a = tau_plateau_a # % improvement needed to avoid plateau
        self.tau_spike = tau_spike      # % degradation that triggers phase switch
        self.p_m = p_m
        self.p_a = p_a

        self.phi = initial_phase  # 1 for main phase, 2 for compression phase
        self.static_phase = static_phase
        self.s_m = 0  # steps since improvement in main
        self.s_a = 0  # steps since improvement in auxiliary

        self.min_m = float('inf')
        self.min_a = float('inf')
        self.clamp_a = clamp_a
        self.use_softplus = use_softplus

    def _relative_gain(self, current_loss: torch.Tensor, min_loss: float) -> float:
        """Calculate percentage improvement (negative = degradation)"""
        if min_loss == float('inf') or abs(min_loss) < 1e-9:
            return 0.0
        current_val = current_loss.detach().item() if torch.is_tensor(current_loss) else current_loss
        return (min_loss - current_val) / max(abs(min_loss), 1e-6)

    def _weight_loss(self, main_loss: torch.Tensor, auxiliary_loss: torch.Tensor) -> torch.Tensor:
        """Weight the loss based on the phase"""
        if self.phi == 1:
            return main_loss
        elif self.phi == 2:
            if not self.use_softplus:
                return main_loss + auxiliary_loss.clamp(min=self.clamp_a)
            else:
                softness = 0.1  # controls sharpness
                soft_aux = self.clamp_a + softness * torch.nn.functional.softplus(
                    (auxiliary_loss - self.clamp_a) / softness
                )
                return main_loss + soft_aux
        return main_loss # fallback
    
    def step(self, main_loss: torch.Tensor, auxiliary_loss: torch.Tensor, 
             verbose: bool = False) -> torch.Tensor:
        """
        Update phase based on loss dynamics.
        
        Returns:
             weighted_loss: Weighted loss tensor suitable for .backward()
        """
        main_val = main_loss.detach().item()
        aux_val = auxiliary_loss.detach().item()

        if not (torch.isfinite(main_loss) and torch.isfinite(auxiliary_loss)):
            print(f"WARNING: Non-finite loss detected! main={main_val}, aux={aux_val}")
            return main_loss  # Fallback to main loss only

        gain_m = self._relative_gain(main_loss, self.min_m)
        gain_a = self._relative_gain(auxiliary_loss, self.min_a)
        
        self.min_m = min(self.min_m, main_val)
        self.min_a = min(self.min_a, aux_val)

        prev_phi = self.phi

        if self.phi == 1 and not self.static_phase:  # Main objective phase
            if gain_m > self.tau_plateau_m: 
                self.s_m = 0
            else: 
                self.s_m += 1
            
            if self.s_m >= self.p_m:
                self.s_m = 0
                self.phi = 2

        elif self.phi == 2 and not self.static_phase:  # Main + Auxiliary phase
            if gain_m < -self.tau_spike:  
                if verbose:
                    print(f"  [GAPT] Main loss spiked: {-gain_m*100:.2f}% > {self.tau_spike*100:.2f}%")
                self.s_a = 0
                self.phi = 1
            else:
                if gain_a > self.tau_plateau_a:  
                    self.s_a = 0
                else: 
                    self.s_a += 1
                
                if self.s_a >= self.p_a:
                    if verbose:
                        print(f"  [GAPT] Auxiliary loss plateaued for {self.p_a} steps")
                    self.s_a = 0
                    self.phi = 1
                    
        
        if verbose and prev_phi != self.phi:
            print(f"  [GAPT] Phase transition: {prev_phi} → {self.phi}")
            print(f"         main_loss={main_val:.4f}, aux_loss={aux_val:.4f}")

        return self._weight_loss(main_loss, auxiliary_loss)

    

    # def comp_loss(self, main_loss: torch.Tensor, auxiliary_loss: torch.Tensor) -> torch.Tensor:
        


# ---- mbe layer mask ablation ---- 
import torch

def get_mbe_layer_mask(
    step: int,
    accum_step: int,
    total_num_accum_steps: int,
    n_layer: int,
    mode: str = "rotate",
    skip_first: int = 1,
    skip_last: int = 1,
    device: str = "cuda"
) -> torch.Tensor:
    """    
    mode: masking strategy
        - "all_middle": Regularize all middle layers (skip first/last N)
        - "rotate": Rotate through one middle layer at a time per step
        - "rotate_accum": Rotate through one layer per accumulation step (faster)
        - "progressive": Start from center, expand outward over training
        - "weighted_valley": Valley-shaped weights (high middle, low edges)
        - "weighted_mountain": Mountain-shaped weights (low middle, high edges for compression)
    skip_first: Number of early layers to always skip (default 1)
    skip_last: Number of late layers to always skip (default 1)
    """
    mask = torch.zeros(n_layer, device=device)
    
    start_layer = skip_first
    end_layer = n_layer - skip_last
    n_active = end_layer - start_layer

    # A simpler idea: we'd pick the layer with highest MBE at each step, this bake in 'uniformity' naturally
    
    if mode == "all_middle":
        # Regularize all middle layers equally
        mask[start_layer:end_layer] = 1.0
        
    elif mode == "rotate":
        # Rotate through one layer at a time per training step
        active_idx = start_layer + (step % n_active)
        mask[active_idx] = 1.0
        
    elif mode == "rotate_accum":
        # Rotate through layers per accumulation step (faster cycling)
        # Combine step and accum_step for finer granularity
        combined_step = step * total_num_accum_steps + accum_step  # Assuming typical accum_steps <= 8
        active_idx = start_layer + (combined_step % n_active)
        mask[active_idx] = 1.0
        
    elif mode == "progressive":
        # Start from center, expand outward over training
        # Assumes training progresses from step 0 to max_steps
        center = (start_layer + end_layer) // 2
        # Expand radius based on step (needs max_steps to be meaningful)
        # For now, use step modulo to cycle through expansion
        radius = (step % (n_active // 2 + 1))
        for i in range(center - radius, center + radius + 1):
            if start_layer <= i < end_layer:
                mask[i] = 1.0
                
    elif mode == "weighted_valley":
        # Valley shape: strongest regularization in the middle
        # Use smooth weighting based on distance from center
        center = (start_layer + end_layer) / 2
        for i in range(start_layer, end_layer):
            # Gaussian-like weight centered at middle
            dist_from_center = abs(i - center)
            max_dist = (end_layer - start_layer) / 2
            # Weight peaks at center (1.0), falls off toward edges (0.3)
            mask[i] = 0.3 + 0.7 * (1 - (dist_from_center / max_dist) ** 2)
            
    elif mode == "weighted_mountain":
        # Mountain shape: strongest regularization at edges of active zone
        # Forces bottleneck at layer boundaries
        center = (start_layer + end_layer) / 2
        for i in range(start_layer, end_layer):
            dist_from_center = abs(i - center)
            max_dist = (end_layer - start_layer) / 2
            # Weight low at center (0.3), high at edges (1.0)
            mask[i] = 0.3 + 0.7 * (dist_from_center / max_dist) ** 2
            
    elif mode == "alternating":
        # Alternate between even/odd layers each step
        # Useful for creating compression-expansion cycles
        parity = step % 2
        for i in range(start_layer, end_layer):
            if i % 2 == parity:
                mask[i] = 1.0
                
    elif mode == "block":
        # Divide active layers into blocks, rotate through blocks
        n_blocks = 3  # Early-middle, middle, middle-late
        block_size = n_active // n_blocks
        block_idx = step % n_blocks
        block_start = start_layer + block_idx * block_size
        block_end = block_start + block_size if block_idx < n_blocks - 1 else end_layer
        mask[block_start:block_end] = 1.0

    elif mode == "slope": 
        # Input layers get HIGH MBE weight, output layers get LOW MBE weight
        # Linear decay from 1.0 at start_layer to min_weight at end_layer-1
        min_weight = 0.1
        for i in range(start_layer, end_layer):
            progress = (i - start_layer) / max(n_active - 1, 1)
            mask[i] = 1.0 - progress * (1.0 - min_weight)
        
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from: all_middle, rotate, rotate_accum, "
                        "progressive, weighted_valley, weighted_mountain, alternating, block")
    
    return mask