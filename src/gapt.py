# Gated Phase Transition & GPT with MBE regularization
# -------------------------------------------------------------------
import torch
torch.set_float32_matmul_precision('high')
from src.mbe import patch_mbe
RANK_REG_LOSS = "mbe"

# Customized GPT model with low-rank regularization loss 
# -------------------------------------------------------------------
from torch import nn
from typing import Optional 
import torch.nn.functional as F
from dataclasses import dataclass, field
from .model import CastedLinear, Block, create_block_mask, norm
import time

@dataclass
class GPTConfig:
    vocab_size : int = 50304
    n_layer : int = 12
    n_head : int = 6
    n_embd : int = 768
    flex_kernel_options: Optional[dict] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    _compile: bool = True if device == "cuda" else False


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

        # ===== Encoder Layers =====
        skip_connections = []
        for i in range(self.num_encoder_layers):
            if self.enable_timing:
                t0 = time.perf_counter()
            
            x, v1 = self.transformer.h[i](x, v1, x0, block_mask)
            
            if self.enable_timing:
                timings[f'encoder_layer_{i}_forward'] = (time.perf_counter() - t0) * 1000
                t0 = time.perf_counter()
            
            loss_dict[f"{RANK_REG_LOSS}_{i}"] = patch_mbe(x, patch_size)
            
            if self.enable_timing:
                timings[f'encoder_layer_{i}_mbe'] = (time.perf_counter() - t0) * 1000
            
            skip_connections.append(x)
        
        # ===== Decoder Layers =====
        for i in range(self.num_decoder_layers):
            if self.enable_timing:
                t0 = time.perf_counter()
            
            x = x + self.skip_weights[i] * skip_connections.pop()
            x, v1 = self.transformer.h[self.num_encoder_layers + i](x, v1, x0, block_mask)
            
            if self.enable_timing:
                timings[f'decoder_layer_{i}_forward'] = (time.perf_counter() - t0) * 1000
                t0 = time.perf_counter()
            
            loss_dict[f"{RANK_REG_LOSS}_{self.num_encoder_layers + i}"] = patch_mbe(x, patch_size)
            
            if self.enable_timing:
                timings[f'decoder_layer_{i}_mbe'] = (time.perf_counter() - t0) * 1000
        
        # ===== Output Head + Loss =====
        if self.enable_timing:
            t0 = time.perf_counter()
        
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


# Gated Phase Transition 
# ---------------------------------------------------------------------------------------
class GatedPhaseTransition:
    """
    Gated Phase Transition (GAPT) : https://arxiv.org/pdf/2505.08727
    with percentage-based thresholds.
    """
    def __init__(self, tau_plateau: float = 0.01, tau_spike: float = 0.1, 
                 p_m: int = 5, p_a: int = 5):
        """
        Args:
            tau_plateau: Relative threshold for detecting plateau (e.g., 0.01 = 1% improvement)
            tau_spike: Relative threshold for detecting spike (e.g., 0.1 = 10% degradation)
            p_m: Patience for main objective (steps without improvement)
            p_c: Patience for auxiliary objective (steps without improvement)
        """
        self.tau_plateau = tau_plateau  # % improvement needed to avoid plateau
        self.tau_spike = tau_spike      # % degradation that triggers phase switch
        self.p_m = p_m
        self.p_a = p_a

        self.phi = 1  # 1 for main phase, 2 for compression phase
        self.s_m = 0  # steps since improvement in main
        self.s_a = 0  # steps since improvement in auxiliary

        self.min_m = float('inf')
        self.min_a = float('inf')

    def _relative_gain(self, current_loss: float, min_loss: float) -> float:
        """Calculate percentage improvement (negative = degradation)"""
        if min_loss == float('inf') or min_loss == 0:
            return 0.0
        return (min_loss - current_loss) / min_loss.clamp(min=1e-6)

    def _weight_loss(self, main_loss: float, auxiliary_loss: float) -> float:
        """Weight the loss based on the phase"""
        if self.phi == 1:
            return main_loss
        elif self.phi == 2:
            return main_loss + auxiliary_loss
    
    def step(self, main_loss: float, auxiliary_loss: float, verbose: bool = False):
        """
        Update phase based on loss dynamics.
        
        Returns:
            phi: Current phase (1=main, 2=compression)
        """
        gain_m = self._relative_gain(main_loss, self.min_m)
        gain_a = self._relative_gain(auxiliary_loss, self.min_a)
        
        self.min_m = min(self.min_m, main_loss)
        self.min_a = min(self.min_a, auxiliary_loss)

        prev_phi = self.phi

        if self.phi == 1:  # Main objective phase
            if gain_m > self.tau_plateau: 
                self.s_m = 0
            else: 
                self.s_m += 1
            
            if self.s_m >= self.p_m:
                self.s_m = 0
                self.phi = 2

        elif self.phi == 2:  # Main + Auxiliary phase
            if gain_m < -self.tau_spike:  
                if verbose:
                    print(f"  [GAPT] Main loss spiked: {gain_m:.3f} < {-self.tau_spike:.3f}")
                self.s_a = 0
                self.phi = 1
            else:
                if gain_a > self.tau_plateau:  
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
            print(f"         main_loss={main_loss:.4f}, aux_loss={auxiliary_loss:.4f}")

        return self._weight_loss(main_loss, auxiliary_loss)