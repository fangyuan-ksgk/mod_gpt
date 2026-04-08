"""
Steering-based abstraction wrappers for causal LMs.

Two variants:
  StackedAbstractionWrapper   — VQ-coded: pre-computed chunk codes from VQ-VAE
  StackedAbstractionWrapperV6 — Self-routed: diagonal projection of hidden states

Both variants add learnable steering vectors to hidden states at selected
transformer layers.  The token sequence is never modified — steering operates
purely in representation space.
"""

import os
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# VQ-coded steering
# ---------------------------------------------------------------------------

class StackedAbstractionWrapper(nn.Module):
    """
    Wraps a HuggingFace causal LM and injects learnable steering vectors
    into selected transformer layers based on pre-computed VQ chunk codes.

    For each L-token chunk assigned code k, steering_emb[k] is added to
    the hidden states of those L positions at the hooked layer(s).
    """

    def __init__(self, model, C_SIZE, D_MODEL, inject_layers=None,
                 scale=0.1, L=16):
        super().__init__()
        self.model = model
        self.L = L
        self.scale = scale
        self.C_SIZE = C_SIZE
        self.D_MODEL = D_MODEL

        self.steering_emb = nn.Embedding(C_SIZE, D_MODEL)
        nn.init.zeros_(self.steering_emb.weight)

        n_layers = model.config.num_hidden_layers
        if inject_layers is None:
            inject_layers = [n_layers // 2]
        self.inject_layers = inject_layers

        self._steering_map = None   # (B, S) code indices; -1 = no steering
        self._hooks = []
        self._register_hooks()

    # ---- hooks ----

    def _register_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        for layer_idx in self.inject_layers:
            layer = self.model.model.layers[layer_idx]
            hook = layer.register_forward_hook(self._steering_hook)
            self._hooks.append(hook)

    def _steering_hook(self, module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None

        if self._steering_map is not None:
            # Training path: use pre-computed VQ codes
            steer_map = self._steering_map          # (B, S)
        else:
            # Generation fallback: V6-style diagonal routing over all positions
            B, S, D = hidden_states.shape
            with torch.no_grad():
                pos_codes = hidden_states[..., -self.C_SIZE:].argmax(dim=-1)
            steer_map = torch.full((B, S), -1, dtype=torch.long,
                                   device=hidden_states.device)
            for b in range(B):
                n_chunks = S // self.L
                for c in range(n_chunks):
                    ps = c * self.L
                    steer_map[b, ps:ps + self.L] = pos_codes[b, ps]

        mask = steer_map >= 0
        if mask.any():
            safe_codes = steer_map.clamp(min=0)
            steer_vecs = self.steering_emb(safe_codes)              # (B, S, D)
            steer_vecs = steer_vecs * mask.unsqueeze(-1).float() * self.scale
            hidden_states = hidden_states + steer_vecs.to(hidden_states.dtype)

        return (hidden_states,) + rest if rest is not None else hidden_states

    # ---- steering map construction ----

    def _build_steering_map(self, batch_size, seq_len, chunk_codes_list,
                            cot_starts, device):
        steer_map = torch.full((batch_size, seq_len), -1,
                               dtype=torch.long, device=device)
        for b in range(batch_size):
            codes = chunk_codes_list[b]
            cot_s = cot_starts[b]
            for c_idx, code in enumerate(codes):
                pos_start = cot_s + c_idx * self.L
                pos_end = min(pos_start + self.L, seq_len)
                if pos_start < seq_len:
                    steer_map[b, pos_start:pos_end] = code
        return steer_map

    # ---- forward / generate ----

    def forward(self, input_ids, attention_mask, labels,
                chunk_codes_list, cot_starts, **kwargs):
        B, S = input_ids.shape
        self._steering_map = self._build_steering_map(
            B, S, chunk_codes_list, cot_starts, input_ids.device
        )
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            labels=labels, **kwargs
        )
        self._steering_map = None
        return outputs

    def generate(self, **kwargs):
        # _steering_map=None → hook falls back to V6-style diagonal routing
        self._steering_map = None
        out = self.model.generate(**kwargs)
        return out

    # ---- save / load ----

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save({
            'steering_emb': self.steering_emb.state_dict(),
            'config': {
                'C_SIZE': self.C_SIZE, 'D_MODEL': self.D_MODEL,
                'inject_layers': self.inject_layers,
                'scale': self.scale, 'L': self.L,
            },
        }, os.path.join(path, 'steer_vq.pt'))

    @classmethod
    def from_pretrained(cls, model, path):
        ckpt = torch.load(os.path.join(path, 'steer_vq.pt'),
                          map_location='cpu', weights_only=False)
        cfg = ckpt['config']
        wrapper = cls(model, **cfg)
        wrapper.steering_emb.load_state_dict(ckpt['steering_emb'])
        return wrapper

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []


# ---------------------------------------------------------------------------
# Self-routed steering (v6-style)
# ---------------------------------------------------------------------------

class StackedAbstractionWrapperV6(nn.Module):
    """
    Self-routed steering: uses diagonal projection of hidden states to
    determine abstract codes at runtime.  No VQ-VAE needed.

    At selected transformer layer(s):
      1. Chunk response hidden states into L-token windows.
      2. code = argmax(h_first[-C_SIZE:])   (diagonal routing, detached)
      3. hidden += steering_emb[code] * scale

    The routing is non-differentiable (fixed diagonal + argmax).
    Only the steering embeddings and (optionally) model params are learned.
    """

    def __init__(self, model, C_SIZE, D_MODEL, inject_layers=None,
                 scale=0.1, L=16):
        super().__init__()
        self.model = model
        self.L = L
        self.scale = scale
        self.C_SIZE = C_SIZE
        self.D_MODEL = D_MODEL

        self.steering_emb = nn.Embedding(C_SIZE, D_MODEL)
        nn.init.zeros_(self.steering_emb.weight)

        n_layers = model.config.num_hidden_layers
        if inject_layers is None:
            inject_layers = [n_layers // 2]
        self.inject_layers = inject_layers

        self._prompt_lens = None    # (B,) int tensor — positions < pl not steered
        self._hooks = []
        self._register_hooks()

    # ---- hooks ----

    def _register_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        for layer_idx in self.inject_layers:
            layer = self.model.model.layers[layer_idx]
            hook = layer.register_forward_hook(self._steering_hook)
            self._hooks.append(hook)

    def _steering_hook(self, module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None

        B, S, D = hidden_states.shape

        # Diagonal routing: last C_SIZE dims → code per position (detached)
        with torch.no_grad():
            pos_codes = hidden_states[..., -self.C_SIZE:].argmax(dim=-1)   # (B, S)

        # Build chunk-level code map: first position determines chunk code
        # When _prompt_lens is None (generation), steer ALL positions (start=0)
        chunk_codes = torch.full((B, S), -1, dtype=torch.long,
                                 device=hidden_states.device)
        for b in range(B):
            start = 0 if self._prompt_lens is None else int(self._prompt_lens[b])
            n_resp = S - start
            n_chunks = n_resp // self.L
            for c in range(n_chunks):
                ps = start + c * self.L
                pe = ps + self.L
                chunk_codes[b, ps:pe] = pos_codes[b, ps]

        mask = chunk_codes >= 0
        if mask.any():
            safe_codes = chunk_codes.clamp(min=0)
            steer_vecs = self.steering_emb(safe_codes)          # (B, S, D)
            steer_vecs = steer_vecs * mask.unsqueeze(-1).float() * self.scale
            hidden_states = hidden_states + steer_vecs.to(hidden_states.dtype)

        return (hidden_states,) + rest if rest is not None else hidden_states

    # ---- forward / generate ----

    def forward(self, input_ids, attention_mask=None, labels=None,
                prompt_lens=None, **kwargs):
        self._prompt_lens = prompt_lens
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            labels=labels, **kwargs
        )
        self._prompt_lens = None
        return outputs

    def generate(self, **kwargs):
        # _prompt_lens=None → hook steers ALL positions (start=0)
        self._prompt_lens = None
        out = self.model.generate(**kwargs)
        return out

    # ---- save / load ----

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save({
            'steering_emb': self.steering_emb.state_dict(),
            'config': {
                'C_SIZE': self.C_SIZE, 'D_MODEL': self.D_MODEL,
                'inject_layers': self.inject_layers,
                'scale': self.scale, 'L': self.L,
            },
        }, os.path.join(path, 'steer_v6.pt'))

    @classmethod
    def from_pretrained(cls, model, path):
        ckpt = torch.load(os.path.join(path, 'steer_v6.pt'),
                          map_location='cpu', weights_only=False)
        cfg = ckpt['config']
        wrapper = cls(model, **cfg)
        wrapper.steering_emb.load_state_dict(ckpt['steering_emb'])
        return wrapper

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
