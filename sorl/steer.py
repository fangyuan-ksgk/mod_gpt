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
      1. Chunk ALL hidden states into L-token windows.
      2. code = argmax(h_first[-C_SIZE:])   (diagonal routing, detached)
      3. hidden += steering_emb[code] * scale

    Steering is unconditional — every representable position is steered.
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

        # Chunk-level codes: first position of each L-window determines code
        n_chunks = S // self.L
        if n_chunks > 0:
            chunk_codes = torch.full((B, S), -1, dtype=torch.long,
                                     device=hidden_states.device)
            for c in range(n_chunks):
                ps = c * self.L
                chunk_codes[:, ps:ps + self.L] = pos_codes[:, ps:ps + 1]

            mask = chunk_codes >= 0
            safe_codes = chunk_codes.clamp(min=0)
            steer_vecs = self.steering_emb(safe_codes)          # (B, S, D)
            steer_vecs = steer_vecs * mask.unsqueeze(-1).float() * self.scale
            hidden_states = hidden_states + steer_vecs.to(hidden_states.dtype)

        return (hidden_states,) + rest if rest is not None else hidden_states

    # ---- forward / generate ----

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            labels=labels, **kwargs
        )

    def generate(self, **kwargs):
        return self.model.generate(**kwargs)

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


# ---------------------------------------------------------------------------
# V7: two-pass backward steering
# ---------------------------------------------------------------------------

class StackedAbstractionWrapperV7(nn.Module):
    """
    Two-pass backward steering.

    Pass 1 (read, no_grad):
        Full forward through the model.  A hook on *read_layer* (default:
        last transformer layer) extracts per-chunk abstract codes from
        hidden states via diagonal routing (argmax of last C_SIZE dims).

    Pass 2 (steer, with grad):
        Full forward again.  At each *inject_layer*, a steering vector
        looked up from the codes is added to the hidden states.

    Information flow is strictly backward: deep-layer representations from
    pass 1 determine the steering applied at (potentially earlier) layers
    in pass 2.  This mirrors the SoRL recursion pattern where the model
    first processes a chunk, then produces an abstract code that steers
    the next forward pass.

    Parameters
    ----------
    model         – HuggingFace causal LM
    C_SIZE        – number of abstract codes (routing width)
    D_MODEL       – hidden dimension
    inject_layers – layer indices where steering is applied (pass 2)
    scale         – steering vector magnitude multiplier
    L             – chunk size in tokens
    per_layer_emb – separate steering embedding per inject layer
    read_layer    – which layer to read codes from (default: last)
    code_position – "first" or "last" position in chunk for routing
    """

    def __init__(self, model, C_SIZE, D_MODEL, inject_layers=None,
                 scale=0.1, L=16, per_layer_emb=False,
                 read_layer=None, code_position="last"):
        super().__init__()
        self.model = model
        self.L = L
        self.scale = scale
        self.C_SIZE = C_SIZE
        self.D_MODEL = D_MODEL
        self.per_layer_emb = per_layer_emb
        self.code_position = code_position

        n_layers = model.config.num_hidden_layers
        self.read_layer = read_layer if read_layer is not None else n_layers - 1

        if inject_layers is None:
            inject_layers = [n_layers // 2]
        self.inject_layers = inject_layers

        # Steering embeddings
        if per_layer_emb:
            self.steering_emb = nn.ModuleList([
                nn.Embedding(C_SIZE, D_MODEL) for _ in inject_layers])
        else:
            self.steering_emb = nn.Embedding(C_SIZE, D_MODEL)

        # Zero-init all embeddings
        for p in self.parameters():
            if p.dim() == 2:
                nn.init.zeros_(p)

        # Two-pass state
        self._chunk_codes = None          # (B, S) from read pass
        self._pass = "idle"               # "read" | "steer" | "idle"

        self._hooks = []
        self._register_hooks()

    # ---- hooks ----

    def _register_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

        # Read hook — on read_layer (fires in pass 1 only)
        read_mod = self.model.model.layers[self.read_layer]
        self._hooks.append(
            read_mod.register_forward_hook(self._read_hook))

        # Steer hooks — on inject_layers (fire in pass 2 only)
        for hi, li in enumerate(self.inject_layers):
            layer = self.model.model.layers[li]
            self._hooks.append(
                layer.register_forward_hook(
                    lambda m, i, o, _hi=hi: self._steer_hook(m, i, o, _hi)))

    def _read_hook(self, module, input, output):
        """Extract per-chunk codes from read_layer output (pass 1 only)."""
        if self._pass != "read":
            return
        hidden = output[0] if isinstance(output, tuple) else output
        B, S, D = hidden.shape

        pos_codes = hidden[..., -self.C_SIZE:].argmax(dim=-1)   # (B, S)

        n_chunks = S // self.L
        chunk_codes = torch.full((B, S), -1, dtype=torch.long,
                                  device=hidden.device)
        for c in range(n_chunks):
            if self.code_position == "last":
                src = c * self.L + self.L - 1
            else:                                       # "first"
                src = c * self.L
            chunk_codes[:, c * self.L:(c + 1) * self.L] = pos_codes[:, src:src + 1]

        self._chunk_codes = chunk_codes
        # Don't modify output — read pass is observation only

    def _steer_hook(self, module, input, output, hook_idx):
        """Inject steering vectors at inject_layers (pass 2 only)."""
        if self._pass != "steer" or self._chunk_codes is None:
            return
        hidden = output[0] if isinstance(output, tuple) else output
        rest = output[1:] if isinstance(output, tuple) else None

        B, S, D = hidden.shape
        # Guard: skip if shape mismatch (e.g. generation beyond prompt)
        if S != self._chunk_codes.shape[1]:
            return (hidden,) + rest if rest else hidden

        emb = self.steering_emb[hook_idx] if self.per_layer_emb else self.steering_emb

        mask = self._chunk_codes >= 0
        if mask.any():
            safe = self._chunk_codes.clamp(min=0)
            sv = emb(safe) * mask.unsqueeze(-1).float() * self.scale
            hidden = hidden + sv.to(hidden.dtype)

        return (hidden,) + rest if rest else hidden

    # ---- forward / generate ----

    def read_pass(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """Pass 1: forward through model, extract codes at read_layer.

        Returns model outputs (with .loss if labels given).
        After this call, self._chunk_codes holds the detached (B,S) codes.
        """
        self._pass = "read"
        self._chunk_codes = None
        out = self.model(input_ids=input_ids,
                         attention_mask=attention_mask,
                         labels=labels, **kwargs)
        self._pass = "idle"
        return out

    def steer_pass(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """Pass 2: forward through model with steering vectors injected.

        Requires self._chunk_codes to be populated by a prior read_pass().
        Returns model outputs (with .loss if labels given).
        """
        self._pass = "steer"
        out = self.model(input_ids=input_ids,
                         attention_mask=attention_mask,
                         labels=labels, **kwargs)
        self._pass = "idle"
        return out

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """Convenience: read_pass + steer_pass, return steer outputs.

        For deep supervision (separate backward per pass), call
        read_pass() and steer_pass() explicitly instead.
        """
        self.read_pass(input_ids, attention_mask, labels, **kwargs)
        return self.steer_pass(input_ids, attention_mask, labels, **kwargs)

    def generate(self, **kwargs):
        """Read codes from prompt, then generate with steering on prompt."""
        input_ids = kwargs.get('input_ids', None)
        attn_mask = kwargs.get('attention_mask', None)

        # Pass 1: read codes from prompt
        if input_ids is not None:
            self._pass = "read"
            self._chunk_codes = None
            with torch.no_grad():
                self.model(input_ids=input_ids, attention_mask=attn_mask)

        # Pass 2: generate with steering (codes cover prompt only;
        # new tokens beyond prompt length are not steered)
        self._pass = "steer"
        out = self.model.generate(**kwargs)
        self._pass = "idle"
        self._chunk_codes = None
        return out

    # ---- save / load ----

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        state = {}
        for name, param in self.named_parameters():
            if 'steering_emb' in name:
                state[name] = param.data
        torch.save({
            'steering_state': state,
            'config': {
                'C_SIZE': self.C_SIZE, 'D_MODEL': self.D_MODEL,
                'inject_layers': self.inject_layers,
                'scale': self.scale, 'L': self.L,
                'per_layer_emb': self.per_layer_emb,
                'read_layer': self.read_layer,
                'code_position': self.code_position,
            },
        }, os.path.join(path, 'steer_v7.pt'))

    @classmethod
    def from_pretrained(cls, model, path):
        ckpt = torch.load(os.path.join(path, 'steer_v7.pt'),
                          map_location='cpu', weights_only=False)
        cfg = ckpt['config']
        wrapper = cls(model, **cfg)
        for name, param in wrapper.named_parameters():
            if name in ckpt['steering_state']:
                param.data.copy_(ckpt['steering_state'][name])
        return wrapper

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def get_steer_params(self):
        """Return all steering embedding parameters (for separate optimizer group)."""
        return [p for n, p in self.named_parameters() if 'steering_emb' in n]
