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
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F


def _find_similar_magnitude_dims(lm_weight, V):
    """Find V hidden dims whose lm_head column importance is most uniform.

    Importance of dim j = ||lm_head.weight[:, j]||_2 (L2 norm of column j).
    We sort dims by importance (descending), slide a V-wide window, and
    pick the window with minimal coefficient of variation (CV = std / mean).
    This selects dims with the most *relatively* uniform importance, so that
    each abstract token's routing dimension competes on a level playing field.

    Returns: (selected_dim_indices, importances, cv)
    """
    with torch.no_grad():
        dim_importance = lm_weight.float().norm(dim=0)  # (d,)
        sorted_vals, sorted_idxs = dim_importance.sort(descending=True)

        best_cv = float('inf')
        best_start = 0
        for i in range(len(sorted_vals) - V + 1):
            window = sorted_vals[i : i + V]
            w_mean = window.mean().item()
            if w_mean < 1e-9:
                continue
            w_std = window.std().item()
            cv = w_std / w_mean
            if cv < best_cv:
                best_cv = cv
                best_start = i

        dims = sorted_idxs[best_start : best_start + V]
        importances = sorted_vals[best_start : best_start + V]
        return dims, importances, best_cv


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

    Variants (all backward-compatible, defaults match original behaviour):
      routing_mode   – "diagonal" (last C_SIZE dims) | "similar_magnitude"
      per_layer_emb  – False → shared embedding | True → one per inject layer
      code_position  – "first" → first token routes chunk (forward steering)
                       "last"  → last token routes chunk (backward steering)
    """

    def __init__(self, model, C_SIZE, D_MODEL, inject_layers=None,
                 scale=0.1, L=16, routing_mode="diagonal",
                 per_layer_emb=False, code_position="first",
                 routing_temperature=None):
        super().__init__()
        self.model = model
        self.L = L
        self.scale = scale
        self.C_SIZE = C_SIZE
        self.D_MODEL = D_MODEL
        self.routing_mode = routing_mode
        self.per_layer_emb = per_layer_emb
        self.code_position = code_position
        self.routing_temperature = routing_temperature

        n_layers = model.config.num_hidden_layers
        if inject_layers is None:
            inject_layers = [n_layers // 2]
        self.inject_layers = inject_layers

        # Routing dims
        if routing_mode == "similar_magnitude":
            lm_weight = model.lm_head.weight
            self._routing_dims, self._routing_importances, self._routing_cv = \
                _find_similar_magnitude_dims(lm_weight, C_SIZE)
            print(f"V6 similar_magnitude routing: selected {C_SIZE} dims, "
                  f"importance range=[{self._routing_importances.min():.4f}, {self._routing_importances.max():.4f}], "
                  f"CV={self._routing_cv:.6f}")
        else:
            self._routing_dims = None

        # Steering embeddings
        if per_layer_emb:
            self.steering_emb = nn.ModuleList([
                nn.Embedding(C_SIZE, D_MODEL) for _ in inject_layers])
            for emb in self.steering_emb:
                nn.init.zeros_(emb.weight)
        else:
            self.steering_emb = nn.Embedding(C_SIZE, D_MODEL)
            nn.init.zeros_(self.steering_emb.weight)

        self._hooks = []
        self._last_codes = None  # (B, S) chunk-level codes from last forward (per-token, repeated within a chunk)
        self._last_chunk_codes = None  # (B, n_chunks) per-chunk codes — mirrors V9 for analysis tooling

        # ---- decode-time state (used only inside self.generate) ------------
        # V6 was trained with prefill-only chunking; applying steering to
        # autoregressive decode tokens is a distribution shift the model never
        # saw, so `_decode_scale_override` defaults to 0.0 — decode-time codes
        # are observed (and logged) but NOT injected. Flip to None (use
        # self.scale) or any float to opt in.
        self._in_generate = False
        self._decode_tail = 0
        self._decode_current_code = None
        self._decode_codes_log = None
        self._decode_scale_override = 0.0

        self._register_hooks()

    # ---- hooks ----

    def _register_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        for i, layer_idx in enumerate(self.inject_layers):
            layer = self.model.model.layers[layer_idx]
            hook = layer.register_forward_hook(
                functools.partial(self._steering_hook, hook_idx=i))
            self._hooks.append(hook)

    def _steering_hook(self, module, input, output, hook_idx=0):
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None

        B, S, D = hidden_states.shape
        emb = self.steering_emb[hook_idx] if self.per_layer_emb else self.steering_emb

        # ---- Decode-time path: S == 1 during autoregressive generation --------
        # V6 was trained with prefill-only chunking; we still want to observe
        # which code the model's hidden state would route to at decode time
        # (for inner-monologue analysis), but by default we do NOT inject —
        # that's a distribution shift V6 never saw during training.
        if self._in_generate and S == 1:
            decode_scale = (self._decode_scale_override
                            if self._decode_scale_override is not None
                            else self.scale)

            # Re-route at each chunk boundary (every L generated tokens).
            if self._decode_tail == 0:
                with torch.no_grad():
                    if self._routing_dims is not None:
                        routed_one = hidden_states[..., self._routing_dims]
                    else:
                        routed_one = hidden_states[..., -self.C_SIZE:]
                    code_one = routed_one[:, 0, :].argmax(dim=-1)   # (B,)
                self._decode_current_code = code_one
                if self._decode_codes_log is not None:
                    self._decode_codes_log.append(code_one.cpu())

            # Apply steering only if decode_scale is nonzero (default: skip).
            if self._decode_current_code is not None and decode_scale != 0.0:
                steer = emb(self._decode_current_code)              # (B, D)
                hidden_states = hidden_states + (
                    steer.unsqueeze(1) * decode_scale
                ).to(hidden_states.dtype)

            self._decode_tail = (self._decode_tail + 1) % self.L
            return (hidden_states,) + rest if rest is not None else hidden_states

        # ---- Prefill / training path -----------------------------------------
        # Routing: select dims for code extraction
        with torch.no_grad():
            if self._routing_dims is not None:  # similar_magnitude
                routed = hidden_states[..., self._routing_dims]
            else:  # diagonal: last C_SIZE dims
                routed = hidden_states[..., -self.C_SIZE:]

            if self.routing_temperature is not None and self.training:
                probs = F.softmax(routed / self.routing_temperature, dim=-1)  # (B, S, C)
                pos_codes = torch.multinomial(probs.view(-1, self.C_SIZE), 1).view(B, S) # -> not tested
            else:
                pos_codes = routed.argmax(dim=-1)   # (B, S)

        # Chunk-level codes.
        # During prefill (_in_generate), also route the partial trailing chunk
        # whose start position sits inside the prompt, so:
        #   (a) its prompt-side tokens receive the correct steering, and
        #   (b) `_decode_current_code` is the partial chunk's code, which the
        #       remaining decode tokens in that chunk will continue to use.
        # Restricted to code_position="first" — "last" can't route without
        # seeing the chunk's final token.
        n_chunks_full = S // self.L
        has_partial = (
            self._in_generate
            and (S % self.L != 0)
            and (self.code_position == "first")
        )
        n_chunks = n_chunks_full + (1 if has_partial else 0)

        if n_chunks > 0:
            chunk_codes = torch.full((B, S), -1, dtype=torch.long,
                                     device=hidden_states.device)
            if self.code_position == "last":
                src_idx = [c * self.L + self.L - 1 for c in range(n_chunks)]
            else:  # "first"
                src_idx = [c * self.L for c in range(n_chunks)]
            for c, src in enumerate(src_idx):
                start = c * self.L
                end = min((c + 1) * self.L, S)
                chunk_codes[:, start:end] = pos_codes[:, src:src + 1]

            mask = chunk_codes >= 0
            safe_codes = chunk_codes.clamp(min=0)
            steer_vecs = emb(safe_codes)          # (B, S, D)
            steer_vecs = steer_vecs * mask.unsqueeze(-1).float() * self.scale
            hidden_states = hidden_states + steer_vecs.to(hidden_states.dtype)
            self._last_codes = chunk_codes                         # (B, S) legacy per-token
            self._last_chunk_codes = pos_codes[:, src_idx].detach()  # (B, n_chunks)

            # Seed decode state at end of prefill so autoregressive steps
            # continue chunking from where the prompt left off.
            if self._in_generate:
                self._decode_tail = S % self.L
                self._decode_current_code = self._last_chunk_codes[:, -1]
        elif self._in_generate:
            # Prefill shorter than one chunk — remember offset for first decode step.
            self._decode_tail = S % self.L

        return (hidden_states,) + rest if rest is not None else hidden_states

    # ---- forward / generate ----

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            labels=labels, **kwargs
        )

    def generate(self, log_decode_codes=False, decode_scale=None, **kwargs):
        """Run autoregressive generation with code logging.
        """
        self._in_generate = True
        self._decode_tail = 0
        self._decode_current_code = None
        self._decode_codes_log = [] if log_decode_codes else None
        prev_override = self._decode_scale_override
        if decode_scale is not None:
            self._decode_scale_override = float(decode_scale)
        try:
            out = self.model.generate(**kwargs)
        finally:
            self._in_generate = False
            self._decode_scale_override = prev_override
        return out

    # ---- save / load ----

    def get_steer_params(self):
        """Return all steering embedding parameters (for separate optimizer group)."""
        return [p for n, p in self.named_parameters() if 'steering_emb' in n]

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save({
            'steering_emb': self.steering_emb.state_dict(),
            'config': {
                'C_SIZE': self.C_SIZE, 'D_MODEL': self.D_MODEL,
                'inject_layers': self.inject_layers,
                'scale': self.scale, 'L': self.L,
                'routing_mode': self.routing_mode,
                'per_layer_emb': self.per_layer_emb,
                'code_position': self.code_position,
                'routing_temperature': self.routing_temperature,
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
# V8: V6 + STE-trainable routing
# ---------------------------------------------------------------------------

class StackedAbstractionWrapperV8(nn.Module):
    """
    End-to-end trainable routing via Straight-Through Estimator (STE).

    Like V6, but routing is differentiable:
      1. logits = routing_proj(h)                         (learned projection)
      2. soft   = softmax(logits)                         (differentiable)
      3. hard   = one_hot(argmax(logits))                 (discrete, forward path)
      4. ste    = hard - soft.detach() + soft             (STE: fwd=hard, bwd=soft)
      5. steer  = ste @ steering_emb.weight * scale       (differentiable lookup)

    No separate routing loss needed — routing_proj and steering_emb both
    receive gradients from the main CE loss via STE.

    During eval / generate: plain argmax, no STE overhead.
    """

    def __init__(self, model, C_SIZE, D_MODEL, inject_layers=None,
                 scale=0.1, L=16, per_layer_emb=False,
                 code_position="first"):
        super().__init__()
        self.model = model
        self.L = L
        self.scale = scale
        self.C_SIZE = C_SIZE
        self.D_MODEL = D_MODEL
        self.per_layer_emb = per_layer_emb
        self.code_position = code_position

        n_layers = model.config.num_hidden_layers
        if inject_layers is None:
            inject_layers = [n_layers // 2]
        self.inject_layers = inject_layers

        # Learned routing projection (match model dtype)
        dtype = next(model.parameters()).dtype
        self.routing_proj = nn.Linear(D_MODEL, C_SIZE, bias=False, dtype=dtype)
        nn.init.normal_(self.routing_proj.weight, std=0.01)

        # Steering embeddings
        if per_layer_emb:
            self.steering_emb = nn.ModuleList([
                nn.Embedding(C_SIZE, D_MODEL) for _ in inject_layers])
            for emb in self.steering_emb:
                nn.init.zeros_(emb.weight)
        else:
            self.steering_emb = nn.Embedding(C_SIZE, D_MODEL)
            nn.init.zeros_(self.steering_emb.weight)

        self._hooks = []
        self._register_hooks()

    # ---- hooks ----

    def _register_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        for i, layer_idx in enumerate(self.inject_layers):
            layer = self.model.model.layers[layer_idx]
            hook = layer.register_forward_hook(
                functools.partial(self._steering_hook, hook_idx=i))
            self._hooks.append(hook)

    def _steering_hook(self, module, input, output, hook_idx=0):
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None

        B, S, D = hidden_states.shape
        n_chunks = S // self.L
        if n_chunks == 0:
            return (hidden_states,) + rest if rest is not None else hidden_states

        # Routing logits — WITH gradient
        logits = self.routing_proj(hidden_states)  # (B, S, C)

        # Source positions per chunk
        if self.code_position == "last":
            src_idx = [c * self.L + self.L - 1 for c in range(n_chunks)]
        else:  # "first"
            src_idx = [c * self.L for c in range(n_chunks)]

        src_logits = logits[:, src_idx, :]  # (B, n_chunks, C)

        if self.training:
            # STE: forward = hard one-hot, backward = soft probabilities
            soft = F.softmax(src_logits.float(), dim=-1)       # (B, nc, C) fp32
            hard = F.one_hot(soft.argmax(dim=-1),
                             self.C_SIZE).to(soft.dtype)       # (B, nc, C)
            ste = (hard - soft).detach() + soft                # STE trick

            emb = self.steering_emb[hook_idx] if self.per_layer_emb else self.steering_emb
            # (B, nc, C) @ (C, D) → (B, nc, D)
            chunk_steer = torch.matmul(ste, emb.weight.float())
        else:
            # Eval: plain argmax, no STE
            codes = src_logits.argmax(dim=-1)  # (B, nc)
            emb = self.steering_emb[hook_idx] if self.per_layer_emb else self.steering_emb
            chunk_steer = emb(codes)           # (B, nc, D)

        # Expand chunk vectors to all positions in each chunk
        # (B, nc, D) → (B, nc, L, D) → (B, nc*L, D)
        steer_vecs = chunk_steer.unsqueeze(2).expand(-1, -1, self.L, -1)
        steer_vecs = steer_vecs.reshape(B, n_chunks * self.L, D)

        # Pad tail tokens (< 1 full chunk) with zeros
        if n_chunks * self.L < S:
            pad = steer_vecs.new_zeros(B, S - n_chunks * self.L, D)
            steer_vecs = torch.cat([steer_vecs, pad], dim=1)

        hidden_states = hidden_states + (steer_vecs * self.scale).to(hidden_states.dtype)

        return (hidden_states,) + rest if rest is not None else hidden_states

    # ---- forward / generate ----

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            labels=labels, **kwargs
        )

    def generate(self, **kwargs):
        return self.model.generate(**kwargs)

    # ---- param groups ----

    def get_steer_params(self):
        """Return steering + routing params."""
        return [p for n, p in self.named_parameters()
                if 'steering_emb' in n or 'routing_proj' in n]

    # ---- save / load ----

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save({
            'steering_emb': self.steering_emb.state_dict(),
            'routing_proj': self.routing_proj.state_dict(),
            'config': {
                'C_SIZE': self.C_SIZE, 'D_MODEL': self.D_MODEL,
                'inject_layers': self.inject_layers,
                'scale': self.scale, 'L': self.L,
                'per_layer_emb': self.per_layer_emb,
                'code_position': self.code_position,
            },
        }, os.path.join(path, 'steer_v8.pt'))

    @classmethod
    def from_pretrained(cls, model, path):
        ckpt = torch.load(os.path.join(path, 'steer_v8.pt'),
                          map_location='cpu', weights_only=False)
        cfg = ckpt['config']
        wrapper = cls(model, **cfg)
        wrapper.steering_emb.load_state_dict(ckpt['steering_emb'])
        wrapper.routing_proj.load_state_dict(ckpt['routing_proj'])
        return wrapper

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []



class StackedAbstractionWrapperV9(nn.Module):
    """
    Learnable abstraction routing (V9).

    Like V6, but routing uses a trainable linear projection instead of
    fixed diagonal dims:
      1. Chunk hidden states into L-token windows.
      2. logits = abs_proj(h)              (learned D → C_SIZE projection)
      3. code = argmax(logits)             (detached, non-differentiable)
      4. hidden += steering_emb[code] * scale

    The abs_proj head and steering embeddings are both learned via the
    main CE loss.  Unlike V8 (STE), routing gradients do NOT flow through
    the argmax — only the steering embeddings receive direct gradients.
    The abs_proj learns from the indirect signal of which codes reduce loss.

    Variants:
      per_layer_emb  – False → shared embedding | True → one per inject layer
      code_position  – "first" | "last"
      routing_temperature – if set, use multinomial sampling during training
    """

    def __init__(self, model, C_SIZE, D_MODEL, inject_layers=None,
                 scale=0.1, L=16, routing_mode="learned",
                 per_layer_emb=False, code_position="first"):
        super().__init__()
        self.model = model
        self.L = L
        self.scale = scale
        self.C_SIZE = C_SIZE
        self.D_MODEL = D_MODEL
        self.routing_mode = routing_mode
        self.per_layer_emb = per_layer_emb
        self.code_position = code_position

        n_layers = model.config.num_hidden_layers
        if inject_layers is None:
            inject_layers = [n_layers // 2]
        self.inject_layers = inject_layers

        # Trainable routing head: D_MODEL → C_SIZE
        dtype = next(model.parameters()).dtype
        self.abs_proj = nn.Linear(D_MODEL, C_SIZE, bias=False, dtype=dtype)
        nn.init.normal_(self.abs_proj.weight, std=0.01)

        # Steering embeddings
        if per_layer_emb:
            self.steering_emb = nn.ModuleList([
                nn.Embedding(C_SIZE, D_MODEL) for _ in inject_layers])
            for emb in self.steering_emb:
                nn.init.zeros_(emb.weight)
        else:
            self.steering_emb = nn.Embedding(C_SIZE, D_MODEL)
            nn.init.zeros_(self.steering_emb.weight)

        # State populated by hooks during forward pass
        self._last_routing_logits = None  # (B, n_chunks, C) — abs_proj logits at src positions
        self._last_codes = None     # (B, n_chunks)     — argmax codes actually used
        self._forced_codes = None         # set externally to override routing in next forward
        self._forward_temperature = None   # set by forward(), consumed by hook
        self._detach_routing = False      # Whether to detach training of policy model from rep

        # ---- decode-time state (used only inside self.generate) ----
        # During autoregressive decoding with KV cache, the hook sees S=1 per
        # step. We maintain a rolling chunk offset + current code so that every
        # L new tokens we re-route, and every decode step gets steered.
        self._in_generate = False
        self._decode_tail = 0         # 0..L-1, position within current chunk
        self._decode_current_code = None  # (B,) long, code to apply each step
        self._decode_codes_log = None     # list of (B,) per decoded chunk, for inspection
        # None → decode uses self.scale (current behaviour). Set a float via
        # `generate(decode_scale=...)` to override (e.g. 0.0 to read codes
        # without injecting, symmetric with V6 defaults).
        self._decode_scale_override = 0.0

        # ---- n-gram ablation patch state (opt-in, used by analyze helpers) ----
        # self._ablate_ngrams: list[tuple[int,...]] of code-sequences to veto.
        # Whenever the router's rolling history of committed codes ends with
        # one of these patterns, the final code is replaced with either:
        #   - the fixed replacement from self._ablate_replacements[pattern]
        #     if that entry is an int, else
        #   - a uniformly random codebook id != the matched code
        # before the steering lookup.
        self._ablate_ngrams = None
        self._ablate_replacements = None  # dict[tuple -> int|None]; None → random
        self._ablate_rng = None
        self._ablate_exclude = None  # set[int] of codes never drawn as random replacement
        self._ablate_history = {}   # batch_idx -> list[int]
        self._ablate_hits = []      # [(phase, batch_idx, chunk_idx, old, new)]

        self._hooks = []
        self._register_hooks()

    # ---- n-gram ablation patch ----

    def _ablate_patch_codes(self, codes, phase):
        """In-place swap of codes that complete any pattern in
        ``self._ablate_ngrams``. Called immediately after argmax in both
        the prefill and decode branches of ``_steering_hook``.

        codes: (B, nc) long on self's device.
        phase: "prefill" or "decode" (recorded in hits for diagnostics).
        """
        if not self._ablate_ngrams:
            return codes
        B, nc = codes.shape
        for b in range(B):
            hist = self._ablate_history.setdefault(b, [])
            for k in range(nc):
                c = int(codes[b, k].item())
                hist.append(c)
                for pat in self._ablate_ngrams:
                    N = len(pat)
                    if len(hist) >= N and tuple(hist[-N:]) == pat:
                        reps = self._ablate_replacements or {}
                        has_fixed = pat in reps and reps[pat] is not None
                        if has_fixed:
                            # Deterministic swap to the requested code.
                            # If it equals c, this is a no-op (no random fallback).
                            new = int(reps[pat])
                            if new == c:
                                break
                        else:
                            # Random replacement: draw uniformly over
                            # codebook \ ({c} ∪ self._ablate_exclude).
                            excl = self._ablate_exclude or set()
                            forbidden = excl | {c}
                            allowed = [i for i in range(self.C_SIZE)
                                       if i not in forbidden]
                            if not allowed:
                                # Degenerate (everything forbidden): keep c.
                                break
                            new = self._ablate_rng.choice(allowed)
                        codes[b, k] = new
                        hist[-1] = new
                        self._ablate_hits.append((phase, b, len(hist) - 1, c, new))
                        break
        return codes

    # ---- hooks ----

    def _register_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        for i, layer_idx in enumerate(self.inject_layers):
            layer = self.model.model.layers[layer_idx]
            hook = layer.register_forward_hook(
                functools.partial(self._steering_hook, hook_idx=i))
            self._hooks.append(hook)

    def _steering_hook(self, module, input, output, hook_idx=0):
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None

        B, S, D = hidden_states.shape

        # ---- Decode-time path: S == 1 during autoregressive generation --------
        # During HF generate with KV cache, the hook fires once per new token
        # with S=1. The prefill-style chunking (n_chunks = S // L) short-circuits
        # here, which means generated tokens never receive steering unless we
        # handle this branch explicitly.
        if self._in_generate and S == 1:
            emb_mod = self.steering_emb[hook_idx] if self.per_layer_emb else self.steering_emb
            decode_scale = (self._decode_scale_override
                            if self._decode_scale_override is not None
                            else self.scale)

            # Re-route at each chunk boundary (every L generated tokens).
            if self._decode_tail == 0:
                routed_one = (self.abs_proj(hidden_states.detach())
                              if self._detach_routing
                              else self.abs_proj(hidden_states))       # (B, 1, C)
                logits_one = routed_one[:, 0, :]                        # (B, C)
                with torch.no_grad():
                    code_one = logits_one.argmax(dim=-1)                # (B,)
                # n-gram ablation patch (no-op unless ablate state populated)
                code_one = self._ablate_patch_codes(
                    code_one.unsqueeze(1), phase="decode"
                ).squeeze(1)
                self._decode_current_code = code_one
                if self._decode_codes_log is not None:
                    self._decode_codes_log.append(code_one.cpu())

            # Apply the current chunk's steering to this single decode token.
            if self._decode_current_code is not None and decode_scale != 0.0:
                steer = emb_mod(self._decode_current_code)              # (B, D)
                hidden_states = hidden_states + (
                    steer.unsqueeze(1) * decode_scale
                ).to(hidden_states.dtype)

            # Advance within-chunk offset.
            self._decode_tail = (self._decode_tail + 1) % self.L
            return (hidden_states,) + rest if rest is not None else hidden_states

        # Routing: learned projection → logits
        if self._detach_routing: 
            routed = self.abs_proj(hidden_states.detach())  # (B, S, C) — gradient ONLY on policy/routing
        else: 
            routed = self.abs_proj(hidden_states)  # (B, S, C) — gradient on policy & representation

        # Chunk-source positions.
        # During prefill (i.e. `_in_generate` is True), the prompt may end
        # mid-chunk — e.g. S=6 with L=4 means chunk 0 spans 0..3 and chunk 1
        # starts at position 4 with only 2 tokens present. We still want to
        # route that partial chunk so:
        #   (a) its prompt-side tokens (4..5) receive the right steering, and
        #   (b) `_decode_current_code` carries its code forward to the
        #       still-to-be-generated tokens (6..7) before the next boundary.
        # Only "first" can be routed partially (its src position is c*L, which
        # is always < S if c*L < S); "last" needs the chunk's final token
        # which doesn't exist yet, so we skip partial chunks there.
        n_chunks_full = S // self.L
        has_partial = (
            self._in_generate
            and (S % self.L != 0)
            and (self.code_position == "first")
        )
        n_chunks = n_chunks_full + (1 if has_partial else 0)

        if n_chunks == 0:
            # Prefill of an empty / sub-chunk input with "last" routing, or
            # S==0: remember tail so the first decode step knows how many
            # tokens until the next chunk boundary.
            if self._in_generate:
                self._decode_tail = S % self.L
            return (hidden_states,) + rest if rest is not None else hidden_states

        if self.code_position == "last":
            src_idx = [c * self.L + self.L - 1 for c in range(n_chunks)]
        else:  # "first"
            src_idx = [c * self.L for c in range(n_chunks)]

        src_logits = routed[:, src_idx, :]  # (B, n_chunks, C)

        # Code selection: forced_codes > temperature sampling > argmax
        if self._forced_codes is not None:
            codes = self._forced_codes  # (B, n_chunks) — externally supplied
        elif self._forward_temperature is not None:
            with torch.no_grad():
                probs = F.softmax(src_logits / self._forward_temperature, dim=-1)
                codes = torch.multinomial(probs.view(-1, self.C_SIZE), 1).view(B, n_chunks)
        else:
            with torch.no_grad():
                codes = src_logits.argmax(dim=-1)  # (B, n_chunks)

        # n-gram ablation patch (no-op unless ablate state populated)
        codes = self._ablate_patch_codes(codes, phase="prefill")

        self._last_routing_logits = src_logits  # (B, n_chunks, C) — retains gradient
        self._last_codes = codes.detach()  # (B, n_chunks)
        self._last_chunk_codes = self._last_codes  # alias — matches V6 analysis API

        emb = self.steering_emb[hook_idx] if self.per_layer_emb else self.steering_emb
        chunk_steer = emb(codes)  # (B, n_chunks, D)

        steer_vecs = chunk_steer.unsqueeze(2).expand(-1, -1, self.L, -1)
        steer_vecs = steer_vecs.reshape(B, n_chunks * self.L, D)

        # Align with S:
        #   - full-chunk case (n*L == S)       → no-op
        #   - partial-chunk case (n*L > S)     → trim trailing positions that
        #                                         don't exist in the prompt
        #   - sub-chunk / legacy case (n*L<S)  → zero-pad the tail (e.g. when
        #                                         has_partial=False for "last"
        #                                         or during training)
        if n_chunks * self.L > S:
            steer_vecs = steer_vecs[:, :S, :]
        elif n_chunks * self.L < S:
            pad = steer_vecs.new_zeros(B, S - n_chunks * self.L, D)
            steer_vecs = torch.cat([steer_vecs, pad], dim=1)

        hidden_states = hidden_states + (steer_vecs * self.scale).to(hidden_states.dtype)

        # Seed decode state at end of prefill so autoregressive steps continue
        # chunking from where the prompt left off.
        if self._in_generate:
            self._decode_tail = S % self.L
            # Carry the last prompt chunk's code so any partial-tail decode
            # tokens still receive steering before the next chunk boundary.
            self._decode_current_code = codes[:, -1]

        return (hidden_states,) + rest if rest is not None else hidden_states

    # ---- forward / generate ----

    def forward(self, input_ids, attention_mask=None, labels=None,
                forced_codes=None, temperature=None, reduction='mean', **kwargs):
        """Forward pass. After calling, inspect:
           - self._last_codes    (B, n_chunks)  — codes used for steering
           - self._last_routing_logits  (B, n_chunks, C) — logits (with grad) for abs_loss

        Args:
            forced_codes: (B, n_chunks) int tensor — override routing with these codes.
            temperature:  float — if set, sample codes via softmax(logits/T). Otherwise argmax.
            reduction: 'mean' (default HF) or 'none' → return per-sample losses.
        """
        self._forced_codes = forced_codes
        self._forward_temperature = temperature

        if reduction == 'none' and labels is not None:
            # Skip labels → HF won't compute redundant scalar CE
            out = self.model(
                input_ids=input_ids, attention_mask=attention_mask,
                **kwargs
            )
            shift_logits = out.logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            per_token = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100, reduction='none',
            ).view(input_ids.size(0), -1)
            valid = (shift_labels != -100).float()
            out.per_sample_loss = (per_token * valid).sum(-1) / valid.sum(-1).clamp(min=1)
        else:
            out = self.model(
                input_ids=input_ids, attention_mask=attention_mask,
                labels=labels, **kwargs
            )

        self._forced_codes = None
        self._forward_temperature = None
        return out

    def compute_abs_loss(self, target_codes=None):
        """Compute abs_loss: CE(routing_logits, target_codes) at chunk positions.

        Args:
            target_codes: (B, n_chunks) int tensor of target codes from search.
                          If None, uses self._last_codes (self-distillation).
        Returns:
            scalar loss, or 0 if no routing logits were captured.
        """
        logits = self._last_routing_logits  # (B, n_chunks, C) — has gradient
        if logits is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        if target_codes is None:
            target_codes = self._last_codes
        return F.cross_entropy(
            logits.reshape(-1, self.C_SIZE),
            target_codes.reshape(-1),
        )

    def generate(self, log_decode_codes=False, decode_scale=None, **kwargs):
        """Generate with steering applied on every decode step.

        The forward hook normally short-circuits when S < L; during KV-cached
        generation that means every decoded token is unsteered. We flip
        `_in_generate` so the hook takes the decode-time path: the prompt is
        steered in the prefill pass, and each subsequent S=1 step re-uses the
        current chunk's code, re-routing every L tokens.

        Args:
            log_decode_codes: if True, each re-routed decode chunk code is
                appended to `self._decode_codes_log` (cpu) for inspection.
            decode_scale: optional float override for decode-time injection
                scale (leaves prefill untouched). Pass 0.0 to read codes
                without perturbing generation. Default (None) uses self.scale.
        """
        self._in_generate = True
        self._decode_tail = 0
        self._decode_current_code = None
        self._decode_codes_log = [] if log_decode_codes else None
        prev_override = self._decode_scale_override
        if decode_scale is not None:
            self._decode_scale_override = float(decode_scale)
        try:
            return self.model.generate(**kwargs)
        finally:
            self._in_generate = False
            self._decode_tail = 0
            self._decode_current_code = None
            self._decode_scale_override = prev_override

    # ---- save / load ----

    def get_steer_params(self):
        """Return all steering + routing parameters (for separate optimizer group)."""
        return [p for n, p in self.named_parameters()
                if 'steering_emb' in n or 'abs_proj' in n]

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save({
            'steering_emb': self.steering_emb.state_dict(),
            'abs_proj': self.abs_proj.state_dict(),
            'config': {
                'C_SIZE': self.C_SIZE, 'D_MODEL': self.D_MODEL,
                'inject_layers': self.inject_layers,
                'scale': self.scale, 'L': self.L,
                'routing_mode': self.routing_mode,
                'per_layer_emb': self.per_layer_emb,
                'code_position': self.code_position,
            },
        }, os.path.join(path, 'steer_v9.pt'))

    @classmethod
    def from_pretrained(cls, model, path):
        ckpt = torch.load(os.path.join(path, 'steer_v9.pt'),
                          map_location='cpu', weights_only=False)
        cfg = ckpt['config']
        wrapper = cls(model, **cfg)
        wrapper.steering_emb.load_state_dict(ckpt['steering_emb'])
        wrapper.abs_proj.load_state_dict(ckpt['abs_proj'])
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
    read_layer    – which layer to read codes from (default: last)
    code_position – "first" or "last" position in chunk for routing
    routing_mode  – "diagonal" (last C_SIZE dims) or "similar_magnitude" (select dims with uniform importance)
    """

    def __init__(self, model, C_SIZE, D_MODEL, inject_layers=None,
                 scale=0.1, L=16,
                 read_layer=None, code_position="last", routing_mode="diagonal"):
        super().__init__()
        self.model = model
        self.L = L
        self.scale = scale
        self.C_SIZE = C_SIZE
        self.D_MODEL = D_MODEL
        self.code_position = code_position
        self.routing_mode = routing_mode

        n_layers = model.config.num_hidden_layers
        self.read_layer = read_layer if read_layer is not None else n_layers - 1

        if inject_layers is None:
            inject_layers = [n_layers // 2]
        self.inject_layers = inject_layers

        # Select routing dimensions based on mode
        if routing_mode == "similar_magnitude":
            lm_weight = model.lm_head.weight
            self._routing_dims, self._routing_importances, self._routing_cv = \
                _find_similar_magnitude_dims(lm_weight, C_SIZE)
            print(f"V7 similar_magnitude routing: selected {C_SIZE} dims, "
                  f"importance range=[{self._routing_importances.min():.4f}, {self._routing_importances.max():.4f}], "
                  f"CV={self._routing_cv:.6f}")
        else:  # diagonal
            self._routing_dims = None  # use last C_SIZE dims

        # Per-layer steering: one embedding per inject layer
        self.steering_emb = nn.ModuleList([
            nn.Embedding(C_SIZE, D_MODEL) for _ in inject_layers])
        for emb in self.steering_emb:
            nn.init.zeros_(emb.weight)

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

        # Clear stale hooks from previous wrapper instances on the same layers
        target_layers = set([self.read_layer] + list(self.inject_layers))
        for li in target_layers:
            layer = self.model.model.layers[li]
            stale = [k for k, v in layer._forward_hooks.items()
                     if 'StackedAbstractionWrapperV7' in repr(v)]
            for k in stale:
                del layer._forward_hooks[k]

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

        # Extract codes from routing dimensions
        if self.routing_mode == "similar_magnitude":
            # Use pre-selected dimensions with uniform importance
            routing_hidden = hidden[..., self._routing_dims]  # (B, S, C_SIZE)
        else:  # diagonal
            # Use last C_SIZE dimensions
            routing_hidden = hidden[..., -self.C_SIZE:]  # (B, S, C_SIZE)

        pos_codes = routing_hidden.argmax(dim=-1)  # (B, S)

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
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output

        B, S, D = hidden.shape
        # Guard: skip if shape mismatch (e.g. generation beyond prompt)
        if S != self._chunk_codes.shape[1]:
            return output

        emb = self.steering_emb[hook_idx]

        mask = self._chunk_codes >= 0
        if mask.any():
            safe = self._chunk_codes.clamp(min=0)
            sv = emb(safe) * mask.unsqueeze(-1).float() * self.scale
            hidden = hidden + sv.to(hidden.dtype)

        if is_tuple:
            return (hidden,) + output[1:]
        return hidden

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
                'read_layer': self.read_layer,
                'code_position': self.code_position,
                'routing_mode': self.routing_mode,
                '_routing_dims': self._routing_dims,
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
