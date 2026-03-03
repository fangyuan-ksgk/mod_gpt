"""
SorlModelWrapperV2: Decoupled NL/abstract token heads.

Key difference from SorlModelWrapper (v1):
  - Does NOT call resize_token_embeddings. The base model's lm_head and
    embed_tokens stay at base_vocab_size — NL generation is never degraded.
  - Adds separate abs_embed (nn.Embedding) and abs_head (nn.Linear) for
    abstract tokens. These are the only new trainable parameters.
  - forward() returns concatenated [nl_logits, abs_logits] for backward
    compatibility, but they come from independent projection heads so
    softmax over NL tokens never competes with abstract tokens.

Drop-in replacement: same API as SorlModelWrapper (same method signatures).
"""

from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig, AutoConfig
from transformers import AutoModelForCausalLM
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
import torch.nn as nn
import torch

from sorl.sorl_wrapper import infer_level, SUPPORTED_MODELS


class SorlModelWrapperV2(PreTrainedModel, GenerationMixin):
    config_class = PretrainedConfig

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)

        model_type = getattr(config, "model_type", None)
        if model_type not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.model_type = model_type
        self.model = AutoModelForCausalLM.from_config(config)
        self.full_vocab_size_list = None
        self.abs_embed = None
        self.abs_head = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, abstract_vocab_size_list: List[int], **kwargs) -> "SorlModelWrapperV2":
        config = AutoConfig.from_pretrained(model_name_or_path, **kwargs)

        wrapper = cls(config)
        wrapper.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **kwargs
        )

        base_vocab_size = config.vocab_size
        wrapper.full_vocab_size_list = [base_vocab_size] + abstract_vocab_size_list
        wrapper._setup_vocabulary()
        wrapper._setup_abstract_heads()
        # NOTE: we do NOT call resize_token_embeddings
        return wrapper

    @classmethod
    def from_scratch(cls, config: PretrainedConfig, full_vocab_size_list: List[int], pad_token_id: int) -> "SorlModelWrapperV2":
        wrapper = cls(config)
        wrapper.full_vocab_size_list = full_vocab_size_list
        wrapper._setup_vocabulary()
        wrapper._setup_abstract_heads()
        return wrapper

    # ------------------------------------------------------------------
    # Abstract token heads (decoupled from base lm_head)
    # ------------------------------------------------------------------
    def _setup_abstract_heads(self):
        """Create separate embedding and lm_head for abstract tokens."""
        hidden_size = self.config.hidden_size
        abs_vocab_size = int(self.vocab_sizes[1].item()) if len(self.vocab_sizes) > 1 else 0
        if abs_vocab_size > 0:
            self.abs_embed = nn.Embedding(abs_vocab_size, hidden_size)
            self.abs_head = nn.Linear(hidden_size, abs_vocab_size, bias=False)
            nn.init.normal_(self.abs_embed.weight, std=0.02)
            nn.init.normal_(self.abs_head.weight, std=0.02)

    # ------------------------------------------------------------------
    # Embedding routing
    # ------------------------------------------------------------------
    def _get_inputs_embeds(self, input_ids):
        """Route NL token IDs through base embed_tokens, abstract IDs through abs_embed."""
        base_vocab = int(self.vocab_sizes[0].item())

        # self.model = Qwen2ForCausalLM, self.model.model = Qwen2Model, .embed_tokens = Embedding
        embed_layer = self.model.model.embed_tokens

        if self.abs_embed is None:
            return embed_layer(input_ids)

        nl_ids = input_ids.clamp(max=base_vocab - 1)
        abs_ids = (input_ids - base_vocab).clamp(min=0, max=self.abs_embed.num_embeddings - 1)

        nl_embeds = embed_layer(nl_ids)
        abs_embeds = self.abs_embed(abs_ids)

        is_abstract = (input_ids >= base_vocab).unsqueeze(-1)
        return torch.where(is_abstract, abs_embeds, nl_embeds)

    # ------------------------------------------------------------------
    # Logit heads
    # ------------------------------------------------------------------
    def _compute_logits(self, hidden_states):
        """Compute NL logits from base lm_head, abstract logits from abs_head.
        
        Returns concatenated (B, L, base_vocab + abs_vocab) but from independent heads.
        """
        nl_logits = self.model.lm_head(hidden_states)

        if self.abs_head is not None:
            abs_logits = self.abs_head(hidden_states)
            return torch.cat([nl_logits, abs_logits], dim=-1)
        return nl_logits

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, input_ids, attention_mask=None, memory_span_abs=1792, memory_span_traj=1792, **kwargs):
        from torch.nn.attention.flex_attention import create_block_mask

        sorl_block_mask = self._create_sorl_block_mask(input_ids, memory_span_abs, memory_span_traj)

        if self.training:
            kwargs.setdefault("use_cache", False)

        inputs_embeds = self._get_inputs_embeds(input_ids)

        outputs = self.model.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            block_mask=sorl_block_mask,
            output_hidden_states=True,
            **kwargs
        )

        hidden_states = outputs.hidden_states[-1]
        outputs.logits = self._compute_logits(hidden_states)
        return outputs

    # ------------------------------------------------------------------
    # Block mask (identical to v1)
    # ------------------------------------------------------------------
    def _create_sorl_block_mask(self, input_ids: torch.Tensor, memory_span_abs: int = 1792, memory_span_traj: int = 1792):
        from torch.nn.attention.flex_attention import create_block_mask

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        doc_boundary_token_id = self._get_doc_boundary_token_id()
        docs = (input_ids == doc_boundary_token_id).cumsum(1)
        levels = infer_level(input_ids, self.vocab_sizes)
        accum_levels = levels.cumsum(1)

        max_idx = seq_len - 1

        def _safe_idx(i):
            if torch.is_tensor(i):
                return torch.clamp(i, min=0, max=max_idx)
            return max(0, min(i, max_idx))

        def sorl_mask_fn(b, h, q_idx, kv_idx):
            in_bounds = (q_idx >= 0) & (q_idx < seq_len) & (kv_idx >= 0) & (kv_idx < seq_len)
            q_safe = _safe_idx(q_idx)
            kv_safe = _safe_idx(kv_idx)

            causal_mask = q_idx >= kv_idx
            document_mask = docs[b, q_safe] == docs[b, kv_safe]
            window_mask = q_idx - kv_idx < 1792

            to_abstract = levels[b, kv_safe] > 0
            from_abstract = levels[b, q_safe] > 0
            skip_abs = accum_levels[b, q_safe] > accum_levels[b, kv_safe]

            traj_memory_span = (q_idx - kv_idx) <= memory_span_traj
            abs_memory_span = (q_idx - kv_idx) <= memory_span_abs

            memory_compression_mask = (
                to_abstract |
                (from_abstract & abs_memory_span) |
                (~from_abstract & traj_memory_span & ~skip_abs)
            )

            return in_bounds & causal_mask & document_mask & window_mask & memory_compression_mask

        block_mask = create_block_mask(
            sorl_mask_fn,
            B=batch_size,
            H=self.model.config.num_attention_heads,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=device,
        )
        return block_mask

    def _get_doc_boundary_token_id(self) -> int:
        doc_id = getattr(self.config, "doc_boundary_token_id", None)
        if doc_id is None:
            doc_id = getattr(self.model.config, "bos_token_id", None)
        if doc_id is None:
            doc_id = getattr(self.model.config, "eos_token_id", None)
        if isinstance(doc_id, (list, tuple)):
            doc_id = doc_id[0] if len(doc_id) > 0 else None
        if doc_id is None:
            doc_id = 50256
        return int(doc_id)

    # ------------------------------------------------------------------
    # Vocabulary setup (identical to v1)
    # ------------------------------------------------------------------
    def _setup_vocabulary(self):
        device = self.device

        base_vocab_size = self.full_vocab_size_list[0]
        abstract_vocab_size = self.full_vocab_size_list[1] if len(self.full_vocab_size_list) > 1 else 0

        vocab_sizes_list = [base_vocab_size, abstract_vocab_size + 1] if abstract_vocab_size > 0 else [base_vocab_size]
        self.register_buffer("vocab_sizes", torch.tensor(vocab_sizes_list, device=device))

        self.total_vocab_size = self.vocab_sizes.sum()

        level_starts = torch.cat([torch.tensor([0]), torch.cumsum(self.vocab_sizes, dim=0)[:-1] + 1])
        level_ends = torch.cumsum(self.vocab_sizes, dim=0)
        self.register_buffer("level_starts", level_starts)
        self.register_buffer("level_ends", level_ends)

        l0_mask = torch.zeros(self.total_vocab_size.item(), dtype=torch.bool, device=device)
        l0_mask[:self.vocab_sizes[0]] = True
        self.register_buffer("l0_mask", l0_mask)

        abs_mask = ~self.l0_mask
        abs_mask[self.vocab_sizes[0]] = False
        self.register_buffer("abs_mask", abs_mask)

    # ------------------------------------------------------------------
    # Generation (uses split heads properly)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        temperature: float = 0.7,
        top_k: int = 50,
        K: Optional[int] = None,
        free_form: bool = False,
        memory_span_abs: int = 1792,
        memory_span_traj: int = 1792,
    ):
        self.model.eval()
        generated_ids = input_ids.clone()
        levels_cache = infer_level(generated_ids, self.vocab_sizes)
        masks = torch.stack([self.l0_mask, self.abs_mask], dim=0).to(generated_ids.device)

        for _ in range(max_new_tokens):
            outputs = self.forward(
                input_ids=generated_ids,
                memory_span_abs=memory_span_abs,
                memory_span_traj=memory_span_traj,
            )
            next_token_logits = outputs.logits[:, -1, :]

            if not free_form:
                if K is None:
                    next_token_level = torch.zeros(generated_ids.size(0), dtype=torch.long, device=generated_ids.device)
                else:
                    next_token_level = 1 - (levels_cache[:, -K:] > 0).any(dim=-1).long()

                allowed_mask = masks[next_token_level]
                next_token_logits = next_token_logits.masked_fill(~allowed_mask, -float("inf"))
            else:
                placeholder_id = self.vocab_sizes[0].item()
                next_token_logits[:, placeholder_id] = -float("inf")

            if temperature > 0:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                topk_probs, topk_idx = torch.topk(probs, min(top_k, probs.size(-1)), dim=-1)
                next_token_id = torch.gather(topk_idx, dim=1, index=torch.multinomial(topk_probs, 1))
            else:
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            if free_form:
                next_token_level = (next_token_id.squeeze(-1) > self.vocab_sizes[0].item()).long()

            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            levels_cache = torch.cat([levels_cache, next_token_level[:, None]], dim=1)

        return generated_ids

    @torch.no_grad()
    def generate_inner_cot(
        self,
        input_ids: torch.LongTensor,
        n_inner_cot_tokens: int = 8,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_k: int = 50,
        memory_span_abs: int = 1792,
        memory_span_traj: int = 1792,
    ):
        """Inner-CoT generation: two-phase autoregressive decoding."""
        self.model.eval()
        generated_ids = input_ids.clone()
        vocab_size_0 = self.vocab_sizes[0].item()
        eos_token_id = getattr(self.model.config, "eos_token_id", None)
        if isinstance(eos_token_id, (list, tuple)):
            eos_token_id = eos_token_id[0] if eos_token_id else None

        # Phase 1: generate abstract tokens
        for _ in range(n_inner_cot_tokens):
            outputs = self.forward(
                input_ids=generated_ids,
                memory_span_abs=memory_span_abs,
                memory_span_traj=memory_span_traj,
            )
            logits = outputs.logits[:, -1, :]
            logits[:, :vocab_size_0 + 1] = -float("inf")
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                topk_probs, topk_idx = torch.topk(probs, min(top_k, probs.size(-1)), dim=-1)
                next_id = torch.gather(topk_idx, dim=1, index=torch.multinomial(topk_probs, 1))
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_id], dim=1)

        # Phase 2: generate NL answer tokens
        for _ in range(max_new_tokens):
            outputs = self.forward(
                input_ids=generated_ids,
                memory_span_abs=memory_span_abs,
                memory_span_traj=memory_span_traj,
            )
            logits = outputs.logits[:, -1, :]
            logits[:, vocab_size_0:] = -float("inf")
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                topk_probs, topk_idx = torch.topk(probs, min(top_k, probs.size(-1)), dim=-1)
                next_id = torch.gather(topk_idx, dim=1, index=torch.multinomial(topk_probs, 1))
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_id], dim=1)
            if eos_token_id is not None and (next_id == eos_token_id).all():
                break

        return generated_ids

    # ------------------------------------------------------------------
    # Abstract-only generation
    # ------------------------------------------------------------------
    def generate_abstract_only(self, idx, attention_mask, memory_span_abs=1792, memory_span_traj=1792, temperature=0.0, prompt_len=None):
        vocab_size_0 = self.vocab_sizes[0].to(idx.device)
        abs_mask = (idx >= vocab_size_0)
        abs_mask[:, 0] = False

        abs_cols = abs_mask[0].nonzero(as_tuple=True)[0]

        if isinstance(temperature, torch.Tensor) and temperature.ndim == 1:
            temp_batch = temperature.float().clamp(min=1e-10)
        else:
            temp_batch = None
            scalar_temp = max(float(temperature), 1e-10)

        for col in abs_cols:
            outputs = self.forward(
                input_ids=idx,
                attention_mask=attention_mask,
                memory_span_abs=memory_span_abs,
                memory_span_traj=memory_span_traj,
            )
            pred_pos = col - 1
            logits_at = outputs.logits[:, pred_pos, :]

            logits_at[:, :vocab_size_0 + 1] = float('-inf')

            if temp_batch is not None:
                probs = F.softmax(logits_at / temp_batch.unsqueeze(1), dim=-1)
            else:
                probs = F.softmax(logits_at / scalar_temp, dim=-1)
            new_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            idx[:, col] = new_token.to(idx.dtype)

        # Final forward for per-token loss
        outputs = self.forward(
            input_ids=idx,
            attention_mask=attention_mask,
            memory_span_abs=memory_span_abs,
            memory_span_traj=memory_span_traj,
        )
        labels = idx.clone()
        labels[attention_mask == 0] = -100
        if prompt_len is not None:
            seq_idx = torch.arange(labels.size(1), device=labels.device).unsqueeze(0)
            labels[seq_idx < prompt_len.unsqueeze(1)] = -100

        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        per_token_loss = per_token_loss.view(idx.shape[0], -1)

        return idx, per_token_loss

    # ------------------------------------------------------------------
    # Extract and sample (recursion helper)
    # ------------------------------------------------------------------
    def extract_and_sample(self, logits, idx, recursion_mask, temperature):
        predict_mask = torch.roll(recursion_mask, -1, dims=1)
        predict_mask[:, -1] = False
        recursion_logits = logits[predict_mask]

        abstract_start = self.vocab_sizes[0].to(logits.device)
        recursion_logits[:, :abstract_start + 1] = float('-inf')

        if isinstance(temperature, torch.Tensor) and temperature.ndim > 0:
            recursion_temp = temperature[predict_mask]
        else:
            recursion_temp = temperature

        temp = torch.clamp(recursion_temp, min=1e-10).view(-1, 1) if isinstance(recursion_temp, torch.Tensor) else max(temperature, 1e-10)

        probs = F.softmax(recursion_logits / temp, dim=-1)
        new_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

        idx[recursion_mask] = new_tokens.to(idx.dtype)
        return idx

    # ------------------------------------------------------------------
    # Recursion
    # ------------------------------------------------------------------
    def recursion(self, idx, attention_mask, max_iterations=5, memory_span_abs=1792, memory_span_traj=1792, attn_blocksize=1792, temperature=0.0, prompt_len=None):
        vocab_size_0 = self.vocab_sizes[0].to(idx.device)
        recursion_mask = (idx >= vocab_size_0)
        recursion_mask[:, 0] = False

        if isinstance(temperature, torch.Tensor) and temperature.ndim == 1:
            temp_expanded = temperature.view(-1, 1).expand_as(idx)
        else:
            temp_expanded = temperature

        for _ in range(max_iterations):
            outputs = self.forward(
                input_ids=idx,
                attention_mask=attention_mask,
                memory_span_abs=memory_span_abs,
                memory_span_traj=memory_span_traj,
            )
            logits = outputs.logits
            idx = self.extract_and_sample(logits, idx, recursion_mask, temp_expanded)

        # Evaluation forward
        labels = idx.clone()
        labels[attention_mask == 0] = -100
        if prompt_len is not None:
            seq_idx = torch.arange(labels.size(1), device=labels.device).unsqueeze(0)
            labels[seq_idx < prompt_len.unsqueeze(1)] = -100

        outputs = self.forward(
            input_ids=idx,
            attention_mask=attention_mask,
            memory_span_abs=memory_span_abs,
            memory_span_traj=memory_span_traj,
        )

        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        per_token_loss = per_token_loss.view(idx.shape[0], -1)

        return idx, per_token_loss
