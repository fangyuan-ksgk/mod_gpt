from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig, AutoConfig
from transformers import AutoModelForCausalLM, AutoConfig as TransformersAutoConfig
from typing import List, Optional, Tuple
import torch.nn.functional as F
import torch.nn as nn
import torch

def left_pad_and_mask(
    sequences: List[torch.Tensor], pad_id: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Left-pad a list of 1-D token tensors to equal length and return (input_ids, attention_mask).

    Args:
        sequences: list of 1-D LongTensors, each of shape (seq_len_i,).
        pad_id: token id used for padding (default 0).

    Returns:
        input_ids:      (B, max_len) left-padded token ids.
        attention_mask:  (B, max_len) with 1 for real tokens and 0 for padding.
    """
    lengths = torch.tensor([s.size(0) for s in sequences])
    max_len = lengths.max().item()
    batch_size = len(sequences)
    device = sequences[0].device
    dtype = sequences[0].dtype

    input_ids = torch.full((batch_size, max_len), pad_id, dtype=dtype, device=device)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)

    # Vectorised fill: compute pad lengths, then scatter each sequence into the right-aligned slot
    pad_lens = max_len - lengths  # (B,)
    for i, (seq, pl) in enumerate(zip(sequences, pad_lens)):
        input_ids[i, pl:] = seq
        attention_mask[i, pl:] = 1

    return input_ids, attention_mask


SUPPORTED_MODELS = {
    "qwen2": "qwen2",
    "qwen3": "qwen3",  # Use AutoModelForCausalLM for both
    "llama": "llama",  # Llama-3.x (same embed_tokens/lm_head layout as Qwen)
}

def infer_level(indices: torch.Tensor, vocab_sizes: torch.Tensor):
    if indices.dtype in [torch.uint8, torch.uint16, torch.uint32, torch.uint64]:
        indices = indices.long()
    vocab_sizes = vocab_sizes.to(indices.device)
    levels = (indices >= vocab_sizes[0]).int()  # Fixed: use >= instead of >
    return levels

# Issue #1. pad_token_id needs to be removed, we do NOT do packed training in post-training time
# Issue #2. bottleneck block mask needs to be implemented, current compression mask is not ideal

class SorlModelWrapper(PreTrainedModel, GenerationMixin):
    config_class = PretrainedConfig

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        
        model_type = getattr(config, "model_type", None)
        if model_type not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.model_type = model_type
        
        # Use AutoModelForCausalLM to load the correct model class
        self.model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
        self.full_vocab_size_list = None
        self.use_memory_compression = False  # set True to enable bottleneck block mask
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, abstract_vocab_size_list: List[int], untie_embeddings: bool = False, **kwargs) -> "SorlModelWrapper":
        config = AutoConfig.from_pretrained(model_name_or_path, **kwargs)
        
        wrapper = cls(config)
        wrapper.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            attn_implementation="eager",
            **kwargs
        )

        base_vocab_size = config.vocab_size
        wrapper.full_vocab_size_list = [base_vocab_size] + abstract_vocab_size_list
        wrapper._setup_vocabulary()
        
        new_total_vocab_size = wrapper.total_vocab_size.item()
        if wrapper.model.config.vocab_size != new_total_vocab_size:
            wrapper.model.resize_token_embeddings(new_total_vocab_size)
            wrapper.model.config.vocab_size = new_total_vocab_size
            wrapper.config.vocab_size = new_total_vocab_size
            # Optionally untie lm_head from embed_tokens so abstract rows are independent
            # parameters. Qwen3 has tie_word_embeddings=True; resize_token_embeddings
            # re-ties them, making lm_head.weight IS embed_tokens.weight (same storage).
            # This prevents abstract embed rows from being trained separately.
            if untie_embeddings and getattr(wrapper.model.config, "tie_word_embeddings", False):
                wrapper.model.lm_head.weight = nn.Parameter(
                    wrapper.model.model.embed_tokens.weight.detach().clone()
                )
                wrapper.model.config.tie_word_embeddings = False
            wrapper._init_abstract_embeddings_orthogonal()

        return wrapper
    
    @classmethod
    def from_scratch(cls, config: PretrainedConfig, full_vocab_size_list: List[int], pad_token_id: int) -> "SorlModelWrapper":
        """A custom initializer for creating a SORL model from scratch."""
        wrapper = cls(config)
        wrapper.full_vocab_size_list = full_vocab_size_list
        wrapper._setup_vocabulary()
        
        new_total_vocab_size = wrapper.total_vocab_size.item()
        wrapper.model.resize_token_embeddings(new_total_vocab_size)
        wrapper.config.vocab_size = new_total_vocab_size
        wrapper._init_abstract_embeddings_orthogonal()
        return wrapper

    @torch.no_grad()
    def _init_abstract_embeddings_orthogonal(self):
        """Reinitialize abstract embedding rows with orthogonal vectors.

        After resize_token_embeddings (mean_resizing), abstract rows are
        near-identical samples from N(mu, Sigma).  This creates a symmetry
        trap for ortho_loss: all rows receive the same gradient direction,
        so they never diverge.  Orthogonal init breaks this symmetry.
        """
        base_vocab = int(self.vocab_sizes[0].item())
        n_abs = int(self.total_vocab_size.item()) - base_vocab  # includes placeholder
        if n_abs <= 1:
            return

        embed_w = self.model.model.embed_tokens.weight
        lm_head_w = self.model.lm_head.weight
        hidden = embed_w.shape[1]

        # Measure scale from existing base embeddings
        base_norm = embed_w[:base_vocab].norm(dim=1).mean().item()

        # Generate orthogonal matrix (n_abs x hidden), scale to match base norms
        ortho = torch.empty(max(n_abs, hidden), hidden, device=embed_w.device)
        nn.init.orthogonal_(ortho)
        ortho = ortho[:n_abs] * base_norm

        embed_w[base_vocab:] = ortho
        lm_head_w[base_vocab:] = ortho

    @property
    def has_separate_abs_params(self) -> bool:
        """Whether abstract embeddings/projections are separate from NL params."""
        return False

    def forward(self, input_ids, attention_mask=None, memory_span_abs=1792, memory_span_traj=1792, **kwargs):
        sorl_attention_mask = self._create_sorl_attention_mask(
            input_ids, attention_mask, memory_span_abs, memory_span_traj
        )
        if self.training:
            kwargs.setdefault("use_cache", False)
        kwargs.pop("block_mask", None)
        return self.model.forward(input_ids=input_ids, attention_mask=sorl_attention_mask, **kwargs)

    def _create_sorl_attention_mask(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, memory_span_abs: int = 1792, memory_span_traj: int = 1792):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        doc_boundary_token_id = self._get_doc_boundary_token_id()
        docs = (input_ids == doc_boundary_token_id).cumsum(1)
        levels = infer_level(input_ids, self.vocab_sizes)
        accum_levels = levels.cumsum(1)
        pos = torch.arange(seq_len, device=device)
        q_pos = pos.view(1, seq_len, 1)
        kv_pos = pos.view(1, 1, seq_len)

        causal_mask = q_pos >= kv_pos
        document_mask = docs[:, :, None] == docs[:, None, :]
        window_mask = (q_pos - kv_pos) < 1792

        to_abstract = levels[:, None, :] > 0
        from_abstract = levels[:, :, None] > 0
        skip_abs = accum_levels[:, :, None] > accum_levels[:, None, :]

        traj_memory_span = (q_pos - kv_pos) <= memory_span_traj
        abs_memory_span = (q_pos - kv_pos) <= memory_span_abs

        if self.use_memory_compression:
            memory_compression_mask = (
                to_abstract |
                (from_abstract & abs_memory_span) |
                (~from_abstract & traj_memory_span & ~skip_abs)
            )
            allowed = causal_mask & document_mask & window_mask & memory_compression_mask
        else:
            allowed = causal_mask & document_mask & window_mask

        if attention_mask is not None:
            if attention_mask.dim() == 4:
                return attention_mask
            key_valid = attention_mask.to(torch.bool)[:, None, :]
            allowed = allowed & key_valid

        dtype = self.model.get_input_embeddings().weight.dtype
        additive_mask = torch.zeros((batch_size, 1, seq_len, seq_len), dtype=dtype, device=device)
        additive_mask = additive_mask.masked_fill(~allowed[:, None, :, :], torch.finfo(dtype).min)
        return additive_mask

    def _get_doc_boundary_token_id(self) -> int:
        # Prefer explicit override, then model config defaults.
        doc_id = getattr(self.config, "doc_boundary_token_id", None)
        if doc_id is None:
            doc_id = getattr(self.model.config, "bos_token_id", None)
        if doc_id is None:
            doc_id = getattr(self.model.config, "eos_token_id", None)

        # Keep backward-compatible behavior if config is missing this info.
        if isinstance(doc_id, (list, tuple)):
            doc_id = doc_id[0] if len(doc_id) > 0 else None
        if doc_id is None:
            doc_id = 50256
        return int(doc_id)


    def _setup_vocabulary(self):
        device = self.device
        
        # Single abstraction level - base + abstract
        base_vocab_size = self.full_vocab_size_list[0]
        abstract_vocab_size = self.full_vocab_size_list[1] if len(self.full_vocab_size_list) > 1 else 0
        
        # Register vocab sizes: [base, abstract+1]
        vocab_sizes_list = [base_vocab_size, abstract_vocab_size + 1] if abstract_vocab_size > 0 else [base_vocab_size]
        self.register_buffer("vocab_sizes", torch.tensor(vocab_sizes_list, device=device))
        
        self.total_vocab_size = self.vocab_sizes.sum()
        
        # Level starts and ends (skip mask token at position 0 of each level)
        level_starts = torch.cat([torch.tensor([0]), torch.cumsum(self.vocab_sizes, dim=0)[:-1] + 1])
        level_ends = torch.cumsum(self.vocab_sizes, dim=0)
        self.register_buffer("level_starts", level_starts)
        self.register_buffer("level_ends", level_ends)

        # Masks for base vs abstract tokens
        l0_mask = torch.zeros(self.total_vocab_size.item(), dtype=torch.bool, device=device)
        l0_mask[:self.vocab_sizes[0]] = True
        self.register_buffer("l0_mask", l0_mask)
        
        abs_mask = ~self.l0_mask
        abs_mask[self.vocab_sizes[0]] = False  # exclude placeholder/mask token
        self.register_buffer("abs_mask", abs_mask)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        attention_mask: Optional[torch.LongTensor] = None,
        temperature: float = 0.7,
        top_k: int = 50,
        K: Optional[int] = None,
        free_form: bool = False,
        memory_span_abs: int = 1792,
        memory_span_traj: int = 1792,
        response_only_abs: bool = False,
        cot_only_abs: bool = False,
        answer_token_id: int = 820,
        abs_prefix_max: Optional[int] = None,
    ):
        import torch.nn.functional as F
        
        self.model.eval()
        generated_ids = input_ids.clone()
        if attention_mask is None:
            attention_mask = torch.ones_like(generated_ids)
        else:
            attention_mask = attention_mask.clone()
        levels_cache = infer_level(generated_ids, self.vocab_sizes)
        masks = torch.stack([self.l0_mask, self.abs_mask], dim=0).to(generated_ids.device)

        # For response_only_abs / cot_only_abs: track NL tokens since last abstract.
        nl_since_abs = torch.zeros(generated_ids.size(0), dtype=torch.long, device=generated_ids.device)
        # For cot_only_abs: track whether the answer delimiter has been emitted per sequence.
        past_answer = torch.zeros(generated_ids.size(0), dtype=torch.bool, device=generated_ids.device)
        # For abs_prefix_max: count ABS tokens generated in the response so far.
        n_response_abs = torch.zeros(generated_ids.size(0), dtype=torch.long, device=generated_ids.device)

        use_response_counter = (response_only_abs or cot_only_abs) and K is not None
        use_abs_prefix = abs_prefix_max is not None and K is not None

        for _ in range(max_new_tokens):
            sorl_attention_mask = self._create_sorl_attention_mask(
                generated_ids, attention_mask, memory_span_abs, memory_span_traj
            )
            outputs = self.model.forward(
                input_ids=generated_ids,
                attention_mask=sorl_attention_mask,
                use_cache=False,
            )
            next_token_logits = outputs.logits[:, -1, :]

            if use_abs_prefix:
                # Phase 1: force ABS token until abs_prefix_max abs tokens generated.
                # Phase 2: free-form NL after that (mask out placeholder only).
                in_prefix = n_response_abs < abs_prefix_max
                next_token_level = in_prefix.long()  # 1=ABS, 0=NL
                allowed_mask = masks[next_token_level]
                next_token_logits = next_token_logits.masked_fill(~allowed_mask, -float("inf"))
            elif not free_form:
                # Periodic: force abstract token every K trajectory tokens
                if K is None:
                    next_token_level = torch.zeros(generated_ids.size(0), dtype=torch.long, device=generated_ids.device)
                elif use_response_counter:
                    # Count NL tokens generated since start of response (or last abstract).
                    # Force abstract after K NL tokens, matching training pattern.
                    # cot_only_abs: also suppress abstract insertion once answer delimiter seen.
                    want_abs = (nl_since_abs >= K) & (~past_answer if cot_only_abs else torch.ones_like(past_answer))
                    next_token_level = want_abs.long()
                else:
                    next_token_level = 1 - (levels_cache[:, -K:] > 0).any(dim=-1).long()

                # Apply level-based masking
                allowed_mask = masks[next_token_level]  # True = allowed tokens
                next_token_logits = next_token_logits.masked_fill(~allowed_mask, -float("inf"))
            else:
                # Free-form: only mask out the placeholder token at vocab_sizes[0]
                placeholder_id = self.vocab_sizes[0].item()
                next_token_logits[:, placeholder_id] = -float("inf")

            # Sample next token
            if temperature > 0:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                topk_probs, topk_idx = torch.topk(probs, min(top_k, probs.size(-1)), dim=-1)
                next_token_id = torch.gather(topk_idx, dim=1, index=torch.multinomial(topk_probs, 1))
            else:
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Infer level from sampled token for levels_cache
            if free_form:
                next_token_level = (next_token_id.squeeze(-1) > self.vocab_sizes[0].item()).long()

            # Update counters
            if use_abs_prefix:
                is_abs = (next_token_level > 0)
                n_response_abs = n_response_abs + is_abs.long()
            if use_response_counter:
                is_abs = (next_token_level > 0)
                nl_since_abs = torch.where(is_abs, torch.zeros_like(nl_since_abs), nl_since_abs + 1)
            if cot_only_abs:
                hit_answer = (next_token_id.squeeze(-1) == answer_token_id)
                past_answer = past_answer | hit_answer

            # Update sequences
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones(generated_ids.size(0), 1, dtype=attention_mask.dtype, device=attention_mask.device)], dim=1)
            levels_cache = torch.cat([levels_cache, next_token_level[:, None]], dim=1)

            # Release stale CUDA reserved blocks from variable-length forward passes
            torch.cuda.empty_cache()

        return generated_ids

    @torch.no_grad()
    def generate_inner_cot(
        self,
        input_ids: torch.LongTensor,
        n_inner_cot_tokens: int = 8,
        max_new_tokens: int = 128,
        attention_mask: Optional[torch.LongTensor] = None,
        temperature: float = 0.0,
        top_k: int = 50,
        memory_span_abs: int = 1792,
        memory_span_traj: int = 1792,
    ):
        """
        Inner-CoT generation: two-phase autoregressive decoding.
        """
        self.model.eval()
        generated_ids = input_ids.clone()
        if attention_mask is None:
            attention_mask = torch.ones_like(generated_ids)
        else:
            attention_mask = attention_mask.clone()
        vocab_size_0 = self.vocab_sizes[0].item()
        eos_token_id = getattr(self.model.config, "eos_token_id", None)
        if isinstance(eos_token_id, (list, tuple)):
            eos_token_id = eos_token_id[0] if eos_token_id else None

        # Phase 1: generate abstract tokens
        for _ in range(n_inner_cot_tokens):
            sorl_attention_mask = self._create_sorl_attention_mask(
                generated_ids, attention_mask, memory_span_abs, memory_span_traj
            )
            outputs = self.model.forward(
                input_ids=generated_ids, attention_mask=sorl_attention_mask, use_cache=False,
            )
            logits = outputs.logits[:, -1, :]
            # Mask base vocab — only allow abstract tokens
            logits[:, :vocab_size_0 + 1] = -float("inf")
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                topk_probs, topk_idx = torch.topk(probs, min(top_k, probs.size(-1)), dim=-1)
                next_id = torch.gather(topk_idx, dim=1, index=torch.multinomial(topk_probs, 1))
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_id], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones(generated_ids.size(0), 1, dtype=attention_mask.dtype, device=attention_mask.device)], dim=1)

        # Phase 2: generate NL answer tokens
        for _ in range(max_new_tokens):
            sorl_attention_mask = self._create_sorl_attention_mask(
                generated_ids, attention_mask, memory_span_abs, memory_span_traj
            )
            outputs = self.model.forward(
                input_ids=generated_ids, attention_mask=sorl_attention_mask, use_cache=False,
            )
            logits = outputs.logits[:, -1, :]
            # Mask abstract vocab — only allow base (NL) tokens
            logits[:, vocab_size_0:] = -float("inf")
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                topk_probs, topk_idx = torch.topk(probs, min(top_k, probs.size(-1)), dim=-1)
                next_id = torch.gather(topk_idx, dim=1, index=torch.multinomial(topk_probs, 1))
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_id], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones(generated_ids.size(0), 1, dtype=attention_mask.dtype, device=attention_mask.device)], dim=1)
            # Stop on EOS
            if eos_token_id is not None and (next_id == eos_token_id).all():
                break

        return generated_ids

    def generate_abstract_only(self, idx, attention_mask, memory_span_abs=1792, memory_span_traj=1792, temperature=0.0, prompt_len=None):

        vocab_size_0 = self.vocab_sizes[0].to(idx.device)
        abs_mask = (idx >= vocab_size_0)  # (B, L)
        abs_mask[:, 0] = False

        abs_cols = abs_mask[0].nonzero(as_tuple=True)[0]  # (n_abs,)

        if isinstance(temperature, torch.Tensor) and temperature.ndim == 1:
            temp_batch = temperature.float().clamp(min=1e-10)  # (B,)
        else:
            temp_batch = None
            scalar_temp = max(float(temperature), 1e-10)

        for col in abs_cols:
            sorl_attention_mask = self._create_sorl_attention_mask(
                idx, attention_mask, memory_span_abs, memory_span_traj
            )
            outputs = self.model.forward(
                input_ids=idx, attention_mask=sorl_attention_mask,
                use_cache=False,
            )
            pred_pos = col - 1  # next-token prediction: logits[t] predicts token[t+1]
            logits_at = outputs.logits[:, pred_pos, :]  # (B, V)

            logits_at[:, :vocab_size_0 + 1] = float('-inf')

            if temp_batch is not None:
                probs = F.softmax(logits_at / temp_batch.unsqueeze(1), dim=-1)
            else:
                probs = F.softmax(logits_at / scalar_temp, dim=-1)
            new_token = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B,)
            idx[:, col] = new_token.to(idx.dtype)

        sorl_attention_mask = self._create_sorl_attention_mask(
            idx, attention_mask, memory_span_abs, memory_span_traj
        )
        labels = idx.clone()
        labels[attention_mask == 0] = -100
        if prompt_len is not None:
            seq_idx = torch.arange(labels.size(1), device=labels.device).unsqueeze(0)
            labels[seq_idx < prompt_len.unsqueeze(1)] = -100

        outputs = self.model.forward(
            input_ids=idx, attention_mask=sorl_attention_mask,
            use_cache=False,
        )
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        per_token_loss = per_token_loss.view(idx.shape[0], -1)

        return idx, per_token_loss

    def extract_and_sample(self, logits, idx, recursion_mask, temperature):
        """Extract masked logits and sample new tokens."""
        
        predict_mask = torch.roll(recursion_mask, -1, dims=1)
        predict_mask[:, -1] = False
        recursion_logits = logits[predict_mask]
        
        # Mask out base tokens (only allow abstract tokens) - ensure device consistency
        abstract_start = self.vocab_sizes[0].to(logits.device)
        recursion_logits[:, :abstract_start + 1] = float('-inf')
        
        # Handle temperature
        if isinstance(temperature, torch.Tensor) and temperature.ndim > 0:
            recursion_temp = temperature[predict_mask]
        else:
            recursion_temp = temperature
            
        temp = torch.clamp(recursion_temp, min=1e-10).view(-1, 1) if isinstance(recursion_temp, torch.Tensor) else max(temperature, 1e-10)
        
        probs = F.softmax(recursion_logits / temp, dim=-1)
        new_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        idx[recursion_mask] = new_tokens.to(idx.dtype)
        return idx

    def recursion_step(self, idx, attention_mask, recursion_mask, temperature,
                       memory_span_abs=1792, memory_span_traj=1792, prompt_len=None):
        """Single recursion iteration for deep supervision.

        Returns (idx_detached, per_token_loss, logits):
          - idx_detached: updated sequence with new abstract tokens, detached for next step
          - per_token_loss: (B, L-1) per-token CE loss from this iteration's forward pass
          - logits: raw logits from this iteration

        Usage (deep supervision, matches HRM Fig. 4):
            recursion_mask = (idx >= model.vocab_sizes[0])
            recursion_mask[:, 0] = False
            for step in range(N_supervision):
                idx, loss, logits = model.recursion_step(
                    idx, attn, recursion_mask, temperature=0.0, prompt_len=pl)
                loss.mean().backward()
                opt.step(); opt.zero_grad()
        """
        sorl_attention_mask = self._create_sorl_attention_mask(
            idx, attention_mask, memory_span_abs, memory_span_traj
        )
        outputs = self.model.forward(
            input_ids=idx, attention_mask=sorl_attention_mask, use_cache=False,
        )
        logits = outputs.logits

        # Sample new abstract tokens (clone to avoid in-place modification of forward input)
        idx_new = self.extract_and_sample(logits, idx.clone(), recursion_mask, temperature)

        # Compute per-token loss (NL tokens only — v6 does not predict abstract tokens)
        labels = idx_new.clone()
        labels[attention_mask == 0] = -100
        labels[recursion_mask] = -100  # mask abstract token positions
        if prompt_len is not None:
            seq_idx = torch.arange(labels.size(1), device=labels.device).unsqueeze(0)
            labels[seq_idx < prompt_len.unsqueeze(1)] = -100

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        ).view(idx.shape[0], -1)

        return idx_new.detach(), per_token_loss, logits

    def recursion(self, idx, attention_mask, max_iterations=5, memory_span_abs=1792, memory_span_traj=1792, attn_blocksize=1792, temperature=0.0, prompt_len=None, differentiable=False):
        """Perform recursion on abstract tokens with information bottleneck mask.
        If differentiable=True, uses Straight-Through Estimator (STE) on the final iteration
        to allow gradients to flow back into the abstraction prediction logits.
        """
        # Ensure vocab_sizes is on the same device as idx
        vocab_size_0 = self.vocab_sizes[0].to(idx.device)
        total_vocab_size = self.total_vocab_size.item()
        
        recursion_mask = (idx >= vocab_size_0)
        recursion_mask[:, 0] = False
        
        predict_mask = torch.roll(recursion_mask, -1, dims=1)
        predict_mask[:, -1] = False
        
        # Expand temperature if needed
        if isinstance(temperature, torch.Tensor) and temperature.ndim == 1:
            temp_expanded = temperature.view(-1, 1).expand_as(idx)
        else:
            temp_expanded = temperature

        for it in range(max_iterations): 
            sorl_attention_mask = self._create_sorl_attention_mask(
                idx, attention_mask, memory_span_abs, memory_span_traj
            )
            outputs = self.model.forward(
                input_ids=idx, 
                attention_mask=sorl_attention_mask,
                use_cache=False,
            )
            logits = outputs.logits
            
            idx = self.extract_and_sample(logits, idx, recursion_mask, temp_expanded)
        
        # Evaluation — mask padding AND question tokens in labels
        labels = idx.clone()
        labels[attention_mask == 0] = -100
        if prompt_len is not None:
            seq_idx = torch.arange(labels.size(1), device=labels.device).unsqueeze(0)
            labels[seq_idx < prompt_len.unsqueeze(1)] = -100

        sorl_attention_mask = self._create_sorl_attention_mask(
            idx, attention_mask, memory_span_abs, memory_span_traj
        )
        
        outputs = self.model.forward(
            input_ids=idx, 
            attention_mask=sorl_attention_mask,
            use_cache=False,
        )

        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        per_token_loss = per_token_loss.view(idx.shape[0], -1)  # [batch_size, seq_len-1]
                    
        return idx, per_token_loss, outputs.logits


# ---------------------------------------------------------------------------
# V2: separate abstract embedding / projection
# ---------------------------------------------------------------------------

class _SplitEmbedding(nn.Module):
    """Drop-in ``nn.Embedding`` replacement that routes abstract token IDs
    to a separate embedding table while keeping NL embeddings untouched.

    NL tokens   (id < base_vocab) → ``nl_embed``
    Abstract    (id ≥ base_vocab) → ``abs_embed`` (remapped to 0-indexed)
    """

    def __init__(self, nl_embed: nn.Embedding, abs_embed: nn.Embedding, base_vocab: int):
        super().__init__()
        self.nl_embed = nl_embed
        self.abs_embed = abs_embed
        self.base_vocab = base_vocab
        self.embedding_dim = nl_embed.embedding_dim

    @property
    def weight(self):
        """Concatenated view for code that reads ``.weight`` (e.g. norm stats)."""
        return torch.cat([self.nl_embed.weight, self.abs_embed.weight], dim=0)

    @property
    def num_embeddings(self):
        return self.nl_embed.num_embeddings + self.abs_embed.num_embeddings

    def forward(self, input_ids):
        abs_mask = (input_ids >= self.base_vocab)
        safe_ids = input_ids.clone()
        safe_ids[abs_mask] = 0  # placeholder — will be overwritten
        embeds = self.nl_embed(safe_ids)
        if abs_mask.any():
            abs_ids = input_ids[abs_mask] - self.base_vocab
            embeds = embeds.clone()  # avoid in-place on nl_embed output
            embeds[abs_mask] = self.abs_embed(abs_ids)
        return embeds


class _SplitLMHead(nn.Module):
    """Drop-in ``nn.Linear`` replacement that appends abstract logits
    (from a separate projection) to the NL logits from the original head.
    """

    def __init__(self, nl_head: nn.Linear, abs_proj: nn.Linear):
        super().__init__()
        self.nl_head = nl_head
        self.abs_proj = abs_proj

    @property
    def weight(self):
        """Concatenated view for code that reads ``.weight``."""
        return torch.cat([self.nl_head.weight, self.abs_proj.weight], dim=0)

    @property
    def in_features(self):
        return self.nl_head.in_features

    @property
    def out_features(self):
        return self.nl_head.out_features + self.abs_proj.out_features

    def forward(self, hidden_states):
        nl_logits = self.nl_head(hidden_states)
        abs_logits = self.abs_proj(hidden_states)
        return torch.cat([nl_logits, abs_logits], dim=-1)


class SorlModelWrapperV2(SorlModelWrapper):
    """SoRL wrapper with **separate** abstract embedding and projection.

    The base ``SorlModelWrapper`` expands the HF model's ``embed_tokens``
    and ``lm_head`` to include abstract token rows.  When the base model
    ties weights (Qwen3), freezing abstract ``lm_head`` rows also freezes
    abstract ``embed_tokens`` rows — making them untrainable.

    V2 keeps ``embed_tokens`` and ``lm_head`` at the original NL vocab
    size and introduces two independent parameter groups:

        * ``abs_embed`` : ``nn.Embedding(V_abs+1, d)`` — abstract input embeddings
        * ``abs_proj``  : ``nn.Linear(d, V_abs+1)``    — abstract output projection

    These are transparently spliced in via ``_SplitEmbedding`` /
    ``_SplitLMHead`` drop-in replacements so that **all** parent methods
    (``forward``, ``generate``, ``recursion``, ``recursion_step``, etc.)
    work without any changes.

    The base model's ``embed_tokens ↔ lm_head`` tying is **preserved** for
    NL tokens.
    """

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        abstract_vocab_size_list: List[int],
        untie_embeddings: bool = False,
        **kwargs,
    ) -> "SorlModelWrapperV2":
        config = AutoConfig.from_pretrained(model_name_or_path, **kwargs)

        wrapper = cls(config)
        wrapper.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, attn_implementation="eager", **kwargs
        )

        base_vocab_size = config.vocab_size
        wrapper.full_vocab_size_list = [base_vocab_size] + abstract_vocab_size_list
        wrapper._setup_vocabulary()

        # HF from_pretrained may load both embed_tokens.weight and lm_head.weight
        # from the checkpoint, breaking the tie even when config says tied.
        # Re-tie explicitly so NL params share storage.
        if getattr(wrapper.model.config, "tie_word_embeddings", False):
            wrapper.model.lm_head.weight = wrapper.model.model.embed_tokens.weight

        # Abstract token count (including placeholder at index 0)
        abs_total = int(wrapper.total_vocab_size.item()) - base_vocab_size
        hidden_size = config.hidden_size

        # Create separate abstract parameters in the same dtype as the base model
        dtype = wrapper.model.model.embed_tokens.weight.dtype
        abs_embed = nn.Embedding(abs_total, hidden_size, dtype=dtype)
        abs_proj = nn.Linear(hidden_size, abs_total, bias=False, dtype=dtype)

        # Replace embed_tokens and lm_head with split versions
        orig_embed = wrapper.model.model.embed_tokens
        orig_head = wrapper.model.lm_head
        wrapper.model.model.embed_tokens = _SplitEmbedding(orig_embed, abs_embed, base_vocab_size)
        wrapper.model.lm_head = _SplitLMHead(orig_head, abs_proj)

        # Update config vocab size to total (logit dim = V_nl + V_abs + 1)
        wrapper.model.config.vocab_size = int(wrapper.total_vocab_size.item())
        wrapper.config.vocab_size = int(wrapper.total_vocab_size.item())

        # Initialise abstract params with orthogonal vectors
        wrapper._init_abstract_embeddings_orthogonal()

        return wrapper

    # -- properties for convenient external access ----------------------------

    @property
    def abs_embed(self) -> nn.Embedding:
        """The abstract embedding table (lives inside _SplitEmbedding)."""
        embed_mod = self.model.model.embed_tokens
        if isinstance(embed_mod, _SplitEmbedding):
            return embed_mod.abs_embed
        raise AttributeError("Model does not use _SplitEmbedding")

    @property
    def abs_proj(self) -> nn.Linear:
        """The abstract projection head (lives inside _SplitLMHead)."""
        head_mod = self.model.lm_head
        if isinstance(head_mod, _SplitLMHead):
            return head_mod.abs_proj
        raise AttributeError("Model does not use _SplitLMHead")

    @property
    def has_separate_abs_params(self) -> bool:
        return True

    # -- init -----------------------------------------------------------------

    @torch.no_grad()
    def _init_abstract_embeddings_orthogonal(self):
        """Initialise ``abs_embed`` and ``abs_proj`` with orthogonal vectors
        scaled to match the norm of the base NL embeddings."""
        base_vocab = int(self.vocab_sizes[0].item())
        abs_embed_w = self.abs_embed.weight
        abs_proj_w = self.abs_proj.weight
        n_abs = abs_embed_w.shape[0]
        hidden = abs_embed_w.shape[1]

        # Measure scale from NL embeddings
        nl_embed_w = self.model.model.embed_tokens.nl_embed.weight
        base_norm = nl_embed_w[:base_vocab].norm(dim=1).mean().item()

        # Orthogonal init for abs_embed (compute in fp32, cast to param dtype)
        ortho_e = torch.empty(max(n_abs, hidden), hidden, device=abs_embed_w.device)
        nn.init.orthogonal_(ortho_e)
        abs_embed_w.copy_((ortho_e[:n_abs] * base_norm).to(abs_embed_w.dtype))

        # Orthogonal init for abs_proj (independent from abs_embed)
        ortho_p = torch.empty(max(n_abs, hidden), hidden, device=abs_proj_w.device)
        nn.init.orthogonal_(ortho_p)
        abs_proj_w.copy_((ortho_p[:n_abs] * base_norm).to(abs_proj_w.dtype))
