from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig, AutoConfig
from transformers import AutoModelForCausalLM, AutoConfig as TransformersAutoConfig
from typing import List, Optional, Tuple
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.nn.attention.flex_attention import create_block_mask

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
        self.model = AutoModelForCausalLM.from_config(config)
        self.full_vocab_size_list = None
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, abstract_vocab_size_list: List[int], **kwargs) -> "SorlModelWrapper":
        config = AutoConfig.from_pretrained(model_name_or_path, **kwargs)
        
        wrapper = cls(config)
        wrapper.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
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

    def forward(self, input_ids, attention_mask=None, memory_span_abs=1792, memory_span_traj=1792, **kwargs):
        # Create SORL block mask for flex attention with information bottleneck
        sorl_block_mask = self._create_sorl_block_mask(input_ids, memory_span_abs, memory_span_traj)

        # Training does not benefit from KV cache and it inflates memory usage.
        if self.training:
            kwargs.setdefault("use_cache", False)
        
        # attention_mask: masks padding tokens; block_mask: handles SoRL-specific attention
        return self.model.forward(input_ids=input_ids, attention_mask=attention_mask, block_mask=sorl_block_mask, **kwargs)

    def _create_sorl_block_mask(self, input_ids: torch.Tensor, memory_span_abs: int = 1792, memory_span_traj: int = 1792):
        """Create SORL block mask with information bottleneck - separate memory spans for abstract vs trajectory tokens"""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device        
        
        # Pre-compute document boundaries and levels
        doc_boundary_token_id = self._get_doc_boundary_token_id()
        docs = (input_ids == doc_boundary_token_id).cumsum(1)
        levels = infer_level(input_ids, self.vocab_sizes)  # Vocabulary levels
        accum_levels = levels.cumsum(1)

        max_idx = seq_len - 1

        def _safe_idx(i):
            # create_block_mask may probe indices at block boundaries (e.g. one-past-end).
            # Clamp before tensor indexing to avoid out-of-bounds while gating with in_bounds.
            if torch.is_tensor(i):
                return torch.clamp(i, min=0, max=max_idx)
            return max(0, min(i, max_idx))
        
        def sorl_mask_fn(b, h, q_idx, kv_idx):
            """SORL mask function with information bottleneck"""
            in_bounds = (q_idx >= 0) & (q_idx < seq_len) & (kv_idx >= 0) & (kv_idx < seq_len)
            q_safe = _safe_idx(q_idx)
            kv_safe = _safe_idx(kv_idx)

            # Causal constraint
            causal_mask = q_idx >= kv_idx
            
            # Document constraint
            document_mask = docs[b, q_safe] == docs[b, kv_safe]
            
            # Window constraint
            window_mask = q_idx - kv_idx < 1792  # attn_blocksize
            
            # Level-based memory compression (information bottleneck)
            to_abstract = levels[b, kv_safe] > 0
            from_abstract = levels[b, q_safe] > 0
            
            # Skip accumulated abstract levels
            skip_abs = accum_levels[b, q_safe] > accum_levels[b, kv_safe]
            
            # Different memory spans for abstract vs trajectory tokens
            traj_memory_span = (q_idx - kv_idx) <= memory_span_traj
            abs_memory_span = (q_idx - kv_idx) <= memory_span_abs
            
            # Information bottleneck mask
            memory_compression_mask = (
                to_abstract |  # Always attend to abstract tokens
                (from_abstract & abs_memory_span) |  # From abstract: use abstract memory span
                (~from_abstract & traj_memory_span & ~skip_abs)  # From trajectory: use traj memory span, skip accumulated abs
            )
            
            return in_bounds & causal_mask & document_mask & window_mask & memory_compression_mask
        
        # Create block mask using the mask function (compiled to avoid materializing full seq x seq tensor)
        block_mask = create_block_mask(
            sorl_mask_fn, 
            B=batch_size, 
            H=self.model.config.num_attention_heads,
            Q_LEN=seq_len, 
            KV_LEN=seq_len, 
            device=device,
            # _compile=True if "cuda" in str(device) else False,
        )
        
        return block_mask

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

        # For response_only_abs: track NL tokens generated since start/last abstract.
        # This avoids looking back into the prompt (which has no abstract tokens)
        # and matches the training pattern: K NL tokens, then 1 ABS, repeat.
        nl_since_abs = torch.zeros(generated_ids.size(0), dtype=torch.long, device=generated_ids.device)

        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self.model.forward(
                input_ids=generated_ids,
                attention_mask=attention_mask,
                block_mask=self._create_sorl_block_mask(generated_ids, memory_span_abs, memory_span_traj),
                use_cache=False,
            )
            next_token_logits = outputs.logits[:, -1, :]

            if not free_form:
                # Periodic: force abstract token every K trajectory tokens
                if K is None:
                    next_token_level = torch.zeros(generated_ids.size(0), dtype=torch.long, device=generated_ids.device)
                elif response_only_abs:
                    # Count NL tokens generated since start of response (or last abstract).
                    # Force abstract after K NL tokens, matching training pattern.
                    next_token_level = (nl_since_abs >= K).long()
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

            # Update NL counter for response_only_abs mode
            if response_only_abs and K is not None:
                is_abs = (next_token_level > 0)
                nl_since_abs = torch.where(is_abs, torch.zeros_like(nl_since_abs), nl_since_abs + 1)

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
            block_mask = self._create_sorl_block_mask(generated_ids, memory_span_abs, memory_span_traj)
            outputs = self.model.forward(
                input_ids=generated_ids, attention_mask=attention_mask, block_mask=block_mask, use_cache=False,
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
            block_mask = self._create_sorl_block_mask(generated_ids, memory_span_abs, memory_span_traj)
            outputs = self.model.forward(
                input_ids=generated_ids, attention_mask=attention_mask, block_mask=block_mask, use_cache=False,
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
            block_mask = self._create_sorl_block_mask(idx, memory_span_abs, memory_span_traj)
            outputs = self.model.forward(
                input_ids=idx, attention_mask=attention_mask,
                block_mask=block_mask, use_cache=False,
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

        block_mask = self._create_sorl_block_mask(idx, memory_span_abs, memory_span_traj)
        labels = idx.clone()
        labels[attention_mask == 0] = -100
        if prompt_len is not None:
            seq_idx = torch.arange(labels.size(1), device=labels.device).unsqueeze(0)
            labels[seq_idx < prompt_len.unsqueeze(1)] = -100

        outputs = self.model.forward(
            input_ids=idx, attention_mask=attention_mask,
            block_mask=block_mask, use_cache=False,
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

        block_mask = self._create_sorl_block_mask(idx, memory_span_abs, memory_span_traj)

        for it in range(max_iterations): 
            outputs = self.model.forward(
                input_ids=idx, 
                attention_mask=attention_mask,
                block_mask=block_mask,
                use_cache=False,
            )
            logits = outputs.logits
            
            if not differentiable or it < max_iterations - 1:
                idx = self.extract_and_sample(logits, idx, recursion_mask, temp_expanded)
            else:
                # Last iteration with STE
                recursion_logits = logits[predict_mask]
                recursion_logits[:, :vocab_size_0 + 1] = float('-inf')
                
                if isinstance(temp_expanded, torch.Tensor):
                    temp = temp_expanded[predict_mask].clamp(min=1e-10).unsqueeze(-1)
                else:
                    temp = max(float(temperature), 1e-10)
                    
                soft_probs = F.softmax(recursion_logits / temp, dim=-1)
                new_tokens = torch.multinomial(soft_probs, num_samples=1).squeeze(-1)
                         
                one_hot = F.one_hot(new_tokens, num_classes=total_vocab_size).to(soft_probs.dtype)
                ste_probs = one_hot + soft_probs - soft_probs.detach()
                
                idx[recursion_mask] = new_tokens.to(idx.dtype)
        
        # Evaluation — mask padding AND question tokens in labels
        labels = idx.clone()
        labels[attention_mask == 0] = -100
        if prompt_len is not None:
            seq_idx = torch.arange(labels.size(1), device=labels.device).unsqueeze(0)
            labels[seq_idx < prompt_len.unsqueeze(1)] = -100

        block_mask = self._create_sorl_block_mask(idx, memory_span_abs, memory_span_traj)

        if not differentiable:
            outputs = self.model.forward(
                input_ids=idx, 
                attention_mask=attention_mask,
                block_mask=block_mask,
                use_cache=False,
            )
        else:
            embed_weight = self.model.get_input_embeddings().weight
            inputs_embeds = self.model.get_input_embeddings()(idx)
            abs_embeds = ste_probs.to(embed_weight.dtype) @ embed_weight
            inputs_embeds[recursion_mask] = abs_embeds
            
            outputs = self.model.forward(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                block_mask=block_mask,
                use_cache=False,
            )

        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        per_token_loss = per_token_loss.view(idx.shape[0], -1)  # [batch_size, seq_len-1]
                    
        return idx, per_token_loss, outputs.logits
