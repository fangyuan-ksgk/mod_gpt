from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig, AutoConfig
from transformers import AutoModelForCausalLM, AutoConfig as TransformersAutoConfig
from typing import List, Optional, Tuple
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.nn.attention.flex_attention import create_block_mask

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
        config.tie_word_embeddings = False
        
        wrapper = cls(config)
        wrapper.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            tie_word_embeddings=False,  # Also pass to model loading
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
        return wrapper

    def forward(self, input_ids, attention_mask=None, memory_span_abs=1792, memory_span_traj=1792, **kwargs):
        # Create SORL block mask for flex attention with information bottleneck
        sorl_block_mask = self._create_sorl_block_mask(input_ids, memory_span_abs, memory_span_traj)
        
        # attention_mask: masks padding tokens; block_mask: handles SoRL-specific attention
        return self.model.forward(input_ids=input_ids, attention_mask=attention_mask, block_mask=sorl_block_mask, **kwargs)

    def _create_sorl_block_mask(self, input_ids: torch.Tensor, memory_span_abs: int = 1792, memory_span_traj: int = 1792):
        """Create SORL block mask with information bottleneck - separate memory spans for abstract vs trajectory tokens"""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device        
        
        # Pre-compute document boundaries and levels
        levels = infer_level(input_ids, self.vocab_sizes)  # Vocabulary levels
        accum_levels = levels.cumsum(1)
        
        def sorl_mask_fn(b, h, q_idx, kv_idx):
            """SORL mask function with information bottleneck"""
            # Causal constraint
            causal_mask = q_idx >= kv_idx
            
            # Document constraint
            docs = (input_ids == 50256).cumsum(1)  # Assuming BOS_TOKEN_ID=50256 for Qwen
            document_mask = docs[b, q_idx] == docs[b, kv_idx]
            
            # Window constraint
            window_mask = q_idx - kv_idx < 1792  # attn_blocksize
            
            # Level-based memory compression (information bottleneck)
            to_abstract = levels[b, kv_idx] > 0
            from_abstract = levels[b, q_idx] > 0
            
            # Skip accumulated abstract levels
            skip_abs = accum_levels[b, q_idx] > accum_levels[b, kv_idx]
            
            # Different memory spans for abstract vs trajectory tokens
            traj_memory_span = (q_idx - kv_idx) <= memory_span_traj
            abs_memory_span = (q_idx - kv_idx) <= memory_span_abs
            
            # Information bottleneck mask
            memory_compression_mask = (
                to_abstract |  # Always attend to abstract tokens
                (from_abstract & abs_memory_span) |  # From abstract: use abstract memory span
                (~from_abstract & traj_memory_span & ~skip_abs)  # From trajectory: use traj memory span, skip accumulated abs
            )
            
            return causal_mask & document_mask & window_mask & memory_compression_mask
        
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
        self.register_buffer("abs_mask", abs_mask)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        temperature: float = 0.7,
        top_k: int = 50,
        K: Optional[int] = None,
        memory_span_abs: int = 1792,
        memory_span_traj: int = 1792,
    ):
        import torch.nn.functional as F
        
        self.model.eval()
        generated_ids = input_ids.clone()
        levels_cache = infer_level(generated_ids, self.vocab_sizes)
        masks = torch.stack([self.l0_mask, self.abs_mask], dim=0).to(generated_ids.device)

        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self.model.forward(
                input_ids=generated_ids,
                block_mask=self._create_sorl_block_mask(generated_ids, memory_span_abs, memory_span_traj),
            )
            next_token_logits = outputs.logits[:, -1, :]

            # Determine next token level
            if K is None:
                next_token_level = torch.zeros(generated_ids.size(0), dtype=torch.long, device=generated_ids.device)
            else:
                next_token_level = 1 - (levels_cache[:, -K:] > 0).any(dim=-1).long()

            # Apply level-based masking
            allowed_mask = masks[next_token_level]  # True = allowed tokens
            next_token_logits = next_token_logits.masked_fill(~allowed_mask, -float("inf"))

            # Sample next token
            if temperature > 0:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                topk_probs, topk_idx = torch.topk(probs, min(top_k, probs.size(-1)), dim=-1)
                next_token_id = torch.gather(topk_idx, dim=1, index=torch.multinomial(topk_probs, 1))
            else:
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Update sequences
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            levels_cache = torch.cat([levels_cache, next_token_level[:, None]], dim=1)
        
        return generated_ids

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

    def recursion(self, idx, attention_mask, max_iterations=5, memory_span_abs=1792, memory_span_traj=1792, attn_blocksize=1792, temperature=0.0, prompt_len=None):
        """Perform recursion on abstract tokens with information bottleneck mask."""
        # Ensure vocab_sizes is on the same device as idx
        vocab_size_0 = self.vocab_sizes[0].to(idx.device)
        recursion_mask = (idx >= vocab_size_0)
        recursion_mask[:, 0] = False
        
        # Expand temperature if needed
        if isinstance(temperature, torch.Tensor) and temperature.ndim == 1:
            temp_expanded = temperature.view(-1, 1).expand_as(idx)
        else:
            temp_expanded = temperature

        block_mask = self._create_sorl_block_mask(idx, memory_span_abs, memory_span_traj)

        for _ in range(max_iterations): 
            outputs = self.model.forward(
                input_ids=idx, 
                attention_mask=attention_mask,
                block_mask=block_mask
            )
            logits = outputs.logits
            idx = self.extract_and_sample(logits, idx, recursion_mask, temp_expanded)
        
        # Evaluation — mask padding AND question tokens in labels
        labels = idx.clone()
        labels[attention_mask == 0] = -100
        if prompt_len is not None:
            seq_idx = torch.arange(labels.size(1), device=labels.device).unsqueeze(0)
            labels[seq_idx < prompt_len.unsqueeze(1)] = -100

        outputs = self.model.forward(
            input_ids=idx, 
            attention_mask=attention_mask,
            block_mask=block_mask,
        )

        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        per_token_loss = per_token_loss.view(idx.shape[0], -1)  # [batch_size, seq_len-1]
                    
        return idx, per_token_loss