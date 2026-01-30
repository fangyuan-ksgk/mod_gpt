from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig, AutoConfig
from transformers import AutoModelForCausalLM, AutoConfig as TransformersAutoConfig
from typing import List, Optional, Tuple
import torch
from torch.nn.attention.flex_attention import create_block_mask

SUPPORTED_MODELS = {
    "qwen2": "qwen2",
    "qwen3": "qwen3",  # Use AutoModelForCausalLM for both
}

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
        self.memory_span = None
        self.full_vocab_size_list = None
        self.pad_token_id = None
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, abstract_vocab_size_list: List[int], memory_span: int, pad_token_id: int, **kwargs) -> "SorlModelWrapper":
        config = AutoConfig.from_pretrained(model_name_or_path, **kwargs)
        
        # Fix Qwen weight tying warning - Qwen doesn't tie embeddings to lm_head
        # This prevents the warning: "The tied weights mapping and config for this model 
        # specifies to tie model.embed_tokens.weight to lm_head.weight, but both are 
        # present in the checkpoints, so we will NOT tie them"
        config.tie_word_embeddings = False
        
        wrapper = cls(config)
        wrapper.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            tie_word_embeddings=False,  # Also pass to model loading
            **kwargs
        )

        wrapper.memory_span = memory_span
        wrapper.pad_token_id = pad_token_id
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
    def from_scratch(cls, config: PretrainedConfig, full_vocab_size_list: List[int], memory_span: int, pad_token_id: int) -> "SorlModelWrapper":
        """A custom initializer for creating a SORL model from scratch."""
        wrapper = cls(config)
        wrapper.memory_span = memory_span
        wrapper.pad_token_id = pad_token_id
        wrapper.full_vocab_size_list = full_vocab_size_list
        wrapper._setup_vocabulary()
        
        new_total_vocab_size = wrapper.total_vocab_size.item()
        wrapper.model.resize_token_embeddings(new_total_vocab_size)
        wrapper.config.vocab_size = new_total_vocab_size
        return wrapper

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Create SORL block mask for flex attention
        sorl_block_mask = self._create_sorl_block_mask(input_ids)
        
        # Use flex attention with block_mask instead of materialized attention_mask
        return self.model.forward(input_ids=input_ids, attention_mask=attention_mask, block_mask=sorl_block_mask, **kwargs)

    def _create_sorl_block_mask(self, input_ids: torch.Tensor):
        """Create SORL block mask for flex attention"""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Infer vocabulary levels for tokens
        def infer_level(indices: torch.Tensor, vocab_sizes: torch.Tensor):
            if indices.dtype in [torch.uint8, torch.uint16, torch.uint32, torch.uint64]:
                indices = indices.long()
            
            vocab_sizes = vocab_sizes.to(indices.device)
            indices_expanded = indices.unsqueeze(-1)
            levels = (indices_expanded < vocab_sizes.cumsum(dim=0)).int().argmax(dim=-1)
            return levels
        
        # Pre-compute document boundaries and levels
        docs = (input_ids == self.pad_token_id).cumsum(1)  # Document boundaries
        levels = infer_level(input_ids, self.vocab_sizes)  # Vocabulary levels
        
        def sorl_mask_fn(b, h, q_idx, kv_idx):
            """SORL mask function for flex attention"""
            # Causal constraint
            causal_mask = q_idx >= kv_idx
            
            # Document boundary constraint
            document_mask = docs[b, q_idx] == docs[b, kv_idx]
            
            # Memory compression constraint
            is_higher_level = levels[b, kv_idx] > 0
            is_recent = (q_idx - kv_idx) <= self.memory_span
            memory_compression_mask = is_higher_level | is_recent
            
            return causal_mask & document_mask & memory_compression_mask
        
        # Create block mask using the mask function
        block_mask = create_block_mask(
            sorl_mask_fn, 
            B=batch_size, 
            H=self.model.config.num_attention_heads,
            Q_LEN=seq_len, 
            KV_LEN=seq_len, 
            device=device
        )
        
        return block_mask


    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None, **kwargs):
        if past_key_values:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "attention_mask": kwargs.get("attention_mask"),
        }

    def _setup_vocabulary(self):
        device = self.device
        
        base_vocab_size = self.full_vocab_size_list[0]
        abstract_vocab_sizes = self.full_vocab_size_list[1:]

        vocab_sizes_list = [base_vocab_size] + [size + 1 for size in abstract_vocab_sizes]
        # Register all derived tensors as buffers. This ensures they are moved
        # to the correct device when you call .to() on the model.
        self.register_buffer("vocab_sizes", torch.tensor(vocab_sizes_list, device=device))

        self.total_vocab_size = self.vocab_sizes.sum()
        self.register_buffer("level_vocab_ends", self.vocab_sizes.cumsum(dim=0))
        self.register_buffer("level_vocab_starts", torch.cat([torch.tensor([0], device=device), self.level_vocab_ends[:-1]]))

        l0_mask_token = torch.tensor([self.pad_token_id], device=device, dtype=torch.long)
        
        if len(self.vocab_sizes) > 1:
            abstract_mask_tokens = (self.level_vocab_ends - 1)[1:]
            self.register_buffer("level_mask_tokens", torch.cat([l0_mask_token, abstract_mask_tokens]))
        else:
            self.register_buffer("level_mask_tokens", l0_mask_token)

        l0_mask = torch.zeros(self.total_vocab_size.item(), dtype=torch.bool, device=device)
        l0_mask[:self.vocab_sizes[0]] = True
        self.register_buffer("l0_mask", l0_mask)
        
        abs_mask = ~self.l0_mask
        # Note: self.level_mask_tokens may have length 1 if there are no abstract vocabs
        if len(self.level_mask_tokens) > 1:
            abs_mask[self.level_mask_tokens[1:]] = False
        self.register_buffer("abs_mask", abs_mask)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        temperature: float = 0.7,
        top_k: int = 50,
        force_abstraction_every_n: Optional[int] = None,
    ):

        self.model.eval()

        generated_ids = input_ids.clone()
        past_key_values = None
        levels_cache = infer_level(generated_ids, self.vocab_sizes, -1) # Use -1 for pad_token, assuming it won't be in the prompt

        for step in range(max_new_tokens):
            model_inputs = self.prepare_inputs_for_generation(
                input_ids=generated_ids, past_key_values=past_key_values
            )
            
            outputs = self.model.forward(**model_inputs)
            next_token_logits = outputs.logits[:, -1, :]
            current_pkv = outputs.past_key_values

            past_key_values, levels_cache, generated_ids = memory_pruning(current_pkv, levels_cache, generated_ids, self.memory_span)

            if force_abstraction_every_n is not None and (step + 1) % force_abstraction_every_n == 0:
                next_token_logits.masked_fill_(self.l0_mask, -float("inf"))
            else: 
                next_token_logits.masked_fill_(self.abs_mask, -float("inf"))

            if temperature > 0:
                vocab_size = next_token_logits.shape[-1]
                top_k = min(top_k, vocab_size)

                probs = F.softmax(next_token_logits / temperature, dim=-1)
                top_k_probs, top_k_indices = torch.topk(probs, top_k)
                next_token_id = top_k_indices[0, torch.multinomial(top_k_probs, num_samples=1)[0]]
            else:
                next_token_id = torch.argmax(next_token_logits, dim=-1)

            next_token_id = next_token_id.unsqueeze(0)
            
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            new_level = infer_level(next_token_id, self.vocab_sizes, -1)
            levels_cache = torch.cat([levels_cache, new_level], dim=1)
            
        return generated_ids


    def _parallel_decode(self, logits: torch.Tensor, levels: Optional[torch.Tensor] = None, temperature: float = 0.0):
        logits = 30 * torch.tanh(logits / 30)
        logits = logits.float()
        
        total_vocab_size = logits.shape[-1]
        valid_mask_tokens = self.level_mask_tokens[self.level_mask_tokens < total_vocab_size]
        logits[:, valid_mask_tokens] = float('-inf')

        if levels is not None:
            assert levels.size(0) == logits.shape[0], f"[Denoise level & mask mismatch] \n - Need to denoise on {logits.shape[0]} tokens, but got {levels.shape[-1]} levels"
            start_logits = self.level_vocab_starts.to(levels.device)[levels]
            end_logits = self.level_vocab_ends.to(levels.device)[levels]
            vocab_indices = torch.arange(logits.size(-1), device=logits.device)
            mask = (vocab_indices >= start_logits.unsqueeze(-1)) & (vocab_indices < end_logits.unsqueeze(-1))
            logits = torch.where(mask, logits, torch.tensor(-float('inf'), device=logits.device))

        if temperature == 0.0:
            next_token = torch.argmax(logits, dim=-1)
        else:
            next_token = torch.multinomial(F.softmax(logits / temperature, dim=-1), num_samples=1).squeeze(-1)
        return next_token


    def denoise(self, idx: torch.Tensor, denoise_mask: torch.Tensor, denoise_levels: torch.Tensor, temperature: float = 0.0): 
        self.model.eval()
        with torch.no_grad():
            if self.model_type == "minimind":
                outputs = self.forward(input_ids=idx, use_cache=False, attention_mask=None)
                hidden_states = outputs.last_hidden_state
            else: 
                outputs = self.forward(input_ids=idx, use_cache=False, attention_mask=None, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
            rep_mask = torch.roll(denoise_mask, -1, dims=1)
            new_tokens = self._parallel_decode(self.model.lm_head(hidden_states[rep_mask]), levels=denoise_levels, temperature=temperature)
            denoised_idx = idx.clone()
            denoised_idx[denoise_mask] = new_tokens

        return denoised_idx