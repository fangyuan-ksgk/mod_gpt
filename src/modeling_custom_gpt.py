from transformers import PretrainedConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

class CustomGPTConfig(PretrainedConfig):
    model_type = "custom_gpt"

    def __init__(
        self,
        vocab_size=50257,
        n_embd=768,
        n_layer=12,
        n_head=12,
        flex_kernel_options=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.flex_kernel_options = flex_kernel_options

        # Auto-map for trust_remote_code loading
        self.auto_map = {
            "AutoConfig": "modeling_custom_gpt.CustomGPTConfig",
            "AutoModel": "modeling_custom_gpt.CustomGPTModel",
            "AutoModelForCausalLM": "modeling_custom_gpt.CustomGPTModel"
        }

    @property
    def num_hidden_layers(self):
        return self.n_layer



import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from torch.nn.attention.flex_attention import flex_attention

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class CastedLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))

class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = None
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim))
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            self.cos_cached = freqs.cos().float()
            self.sin_cached = freqs.sin().float()
        cos, sin = self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]
        # apply_rotary_emb(x, cos, sin)
        assert x.ndim == 4 # multihead attention
        d = x.shape[3]//2
        x1 = x[..., :d]
        x2 = x[..., d:]
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat([y1, y2], 3).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_head, flex_kernel_options=None):
        super().__init__()
        assert dim % n_head == 0, "Embedding dimension must be divisible by number of heads"
        self.dim = dim
        self.n_head = n_head        
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, dim)
        self.c_v = CastedLinear(dim, dim)
        # value residual lambda 
        self.lamb = nn.Parameter(torch.tensor(0.5))  # @Grad62304977
        # rotary embeddings
        self.rotary = Rotary(dim // n_head)
        # output projection
        self.c_proj = CastedLinear(dim, dim)
        self.c_proj.weight.data.zero_()  # zero init suggested by @Grad62304977
        # flex attention kernel options
        self.flex_kernel_options = flex_kernel_options

    def forward(self, x, v1=None, block_mask=None):
        B, T = x.size(0), x.size(1)  
        # Compute Q, K, V
        q = self.c_q(x).view(B, T, self.n_head, -1)
        k = self.c_k(x).view(B, T, self.n_head, -1)
        v = self.c_v(x).view(B, T, self.n_head, -1)        
        if v1 is None:
            v1 = v  # If this is the first block, set v1 to v
        v = (1 - self.lamb) * v + self.lamb * v1.view_as(v)  # @Grad62304977
        q, k = norm(q), norm(k) # QK norm suggested by @Grad62304977
        q, k = self.rotary(q), self.rotary(k)        
        
        y = flex_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            block_mask=block_mask,
            kernel_options=self.flex_kernel_options
        )
        y = y.transpose(1, 2).contiguous().view_as(x)       
        y = self.c_proj(y)
        return y, v1

class MLP(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.c_fc   = CastedLinear(dim, 4 * dim)
        self.c_proj = CastedLinear(4 * dim, dim)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config.n_embd, config.n_head, config.flex_kernel_options)
        self.mlp = MLP(config.n_embd)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x, v1, x0, block_mask):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        x1, v1 = self.attn(norm(x), v1, block_mask)
        x = x + x1
        x = x + self.mlp(norm(x))
        return x, v1


class CustomGPTModel(PreTrainedModel, GenerationMixin):
    config_class = CustomGPTConfig
    # NOTE: Original model does NOT tie weights (lm_head is separate, zero-init)
    # _tied_weights_keys = ["lm_head.weight"]  # DISABLED

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        
        self.layers = nn.ModuleList([
            Block(config) for _ in range(config.n_layer)
        ])

        self.lm_head = CastedLinear(config.n_embd, config.vocab_size)

        self.post_init()  # important for Hugging Face weight init

    def tie_weights(self):
        # EXPLICITLY DO NOTHING - original model has separate embed and lm_head weights
        # lm_head is zero-initialized separately, NOT tied to embedding
        pass

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}
        
    def forward(self, input_ids, labels=None, attn_blocksize=1024, **kwargs):
        # Create block mask with causal + window constraints
        S = input_ids.shape[1]
        def causal_window_mask(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            window_mask = q_idx - kv_idx < attn_blocksize
            return causal_mask & window_mask
        
        from torch.nn.attention.flex_attention import create_block_mask
        block_mask = create_block_mask(
            causal_window_mask, None, None, S, S, 
            device=input_ids.device, _compile=False
        )
        
        x = self.embed_tokens(input_ids)
        x = norm(x)  # norm after embedding (matches local model)
        x0 = x  # save for lambda mixing
        v1 = None  # value residual state
        
        for layer in self.layers:
            x, v1 = layer(x, v1, x0, block_mask=block_mask)

        x = norm(x)
        logits = self.lm_head(x)
        logits = 30 * torch.tanh(logits / 30)  # logit clamping (matches local model)
        logits = logits.float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
        )

def port_weights(custom_state, hf_state, config):
    """
    Map weights from pretrained modgpt into HF CustomGPTModel.
    """
    new_state = hf_state.copy()

    # --- Embeddings ---
    new_state['embed_tokens.weight'] = custom_state['transformer.wte.weight']

    # --- LM head ---
    new_state['lm_head.weight'] = custom_state['lm_head.weight']

    # --- Transformer blocks ---
    for i in range(config.n_layer):
        # Attention Q, K, V
        new_state[f'layers.{i}.attn.c_q.weight'] = custom_state[f'transformer.h.{i}.attn.c_q.weight']
        new_state[f'layers.{i}.attn.c_k.weight'] = custom_state[f'transformer.h.{i}.attn.c_k.weight']
        new_state[f'layers.{i}.attn.c_v.weight'] = custom_state[f'transformer.h.{i}.attn.c_v.weight']
        new_state[f'layers.{i}.attn.c_proj.weight'] = custom_state[f'transformer.h.{i}.attn.c_proj.weight']
        
        # Value residual lambda (CRITICAL!)
        new_state[f'layers.{i}.attn.lamb'] = custom_state[f'transformer.h.{i}.attn.lamb']

        # MLP
        new_state[f'layers.{i}.mlp.c_fc.weight'] = custom_state[f'transformer.h.{i}.mlp.c_fc.weight']
        new_state[f'layers.{i}.mlp.c_proj.weight'] = custom_state[f'transformer.h.{i}.mlp.c_proj.weight']
        
        # Block mixing lambdas (CRITICAL!)
        new_state[f'layers.{i}.lambdas'] = custom_state[f'transformer.h.{i}.lambdas']

    return new_state

# Auto-registration for HuggingFace
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

AutoConfig.register("custom_gpt", CustomGPTConfig)
AutoModel.register(CustomGPTConfig, CustomGPTModel)
AutoModelForCausalLM.register(CustomGPTConfig, CustomGPTModel)