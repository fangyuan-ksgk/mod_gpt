import torch
from sorl.gat_act import BOS_TOKEN_ID, search, GAT, recursion, infer_level
from sorl.gat_sim import BOS_TOKEN_ID, GAT, recursion, extract_and_sample, recursion_v3
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Optional, Union
from torch import nn                
import math


@torch.compile
def infer_rythmic_insert_mask(tokens, K, traj_vocab_size):
    batch_size, seq_len = tokens.shape
    assert batch_size == 1, "only one sample supported"
    positions = torch.arange(1, seq_len + 1, device=tokens.device).unsqueeze(0)

    is_bos = (tokens == BOS_TOKEN_ID).long()
    bos_cumsum = is_bos.cumsum(dim=1)
    is_abstract = (tokens >= traj_vocab_size).long()
    
    # Find last BOS or abstraction in each document
    is_bos_or_abstract = torch.maximum(is_bos, is_abstract)
    last_ref_pos = torch.zeros(batch_size, seq_len + 1, device=tokens.device, dtype=torch.long)
    ref_positions = torch.where(is_bos_or_abstract > 0, positions, torch.tensor(0, device=tokens.device))
    last_ref_pos.scatter_reduce_(1, bos_cumsum, ref_positions, reduce='amax', include_self=False)
    
    reference_pos = last_ref_pos.gather(1, bos_cumsum)
    within_doc_pos = positions - reference_pos
    no_fill = torch.ones_like(is_bos, dtype=torch.bool, device=tokens.device)
    no_fill[:, :-1] = is_bos[:, 1:].bool()

    insert_mask = (within_doc_pos % K == 0) & (within_doc_pos > 0) & (~no_fill)    
    return insert_mask

def insert_tokens(tokens, insert_mask, placeholder_token):
    """Vectorized insertion of placeholder tokens at masked positions."""
    batch_size, seq_len = tokens.shape
    
    cumsum_mask = torch.cumsum(insert_mask.long(), dim=1)
    shift = torch.cat([torch.zeros(batch_size, 1, dtype=torch.long, device=tokens.device), 
                       cumsum_mask[:, :-1]], dim=1)
    
    original_positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
    new_positions = original_positions + shift
    
    max_new_len = seq_len + cumsum_mask[:, -1].max().item()
    expanded_tokens = torch.full((batch_size, max_new_len), placeholder_token,
                                 dtype=tokens.dtype, device=tokens.device)
    
    batch_indices = torch.arange(batch_size, device=tokens.device).unsqueeze(1).expand(-1, seq_len)
    expanded_tokens[batch_indices, new_positions] = tokens
    
    return expanded_tokens

@torch.no_grad()
def sorl_rollout_v2(data: torch.Tensor, model: GAT, n: int, K: int, max_iterations: int, memory_span: int, attn_blocksize: int, temperature: Union[float, torch.Tensor] = 0.0,
                 truncate_seq_len: bool = True):
    """
    Direct rollout without greedy sample. 
    """
    assert data.shape[0] == 1, "only single sample supported"
    data_len = data.shape[1]

    # --- repeat data & add placeholder tokens ---
    insert_mask = infer_rythmic_insert_mask(data, K, model.vocab_sizes[0])
    data = insert_tokens(data, insert_mask, model.vocab_sizes[0].item())
    if truncate_seq_len:
        data = data[:, :data_len] # avoids recompilation
    repeat_data = data.repeat_interleave(n, dim=0)

    # --- search --- 
    search_data, search_ppt = recursion(model, repeat_data, max_iterations=max_iterations, 
                                memory_span=memory_span, attn_blocksize=attn_blocksize, temperature=temperature)

    return search_data, search_ppt

@torch.no_grad()
def sorl_rollout_v3(data: torch.Tensor, model: GAT, n: int, K: int, max_iterations: int, memory_span: int, attn_blocksize: int, temperature: Union[float, torch.Tensor] = 0.0,
                 truncate_seq_len: bool = True):
    """
    Direct rollout without greedy sample. 
    """
    assert data.shape[0] == 1, "only single sample supported"
    data_len = data.shape[1]

    # --- repeat data & add placeholder tokens ---
    insert_mask = infer_rythmic_insert_mask(data, K, model.vocab_sizes[0])
    data = insert_tokens(data, insert_mask, model.vocab_sizes[0].item())
    if truncate_seq_len:
        data = data[:, :data_len] # avoids recompilation
    repeat_data = data.repeat_interleave(n, dim=0)

    # --- search --- 
    search_data, search_ppt, abs_logits = recursion_v3(model, repeat_data, max_iterations=max_iterations, 
                                memory_span=memory_span, attn_blocksize=attn_blocksize, temperature=temperature)

    return search_data, search_ppt, abs_logits

def avg_ppt_per_sample(ppt, ppt_idx):
    """Average perplexity per document, per rollout."""
    n_r = ppt.shape[0]
    n_d = ppt_idx.max().item() + 1  # +1 because 0-indexed
    idx = (torch.arange(n_r, device=ppt.device)[:, None] * n_d + ppt_idx).reshape(-1)
    sums = torch.zeros(n_r * n_d, device=ppt.device).scatter_add_(0, idx, ppt.reshape(-1))
    counts = torch.zeros(n_r * n_d, device=ppt.device).scatter_add_(0, idx, torch.ones_like(ppt.reshape(-1)))
    return (sums / counts.clamp(min=1)).reshape(n_r, n_d)

def normalize_advantage(raw_adv: torch.Tensor): 
    raw_adv = raw_adv.clamp(min=1e-8)
    norm_raw_adv = raw_adv / raw_adv.max(dim=0, keepdim=True).values
    return norm_raw_adv

def select_best_per_doc(search_data, ppt, levels, model):
    trajectory_mask = (levels[:, 1:] == 0).float()
    trajectory_ppt = ppt * trajectory_mask
    abs_ppt = ppt * (1 - trajectory_mask)
    
    doc_idx = (search_data == BOS_TOKEN_ID).cumsum(dim=1)
    doc_idx = doc_idx - doc_idx.min(dim=1, keepdim=True).values # idx starts from 0  
    doc_ppt = avg_ppt_per_sample(trajectory_ppt, doc_idx[:,1:])

    min_doc_ppt, best_rollout_per_doc = doc_ppt.min(dim=0) # argmin
    rollout_for_each_pos = best_rollout_per_doc[doc_idx[0]]

    best_seq = search_data[rollout_for_each_pos, torch.arange(search_data.shape[1], device=search_data.device)]
    best_ppt = ppt[rollout_for_each_pos[1:], torch.arange(ppt.shape[1], device=ppt.device)]

    max_doc_ppt = doc_ppt.max(dim=0).values
    best_ppt_advantage = (max_doc_ppt - min_doc_ppt) / max_doc_ppt.clamp(min=1e-8)

    return best_seq.unsqueeze(0), best_ppt, best_ppt_advantage.mean()

def select_best_per_doc_v2(search_data, ppt, levels, r_min: float = 1.0, reward_mode: int = 0):
    trajectory_mask = (levels[:, 1:] == 0).float()
    trajectory_ppt = ppt * trajectory_mask
    abs_ppt = ppt * (1 - trajectory_mask)
    
    doc_idx = (search_data == BOS_TOKEN_ID).cumsum(dim=1)
    doc_idx = doc_idx - doc_idx.min(dim=1, keepdim=True).values # idx starts from 0  
    doc_ppt = avg_ppt_per_sample(trajectory_ppt, doc_idx[:,1:])

    min_doc_ppt, best_rollout_per_doc = doc_ppt.min(dim=0) # argmin
    rollout_for_each_pos = best_rollout_per_doc[doc_idx[0]]

    best_seq = search_data[rollout_for_each_pos, torch.arange(search_data.shape[1], device=search_data.device)]
    best_ppt = ppt[rollout_for_each_pos[1:], torch.arange(ppt.shape[1], device=ppt.device)]

    max_doc_ppt = doc_ppt.max(dim=0).values
    best_ppt_advantage = (max_doc_ppt - min_doc_ppt) / max_doc_ppt.clamp(min=1e-8)

    # --- utility reward --- 
    if reward_mode == 0: # exponential PMI: r = max(exp(PMI), r_min)
        utility_reward = torch.exp(max_doc_ppt - min_doc_ppt).clamp(min=r_min)
    elif reward_mode == 1: # PMI = log(p(s | a)/p(s))
        utility_reward = (max_doc_ppt - min_doc_ppt).clamp(min=r_min)
    utility_reward = utility_reward[doc_idx[0]]

    return best_seq.unsqueeze(0), best_ppt, best_ppt_advantage.mean(), utility_reward

def sorl_search(tokens, model, n=3, K=3, max_iterations=1,
                memory_span=1792, attn_blocksize=1792, temperature: Union[float, torch.Tensor] = 1.0, truncate_seq_len: bool = True): 
    """
    Hopefully the last ver |:->))
    """
    # --- generate & evaluate rollouts ---
    search_data, search_ppt = sorl_rollout_v2(tokens, model, n=n, K=K, 
                               max_iterations=max_iterations,
                               memory_span=memory_span,
                               attn_blocksize=attn_blocksize,
                               temperature=temperature,
                               truncate_seq_len=truncate_seq_len)
    search_ppt = search_ppt.reshape(search_data.shape[0], -1)

    # --- select best rollouts ---
    levels = (search_data >= model.vocab_sizes[0]).long()
    best_data, best_ppt, best_ppt_advantage = select_best_per_doc(search_data, search_ppt, levels, model)  # stay with default for select mode (for now)

    return best_data, best_ppt, best_ppt_advantage

def sorl_search_v8(tokens, model, n=3, K=3, max_iterations=1,
                memory_span=1792, attn_blocksize=1792, temperature: Union[float, torch.Tensor] = 1.0, truncate_seq_len: bool = True, use_per_abs_selection: bool = False): 
    """
    Hopefully the last ver |:->))
    """
    # --- generate & evaluate rollouts ---
    search_data, search_ppt = sorl_rollout_v2(tokens, model, n=n, K=K, 
                               max_iterations=max_iterations,
                               memory_span=memory_span,
                               attn_blocksize=attn_blocksize,
                               temperature=temperature,
                               truncate_seq_len=truncate_seq_len)
    search_ppt = search_ppt.reshape(search_data.shape[0], -1)

    # --- select best rollouts ---
    levels = (search_data >= model.vocab_sizes[0]).long()
    best_data, best_ppt, best_ppt_advantage, utility_reward = select_best_per_doc_v2(search_data, search_ppt, levels)  # stay with default for select mode (for now)

    return best_data, best_ppt, best_ppt_advantage, utility_reward[1:]

def sorl_evaluate(tokens, model, n=2, K=4, max_iterations=1, memory_span=1792, attn_blocksize=1792, temperature: Union[float, torch.Tensor] = 1.0,
                  loss_mask: Optional[torch.Tensor] = None, truncate_seq_len: bool = True):
    """
    Search & Check greedy rollout advantage
    """
    search_data, search_ppt = sorl_rollout_v2(tokens, model, n=n, K=K, 
                               max_iterations=max_iterations,
                               memory_span=memory_span,
                               attn_blocksize=attn_blocksize,
                               temperature=temperature,
                               truncate_seq_len=truncate_seq_len)
    search_ppt = search_ppt.reshape(search_data.shape[0], -1)

    # Get valid positions
    bos_pos_mask = torch.logical_and(
        search_data[:, :-1] != BOS_TOKEN_ID, 
        search_data[:, 1:] != BOS_TOKEN_ID
    ).float()
    
    traj_mask = (search_data[:, 1:] < model.vocab_sizes[0]).float()
    
    # --- greedy rollout's advantage ---
    valid_traj_mask = bos_pos_mask * traj_mask
    raw_ppt_adv = (search_ppt[1:].mean(dim=0) - search_ppt[0]) / (search_ppt[1:].mean(dim=0) + 1e-8)
    search_adv = (raw_ppt_adv * valid_traj_mask[0]).sum() / valid_traj_mask[0].sum().clamp(min=1)

    # --- losses ---
    greedy_ppt = search_ppt[0]
    abs_mask = 1 - traj_mask[0]
    
    valid_traj = valid_traj_mask[0]
    valid_abs = bos_pos_mask[0] * abs_mask
    
    traj_loss = (greedy_ppt * valid_traj).sum() / valid_traj.sum().clamp(min=1)
    abs_loss = (greedy_ppt * valid_abs).sum() / valid_abs.sum().clamp(min=1)
    
    return search_data[:1], search_adv, traj_loss, abs_loss


def sorl_evaluate_v2(tokens, model, n=2, K=4, max_iterations=1, memory_span=1792, attn_blocksize=1792, temperature: Union[float, torch.Tensor] = 1.0,
                  loss_mask: Optional[torch.Tensor] = None, truncate_seq_len: bool = True):
    """
    Search & Check greedy rollout advantage & topological similarity
    """
    search_data, search_ppt, abs_logits = sorl_rollout_v3(tokens, model, n=n, K=K, 
                                                        max_iterations=max_iterations,
                                                        memory_span=memory_span,
                                                        attn_blocksize=attn_blocksize,
                                                        temperature=temperature,
                                                        truncate_seq_len=truncate_seq_len)
    search_ppt = search_ppt.reshape(search_data.shape[0], -1)

    # Get valid positions
    bos_pos_mask = torch.logical_and(
        search_data[:, :-1] != BOS_TOKEN_ID, 
        search_data[:, 1:] != BOS_TOKEN_ID
    ).float()
    
    traj_mask = (search_data[:, 1:] < model.vocab_sizes[0]).float()
    
    # --- greedy rollout's advantage ---
    valid_traj_mask = bos_pos_mask * traj_mask
    raw_ppt_adv = (search_ppt[1:].mean(dim=0) - search_ppt[0]) / (search_ppt[1:].mean(dim=0) + 1e-8)
    search_adv = (raw_ppt_adv * valid_traj_mask[0]).sum() / valid_traj_mask[0].sum().clamp(min=1)

    # --- losses ---
    greedy_ppt = search_ppt[0]
    valid_traj_mask = valid_traj_mask[0].bool()
    abs_mask = (1 - traj_mask[0]).bool()
    traj_mask = traj_mask[0].bool()
    
    traj_loss = greedy_ppt[valid_traj_mask] 
    abs_loss = greedy_ppt[abs_mask]
    
    greedy_abs_logits = abs_logits[0, abs_mask, :]
    greedy_abs_tokens = search_data[0, 1:][abs_mask]

    normed = F.normalize(greedy_abs_logits, dim=-1)
    sim_matrix = normed @ normed.T  # shape: [num_abs, num_abs]
    avg_logit_sim = (sim_matrix.sum() - sim_matrix.trace()) / (sim_matrix.numel() - sim_matrix.shape[0])  # exclude diagonal


    return search_data[:1], search_adv, traj_loss, abs_loss, greedy_abs_logits, greedy_abs_tokens, avg_logit_sim


def generate(model, idx, K, max_iterations=0, memory_span=1792, attn_blocksize=1792, temperature=0.0):
    
    # --- insert placeholder tokens ---
    insert_mask = infer_rythmic_insert_mask(idx, K, model.vocab_sizes[0])
    idx = insert_tokens(idx, insert_mask, model.vocab_sizes[0].item())

    # --- prefix recursion (abstraction) ---
    # recursion_mask = (idx >= model.vocab_sizes[0]) # abstract recursion
    recursion_mask = (idx == model.vocab_sizes[0]) # abstract pre-fill only
    recursion_mask[:, 0] = False
    for _ in range(max_iterations): 
        _, logits = model.forward(idx, memory_span, attn_blocksize)
        idx = extract_and_sample(
            logits, idx, recursion_mask, model.vocab_sizes, temperature
        )
    
    # --- generate next trajectory token ---
    _, logits = model.forward(idx, memory_span, attn_blocksize)
    next_token_logits = logits[:, -1, :model.vocab_sizes[0]]

    if temperature == 0.0:
        new_tokens = torch.argmax(next_token_logits, dim=-1)
    else:
        probs = F.softmax(next_token_logits / temperature, dim=-1)
        new_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

    idx = torch.cat((idx, new_tokens.unsqueeze(1)), dim=1)
    return idx


def reinit_model(model, mode: int = 0):
    def _normal_slice(tensor, slc=None):
        tgt = tensor.detach() if slc is None else tensor.detach()[slc]
        torch.nn.init.normal_(tgt, mean=0.0, std=1.0)

    def _zero_slice(tensor, slc=None):
        tgt = tensor.detach() if slc is None else tensor.detach()[slc]
        tgt.zero_()

    with torch.no_grad():
        vocab_split = model.vocab_sizes[0].item()
        abstract_slice = slice(vocab_split, None)

        if mode == 0:  # abstract tokens only
            _normal_slice(model.transformer.wte.weight, abstract_slice)
            _zero_slice(model.lm_head.weight, abstract_slice)

        elif mode == 1:  # entire embedding + head
            _normal_slice(model.transformer.wte.weight)
            _zero_slice(model.lm_head.weight)

        elif mode == 2: # abstract head only 
            _normal_slice(model.lm_head.weight, abstract_slice)
        
        elif mode == 3: # abstract embedding only
            _normal_slice(model.transformer.wte.weight, abstract_slice)

        elif mode == 4: # all parameter re-initialized
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                    
                # Special scalar parameters
                if 'skip_weights' in name:
                    torch.nn.init.ones_(param)
                elif 'lamb' in name and param.dim() == 0:  # CausalSelfAttention.lamb
                    param.data.fill_(0.5)
                elif 'lambdas' in name and param.dim() == 1:  # Block.lambdas
                    param.data.copy_(torch.tensor([1., 0.], device=param.device))
                # Projection layers that should be zeroed
                elif 'lm_head.weight' in name or 'c_proj.weight' in name:
                    torch.nn.init.zeros_(param)
                # 2D+ parameters (Linear layers, embeddings)
                elif param.dim() >= 2:
                    if 'wte' in name:  # Embedding
                        torch.nn.init.normal_(param, mean=0.0, std=1.0)
                    else:  # Linear layers - use Kaiming uniform like PyTorch default
                        torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                # Other scalar parameters (biases, etc.) - PyTorch Linear doesn't use bias
                else:
                    torch.nn.init.zeros_(param)

        else:
            raise ValueError(f"Unknown reinit mode {mode}")