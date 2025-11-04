import torch
# from sorl.gat_act import BOS_TOKEN_ID, search, GAT, recursion, infer_level
from sorl.gat_sim import BOS_TOKEN_ID, GAT, recursion, extract_and_sample
import torch.nn.functional as F
from typing import Optional

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
    
    insert_mask = (within_doc_pos % K == 0) & (within_doc_pos > 0)
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
def sorl_rollout(data: torch.Tensor, model: GAT, n: int, K: int, max_iterations: int, memory_span: int, attn_blocksize: int, temperature: float,
                 truncate_seq_len: bool = True):
    """
    Perform rollout with 1 greedy sample and (n-1) stochastic samples.
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
    greedy_data, greedy_ppt = recursion(model, repeat_data[:1], max_iterations=max_iterations, 
                            memory_span=memory_span, attn_blocksize=attn_blocksize, temperature=0.0)

    if n == 1: 
        return greedy_data, greedy_ppt
    else: 
        stochastic_data, stochastic_ppt = recursion(model, repeat_data[1:], max_iterations=max_iterations, 
                                    memory_span=memory_span, attn_blocksize=attn_blocksize, temperature=temperature)

        combined_data = torch.cat([greedy_data, stochastic_data], dim=0)
        combined_ppt = torch.cat([greedy_ppt, stochastic_ppt], dim=0)
        return combined_data, combined_ppt


@torch.no_grad()
def sorl_rollout_v2(data: torch.Tensor, model: GAT, n: int, K: int, max_iterations: int, memory_span: int, attn_blocksize: int, temperature: float,
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

def select_best_per_doc(search_data, ppt, levels):
    """Select best rollout per document and stitch together."""
    trajectory_mask = (levels[:, 1:] == 0).float()
    trajectory_ppt = ppt * trajectory_mask
    
    doc_idx = (search_data == BOS_TOKEN_ID).cumsum(dim=1) - 1    
    doc_ppt = avg_ppt_per_sample(trajectory_ppt, doc_idx[:,1:])

    min_doc_ppt, best_rollout_per_doc = doc_ppt.min(dim=0)
    rollout_for_each_pos = best_rollout_per_doc[doc_idx[0]]

    best_seq = search_data[rollout_for_each_pos, torch.arange(search_data.shape[1], device=search_data.device)]
    best_ppt = ppt[rollout_for_each_pos[1:], torch.arange(ppt.shape[1], device=ppt.device)]

    max_doc_ppt = doc_ppt.max(dim=0).values
    best_ppt_advantage = (max_doc_ppt - min_doc_ppt) / max_doc_ppt.clamp(min=1e-8)

    # --- per document abstraction perplexity (for best rollout) ---
    abs_ppt = ppt * (1 - trajectory_mask)
    doc_abs_ppt = avg_ppt_per_sample(abs_ppt, doc_idx[:, 1:])
    best_doc_abs_ppt = doc_abs_ppt[best_rollout_per_doc, torch.arange(doc_abs_ppt.shape[1], device=doc_abs_ppt.device)] 
    per_pos_curiosity_advantage = best_doc_abs_ppt[doc_idx[0, 1:]]

    return best_seq.unsqueeze(0), best_ppt, best_ppt_advantage.mean(), normalize_advantage(per_pos_curiosity_advantage)


# Reflection 1. what if we have a 'information bottleneck mask' to mute future influence 
#               applied to both ACT & selection? 
import time
from sorl.eval import compute_vocab_utilization_rate

def sorl_search(tokens, model, n=3, K=3, max_iterations=1,
                memory_span=1792, attn_blocksize=1792, temperature=1.0, truncate_seq_len: bool = True, loss_mask: Optional[torch.Tensor] = None):
    """
    Complete SoRL search pipeline:
    1. Generate n rollouts (1 greedy + (n-1) stochastic)
    2. Evaluate rollouts via recursion
    3. Select best rollout per document
    Returns: 
    - best_data: Best rollout per document [1, seq_len]
    - best_ppt: Perplexity of best rollout [seq_len - 1]
    """
    # --- generate & evaluate rollouts ---
    search_data, search_ppt = sorl_rollout(tokens, model, n=n, K=K, 
                               max_iterations=max_iterations,
                               memory_span=memory_span,
                               attn_blocksize=attn_blocksize,
                               temperature=temperature,
                               truncate_seq_len=truncate_seq_len)
    search_ppt = search_ppt.reshape(search_data.shape[0], -1)

    # --- select best rollouts (based on trajectory perplexity) ---
    levels = (search_data >= model.vocab_sizes[0]).long()
    best_data, best_ppt, best_ppt_advantage, per_pos_curiosity = select_best_per_doc(search_data, search_ppt, levels)
    
    return best_data, best_ppt, best_ppt_advantage, per_pos_curiosity

def sorl_search_v2(tokens, model, n=3, K=3, max_iterations=1,
                memory_span=1792, attn_blocksize=1792, temperature=1.0, truncate_seq_len: bool = True, loss_mask: Optional[torch.Tensor] = None):
    """
    Complete SoRL search pipeline:
    1. Direct rollout without greedy sample. 
    2. Evaluate rollouts via recursion
    3. Select best rollout per document
    """
    # --- generate & evaluate rollouts ---
    search_data, search_ppt = sorl_rollout_v2(tokens, model, n=n, K=K, 
                               max_iterations=max_iterations,
                               memory_span=memory_span,
                               attn_blocksize=attn_blocksize,
                               temperature=temperature,
                               truncate_seq_len=truncate_seq_len)
    search_ppt = search_ppt.reshape(search_data.shape[0], -1)

    # --- select best rollouts (based on trajectory perplexity) ---
    levels = (search_data >= model.vocab_sizes[0]).long()
    best_data, best_ppt, best_ppt_advantage, per_pos_curiosity = select_best_per_doc(search_data, search_ppt, levels)
    return best_data, best_ppt, best_ppt_advantage, per_pos_curiosity


def sorl_evaluate(tokens, model, n=2, K=4, max_iterations=1, memory_span=1792, attn_blocksize=1792, temperature=1.0,
                  loss_mask: Optional[torch.Tensor] = None, truncate_seq_len: bool = True):
    """
    Search & Check greedy rollout advantage
    """
    search_data, search_ppt = sorl_rollout(tokens, model, n=n, K=K, 
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


# (TBD). optional 'loss_mask' argument for QA task
# --------------------------------------------------
# - It's bad idea to keep the 'None' input, it's only here for experimental purpose
# - to optimize for GPU runs, we need to fix the input format
# --------------------------------------------------

def compute_loss(best_data, model, memory_span: int, attn_blocksize: int, 
                 per_pos_curiosity: Optional[torch.Tensor] = None,
                 loss_mask: Optional[torch.Tensor] = None):
    """Compute trajectory and abstraction loss from sorl_search output."""

    best_ppt, _ = model.forward(best_data, memory_span, attn_blocksize)
    best_ppt = best_ppt.reshape(best_data.shape[0], -1)
    if per_pos_curiosity is not None:
        best_ppt = best_ppt * per_pos_curiosity

    levels = (best_data >= model.vocab_sizes[0]).long()[:, 1:]

    bos_pos_mask = torch.logical_and(
        best_data[:, :-1] != BOS_TOKEN_ID, 
        best_data[:, 1:] != BOS_TOKEN_ID
    ).float()
    traj_mask = (levels == 0).float()[0]
    abs_mask = 1 - traj_mask

    valid_traj_mask = bos_pos_mask[0] * traj_mask
    valid_abs_mask = bos_pos_mask[0] * abs_mask

    traj_loss = (best_ppt[0] * valid_traj_mask).sum() / valid_traj_mask.sum().clamp(min=1)
    abs_loss = (best_ppt[0] * valid_abs_mask).sum() / valid_abs_mask.sum().clamp(min=1)
    return traj_loss, abs_loss


def generate(model, idx, K, max_iterations=0, memory_span=1792, attn_blocksize=1792, temperature=0.0):
    
    # --- insert placeholder tokens ---
    insert_mask = infer_rythmic_insert_mask(idx, K, model.vocab_sizes[0])
    idx = insert_tokens(idx, insert_mask, model.vocab_sizes[0].item())

    # --- prefix recursion (abstraction) ---
    recursion_mask = (idx >= model.vocab_sizes[0])
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