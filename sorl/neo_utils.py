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

def select_best_per_doc(search_data, ppt, levels, model, alpha_select: float = 0.0, select_mode: str = "abs_ppt"):
    trajectory_mask = (levels[:, 1:] == 0).float()
    trajectory_ppt = ppt * trajectory_mask
    abs_ppt = ppt * (1 - trajectory_mask)
    
    doc_idx = (search_data == BOS_TOKEN_ID).cumsum(dim=1)
    doc_idx = doc_idx - doc_idx.min(dim=1, keepdim=True).values # idx starts from 0  
    doc_ppt = avg_ppt_per_sample(trajectory_ppt, doc_idx[:,1:])

    if select_mode == "abs_ppt": # linear combo of curiosity force
        doc_abs_ppt = avg_ppt_per_sample(abs_ppt, doc_idx[:,1:])
        regularized_ppt = doc_ppt - alpha_select * doc_abs_ppt # ver. 1 
    else: 
        z = (search_data * levels - model.vocab_sizes[0]).clamp_min(0)
        pres = torch.zeros(z.size(0), model.vocab_sizes[1] + 1, device=z.device, dtype=torch.bool).scatter_(1, z, True)
        vocab_util = pres[:, 1:].sum(1).float() / model.vocab_sizes[1]
        regularized_ppt = doc_ppt - alpha_select * vocab_util.unsqueeze(1)

    min_doc_ppt, best_rollout_per_doc = regularized_ppt.min(dim=0) # argmin
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

def select_best_info_gain(tokens, base_traj_ppt, search_data, ppt, levels): 
    """Information Gain SoRL"""
    # --- abstraction conditioned perplexity ---
    trajectory_mask = (levels[:, 1:] == 0).float()
    trajectory_ppt = ppt * trajectory_mask
    abs_ppt = ppt * (1 - trajectory_mask)

    doc_idx = (search_data == BOS_TOKEN_ID).cumsum(dim=1)
    doc_idx = doc_idx - doc_idx.min(dim=1, keepdim=True).values # idx starts from 0  
    doc_ppt = avg_ppt_per_sample(trajectory_ppt, doc_idx[:,1:])

    # --- base modeling perplexity ---
    base_doc_idx = (tokens == BOS_TOKEN_ID).cumsum(dim=1)
    base_doc_idx = base_doc_idx - base_doc_idx.min()
    base_doc_ppt = avg_ppt_per_sample(base_traj_ppt[None, :], base_doc_idx[:, 1:])

    # --- info-gain --- 
    doc_info_gain = base_doc_ppt - doc_ppt

    max_doc_info_gain, best_rollout_per_doc = doc_info_gain.max(dim=0) # argmax
    rollout_for_each_pos = best_rollout_per_doc[doc_idx[0]]

    best_seq = search_data[rollout_for_each_pos, torch.arange(search_data.shape[1], device=search_data.device)]
    best_ppt = ppt[rollout_for_each_pos[1:], torch.arange(ppt.shape[1], device=ppt.device)]

    max_doc_ppt, min_doc_ppt = doc_ppt.max(dim=0).values, doc_ppt.min(dim=0).values
    best_ppt_advantage = (max_doc_ppt - min_doc_ppt) / max_doc_ppt.clamp(min=1e-8)

    # --- info-gain reward --- 
    info_gain_reward_doc = torch.exp(max_doc_info_gain)
    info_gain_reward = info_gain_reward_doc[doc_idx[0, 1:]]

    return best_seq.unsqueeze(0), best_ppt, best_ppt_advantage, info_gain_reward

def compute_abs_util(trajectory_ppt, abs_ppt, K): 
    chunk_means = F.avg_pool1d(F.pad(trajectory_ppt.float()[:, 1:], (0, K)), kernel_size=K, stride=1)
    abs_util = chunk_means[abs_ppt > 0].reshape(abs_ppt.shape[0], -1)
    return abs_util

def select_best_per_abs(search_data, ppt, levels, model, K): 

    trajectory_mask = (levels[:, 1:] == 0).float()
    trajectory_ppt = ppt * trajectory_mask
    abs_ppt = ppt * (1 - trajectory_mask)

    abs_util = compute_abs_util(trajectory_ppt, abs_ppt, K) # compute p(s | a)
    min_abs_util, best_rollout_per_abs = abs_util.min(dim=0)

    best_seq = search_data[0].clone()
    is_abs = (levels[0] != 0)
    abs_positions = torch.where(is_abs)[0]
    chosen_tokens = search_data[best_rollout_per_abs, abs_positions]

    best_seq[is_abs] = chosen_tokens

    return best_seq.unsqueeze(0)

# Question #1. This is still 'document-level', I wonder what'd happen if we do 'per-abs-token' level resampling? 
#              for instance, each abs token is in-charge of the next K tokens? But that'd lose the context

def resample_rollout(search_data, ppt, levels, model, tau: float = 2e-4, resample_mode: int = 0, curiosity_epsilon: float = 0.2): 

    trajectory_mask = (levels[:, 1:] == 0).float()
    trajectory_ppt = ppt * trajectory_mask
    abs_ppt = ppt * (1 - trajectory_mask)
    
    doc_idx = (search_data == BOS_TOKEN_ID).cumsum(dim=1)
    doc_idx = doc_idx - doc_idx.min(dim=1, keepdim=True).values # idx starts from 0  
    doc_ppt = avg_ppt_per_sample(trajectory_ppt, doc_idx[:,1:])
    doc_abs_ppt = avg_ppt_per_sample(abs_ppt, doc_idx[:,1:])

    # --- resampling ---
    if resample_mode == 0: # doc-level high utility preference --> collapsed
        signal = doc_ppt
    elif resample_mode == 1: # doc-level low predictability preference --> volatile
        signal = -doc_abs_ppt
    elif resample_mode == 2: # doc-level low utility preference
        signal = -doc_ppt
    elif resample_mode == 3: # doc-level high predictability preference
        signal = doc_abs_ppt
    elif resample_mode == 4: # relative utility preference
        signal = 1.0 * (doc_ppt - doc_ppt.mean()) + \
         0.0 * (-doc_abs_ppt - (-doc_abs_ppt).mean())
    elif resample_mode == 5: # relative utility preference + abstraction curiosity (weak)
        signal = 1.0 * (doc_ppt - doc_ppt.mean()) + \
         0.1 * (-doc_abs_ppt - (-doc_abs_ppt).mean())
    elif resample_mode == 6: # relative utility preference + abstraction curiosity (strong)
        signal = 1.0 * (doc_ppt - doc_ppt.mean()) + \
         0.5 * (-doc_abs_ppt - (-doc_abs_ppt).mean())
    elif resample_mode == 7: # relative utility preference + abstraction curiosity (extreme)
        signal = 1.0 * (doc_ppt - doc_ppt.mean()) + \
         1.0 * (-doc_abs_ppt - (-doc_abs_ppt).mean())
    elif resample_mode == 8: # relative utility preference + abstraction curiosity (reverse)
        signal = 0.5 * (doc_ppt - doc_ppt.mean()) + \
         1.0 * (-doc_abs_ppt - (-doc_abs_ppt).mean())
    elif resample_mode == 9: # w.p. epsion gets curious, otherwise favor utility
        signal = torch.where(torch.rand(doc_ppt.shape[0], device=doc_ppt.device) < curiosity_epsilon, 
                            -doc_abs_ppt, doc_ppt)

    logits = -(signal / tau)
    probs = torch.softmax(logits, dim=0).transpose(0, 1)
    doc_choices = torch.multinomial(probs, num_samples=1, replacement=True).squeeze(1)
    rollout_for_each_pos = doc_choices[doc_idx[0]]

    select_seq = search_data[rollout_for_each_pos, torch.arange(search_data.shape[1], device=search_data.device)]
    select_ppt = ppt[rollout_for_each_pos[1:], torch.arange(ppt.shape[1], device=ppt.device)]

    min_doc_ppt = doc_ppt.min(dim=0).values
    max_doc_ppt = doc_ppt.max(dim=0).values
    best_ppt_advantage = (max_doc_ppt - min_doc_ppt) / max_doc_ppt.clamp(min=1e-8)

    return select_seq.unsqueeze(0), select_ppt, best_ppt_advantage.mean()


class RunningRewardScaler:
    def __init__(self, device='cpu'):
        self.mean = torch.tensor(0.0, device=device)
        self.var = torch.tensor(1.0, device=device)
        self.count = 0

    def update_and_normalize(self, batch_rewards):
        batch_mean = batch_rewards.mean()
        batch_var = batch_rewards.var(unbiased=False)
        batch_count = batch_rewards.numel()  

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

        return batch_rewards / (torch.sqrt(self.var) + 1e-8)

# --------- "Variants" to hard selection + train on single rollout --------
# 1. Soft selection with 'resampling'
# 2. Return all rollout, but with ppt & advantage for each rollout || here we no longer need "stitching"
#    -> we'd need to reduce the seq_len to use more rollouts, then we'd increase the batch_size accordingly, it's fine
# -------------------------------------------------------------------------
def compute_rollout_reward(search_data, ppt, levels, mode: int = 0, scaler: Optional[RunningRewardScaler] = None): 
    trajectory_mask = (levels[:, 1:] == 0).float()
    trajectory_ppt = ppt * trajectory_mask
    abs_ppt = ppt * (1 - trajectory_mask)

    doc_idx = (search_data == BOS_TOKEN_ID).cumsum(dim=1)
    doc_idx = doc_idx - doc_idx.min(dim=1, keepdim=True).values # idx starts from 0  
    doc_ppt = avg_ppt_per_sample(trajectory_ppt, doc_idx[:,1:])
    doc_abs_ppt = avg_ppt_per_sample(abs_ppt, doc_idx[:,1:])

    # group advantage by documnet | same way GRPO's advantage is computed | like GSPO since adv is sequence level
    doc_ppt_mean = doc_ppt.mean(dim=0, keepdim=True)  # Keep dim for broadcasting
    doc_ppt_std = doc_ppt.std(dim=0, keepdim=True, unbiased=False).clamp(min=1e-8)  # Avoid division by zero
    
    if mode == 0:
        # SGPO ver.
        advantage = (doc_ppt - doc_ppt_mean) / doc_ppt_std
    elif mode == 1:
        # No advantage (baseline MLE) | more stable abstraction | all-rollout SoRL
        advantage = torch.ones_like(doc_ppt)
    elif mode == 2: 
        # for distillation, encourage more familiar abstractions
        doc_abs_ppt_mean = doc_abs_ppt.mean(dim=0, keepdim=True)
        doc_abs_ppt_std = doc_abs_ppt.std(dim=0, keepdim=True, unbiased=False).clamp(min=1e-8)
        advantage = - (doc_abs_ppt - doc_abs_ppt_mean) / doc_abs_ppt_std
    elif mode == 3: 
        # for distillation, encourage more useful abstractions
        advantage = (doc_ppt_mean - doc_ppt) / doc_ppt_std
    elif mode == 4: 
        # curiosity (favor less familiar abstraction)
        doc_abs_ppt_mean = doc_abs_ppt.mean(dim=0, keepdim=True)
        doc_abs_ppt_std = doc_abs_ppt.std(dim=0, keepdim=True, unbiased=False).clamp(min=1e-8)
        advantage = (doc_abs_ppt - doc_abs_ppt_mean) / doc_abs_ppt_std
    elif mode == 5: 
        # stochastic exploration encouragement 
        advantage = torch.relu(doc_ppt[:1] - doc_ppt[1:])
    elif mode == 6: 
        # stochastic exploration encouragement (relative improvement)
        advantage = torch.relu((doc_ppt[:1] - doc_ppt[1:]) / (doc_ppt[:1].clamp(min=1e-8)))
        advantage = advantage.clamp(max=1.0)
    elif mode == 7: 
        # stochastic exploration encouragement (binary reward)
        advantage = torch.where(doc_ppt[:1] > doc_ppt[1:], 1.0, 0.0)

    doc_adv = torch.where(
        doc_ppt_std > 1e-8,
        advantage,
        torch.zeros_like(doc_ppt)
    )

    # broadcast back to per-token advantage
    token_adv = doc_adv.gather(1, doc_idx[:,1:])
    # token_adv = token_adv * (1 - trajectory_mask) # redundantly line to play it safe
    return token_adv


# Reflection 1. what if we have a 'information bottleneck mask' to mute future influence 
#               applied to both ACT & selection? 
import time
from sorl.eval import compute_vocab_utilization_rate

def sorl_search(tokens, model, n=3, K=3, max_iterations=1,
                memory_span=1792, attn_blocksize=1792, temperature=1.0, truncate_seq_len: bool = True, 
                alpha_select: float = 0.0, select_mode: str = "abs_ppt",
                loss_mask: Optional[torch.Tensor] = None,
                tau: float = 2e-4, resample_mode: int = 0, curiosity_epsilon: float = 0.2):
    """
    SoRL with resampling (instead of selection)
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
    select_data, select_ppt, select_ppt_advantage = resample_rollout(search_data, search_ppt, levels, model, tau=tau, resample_mode=resample_mode, curiosity_epsilon=curiosity_epsilon)
    return select_data, select_ppt, select_ppt_advantage

def sorl_search_v2(tokens, model, n=3, K=3, max_iterations=1,
                memory_span=1792, attn_blocksize=1792, temperature: Union[float, torch.Tensor] = 1.0, truncate_seq_len: bool = True, 
                alpha_select: float = 0.0, select_mode: str = "abs_ppt", loss_mask: Optional[torch.Tensor] = None):
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
    best_data, best_ppt, best_ppt_advantage = select_best_per_doc(search_data, search_ppt, levels, model, alpha_select=alpha_select, select_mode=select_mode)
    return best_data, best_ppt, best_ppt_advantage


def sorl_search_v3(tokens, model, n=3, K=3, max_iterations=1,
                memory_span=1792, attn_blocksize=1792, temperature: Union[float, torch.Tensor] = 1.0, truncate_seq_len: bool = True, 
                alpha_select: float = 0.0, select_mode: str = "abs_ppt", # placeholder
                loss_mask: Optional[torch.Tensor] = None, mode: int = 0, scaler: Optional[RunningRewardScaler] = None):
    """
    Complete SoRL search pipeline:
    1. Direct rollout without greedy sample. 
    2. Compute reward for each rollout
    """
    # --- generate & evaluate rollouts ---
    search_data, search_ppt = sorl_rollout_v2(tokens, model, n=n, K=K, 
                               max_iterations=max_iterations,
                               memory_span=memory_span,
                               attn_blocksize=attn_blocksize,
                               temperature=temperature,
                               truncate_seq_len=truncate_seq_len)
    search_ppt = search_ppt.reshape(search_data.shape[0], -1)

    # --- compute reward for each rollout ---
    levels = (search_data >= model.vocab_sizes[0]).long()

    token_adv = compute_rollout_reward(search_data, search_ppt, levels, mode=mode, scaler=scaler) # un-utility reward 
    return search_data, search_ppt, token_adv


def sorl_search_v4(tokens, model, n=3, K=3, max_iterations=1,
                memory_span=1792, attn_blocksize=1792, temperature: Union[float, torch.Tensor] = 1.0, truncate_seq_len: bool = True, 
                alpha_select: float = 0.0, select_mode: str = "abs_ppt", # placeholder
                loss_mask: Optional[torch.Tensor] = None, mode: int = 0, topo_abs_dist_mode: int = 0, ref_model: Optional[nn.Module] = None):
    """
    Complete SoRL search pipeline:
    1. Direct rollout without greedy sample. 
    2. Compute reward for each rollout
    """
    # --- generate & evaluate rollouts ---
    if ref_model is not None:
        search_data, search_ppt = sorl_rollout_v2(tokens, ref_model, n=n, K=K, 
                               max_iterations=max_iterations,
                               memory_span=memory_span,
                               attn_blocksize=attn_blocksize,
                               temperature=temperature,
                               truncate_seq_len=truncate_seq_len)
    else:
        search_data, search_ppt = sorl_rollout_v2(tokens, model, n=n, K=K, 
                                max_iterations=max_iterations,
                                memory_span=memory_span,
                                attn_blocksize=attn_blocksize,
                                temperature=temperature,
                                truncate_seq_len=truncate_seq_len)
    search_ppt = search_ppt.reshape(search_data.shape[0], -1)

    # --- compute reward for each rollout & abs distance matrix ---
    levels = (search_data >= model.vocab_sizes[0]).long()

    token_adv, abs_dist = compute_rollout_reward_v2(search_data, search_ppt, levels, mode=mode, topo_abs_dist_mode=topo_abs_dist_mode) # un-utility reward
 
    return search_data, search_ppt, token_adv, abs_dist


def sorl_search_v5(tokens, model, n=3, K=3, max_iterations=1,
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
    if use_per_abs_selection: 
        best_data = select_best_per_abs(search_data, search_ppt, levels, model, K)
    else: 
        best_data, _, _ = select_best_per_doc(search_data, search_ppt, levels, model)  # stay with default for select mode (for now)

    return best_data

def sorl_search_v6(tokens, model, n=3, K=3, max_iterations=1,
                memory_span=1792, attn_blocksize=1792, temperature: Union[float, torch.Tensor] = 1.0, truncate_seq_len: bool = True, r_min: float = 1.0, reward_mode: int = 0): 
    """
    Hopefully the last ver |:->)) | Utility reward scaling
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

    best_data, _, _, utility_reward = select_best_per_doc_v2(search_data, search_ppt, levels, r_min=r_min, reward_mode=reward_mode)
    return best_data, utility_reward

def sorl_search_v7(tokens, model, ema_model, n=3, K=3, max_iterations=1,
                memory_span=1792, attn_blocksize=1792, temperature: Union[float, torch.Tensor] = 1.0, truncate_seq_len: bool = True, r_min: float = 1.0, reward_mode: int = 0): 
    """
    InfoGain reward SoRL
    """
    # --- compute base trajectory perplexity ---
    base_traj_ppt, _ = ema_model.forward(tokens, memory_span, attn_blocksize)
    
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
    best_data, _, _, info_gain_reward = select_best_info_gain(tokens, base_traj_ppt, search_data, search_ppt, levels)
    
    return best_data, info_gain_reward, base_traj_ppt


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
    abs_mask = 1 - traj_mask[0]
    
    valid_traj = valid_traj_mask[0]
    valid_abs = bos_pos_mask[0] * abs_mask
    
    traj_loss = (greedy_ppt * valid_traj).sum() / valid_traj.sum().clamp(min=1)
    abs_loss = (greedy_ppt * valid_abs).sum() / valid_abs.sum().clamp(min=1)

    greedy_abs_logits = abs_logits[0, abs_mask.bool(), :]
    greedy_abs_tokens = search_data[0, 1:][abs_mask.bool()]

    return search_data[:1], search_adv, traj_loss, abs_loss, greedy_abs_logits, greedy_abs_tokens

def compute_loss(best_data, model, memory_span: int, attn_blocksize: int,
                 loss_mask: Optional[torch.Tensor] = None):
    """Compute trajectory and abstraction loss from sorl_search output."""

    best_ppt, _ = model.forward(best_data, memory_span, attn_blocksize)
    best_ppt = best_ppt.reshape(best_data.shape[0], -1)

    levels = (best_data >= model.vocab_sizes[0]).long()[:, 1:]

    bos_pos_mask = torch.logical_and(
        best_data[:, :-1] != BOS_TOKEN_ID, 
        best_data[:, 1:] != BOS_TOKEN_ID
    ).float()
    traj_mask = (levels == 0).float()[0]
    abs_mask = 1 - traj_mask

    valid_traj_mask = bos_pos_mask * traj_mask
    valid_abs_mask = bos_pos_mask * abs_mask

    # Note: we only compute loss on 'first rollout' (greedy one)
    traj_loss = (best_ppt * valid_traj_mask).sum() / valid_traj_mask.sum().clamp(min=1)
    abs_loss = (best_ppt * valid_abs_mask).sum() / valid_abs_mask.sum().clamp(min=1)
    return traj_loss, abs_loss

def compute_loss_with_entropy(data, model, memory_span: int, attn_blocksize: int, 
                                loss_mask: Optional[torch.Tensor] = None, target_entropy: float = 999.9):

    ppt, logits = model.forward(data, memory_span, attn_blocksize)
    ppt = ppt.reshape(data.shape[0], -1)
    logits = logits.reshape(data.shape[0], -1, logits.size(-1))

    levels = (data >= model.vocab_sizes[0]).long()[:, 1:]
    bos_pos_mask = torch.logical_and(
        data[:, :-1] != BOS_TOKEN_ID, 
        data[:, 1:] != BOS_TOKEN_ID
    ).float()

    traj_mask = (levels[0] == 0).float()
    abs_mask = 1 - traj_mask

    valid_traj_mask = bos_pos_mask * traj_mask
    valid_abs_mask = bos_pos_mask * abs_mask

    traj_loss = (ppt * valid_traj_mask).sum() / valid_traj_mask.sum().clamp(min=1)
    abs_loss = (ppt * valid_abs_mask).sum() / valid_abs_mask.sum().clamp(min=1)


    abs_positions = abs_mask.bool()
    abs_logits = logits[:, :-1][:, abs_positions, model.vocab_sizes[0]:]

    dist = Categorical(logits=abs_logits)
    abs_entropy = dist.entropy()
    entropy_loss = torch.clamp(target_entropy - abs_entropy, min=0.0).mean()

    return traj_loss, abs_loss, entropy_loss

def compute_loss_with_kl(best_data, model, ref_model, memory_span: int, attn_blocksize: int,
                 loss_mask: Optional[torch.Tensor] = None):
    """Compute trajectory and abstraction loss from sorl_search output."""

    best_ppt, _ = model.forward(best_data, memory_span, attn_blocksize)
    best_ppt = best_ppt.reshape(best_data.shape[0], -1)
    
    ref_ppt, _ = ref_model.forward(best_data, memory_span, attn_blocksize)
    ref_ppt = ref_ppt.reshape(best_data.shape[0], -1)

    levels = (best_data >= model.vocab_sizes[0]).long()[:, 1:]

    bos_pos_mask = torch.logical_and(
        best_data[:, :-1] != BOS_TOKEN_ID, 
        best_data[:, 1:] != BOS_TOKEN_ID
    ).float()
    traj_mask = (levels == 0).float()[0]
    abs_mask = 1 - traj_mask

    valid_traj_mask = bos_pos_mask * traj_mask
    valid_abs_mask = bos_pos_mask * abs_mask

    traj_loss = (best_ppt * valid_traj_mask).sum() / valid_traj_mask.sum().clamp(min=1)
    abs_loss = (best_ppt * valid_abs_mask).sum() / valid_abs_mask.sum().clamp(min=1)

    comput_kl = lambda p_new, p_ref: torch.exp(p_ref - p_new) - (p_ref - p_new) - 1
    kl_loss = comput_kl(best_ppt, ref_ppt)
    traj_kl_loss = (kl_loss * valid_traj_mask).sum() / valid_traj_mask.sum().clamp(min=1)
    abs_kl_loss = (kl_loss * valid_abs_mask).sum() / valid_abs_mask.sum().clamp(min=1)

    return traj_loss, abs_loss, traj_kl_loss, abs_kl_loss

def compute_weighted_loss(best_data, token_adv, model, memory_span: int, attn_blocksize: int,
                 loss_mask: Optional[torch.Tensor] = None):
    best_ppt, _ = model.forward(best_data, memory_span, attn_blocksize)
    best_ppt = best_ppt.reshape(best_data.shape[0], -1)
    best_ppt = best_ppt * token_adv

    levels = (best_data >= model.vocab_sizes[0]).long()[:, 1:]

    bos_pos_mask = torch.logical_and(
        best_data[:, :-1] != BOS_TOKEN_ID, 
        best_data[:, 1:] != BOS_TOKEN_ID
    ).float()
    traj_mask = (levels == 0).float()[0]
    abs_mask = 1 - traj_mask

    valid_traj_mask = bos_pos_mask[0] * traj_mask
    valid_abs_mask = bos_pos_mask[0] * abs_mask

    traj_loss = (best_ppt.mean(dim=0) * valid_traj_mask).sum() / valid_traj_mask.sum().clamp(min=1)
    abs_loss = (best_ppt.mean(dim=0) * valid_abs_mask).sum() / valid_abs_mask.sum().clamp(min=1)
    return traj_loss, abs_loss     

# Should GRPO loss mask out trajectory tokens? GAPT requires this separation. 
def compute_grpo_loss(rollout_data, rollout_ppt, reference_ppt, token_adv, epsilon: float = 0.1):
    """Standard GRPO loss :: with gradient clipping (PPO style)"""
    ratio = torch.exp(reference_ppt - rollout_ppt) # log(p) - log(p_ref)
    surr1 = ratio * token_adv
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * token_adv
    surrogate_loss = torch.min(surr1, surr2)

    # mask out trajectory tokens
    valid_mask = (token_adv != 0).float()
    grpo_loss = -(surrogate_loss * valid_mask).sum() / valid_mask.sum().clamp(min=1)
    return grpo_loss

def compute_sgpo_loss(rollout_data, token_adv, model, memory_span: int, attn_blocksize: int):
    """
    SGPO + exploitation (alpha_loss > 0) leads to emergence of abstraction structure (non-collapsed greedy vocab utilization) 
    """
    rollout_ppt, _ = model.forward(rollout_data, memory_span, attn_blocksize)
    rollout_ppt = rollout_ppt.reshape(rollout_data.shape[0], -1)

    levels = (rollout_data >= model.vocab_sizes[0]).long()[:, 1:]  # [n_rollouts, seq_len-1]
    bos_pos_mask = torch.logical_and(
        rollout_data[:, :-1] != BOS_TOKEN_ID, 
        rollout_data[:, 1:] != BOS_TOKEN_ID
    ).float()
    
    traj_mask = (levels == 0).float()  
    abs_mask = (levels > 0).float() 
    valid_traj_mask = bos_pos_mask * traj_mask 
    valid_abs_mask = bos_pos_mask * abs_mask

    traj_loss = (rollout_ppt * valid_traj_mask).sum() / valid_traj_mask.sum().clamp(min=1)    

    # Pick discriminatively
    surrogate_loss = rollout_ppt * token_adv 
    grpo_loss = (surrogate_loss * valid_abs_mask).sum() / valid_abs_mask.sum().clamp(min=1)

    return traj_loss, grpo_loss

def compute_partial_loss(rollout_data, token_adv, model, memory_span: int, attn_blocksize: int, w_util_greedy: float = 1.0, w_util_stochastic: float = 1.0): 
    # Encourage partial exploration (stochastic rollout)
    rollout_ppt, _ = model.forward(rollout_data, memory_span, attn_blocksize)
    rollout_ppt = rollout_ppt.reshape(rollout_data.shape[0], -1)

    levels = (rollout_data >= model.vocab_sizes[0]).long()[:, 1:]  # [n_rollouts, seq_len-1]
    bos_pos_mask = torch.logical_and(
        rollout_data[:, :-1] != BOS_TOKEN_ID, 
        rollout_data[:, 1:] != BOS_TOKEN_ID
    ).float()
    
    traj_mask = (levels == 0).float()  
    abs_mask = (levels > 0).float() 
    valid_traj_mask = bos_pos_mask * traj_mask 
    valid_abs_mask = bos_pos_mask * abs_mask

    greedy_util_loss = (rollout_ppt[:1] * valid_traj_mask[:1]).sum() / valid_traj_mask[:1].sum().clamp(min=1)
    stochastic_util_loss = (rollout_ppt[1:] * valid_traj_mask[1:]).sum() / valid_traj_mask[1:].sum().clamp(min=1)
    traj_loss = (w_util_greedy * greedy_util_loss + w_util_stochastic * stochastic_util_loss) / (w_util_greedy + w_util_stochastic)
    
    surrogat_loss = rollout_ppt[1:] * token_adv[1:]
    partial_loss = (surrogat_loss * valid_abs_mask[1:]).sum() / valid_abs_mask[1:].sum().clamp(min=1)

    greedy_abs_loss = (rollout_ppt[:1] * valid_traj_mask[:1]).sum() / valid_traj_mask[:1].sum().clamp(min=1)
 
    return traj_loss, partial_loss, greedy_abs_loss



def compute_clip_sgpo_loss(rollout_data, reference_ppt, token_adv, model, memory_span: int, attn_blocksize: int, epsilon: float = 0.1): 
    """Simplified SGPO loss :: gradient clipping, reference model"""
    
    rollout_ppt, _ = model.forward(rollout_data, memory_span, attn_blocksize)
    rollout_ppt = rollout_ppt.reshape(rollout_data.shape[0], -1)

    levels = (rollout_data >= model.vocab_sizes[0]).long()[:, 1:]  # [n_rollouts, seq_len-1]
    bos_pos_mask = torch.logical_and(
        rollout_data[:, :-1] != BOS_TOKEN_ID, 
        rollout_data[:, 1:] != BOS_TOKEN_ID
    ).float()
    
    traj_mask = (levels == 0).float()  
    abs_mask = (levels > 0).float() 
    valid_traj_mask = bos_pos_mask * traj_mask 
    valid_abs_mask = bos_pos_mask * abs_mask

    # We make all abstraction useful
    traj_loss = (rollout_ppt * valid_traj_mask).sum() / valid_traj_mask.sum().clamp(min=1)    

    # Pick discriminatively
    ratio = torch.exp(reference_ppt - rollout_ppt) # log(p) - log(p_ref)
    surr1 = ratio * token_adv
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * token_adv
    surrogate_loss = torch.min(surr1, surr2)

    sgpo_loss = -(surrogate_loss * valid_abs_mask).sum() / valid_abs_mask.sum().clamp(min=1)
    return traj_loss, sgpo_loss

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