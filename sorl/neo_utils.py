import torch
# from sorl.gat_act import BOS_TOKEN_ID, search, GAT, recursion, infer_level
from sorl.gat_sim import BOS_TOKEN_ID, GAT, recursion, extract_and_sample, recursion_v2
import torch.nn.functional as F
from typing import Optional, Union
from sorl.topo import doc_hamming_dist_pairwise, compute_correlation, compute_topo_loss, doc_util_dist, doc_levenshtein_dist, compute_util_dist
from torch import nn 


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

    if select_mode == "abs_ppt":
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
        # curiosity + utility preference (naive combo 3:1)
        doc_abs_ppt_mean = doc_abs_ppt.mean(dim=0, keepdim=True)
        doc_abs_ppt_std = doc_abs_ppt.std(dim=0, keepdim=True, unbiased=False).clamp(min=1e-8)
        curiosity_advantage = (doc_abs_ppt - doc_abs_ppt_mean) / doc_abs_ppt_std
        utility_advantage = (doc_ppt_mean - doc_ppt) / doc_ppt_std
        advantage = curiosity_advantage + 0.3 * utility_advantage
    elif mode == 6: 
        # curiosity + utility preference (naive combo 10:1)
        doc_abs_ppt_mean = doc_abs_ppt.mean(dim=0, keepdim=True)
        doc_abs_ppt_std = doc_abs_ppt.std(dim=0, keepdim=True, unbiased=False).clamp(min=1e-8)
        curiosity_advantage = (doc_abs_ppt - doc_abs_ppt_mean) / doc_abs_ppt_std
        utility_advantage = (doc_ppt_mean - doc_ppt) / doc_ppt_std
        advantage = curiosity_advantage + 0.1 * utility_advantage
    elif mode == 7: 
        # SGPO + familiarity preference (3:1)
        doc_abs_ppt_mean = doc_abs_ppt.mean(dim=0, keepdim=True)
        doc_abs_ppt_std = doc_abs_ppt.std(dim=0, keepdim=True, unbiased=False).clamp(min=1e-8)
        familiarity_advantage = (doc_abs_ppt - doc_abs_ppt_mean) / doc_abs_ppt_std
        utility_disadvantage = (doc_ppt - doc_ppt_mean) / doc_ppt_std
        advantage = 0.3 * familiarity_advantage + utility_disadvantage
    elif mode == 8: 
        # curiosity + utility preference (0.1:0.9) | collapsed vocabulary with no advantage
        doc_abs_ppt_mean = doc_abs_ppt.mean(dim=0, keepdim=True)
        doc_abs_ppt_std = doc_abs_ppt.std(dim=0, keepdim=True, unbiased=False).clamp(min=1e-8)
        curiosity_advantage = (doc_abs_ppt - doc_abs_ppt_mean) / doc_abs_ppt_std
        utility_advantage = (doc_ppt_mean - doc_ppt) / doc_ppt_std
        advantage = 0.1 * curiosity_advantage + 0.9 * utility_advantage
    elif mode == 9: 
        # curiosity + utility preference (naive combo 0.25:0.75)
        doc_abs_ppt_mean = doc_abs_ppt.mean(dim=0, keepdim=True)
        doc_abs_ppt_std = doc_abs_ppt.std(dim=0, keepdim=True, unbiased=False).clamp(min=1e-8)
        curiosity_advantage = (doc_abs_ppt - doc_abs_ppt_mean) / doc_abs_ppt_std
        utility_advantage = (doc_ppt_mean - doc_ppt) / doc_ppt_std
        advantage = 0.25 * curiosity_advantage + 0.75 * utility_advantage
    elif mode == 10: # variance scaling 
        # rms # update running mean and variance
        doc_abs_ppt_norm = scaler.update_and_normalize(doc_abs_ppt)
        curiosity_coef = 0.1  # 0.5 fails to exploit | reduce to 0.1
        doc_rew = curiosity_coef * doc_abs_ppt_norm - doc_ppt
        doc_rew_mean = doc_rew.mean(dim=0, keepdim=True)
        doc_rew_std = doc_rew.std(dim=0, keepdim=True, unbiased=False).clamp(min=1e-8)
        advantage = (doc_rew - doc_rew_mean) / doc_rew_std
    elif mode == 11: # conditional curiosity | no search adv observed, exploration
        alpha = 0.3 # damping factor (blur curiosity gain with utility gain)
        doc_rew = - doc_ppt / (1 + alpha * doc_abs_ppt)
        doc_rew_mean = doc_rew.mean(dim=0, keepdim=True)
        doc_rew_std = doc_rew.std(dim=0, keepdim=True, unbiased=False).clamp(min=1e-8)
        advantage = (doc_rew - doc_rew_mean) / doc_rew_std
    elif mode == 12: # clipped curiosity | no search adv observed, exploration
        doc_abs_ppt_clip = torch.minimum(doc_abs_ppt, doc_ppt.abs())
        doc_rew = doc_abs_ppt_clip - doc_ppt
        doc_rew_mean = doc_rew.mean(dim=0, keepdim=True)
        doc_rew_std = doc_rew.std(dim=0, keepdim=True, unbiased=False).clamp(min=1e-8)
        advantage = (doc_rew - doc_rew_mean) / doc_rew_std
    elif mode == 13: # clipped curiosity
        doc_abs_ppt_clip = torch.minimum(doc_abs_ppt, doc_ppt.abs())
        curiosity_coef = 0.5
        doc_rew = curiosity_coef * doc_abs_ppt_clip - doc_ppt
        doc_rew_mean = doc_rew.mean(dim=0, keepdim=True)
        doc_rew_std = doc_rew.std(dim=0, keepdim=True, unbiased=False).clamp(min=1e-8)
        advantage = (doc_rew - doc_rew_mean) / doc_rew_std
    elif mode == 14: 
        # Only apply curiosity bonus when utility is already good (low ppt)
        utility_gate = torch.sigmoid(-doc_ppt + 1.0)  # High when utility good
        curiosity_bonus = 0.3 * doc_abs_ppt * utility_gate
        doc_rew = -doc_ppt + curiosity_bonus
        
        doc_rew_mean = doc_rew.mean(dim=0, keepdim=True)
        doc_rew_std = doc_rew.std(dim=0, keepdim=True, unbiased=False).clamp(min=1e-8)
        advantage = (doc_rew - doc_rew_mean) / doc_rew_std
    elif mode == 15: 
        beta = 0.1 
        gating_factor = torch.tanh(beta * doc_ppt.detach())

        r_max = 5.0
        capped_curiosity = doc_abs_ppt.clamp(max=r_max)
        
        lambda_val = 0.3
        curiosity_bonus = gating_factor * lambda_val * capped_curiosity
        doc_rew = -doc_ppt + curiosity_bonus

        # Debugging stats
        print(f"Gate Openness: {gating_factor.mean():.3f}, Bonus: {curiosity_bonus.mean():.3f}, Cost: {doc_ppt.mean():.3f}")

        # Standard Advantage Calculation
        doc_rew_mean = doc_rew.mean(dim=0, keepdim=True)
        doc_rew_std = doc_rew.std(dim=0, keepdim=True, unbiased=False).clamp(min=1e-8)
        advantage = (doc_rew - doc_rew_mean) / doc_rew_std

    doc_adv = torch.where(
        doc_ppt_std > 1e-8,
        advantage,
        torch.zeros_like(doc_ppt)
    )

    # broadcast back to per-token advantage
    token_adv = doc_adv.gather(1, doc_idx[:,1:])
    # token_adv = token_adv * (1 - trajectory_mask) # redundantly line to play it safe
    return token_adv


def compute_rollout_reward_v2(search_data, ppt, levels, mode: int = 0, topo_abs_dist_mode: int = 0): 
    
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
        # SGPO, encourage picking a with low p(s | a)
        advantage = (doc_ppt - doc_ppt_mean) / doc_ppt_std
    elif mode == 1:
        # No advantage (baseline MLE) | more stable abstraction | all-rollout SoRL
        advantage = torch.ones_like(doc_ppt)
    elif mode == 2: 
        # distillation, encourage picking a with high p(a | s)
        doc_abs_ppt_mean = doc_abs_ppt.mean(dim=0, keepdim=True)
        doc_abs_ppt_std = doc_abs_ppt.std(dim=0, keepdim=True, unbiased=False).clamp(min=1e-8)
        advantage = - (doc_abs_ppt - doc_abs_ppt_mean) / doc_abs_ppt_std
    elif mode == 3: 
        # exploitation, encourage picking a with high p(s | a)
        advantage = (doc_ppt_mean - doc_ppt) / doc_ppt_std
    elif mode == 4:
        # exploration, encourage picking a with low p(a | s)
        doc_abs_ppt_mean = doc_abs_ppt.mean(dim=0, keepdim=True)
        doc_abs_ppt_std = doc_abs_ppt.std(dim=0, keepdim=True, unbiased=False).clamp(min=1e-8)
        advantage = (doc_ppt - doc_ppt_mean) / doc_ppt_std

    doc_adv = torch.where(
        doc_ppt_std > 1e-8,
        advantage,
        torch.zeros_like(doc_ppt)
    )

    token_adv = doc_adv.gather(1, doc_idx[:,1:])

    # --- abs distance matrix --- 
    abs_mask = levels.bool()
    # This is likely the culprit, we need to benchmark the speed of this one ...
    abs_dist = doc_hamming_dist_pairwise(search_data, doc_idx, abs_mask, normalize=True)

    return token_adv, abs_dist


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


def evaluate_topo_similarity(search_data, ppt, model, topo_mode: int = 1, util_dist_mode: int = 0):
    """Again, Levenshtein distance is too slow for GPU, we keep the 'True levenshtein' commented out and use the battle-proof ver. """ 

    levels = (search_data >= model.vocab_sizes[0]).long()
    # (a). document idx
    doc_idx = (search_data == BOS_TOKEN_ID).cumsum(dim=1)
    doc_idx = doc_idx - doc_idx.min(dim=1, keepdim=True).values # idx starts from 0 

    # (b). per-document trajectory ppt
    trajectory_mask = (levels[:, 1:] == 0).float()
    trajectory_ppt = ppt * trajectory_mask
    abs_ppt = ppt * (1 - trajectory_mask)

    doc_ppt = avg_ppt_per_sample(trajectory_ppt, doc_idx[:,1:])  # Shape: (n_rollouts, n_docs)

    # (Alternative: aligned with training setting, use hamming distance)
    abs_mask = levels.bool()
    abs_dist = doc_hamming_dist_pairwise(search_data, doc_idx, abs_mask, normalize=True)
    
    util_dist = compute_util_dist(ppt, mode=util_dist_mode)

    topo_loss = compute_topo_loss(abs_dist, util_dist, mode=topo_mode)

    # # (c). d(a1, a2) :: pairwise levenshtein distance matrix || simplest case, n=2
    # abs_mask = search_data >= model.vocab_sizes[0]  # True for abstraction tokens
    # abs_dist_matrix = doc_levenshtein_dist(search_data, doc_idx, abs_mask, normalize=True)

    # # (d). d(p(s|a1), p(s|a2)) :: utility distance matrix
    # util_dist_matrix = doc_util_dist(doc_ppt)

    # # (e). Compute correlation
    # correlation = compute_correlation(abs_dist_matrix, util_dist_matrix)

    return -topo_loss

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

    # --- topological similarity ---
    correlation = evaluate_topo_similarity(search_data, search_ppt, model, topo_mode = 1)
    
    return search_data[:1], search_adv, traj_loss, abs_loss, correlation

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

    valid_traj_mask = bos_pos_mask[0] * traj_mask
    valid_abs_mask = bos_pos_mask[0] * abs_mask

    traj_loss = (best_ppt[0] * valid_traj_mask).sum() / valid_traj_mask.sum().clamp(min=1)
    abs_loss = (best_ppt[0] * valid_abs_mask).sum() / valid_abs_mask.sum().clamp(min=1)
    return traj_loss, abs_loss

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

    valid_traj_mask = bos_pos_mask[0] * traj_mask
    valid_abs_mask = bos_pos_mask[0] * abs_mask

    traj_loss = (best_ppt[0] * valid_traj_mask).sum() / valid_traj_mask.sum().clamp(min=1)
    abs_loss = (best_ppt[0] * valid_abs_mask).sum() / valid_abs_mask.sum().clamp(min=1)

    comput_kl = lambda p_new, p_ref: torch.exp(p_ref - p_new) - (p_ref - p_new) - 1
    kl_loss = comput_kl(best_ppt[0], ref_ppt[0])
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

    traj_loss = (best_ppt[0] * valid_traj_mask).sum() / valid_traj_mask.sum().clamp(min=1)
    abs_loss = (best_ppt[0] * valid_abs_mask).sum() / valid_abs_mask.sum().clamp(min=1)
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

def compute_sgpo_loss_v2(rollout_data, token_adv, abs_dist, model, memory_span: int, attn_blocksize: int, topo_mode: int = 1, util_dist_mode: int = 1):
    """Default: correlation based topo regularization + stop grad on worse rollout"""
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

    # Compute topological similarity regularization loss
    util_dist = compute_util_dist(rollout_ppt, mode=util_dist_mode)
    topo_loss = compute_topo_loss(abs_dist, util_dist, mode=topo_mode)

    return traj_loss, grpo_loss, topo_loss


# def compute_inverse_sgpo_loss(rollout_data, token_adv, model, memory_span: int, attn_blocksize: int):
#     """
#     Inverse SGPO + exploitation (alpha_loss > 0) leads to emergence of abstraction structure (non-collapsed greedy vocab utilization) 
#     """
#     rollout_ppt, _ = model.forward(rollout_data, memory_span, attn_blocksize)
#     rollout_ppt = rollout_ppt.reshape(rollout_data.shape[0], -1)

#     levels = (rollout_data >= model.vocab_sizes[0]).long()[:, 1:]  # [n_rollouts, seq_len-1]
#     bos_pos_mask = torch.logical_and(
#         rollout_data[:, :-1] != BOS_TOKEN_ID, 
#         rollout_data[:, 1:] != BOS_TOKEN_ID
#     ).float()
    
#     traj_mask = (levels == 0).float()  
#     abs_mask = (levels > 0).float() 
#     valid_traj_mask = bos_pos_mask * traj_mask 
#     valid_abs_mask = bos_pos_mask * abs_mask

#     abs_loss = (rollout_ppt * valid_abs_mask).sum() / valid_abs_mask.sum().clamp(min=1)

#     # Improve utility discriminately 
#     surrogate_loss = rollout_ppt * token_adv 
#     traj_loss = (surrogate_loss * valid_traj_mask).sum() / valid_traj_mask.sum().clamp(min=1)    

#     return traj_loss, abs_loss


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

        else:
            raise ValueError(f"Unknown reinit mode {mode}")