# Information Gain Reward for Abstraction 
# Source: https://github.com/NVlabs/RLP?tab=readme-ov-file#paper

# InfoGain(a) = Policy(x_{i} | x_{<i}, a) - EMA(Policy(x_{i} | x_{<i}))
# (1). GRPO loss + normal reward is 6x worse than SoRL. 
# (2). Adding EMA reference model to-be-tested, but I'm not excited about it, as it complicates implementation 
#      and is unlikely to be 6x better etc. SoRL is simpler. 

from copy import deepcopy
import torch
from src.utils import infer_level
from model.model_sorl import SorlModelWrapper
from typing import List


class EMAModel:
    """
    Maintains an Exponential Moving Average of a model's parameters.
    """
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.ema_model = deepcopy(model).eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, main_model: torch.nn.Module):
        training_mode = main_model.training
        main_model.eval()

        for ema_param, main_param in zip(self.ema_model.parameters(), main_model.parameters()):
            ema_param.data.mul_(self.decay).add_(main_param.data, alpha=1 - self.decay)

        if training_mode:
            main_model.train()

    def __call__(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)

def compute_token_advantage(rewards: torch.Tensor, orig_indices: torch.Tensor, token_indices: torch.Tensor):
    """
    Computes advantage for each token, grouped by original sample index and token position.
    Advantages are normalized per-group (mean=0, std=1).
    """
    # Create a unique ID for each group (original sample, token position)
    grouping_tensor = orig_indices * (token_indices.max() + 1) + token_indices
    unique_groups, inverse_indices = torch.unique(grouping_tensor, return_inverse=True)
    num_groups = len(unique_groups)

    counts = torch.bincount(inverse_indices, minlength=num_groups).float().clamp(min=1)
    
    means = torch.zeros(num_groups, device=rewards.device).scatter_add_(0, inverse_indices, rewards) / counts
    vars_sum = torch.zeros(num_groups, device=rewards.device).scatter_add_(0, inverse_indices, rewards**2)
    variances = (vars_sum / counts) - means**2
    stds = torch.sqrt(variances.clamp(min=1e-8))

    advantages = (rewards - means[inverse_indices]) / (stds[inverse_indices] + 1e-5)
    return advantages

# Info-gain requires 'abstraction free' sequence's ppt, however, when multiple abstraction levels 
# are present, abstraction-free requires 'level-free' sequence, that's a bit more complicated

def compute_abstract_token_rewards(
    model: SorlModelWrapper,
    data: torch.Tensor,
    sample_idx: torch.Tensor,
    ppt: torch.Tensor,
    loss_mask: torch.Tensor,
) -> dict:
    """
    Computes per-token rewards for abstractions based on the average negative perplexity
    of all subsequent, unmasked trajectory tokens at the level below.
    """
    levels = infer_level(data, model.vocab_sizes, model.level_mask_tokens[0])
    indices = torch.arange(data.size(1), device=data.device)

    padded_loss_mask = torch.nn.functional.pad(loss_mask, (0, data.size(1) - loss_mask.size(1)), value=0)
    broadcasted_mask = padded_loss_mask[sample_idx][:, 1:].bool()

    reward_lookups = {l: {} for l in range(1, len(model.vocab_sizes))}

    abs_mask = (levels[:, 1:] > 0)
    if not abs_mask.any():
        return {0: None, **reward_lookups}

    abs_locs = torch.where(abs_mask)
    
    for s_idx, abs_token_idx in zip(*abs_locs):
        s_idx, abs_token_idx = s_idx.item(), abs_token_idx.item()
        abs_level = levels[s_idx, abs_token_idx + 1].item()

        suffix_mask = (indices[1:] > (abs_token_idx + 1))
        level_mask = (levels[s_idx, 1:] == abs_level - 1)
        
        final_reward_mask = suffix_mask & level_mask & broadcasted_mask[s_idx]
        
        reward = 0.0
        if final_reward_mask.sum() > 0:
            reward = -ppt[s_idx][final_reward_mask].mean().item()

        if s_idx not in reward_lookups[abs_level]:
            reward_lookups[abs_level][s_idx] = {}
        reward_lookups[abs_level][s_idx][abs_token_idx] = torch.tensor(reward, device=data.device)
            
    return {0: None, **reward_lookups}

def compute_grpo_loss(
    model: SorlModelWrapper,
    data: torch.Tensor,
    sample_idx: torch.Tensor,
    ppt: torch.Tensor,
    loss_mask: torch.Tensor,
    old_log_probs: List[torch.Tensor],
    epsilon: float = 0.2,
):
    """
    Computes the GRPO loss using per-token rewards derived from masked trajectory perplexity.
    """
    levels = infer_level(data, model.vocab_sizes, model.level_mask_tokens[0])
    reward_lookups = compute_abstract_token_rewards(model, data, sample_idx, ppt.detach(), loss_mask)
    
    total_loss = torch.tensor(0.0, device=data.device)
    L = len(model.vocab_sizes)

    for l in range(1, L): 
        level_mask = (levels[:, 1:] == l)
        if not level_mask.any():
            continue

        new_level_log_probs = -ppt[level_mask]
        old_level_log_probs_l = old_log_probs[l - 1]
        
        if old_level_log_probs_l is None or len(old_level_log_probs_l) == 0:
            continue

        abs_locs = torch.where(level_mask)
        pt_sample_idx = abs_locs[0]
        pt_token_idx = abs_locs[1]
        
        try:
            pt_rewards = torch.stack([
                reward_lookups[l][s_idx.item()][t_idx.item()] 
                for s_idx, t_idx in zip(pt_sample_idx, pt_token_idx)
            ])
        except KeyError:
            continue

        orig_indices = sample_idx[pt_sample_idx]
        advantages = compute_token_advantage(pt_rewards, orig_indices, pt_token_idx)
        
        ratio = torch.exp(new_level_log_probs - old_level_log_probs_l)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
        surrogate_loss = torch.min(surr1, surr2)

        per_token_loss = surrogate_loss
        total_loss -= per_token_loss.mean()

    return total_loss  