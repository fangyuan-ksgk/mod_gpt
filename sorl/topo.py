'''
Topological Similarity: 
1. Levenshtein distance d(a1, a2)
2. end-of-doc representation distance d(r1, r2) (I use logits as proxy, linear projected representation)
3. Topological Similarity = Correlation between topological distance and end-of-doc representation distance
'''

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.init as init

def pairwise_hamming_dist(abs_data, normalize=True):
    """
    Compute pairwise Hamming distance between all sequences.
    """
    mismatches = (abs_data[:, None, :] != abs_data[None, :, :]).float()
    hamming = mismatches.sum(dim=-1)
    if normalize:
        seq_len = abs_data.shape[1]
        hamming = hamming / seq_len
    return hamming

def compute_correlation(levenshtein_dist_matrix, util_dist_matrix):
    n_rollouts = levenshtein_dist_matrix.shape[0]
    mask = torch.triu(torch.ones(n_rollouts, n_rollouts), diagonal=1).bool()

    lev_pairs = levenshtein_dist_matrix[mask, :].flatten()  
    util_pairs = util_dist_matrix[mask, :].flatten()

    lev_centered = lev_pairs - lev_pairs.mean()
    util_centered = util_pairs - util_pairs.mean()
    
    cov = (lev_centered * util_centered).mean()
    std_lev = lev_centered.square().mean().sqrt()
    std_util = util_centered.square().mean().sqrt()
    
    correlation = cov / (std_lev * std_util + 1e-8)
    return correlation

def compute_covariance(dist_matrix: torch.Tensor,
                         util_matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute covariance between distance and utility matrices.
    """
    assert dist_matrix.shape == util_matrix.shape, "Shape mismatch"
    n, _, _ = dist_matrix.shape
    if n < 2:
        return torch.tensor(0.0, device=dist_matrix.device)

    tri_mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=dist_matrix.device), diagonal=1)
    pair_dist = dist_matrix[tri_mask, :]      # (n_pairs, n_docs)
    pair_util = util_matrix[tri_mask, :]      # (n_pairs, n_docs)

    pair_dist_centered = pair_dist - pair_dist.mean(dim=1, keepdim=True)
    pair_util_centered = pair_util - pair_util.mean(dim=1, keepdim=True)

    pair_cov = (pair_dist_centered * pair_util_centered).mean(dim=1)

    coeff = 2.0 / (n * (n - 1))
    return coeff * pair_cov.sum()

# --------- Topological Similarity Regularization Loss: ------------------------
# (Mode 1). - d(a1, a2) * d(p(s|a1), p(s|a2))
# (Mode 2). max(k d(a1, a2) - d(p(s|a1), p(s|a2)), 0)**2
# (Mode 3). - correlation(d(a1, a2), d(p(s|a1), p(s|a2)))
# (Mode 4). - covariance(d(a1, a2), d(p(s|a1), p(s|a2)))
# Any other suggestions? 
# --------- Hope to improve correlation(d(a1, a2), d(p(s|a1), p(s|a2))) ---------

def compute_topo_loss(abs_dist, util_dist, mode: int = 0):
    """
    Compute topological similarity regularization loss. (Using vector)
    """
    if mode == 0: # dot product
        return - (abs_dist * util_dist).mean()
    elif mode == 1: # correlation
        abs_centered = abs_dist - abs_dist.mean()
        util_centered = util_dist - util_dist.mean()
        cov = (abs_centered * util_centered).mean()
        std_abs = abs_centered.square().mean().sqrt()  # standard deviations
        std_util = util_centered.square().mean().sqrt()
        corr = cov / (std_abs * std_util + 1e-8).detach()       # avoid
        return -corr
    elif mode == 2: # covariance
        abs_centered = abs_dist - abs_dist.mean()
        util_centered = util_dist - util_dist.mean()
        cov = (abs_centered * util_centered).mean()
        return -cov
    else: 
        raise ValueError(f"Unknown mode: {mode}")


def orthogonalize_abs_param(model, gain=1.0, do_wte=True, do_head=True):

    traj_vocab_size = model.vocab_sizes[0]
    total_vocab_size = sum(model.vocab_sizes)

    if do_wte:  
        with torch.no_grad():
            # Create temporary matrix for orthogonalization
            abs_wte = model.transformer.wte.weight[traj_vocab_size:total_vocab_size]
            init.orthogonal_(abs_wte, gain=gain)
            model.transformer.wte.weight[traj_vocab_size:total_vocab_size] = abs_wte
    
    if do_head:
        with torch.no_grad():
            abs_head = model.lm_head.weight[traj_vocab_size:total_vocab_size]
            init.orthogonal_(abs_head, gain=gain)
            model.lm_head.weight[traj_vocab_size:total_vocab_size] = abs_head