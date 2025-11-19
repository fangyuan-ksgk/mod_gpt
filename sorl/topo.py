'''
Topological Similarity: 
1. Levenshtein distance d(a1, a2)
2. Utility difference |p(s|a1) - p(s|a2)|
3. Topological Similarity = Correlation between topological distance and utility difference
'''

import numpy as np
import torch

# ratio returns the levenshtein ratio instead of levenshtein distance
# print_matrix prints the matrix
# lowercase compares the strings as lowercase

def levenshtein(a,b,ratio=False,print_matrix=False,lowercase=False) :
	if type(a) != type('') :
		raise TypeError('First argument is not a string!')
	if type(b) != type('') :
		raise TypeError('Second argument is not a string!')
	if a == '' :
		return len(b)
	if b == '' :
		return len(a)
	if lowercase :
		a = a.lower()
		b = b.lower()

	n = len(a)
	m = len(b)
	lev = np.zeros((n+1,m+1))

	for i in range(0,n+1) :
		lev[i,0] = i 
	for i in range(0,m+1) :
		lev[0,i] = i

	for i in range(1,n+1) :
		for j in range(1,m+1) :
			insertion = lev[i-1,j] + 1
			deletion = lev[i,j-1] + 1
			substitution = lev[i-1,j-1] + (1 if a[i-1]!= b[j-1] else 0)
			lev[i,j] = min(insertion,deletion,substitution)

	if print_matrix :
		print(lev)

	if ratio :
		return (n+m-lev[n,m])/(n+m)
	else :
		return lev[n,m]


def levenshtein_torch(a: torch.Tensor, b: torch.Tensor, ratio: bool = False) -> torch.Tensor:
	"""
	Compute Levenshtein distance between two 1D tensors (can have different lengths)
	"""
	n = a.size(0)
	m = b.size(0)
	
	if n == 0:
		return torch.tensor(m, dtype=torch.float32, device=a.device)
	if m == 0:
		return torch.tensor(n, dtype=torch.float32, device=a.device)
	
	# Initialize DP matrix
	lev = torch.zeros(n + 1, m + 1, dtype=torch.float32, device=a.device)
	lev[:, 0] = torch.arange(n + 1, dtype=torch.float32, device=a.device)
	lev[0, :] = torch.arange(m + 1, dtype=torch.float32, device=a.device)
	
	# Fill DP matrix
	for i in range(1, n + 1):
		for j in range(1, m + 1):
			cost = 0.0 if a[i-1] == b[j-1] else 1.0
			lev[i, j] = torch.min(
				torch.stack([
					lev[i-1, j] + 1,      # deletion
					lev[i, j-1] + 1,      # insertion
					lev[i-1, j-1] + cost  # substitution
				])
			)
	
	distance = lev[n, m]
	
	if ratio:
		return (n + m - distance) / (n + m)
	return distance

def doc_levenshtein_dist(tokens: torch.Tensor,
                         doc_idx: torch.Tensor,
                         abs_mask: torch.Tensor,
                         normalize: bool = True) -> torch.Tensor:
    """Symmetric Levenshtein distance matrix per document without intermediate padding"""
    
    n_r, _ = tokens.shape
    n_d = doc_idx.max().item() + 1
    device = tokens.device

    doc_masks = [(doc_idx == d) & abs_mask for d in range(n_d)]

    dist = torch.zeros(n_r, n_r, n_d, device=device)

    for d, mask in enumerate(doc_masks):
        seqs = [tokens[r, mask[r]] for r in range(n_r)]
        lengths = [seq.size(0) for seq in seqs]

        for i in range(n_r):
            dist[i, i, d] = 0.0
            for j in range(i + 1, n_r):
                d_ij = levenshtein_torch(seqs[i], seqs[j], ratio=False)

                if normalize:
                    denom = max(lengths[i], lengths[j], 1)
                    d_ij = d_ij / denom

                dist[i, j, d] = d_ij
                dist[j, i, d] = d_ij  # symmetry

    return dist

def doc_hamming_dist(tokens: torch.Tensor,
                     doc_idx: torch.Tensor,
                     abs_mask: torch.Tensor) -> torch.Tensor:
    """
    Fully vectorized Hamming distance assuming identical placeholder positions.
    """
    device = tokens.device
    n_d = doc_idx.max().item() + 1

    doc_masks = [(doc_idx[0] == d) & abs_mask[0] for d in range(n_d)]

    seqs = []
    for mask in doc_masks:
        idx = mask.nonzero(as_tuple=True)[0]  
        seqs.append(tokens[:, idx])           
    seqs = torch.stack(seqs, dim=1)           
    lengths = torch.tensor([mask.sum().item() for mask in doc_masks],
                           device=device)

    mismatches = (seqs.unsqueeze(1) != seqs.unsqueeze(0)).float().sum(dim=-1)
    lengths = lengths.view(1, 1, n_d).clamp(min=1)
    return mismatches / lengths

def doc_levenshtein_dist_pairwise(tokens: torch.Tensor,
                         doc_idx: torch.Tensor,
                         abs_mask: torch.Tensor,
                         normalize: bool = True) -> torch.Tensor:
    """
    Levenshtein distance per document for exactly two rollouts
    """
    n_r, _ = tokens.shape
    assert n_r == 2, "doc_levenshtein_dist currently supports exactly two rollouts"

    n_d = doc_idx.max().item() + 1
    device = tokens.device

    doc_masks = [(doc_idx[0] == d) & abs_mask[0] for d in range(n_d)]

    dist = torch.zeros(n_d, device=device)

    for d, mask in enumerate(doc_masks):
        idx = mask.nonzero(as_tuple=True)[0]
        seq0 = tokens[0, idx]
        seq1 = tokens[1, idx]

        d_val = levenshtein_torch(seq0, seq1, ratio=False)

        if normalize:
            denom = max(seq0.size(0), seq1.size(0), 1)
            d_val = d_val / denom

        dist[d] = d_val  # scalar distance for document d

    dist = dist[doc_idx[:, 1:]] # broadcast back
    return dist

def doc_hamming_dist_pairwise(tokens: torch.Tensor,
                         doc_idx: torch.Tensor,
                         abs_mask: torch.Tensor,
                         normalize: bool = True) -> torch.Tensor:
    """
    Hamming distance per document for exactly two rollouts (aligned positions).
    Much faster than Levenshtein since abstract tokens are at same positions.
    """
    n_r, _ = tokens.shape
    assert n_r == 2, "doc_hamming_dist_pairwise currently supports exactly two rollouts"

    device = tokens.device
    
    abs_mask_0 = abs_mask[0]
    
    mismatches = (tokens[0] != tokens[1]) & abs_mask_0  # [seq_len]
    
    n_d = doc_idx.max().item() + 1
    doc_idx_flat = doc_idx[0, :]  # [seq_len-1]
    
    dist = torch.zeros(n_d, device=device)
    dist.scatter_add_(0, doc_idx_flat[abs_mask_0], mismatches[abs_mask_0].float())
    
    if normalize:
        counts = torch.zeros(n_d, device=device)
        counts.scatter_add_(0, doc_idx_flat[abs_mask_0], 
                           torch.ones_like(mismatches[abs_mask_0], dtype=torch.float))
        dist = dist / counts.clamp(min=1)
    
    dist = dist[doc_idx[:, 1:]]
    return dist

def doc_transposition_dist_pairwise(tokens: torch.Tensor,
                         doc_idx: torch.Tensor,
                         abs_mask: torch.Tensor,
                         normalize: bool = True) -> torch.Tensor:
    """
    Count transpositions (adjacent swaps) + substitutions.
    Faster than full Damerau-Levenshtein, only for aligned positions.
    """
    n_r, _ = tokens.shape
    assert n_r == 2, "supports exactly two rollouts"
    
    device = tokens.device
    abs_mask_0 = abs_mask[0]
    
    seq0 = tokens[0]
    seq1 = tokens[1]
    
    # Count substitutions
    substitutions = (seq0 != seq1) & abs_mask_0
    
    # Count transpositions (adjacent swaps)
    # A transposition: seq0[i] == seq1[i+1] AND seq0[i+1] == seq1[i]
    transpositions = (
        (seq0[:-1] == seq1[1:]) & 
        (seq0[1:] == seq1[:-1]) & 
        (seq0[:-1] != seq0[1:]) &  # Not the same token
        abs_mask_0[:-1] & abs_mask_0[1:]
    )
    
    # Combine: transposition removes 2 substitutions but adds 1 transposition
    # So we subtract transpositions from substitutions
    substitutions[:-1] = substitutions[:-1] & ~transpositions
    substitutions[1:] = substitutions[1:] & ~transpositions
    
    n_d = doc_idx.max().item() + 1
    doc_idx_flat = doc_idx[0]
    
    dist = torch.zeros(n_d, device=device)
    
    # Add substitutions
    dist.scatter_add_(0, doc_idx_flat[abs_mask_0], substitutions[abs_mask_0].float())
    
    # Add transpositions (weighted - you can adjust this)
    if transpositions.any():
        trans_mask = transpositions & abs_mask_0[:-1]
        dist.scatter_add_(0, doc_idx_flat[:-1][trans_mask], trans_mask.float())
    
    if normalize:
        counts = torch.zeros(n_d, device=device)
        counts.scatter_add_(0, doc_idx_flat[abs_mask_0], 
                           torch.ones_like(substitutions[abs_mask_0], dtype=torch.float))
        dist = dist / counts.clamp(min=1)
    
    dist = dist[doc_idx[:, 1:]]
    return dist

def doc_util_dist(doc_ppt: torch.Tensor, metric: str = "abs_diff") -> torch.Tensor:
    """
    Compute pairwise utility distance matrix between rollouts for each document.
    """
    n_rollouts, n_docs = doc_ppt.shape
    
    diff = doc_ppt.unsqueeze(1) - doc_ppt.unsqueeze(0)
    
    if metric == "abs_diff":
        return torch.abs(diff)
    elif metric == "squared":
        return diff ** 2
    elif metric == "ratio":
        sum_utils = doc_ppt.unsqueeze(1) + doc_ppt.unsqueeze(0)
        return torch.abs(diff) / sum_utils.clamp(min=1e-8)
    else:
        raise ValueError(f"Unknown metric: {metric}")

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

def compute_topo_loss(abs_dist, util_dist, mode: int = 0, kappa: float = 0.1):
    """
    Compute topological similarity regularization loss. (Using vector)
    """
    if mode == 0: 
        return - (abs_dist * util_dist).mean()
    elif mode == 1: 
        return torch.clamp(kappa * abs_dist - util_dist, min=0).pow(2).mean()
    elif mode == 2:
        abs_centered = abs_dist - abs_dist.mean()
        util_centered = util_dist - util_dist.mean()
        cov = (abs_centered * util_centered).mean()
        std_abs = abs_centered.square().mean().sqrt()  # standard deviations
        std_util = util_centered.square().mean().sqrt()
        corr = cov / (std_abs * std_util + 1e-8).detach()       # avoid
        return -corr
    elif mode == 3: # golden | covariance
        abs_centered = abs_dist - abs_dist.mean()
        util_centered = util_dist - util_dist.mean()
        cov = (abs_centered * util_centered).mean()
        return -cov
    else: 
        raise ValueError(f"Unknown mode: {mode}")

# Correct but not efficient
# def compute_topo_loss(levenshtein_dist_matrix, util_dist_matrix, mode: int = 0, kappa: float = 0.1):
	
#     n_rollouts = levenshtein_dist_matrix.shape[0]
#     mask = torch.triu(torch.ones(n_rollouts, n_rollouts), diagonal=1).bool()
#     lev_pairs = levenshtein_dist_matrix[mask, :].flatten()  
#     util_pairs = util_dist_matrix[mask, :].flatten()

#     if mode == 0:
#         topo_loss = - (lev_pairs * util_pairs).mean()
#     elif mode == 1:
#         topo_loss = torch.clamp(kappa * lev_pairs - util_pairs, min=0).pow(2).mean()
#     elif mode == 2:
#         topo_loss = - compute_correlation(levenshtein_dist_matrix, util_dist_matrix)
#     elif mode == 3:
#         topo_loss = - compute_covariance(levenshtein_dist_matrix, util_dist_matrix)
#     else:
#         raise ValueError(f"Unknown mode: {mode}")
#     return topo_loss