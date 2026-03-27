# @Ksgk : Continuous target for rank regularization on representation matrix
# -----------------------------------------------------------------------------

import torch 
from torch import einsum
import math 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Matrix-Based Entropy 
# -----------------------------------------------------------------------------
# Exact MBE calculation
def mbe_alpha2_exact(Z, detach=False, epsilon=1e-5):   
    Z = Z.float()  # Force FP32
      
    gram = torch.bmm(Z, Z.transpose(1,2))
    if detach: 
        gram_trace = torch.diagonal(gram.detach(), dim1=1, dim2=2).sum(dim=1)
    else:
        gram_trace = torch.diagonal(gram, dim1=1, dim2=2).sum(dim=1)
    gram_sq = gram.pow(2).sum(dim=(1,2))

    log_trace = torch.log(gram_trace.abs() + epsilon)
    log_sq = torch.log(gram_sq + epsilon)
    mbe = 2 * log_trace - log_sq 
    return mbe.clamp(min=0.0)


def patch_mbe(x, patch_size=8): 
    B, S, D = x.shape 
    assert S % patch_size == 0, "Sequence length must be divisible by patch size"
    num_patches = S // patch_size
    x_reshaped = x.reshape(B, num_patches, patch_size, D).reshape(-1, patch_size, D)    
    mbe_values = mbe_alpha2_exact(x_reshaped)
    return mbe_values.mean()


def patch_frobenius(x, patch_size=8):
    """Mean squared Frobenius norm of patched hidden states (normalized)."""
    B, S, D = x.shape
    assert S % patch_size == 0, "Sequence length must be divisible by patch size"
    num_patches = S // patch_size
    x_reshaped = x.reshape(B, num_patches, patch_size, D).reshape(-1, patch_size, D)
    x_reshaped = x_reshaped.float()
    frob_sq = x_reshaped.pow(2).sum(dim=(1, 2)) / (patch_size * D)
    return frob_sq.mean()


def mbe_reverse_gram(Z, epsilon=1e-5):
    """MBE via reverse Gram matrix Z^T Z (D×D instead of N×N). Memory-efficient when N >> D."""
    Z = Z.float()
    G = torch.bmm(Z.transpose(1, 2), Z)  # (B, D, D)
    gram_trace = torch.diagonal(G, dim1=1, dim2=2).sum(dim=1)
    gram_sq = G.pow(2).sum(dim=(1, 2))
    log_trace = torch.log(gram_trace.abs() + epsilon)
    log_sq = torch.log(gram_sq + epsilon)
    mbe = 2 * log_trace - log_sq
    return mbe.clamp(min=0.0)


def patch_condition_number(x, patch_size=8, epsilon=1e-6):
    """Mean condition number (sigma_max / sigma_min) of patched hidden states.
    
    Minimizing this encourages a uniform singular value distribution,
    spreading out the spectral energy of the representation.
    """
    B, S, D = x.shape
    assert S % patch_size == 0, "Sequence length must be divisible by patch size"
    num_patches = S // patch_size
    x_reshaped = x.reshape(B, num_patches, patch_size, D).reshape(-1, patch_size, D)
    x_reshaped = x_reshaped.float()
    # SVD: singular values shape (num_patches*B, min(patch_size, D))
    sv = torch.linalg.svdvals(x_reshaped)
    sigma_max = sv[:, 0]
    sigma_min = sv[:, -1]
    cond = sigma_max / (sigma_min + epsilon)
    return cond.mean()


class OnlineMBE:
    """Incremental MBE via running reverse Gram matrix G = Z^T Z (D×D)."""

    def __init__(self, D: int, device=None, dtype=torch.float32):
        self.D = D
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.reset()

    def reset(self):
        self.G = torch.zeros(self.D, self.D, device=self.device, dtype=self.dtype)
        self.n = 0

    def update(self, v: torch.Tensor):
        """v: (D,) or (B, D) — sums all outer products into G."""
        v = v.to(self.dtype)
        if v.dim() == 1:
            v = v.unsqueeze(0)
        self.G = self.G + v.T @ v  # (D, D)
        self.n += v.shape[0]

    def mbe(self, epsilon: float = 1e-5) -> torch.Tensor:
        tr_G = torch.trace(self.G)
        sq_G = self.G.pow(2).sum()
        log_trace = torch.log(tr_G.abs() + epsilon)
        log_sq = torch.log(sq_G + epsilon)
        return (2.0 * log_trace - log_sq).clamp(min=0.0)
