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

    # ratio = (gram_sq + epsilon) / (gram_trace.pow(2) + epsilon)
    # ratio = ratio.clamp(min=0.01, max=0.99999)
    # mbe = -torch.log(ratio)

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


def patch_mbe_detailed(x, patch_size=8):
    """
    Compute per-patch MBE values (not averaged).
    
    Returns:
        mbe_per_patch: (B, num_patches) tensor of MBE values
    """
    B, S, D = x.shape 
    assert S % patch_size == 0, "Sequence length must be divisible by patch size"
    num_patches = S // patch_size
    x_reshaped = x.reshape(B, num_patches, patch_size, D).reshape(-1, patch_size, D)    
    mbe_values = mbe_alpha2_exact(x_reshaped)  # (B * num_patches,)
    return mbe_values.view(B, num_patches)