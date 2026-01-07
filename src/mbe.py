# @Ksgk : Continuous target for rank regularization on representation matrix
# -----------------------------------------------------------------------------

import torch 
from torch import einsum
import math 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Matrix-Based Entropy 
# -----------------------------------------------------------------------------
# Exact MBE calculation
def mbe_alpha2_exact(Z, detach=False, epsilon=1e-7):     
    gram = torch.bmm(Z, Z.transpose(1,2))
    if detach: 
        gram_trace = torch.diagonal(gram.detach(), dim1=1, dim2=2).sum(dim=1)
    else:
        gram_trace = torch.diagonal(gram, dim1=1, dim2=2).sum(dim=1)
    gram_sq = gram.pow(2).sum(dim=(1,2))
    ratio = (gram_sq + epsilon) / (gram_trace.pow(2) + epsilon)
    ratio = ratio.clamp(min=1e-6, max=1.0)
    return -torch.log(ratio)

def patch_mbe(x, patch_size=8): 
    B, S, D = x.shape 
    assert S % patch_size == 0, "Sequence length must be divisible by patch size"
    num_patches = S // patch_size
    x_reshaped = x.reshape(B, num_patches, patch_size, D).reshape(-1, patch_size, D)    
    mbe_values = mbe_alpha2_exact(x_reshaped)
    return mbe_values.mean()