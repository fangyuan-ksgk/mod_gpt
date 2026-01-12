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


def patch_mbe_variance(x, patch_size=8): 
    """Minimize Variance of MBE across patches (alleviate over compression)"""
    B, S, D = x.shape 
    assert S % patch_size == 0, "Sequence length must be divisible by patch size"
    num_patches = S // patch_size
    x_reshaped = x.reshape(B, num_patches, patch_size, D).reshape(-1, patch_size, D)    
    mbe_values = mbe_alpha2_exact(x_reshaped)
    return mbe_values.mean() + 1.0 * mbe_values.var()


def patch_mbe_range(x, patch_size=8, mbe_min=0.45, mbe_max=0.75):
    """
    Range-constrained MBE: penalize values OUTSIDE [mbe_min, mbe_max] sweet spot.
    """
    B, S, D = x.shape 
    assert S % patch_size == 0, "Sequence length must be divisible by patch size"
    num_patches = S // patch_size
    x_reshaped = x.reshape(B, num_patches, patch_size, D).reshape(-1, patch_size, D)    
    mbe_values = mbe_alpha2_exact(x_reshaped)
    
    # Hinge loss: penalize outside range, free inside
    below_floor = torch.relu(mbe_min - mbe_values)  # penalty for < mbe_min
    above_ceiling = torch.relu(mbe_values - mbe_max)  # penalty for > mbe_max
    
    range_loss = (below_floor + above_ceiling).mean()
    return range_loss


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


# -----------------------------------------------------------------------------
# Stronger MBE regularization to prevent collapse
# -----------------------------------------------------------------------------

def patch_mbe_log_barrier(x, patch_size=8, mbe_floor=0.3, eps=1e-6):
    """
    Log barrier loss - creates infinite resistance to collapse.
    
    As MBE → mbe_floor, loss → +∞
    This prevents the model from escaping the regularizer via collapse.
    """
    B, S, D = x.shape 
    assert S % patch_size == 0
    num_patches = S // patch_size
    x_reshaped = x.reshape(B, num_patches, patch_size, D).reshape(-1, patch_size, D)    
    mbe_values = mbe_alpha2_exact(x_reshaped)
    
    # Log barrier: -log(mbe - floor) becomes large as mbe approaches floor
    # Clamp to prevent negative values inside log
    safe_diff = (mbe_values - mbe_floor).clamp(min=eps)
    barrier = -torch.log(safe_diff)
    
    return barrier.mean()


def patch_mbe_softmin(x, patch_size=8, temperature=0.1):
    """
    Maximize the MINIMUM MBE across patches (differentiable via softmin).
    
    Focuses regularization on the worst (most collapsed) patches.
    Lower temperature = sharper focus on true minimum.
    """
    B, S, D = x.shape 
    assert S % patch_size == 0
    num_patches = S // patch_size
    x_reshaped = x.reshape(B, num_patches, patch_size, D).reshape(-1, patch_size, D)    
    mbe_values = mbe_alpha2_exact(x_reshaped)
    
    # Softmin: weighted average focused on smallest values
    weights = torch.softmax(-mbe_values / temperature, dim=0)
    soft_min = (weights * mbe_values).sum()
    
    # Return negative because we want to MAXIMIZE the minimum MBE
    return -soft_min


def patch_mbe_floor(x, patch_size=8, mbe_floor=0.3, steepness=10.0):
    """
    Soft floor with exponential penalty below threshold.
    
    Combines mean MBE (for compression) with strong floor enforcement.
    steepness controls how sharply the penalty increases below floor.
    """
    B, S, D = x.shape 
    assert S % patch_size == 0
    num_patches = S // patch_size
    x_reshaped = x.reshape(B, num_patches, patch_size, D).reshape(-1, patch_size, D)    
    mbe_values = mbe_alpha2_exact(x_reshaped)
    
    # Exponential penalty for values below floor
    # exp(steepness * (floor - mbe)) grows rapidly as mbe drops below floor
    floor_violation = mbe_floor - mbe_values
    floor_penalty = torch.exp(steepness * floor_violation.clamp(min=0)).mean() - 1.0
    
    # Combine: encourage compression (low mean) but enforce floor
    mean_mbe = mbe_values.mean()
    
    return mean_mbe + floor_penalty