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
    return -torch.log((gram_sq + epsilon) / (gram_trace.pow(2) + epsilon))

def patch_mbe2(x, patch_size=8): 
    B, S, D = x.shape 
    assert S % patch_size == 0, "Sequence length must be divisible by patch size"
    num_patches = S // patch_size
    x_reshaped = x.reshape(B, num_patches, patch_size, D).reshape(-1, patch_size, D)    
    mbe_values = mbe_alpha2_exact(x_reshaped)
    return mbe_values.mean() 

def patch_mbe3(x, patch_size=8): 
    B, S, D = x.shape 
    num_patches = S // patch_size
    assert num_patches > 0, "Sequence length should be bigger than patch size"
    S = num_patches * patch_size 
    x_reshaped = x[:, -S:].reshape(B, num_patches, patch_size, D).reshape(-1, patch_size, D)    
    mbe_values = mbe_alpha2_exact(x_reshaped)
    return mbe_values.mean() 

# Differential MBE 
# -----------------------------------------------------------------------------
def mbe_alpha2_exact2(Z, detach=False, epsilon=1e-7):
    gram = torch.einsum("bpid,bpjd->bpij", Z, Z)
    if detach: 
        gram_trace = torch.diagonal(gram.detach(), dim1=2, dim2=3).sum(dim=2)
    else:
        gram_trace = torch.diagonal(gram, dim1=2, dim2=3).sum(dim=2)
    gram_sq = gram.pow(2).sum(dim=(2,3))
    return -torch.log((gram_sq + epsilon) / (gram_trace.pow(2) + epsilon))

def patch_mbe1_diff(x, patch_size=8, stride=4): 
    x_patches = x.unfold(dimension=1, size=patch_size, step=stride)
    mbe_values = mbe_alpha2_exact2(x_patches)
    mbe_diff = torch.diff(mbe_values, dim=1).mean()
    return mbe_diff

# requires 2x patches compared to 'patch_mbe2' 
def patch_mbe2_diff(x, patch_size=8, overlap=4): 
    B, S, D = x.shape 
    prefix_patches = x[:, overlap:].unfold(dimension=1, size=patch_size, step=patch_size).transpose(3, 2).view(B, -1, D)
    suffix_patches = x[:, :-overlap].unfold(dimension=1, size=patch_size, step=patch_size).transpose(3, 2).view(B, -1, D)
    mbe_diff = mbe_alpha2_exact(suffix_patches) - mbe_alpha2_exact(prefix_patches)
    return mbe_diff.mean()

# Non-overlapping Diff MBE
def patch_mbe3_diff(x, patch_size=8): 
    B, S, D = x.shape 
    num_patches = S // patch_size
    assert num_patches > 0, "Sequence length should be bigger than patch size"
    S = num_patches * patch_size 
    x_reshaped = x[:, -S:].reshape(B, num_patches, patch_size, D).reshape(-1, patch_size, D)    
    mbe_vals = mbe_alpha2_exact(x_reshaped)
    return torch.diff(mbe_vals).mean()

# MBE approximatioon 
# -----------------------------------------------------------------------------
def rademacher(shape, dtype=torch.float32, device=DEVICE):
    rand = ((torch.rand(shape) < 0.5)) * 2 - 1
    return rand.to(dtype).to(device)

# MBE Estimate with Hutchison Trace Approximation
def mbe_alpha2_hutchison(Z, N=100, detach=False): 
    B, S, D = Z.shape
    v = rademacher(shape=(N, S), device=Z.device)
    G = torch.bmm(Z, Z.transpose(1,2)) # gram matrix
    if detach: 
        gram_trace = torch.diagonal(G.detach(), dim1=1, dim2=2).sum(dim=1)
    else:
        gram_trace = torch.diagonal(G, dim1=1, dim2=2).sum(dim=1)
    Gv = torch.einsum("bmn,sn->bsm", G, v)
    GGv = torch.einsum("bmn,bsn->bsm", G, Gv)
    vGGv = torch.einsum("sn,bsn->bs", v, GGv) # gram.pow(2) trace estimate
    trace_est = vGGv.mean(dim=1)
    return - torch.log(trace_est / gram_trace.pow(2))


# Stable Rank 
# -----------------------------------------------------------------------------

def calculate_stable_rank(Z): 
    svd_Z = torch.svd(Z)
    return torch.sum(svd_Z[1]**2) / svd_Z[1][0]**2

def estimate_stable_rank(Z, num_samples=10): 
    frob_norm_sq = torch.sum(Z * Z)
    # power iteration
    m, n = Z.shape
    v = torch.randn(n, device=Z.device)
    v = v / torch.norm(v)
    for _ in range(num_samples):
        u = Z @ v  # Z*v
        v = Z.T @ u  # Z^T*Z*v
        v_norm = torch.norm(v)
        v = v / v_norm
    largest_sv = torch.sqrt(v_norm)    
    return frob_norm_sq / (largest_sv ** 2)