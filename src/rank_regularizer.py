# @Ksgk : Continuous target for rank regularization on representation matrix
# -----------------------------------------------------------------------------

import torch 

# Matrix-Based Entropy 
# -----------------------------------------------------------------------------
def calculate_matrix_based_entropy(Z, alpha = 2): 
    # Calculate P = Z Z^T / ||Z||_F^2
    frob_norm_sq = torch.norm(Z, 'fro') ** 2
    P = Z @ Z.T / frob_norm_sq
    
    # Use eigendecomposition to compute P^alpha
    eigenvalues, eigenvectors = torch.linalg.eigh(P)
    P_alpha = eigenvectors @ torch.diag(eigenvalues ** alpha) @ eigenvectors.T
    
    return torch.log(torch.trace(P_alpha)) / (1-alpha)

def hutchinson_power_estimator(Z, alpha, num_samples=10):
    m, n = Z.shape
    frob_norm_sq = torch.sum(Z * Z)
    
    trace_est = 0
    for _ in range(num_samples):
        # Random ±1 vector
        v = (torch.randint(0, 2, (m,), device=Z.device) * 2 - 1).to(Z.dtype)
        
        # Compute v^T P^α v
        u = v.clone()
        for _ in range(alpha):  # For integer alpha
            u = torch.mv(Z, torch.mv(Z.t(), u)) / frob_norm_sq
        
        trace_est += torch.dot(v, u) / num_samples
    
    return torch.log(trace_est) / (1-alpha)

def batch_mbe_estimator(Z, alpha, num_samples=10):
    B, S, D = Z.shape
    frob_norm_sq = torch.sum(Z * Z, dim=(1, 2))
    
    trace_est = torch.zeros(B, device=Z.device)
    for _ in range(num_samples):
        # Random ±1 vector
        v = (torch.randint(0, 2, (B, S), device=Z.device) * 2 - 1).to(Z.dtype)
        
        # Compute v^T P^α v
        u = v.clone()
        for _ in range(alpha):  # For integer alpha
            u = torch.einsum('bsd,bd->bs', Z, torch.einsum('bsd,bs->bd', Z, u)) / frob_norm_sq.unsqueeze(1)
        
        batch_dots = (v * u).sum(dim=1)
        trace_est += batch_dots / num_samples
    
    return torch.log(trace_est) / (1-alpha)

def patch_mbe(x, alpha=2, patch_size=16, num_samples=10): 
    B, S, D = x.shape 
    assert S % patch_size == 0, "Sequence length must be divisible by patch size"
    batch_reg = torch.zeros(B, device=x.device)
    num_patches = S // patch_size
    
    mbe_per_patch = torch.zeros(B, num_patches, device=x.device)
    for i in range(num_patches): 
        patch_start = i * patch_size
        patch_end = patch_start + patch_size
        patch = x[:, patch_start:patch_end, :]  # Use the correct batch index
        mbe_per_patch[:, i] = batch_mbe_estimator(patch, alpha, num_samples)
    
    return mbe_per_patch.mean()

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