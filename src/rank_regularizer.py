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
        v = (torch.randint(0, 2, (m,), device=Z.device) * 2 - 1).float()
        
        # Compute v^T P^α v
        u = v.clone()
        for _ in range(alpha):  # For integer alpha
            u = torch.mv(Z, torch.mv(Z.t(), u)) / frob_norm_sq
        
        trace_est += torch.dot(v, u) / num_samples
    
    return torch.log(trace_est) / (1-alpha)



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