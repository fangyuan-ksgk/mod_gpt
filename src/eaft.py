import torch
import torch.nn.functional as F

def _eaft_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1.0,
    topk: int = 20,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Entropy-Adaptive cross entropy - upweights uncertain predictions."""
    per_token_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), 
        target.view(-1), 
        ignore_index=ignore_index, 
        reduction="none"
    )
    valid_mask = target.view(-1) != ignore_index
    if not valid_mask.any():
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    valid_losses = per_token_loss[valid_mask]
    
    with torch.no_grad():
        logits_flat = logits.view(-1, logits.size(-1))[valid_mask].detach()
        topk_val, _ = torch.topk(logits_flat, k=topk, dim=-1)
        log_probs = F.log_softmax(topk_val, dim=-1)
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum(dim=-1)
        adaptive_weight = torch.pow(entropy / 3.0, alpha)

    return (valid_losses * adaptive_weight).mean()


def collect_token_stats(logits, targets, topk=50):
    """
    Collect (probability, entropy) pairs for each valid token.
    Returns: dict with 'probs' and 'entropies' tensors
    """
    logits_flat = logits.view(-1, logits.size(-1))  # (N, vocab)
    targets_flat = targets.view(-1)  # (N,)
    
    # Probability of correct token
    probs_all = torch.softmax(logits_flat.float(), dim=-1)
    correct_probs = probs_all.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
    
    # Entropy (use top-k for efficiency)
    topk_logits, _ = torch.topk(logits_flat, k=topk, dim=-1)
    topk_probs = torch.softmax(topk_logits.float(), dim=-1)
    entropy = -(topk_probs * topk_probs.log().clamp(min=-100)).sum(dim=-1)
    
    return {
        'probs': correct_probs.cpu(),
        'entropies': entropy.cpu()
    }


def collect_token_stats_with_grads(model, inputs, targets, attn_blocksize, patch_size, topk=50):
    """
    Collect (probability, entropy, gradient_magnitude) for each token.
    Requires enabling gradients - more expensive than collect_token_stats.
    """
    model.zero_grad(set_to_none=True)
    
    # Get embedding layer for gradient tracking
    embed_layer = model.transformer.wte if hasattr(model, 'transformer') else model._orig_mod.transformer.wte
    
    with torch.enable_grad():
        # Forward pass
        loss_dict = model.forward(inputs, targets, attn_blocksize, patch_size)
        logits = loss_dict["logits"]
        
        # Compute per-token loss (unreduced)
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        per_token_loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        
        # Sum and backward to get gradients
        total_loss = per_token_loss.sum()
        total_loss.backward()
        
        # Get gradient magnitude per token from embedding gradients
        # Shape of embed gradient: (batch, seq_len, embed_dim) 
        embed_grad = embed_layer.weight.grad  # (vocab_size, embed_dim)
        
        # Per-token gradient: look up the gradient for each input token
        input_flat = inputs.view(-1)  # (batch * seq_len,)
        per_token_grad = embed_grad[input_flat]  # (batch * seq_len, embed_dim)
        grad_magnitudes = per_token_grad.norm(dim=-1)  # (batch * seq_len,)
    
    # Collect probability and entropy
    with torch.no_grad():
        probs_all = torch.softmax(logits_flat.float(), dim=-1)
        correct_probs = probs_all.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
        
        topk_logits, _ = torch.topk(logits_flat, k=topk, dim=-1)
        topk_probs = torch.softmax(topk_logits.float(), dim=-1)
        entropy = -(topk_probs * topk_probs.log().clamp(min=-100)).sum(dim=-1)
    
    model.zero_grad(set_to_none=True)
    
    return {
        'probs': correct_probs.cpu(),
        'entropies': entropy.cpu(),
        'grad_magnitudes': grad_magnitudes.detach().cpu(),
        'per_token_loss': per_token_loss.detach().cpu(),
    }


# Add the plotting function near the top (after imports):
def plot_entropy_vs_prob(token_stats, save_path="entropy_prob_scatter.png"):
    import matplotlib.pyplot as plt
    
    probs = torch.cat(token_stats['probs']).numpy()
    entropies = torch.cat(token_stats['entropies']).numpy()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(probs, entropies, alpha=0.2, s=2, c='#E74C3C', edgecolors='none')
    ax.set_xlabel('Probability p(correct)', fontsize=12)
    ax.set_ylabel('Entropy H(p)', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(entropies.max() * 1.1, 3))
    ax.set_title('Token Confidence vs Entropy')
    
    # Mark "confident conflicts" region
    from matplotlib.patches import Circle
    circle = Circle((0.3, 1.5), 0.4, fill=False, linestyle='--', color='gray', linewidth=1.5)
    ax.add_patch(circle)
    ax.annotate('Confident\nConflicts', xy=(0.3, 2.0), fontsize=10, ha='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved entropy-prob scatter to {save_path}")