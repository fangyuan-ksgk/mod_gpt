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