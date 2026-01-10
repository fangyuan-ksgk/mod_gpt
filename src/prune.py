# Prune layers away from GPT model
import torch.nn as nn
from typing import List, Dict

class IdentityBlock(nn.Module):
    """Drop-in replacement that passes through unchanged"""
    def forward(self, x, v1, x0, block_mask):
        return x, v1  # Identity: no transformation

def prune_layer(
    model, 
    prune_layer_indices: List[int],
) -> Dict[str, float]:

    device = next(model.parameters()).device
    
    # --- Step 2: Swap layers with identity ---
    original_blocks = {}
    for idx in prune_layer_indices:
        original_blocks[idx] = model.transformer.h[idx]
        model.transformer.h[idx] = IdentityBlock().to(device)
    
    print(f"🔧 Pruned layers: {prune_layer_indices}")