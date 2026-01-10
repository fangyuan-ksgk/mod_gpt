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


# ==== IBLM ckpt ==== 
iblm_gpt = {
    "small": {"hf_repo": "Ksgk-fy/iblm-gpt2-ckpt", "remote_filename": "fineweb10B-iblm-gpt2-small-spike.pt", "b_layer": [5, 10]},
    "medium": {"hf_repo": "Ksgk-fy/iblm-gpt2-ckpt", "remote_filename": "fineweb10B-iblm-gpt2-medium-spike.pt", "b_layer": [11, 15]},
    "large": {"hf_repo": "Ksgk-fy/iblm-gpt2-ckpt", "remote_filename": "fineweb10B-iblm-gpt2-large-softplus-spike.pt", "b_layer": [3, 16, 2]},
    "xl": {"hf_repo": "Ksgk-fy/iblm-gpt2-ckpt", "remote_filename": "fineweb10B-iblm-gpt2-xl-spike.pt", "b_layer": [22, 23]}
}

# --- download from huggingface repo ---
from huggingface_hub import hf_hub_download
from pathlib import Path

def download_iblm_ckpt(size: str, ckpt_dir: str = "ckpt") -> str:
    if size not in iblm_gpt:
        raise ValueError(f"Invalid size '{size}'. Choose from: {list(iblm_gpt.keys())}")
    
    config = iblm_gpt[size]
    ckpt_path = Path(ckpt_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    
    local_path = ckpt_path / config["remote_filename"]
    
    if local_path.exists():
        print(f"✅ Checkpoint already exists: {local_path}")
        return str(local_path)
    
    print(f"⬇️  Downloading {config['remote_filename']} from {config['hf_repo']}...")
    downloaded_path = hf_hub_download(
        repo_id=config["hf_repo"],
        filename=config["remote_filename"],
        local_dir=ckpt_dir,
    )
    print(f"✅ Downloaded to: {downloaded_path}")
    return downloaded_path