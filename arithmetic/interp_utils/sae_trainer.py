"""
SAE trainer wrapper around EleutherAI's sparsify (eai-sparsify).
Uses SparseCoder directly for training on custom activations from our GAT model.

Install: pip install eai-sparsify

Usage:
    from arithmetic.interp_utils.sae_trainer import SAETrainer

    trainer = SAETrainer(d_in=510, k=32, expansion_factor=16)
    for batch in activation_batches:
        loss = trainer.step(batch)
    trainer.save("ckpt/sae/")
"""
import os
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

from sparsify import SparseCoder, SparseCoderConfig


@dataclass
class SAETrainerConfig:
    d_in: int = 510                  # model hidden dim
    k: int = 32                      # top-k sparsity
    expansion_factor: int = 16       # d_sae = d_in * expansion_factor
    normalize_decoder: bool = True
    multi_topk: bool = False         # multi-topk auxiliary loss
    auxk_alpha: float = 1 / 32       # auxiliary loss for dead features
    lr: float = 5e-4
    optimizer: str = "adam"          # "adam", "signum", or "muon"
    weight_decay: float = 0.0
    dead_feature_window: int = 1000  # steps before a feature is considered dead
    grad_clip: float = 1.0
    device: str = "cuda"
    dtype: str = "float32"


class SAETrainer:
    """
    Trains a TopK SAE on custom activation tensors using sparsify's SparseCoder.
    """

    def __init__(self, config: Optional[SAETrainerConfig] = None, **kwargs):
        if config is None:
            config = SAETrainerConfig(**kwargs)
        self.config = config

        dtype = getattr(torch, config.dtype)
        sae_cfg = SparseCoderConfig(
            k=config.k,
            expansion_factor=config.expansion_factor,
            normalize_decoder=config.normalize_decoder,
            multi_topk=config.multi_topk,
        )
        self.sae = SparseCoder(
            d_in=config.d_in,
            cfg=sae_cfg,
            device=config.device,
            dtype=dtype,
        )

        if config.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                self.sae.parameters(), lr=config.lr, weight_decay=config.weight_decay,
            )
        elif config.optimizer == "signum":
            self.optimizer = torch.optim.SGD(
                self.sae.parameters(), lr=config.lr, weight_decay=config.weight_decay,
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.sae.parameters(), lr=config.lr, weight_decay=config.weight_decay,
            )

        self.step_count = 0
        self.dead_mask = None
        self._feature_activity = torch.zeros(
            self.sae.num_latents, device=config.device
        )

    def _update_dead_mask(self, top_indices):
        """Track feature usage and compute dead mask for auxk loss."""
        # Mark active features
        active = torch.zeros_like(self._feature_activity)
        active.scatter_(0, top_indices.flatten().long(), 1.0)
        self._feature_activity = self._feature_activity * 0.99 + active * 0.01

        if self.step_count > self.config.dead_feature_window:
            self.dead_mask = (self._feature_activity < 1e-5)
        else:
            self.dead_mask = None

    def step(self, activations: torch.Tensor) -> dict:
        """
        Train one step on a batch of activations.

        Args:
            activations: (batch, d_in) tensor of model hidden states

        Returns:
            dict with loss, fvu, num_dead_features
        """
        self.optimizer.zero_grad()

        output = self.sae(activations, dead_mask=self.dead_mask)

        # Main loss: fraction of variance unexplained
        loss = output.fvu
        if self.config.auxk_alpha > 0 and output.auxk_loss is not None:
            loss = loss + self.config.auxk_alpha * output.auxk_loss

        loss.backward()

        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.sae.parameters(), self.config.grad_clip)

        self.optimizer.step()

        # Track dead features
        self._update_dead_mask(output.latent_indices)
        self.step_count += 1

        num_dead = self.dead_mask.sum().item() if self.dead_mask is not None else 0

        return {
            "loss": loss.item(),
            "fvu": output.fvu.item(),
            "num_dead": int(num_dead),
            "step": self.step_count,
        }

    def encode(self, activations: torch.Tensor):
        """Encode activations into sparse features."""
        with torch.no_grad():
            return self.sae.encode(activations)

    def decode(self, top_acts, top_indices):
        """Decode sparse features back to activation space."""
        with torch.no_grad():
            return self.sae.decode(top_acts, top_indices)

    def save(self, save_dir: str):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.sae.state_dict(), save_dir / "sae.pt")
        with open(save_dir / "sae_config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=2)

    def load(self, save_dir: str):
        save_dir = Path(save_dir)
        state = torch.load(save_dir / "sae.pt", map_location=self.config.device)
        self.sae.load_state_dict(state)


def collect_activations(model_wrapper, n_digits=6, n_batches=100,
                        batch_size=64, device="cuda"):
    """
    Collect hidden state activations from a trained model for SAE training.

    Args:
        model_wrapper: AdditionGAT instance
        n_digits: number of digits
        n_batches: number of batches to collect
        batch_size: batch size per collection step

    Returns:
        activations: (N, d_model) tensor
    """
    from arithmetic.datasets.addition import generate_batch

    all_acts = []
    model_wrapper.model.eval()

    with torch.no_grad():
        for _ in range(n_batches):
            tokens, _, answer_mask = generate_batch(
                batch_size, n_digits, use_sum9_aug=True, device=device
            )
            hidden = model_wrapper.get_hidden_states(tokens)
            # Collect activations at answer positions
            acts = hidden[answer_mask]  # (N_ans, d_model)
            all_acts.append(acts)

    model_wrapper.model.train()
    return torch.cat(all_acts, dim=0)
