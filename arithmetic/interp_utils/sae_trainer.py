"""
SAE trainer for arithmetic models using EleutherAI sparsify (eai-sparsify).

Usage:
    # Train SAE on a model from HF
    python -m arithmetic.interp_utils.sae_trainer \
        --model_subfolder add_baseline_500K \
        --layer 1 --k 32 --expansion 16

    # Or programmatic
    from arithmetic.interp_utils.sae_trainer import SAETrainer, collect_activations
    acts = collect_activations(model, tokenizer, layer=1, n_batches=200)
    trainer = SAETrainer(d_in=512, k=32, expansion_factor=16)
    for epoch in range(10):
        for batch in acts.split(256):
            trainer.step(batch)
    trainer.save("sae_checkpoints/")
"""
import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List

from sparsify import SparseCoder, SparseCoderConfig


@dataclass
class SAETrainerConfig:
    d_in: int = 512                  # model hidden dim
    k: int = 32                      # top-k sparsity (num active features per input)
    expansion_factor: int = 16       # d_sae = d_in * expansion_factor
    normalize_decoder: bool = True
    multi_topk: bool = False
    auxk_alpha: float = 1 / 32
    lr: float = 5e-4
    weight_decay: float = 0.0
    dead_feature_window: int = 1000
    grad_clip: float = 1.0
    device: str = "cuda"
    dtype: str = "float32"


class SAETrainer:
    """Trains a TopK SAE on activation tensors using sparsify's SparseCoder."""

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
            d_in=config.d_in, cfg=sae_cfg,
            device=config.device, dtype=dtype,
        )
        self.optimizer = torch.optim.Adam(
            self.sae.parameters(), lr=config.lr, weight_decay=config.weight_decay,
        )
        self.step_count = 0
        self.dead_mask = None
        self._feature_activity = torch.zeros(self.sae.num_latents, device=config.device)

    def _update_dead_mask(self, top_indices):
        active = torch.zeros_like(self._feature_activity)
        active.scatter_(0, top_indices.flatten().long(), 1.0)
        self._feature_activity = self._feature_activity * 0.99 + active * 0.01
        if self.step_count > self.config.dead_feature_window:
            self.dead_mask = (self._feature_activity < 1e-5)
        else:
            self.dead_mask = None

    def step(self, activations: torch.Tensor) -> dict:
        self.optimizer.zero_grad()
        output = self.sae(activations, dead_mask=self.dead_mask)
        loss = output.fvu
        if self.config.auxk_alpha > 0 and output.auxk_loss is not None:
            loss = loss + self.config.auxk_alpha * output.auxk_loss
        loss.backward()
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.sae.parameters(), self.config.grad_clip)
        self.optimizer.step()
        self._update_dead_mask(output.latent_indices)
        self.step_count += 1
        num_dead = self.dead_mask.sum().item() if self.dead_mask is not None else 0
        return {"loss": loss.item(), "fvu": output.fvu.item(), "num_dead": int(num_dead)}

    def encode(self, activations: torch.Tensor):
        with torch.no_grad():
            return self.sae.encode(activations)

    def save(self, save_dir: str):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.sae.state_dict(), save_dir / "sae.pt")
        with open(save_dir / "sae_config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=2)

    def load(self, save_dir: str):
        state = torch.load(Path(save_dir) / "sae.pt", map_location=self.config.device)
        self.sae.load_state_dict(state)


# ── Activation collection from Qwen3 models ───────────────────────

def collect_activations(
    model, tokenizer, layer: int = -1,
    n_digits: int = 6, ops: str = "add",
    n_batches: int = 200, batch_size: int = 64,
    positions: str = "answer", device: str = "cuda",
) -> torch.Tensor:
    """
    Collect hidden state activations from a trained Qwen3 SorlModelWrapper.

    Args:
        model: SorlModelWrapper instance
        tokenizer: Qwen3 tokenizer
        layer: which transformer layer (-1 = last, 0 = first, etc.)
        positions: "answer" = only answer positions, "all" = all positions
        n_batches: number of batches
        batch_size: examples per batch

    Returns:
        (N, hidden_size) tensor of activations
    """
    from arithmetic.train import Qwen3ArithmeticDataset, collate_fn

    ds = Qwen3ArithmeticDataset(tokenizer, n_digits, ops, n_batches * batch_size)
    prompt_len = ds.prompt_len
    all_acts = []
    model.eval()

    # Register hook to capture activations at target layer
    n_layers = model.model.config.num_hidden_layers
    target_layer = layer if layer >= 0 else n_layers + layer
    hook_output = []

    def hook_fn(module, input, output):
        hook_output.append(output[0].detach())

    handle = model.model.model.layers[target_layer].register_forward_hook(hook_fn)

    with torch.no_grad():
        for i in range(0, len(ds), batch_size):
            batch_items = [ds[j] for j in range(i, min(i + batch_size, len(ds)))]
            batch = collate_fn(batch_items)
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)

            hook_output.clear()
            model(ids, attention_mask=attn, memory_span_abs=1792, memory_span_traj=1792)

            hidden = hook_output[0]  # (B, seq_len, hidden_size)

            if positions == "answer":
                acts = hidden[:, prompt_len:]  # (B, answer_len, hidden)
            else:
                acts = hidden  # (B, seq_len, hidden)

            all_acts.append(acts.reshape(-1, acts.shape[-1]).cpu())

    handle.remove()
    model.train()
    return torch.cat(all_acts, dim=0)


# ── CLI ────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Train SAE on arithmetic model activations")
    p.add_argument("--model_subfolder", type=str, required=True,
                   help="Subfolder in thoughtworks/arithmetic-sorl (e.g. add_baseline_500K)")
    p.add_argument("--layer", type=int, default=-1, help="Layer to extract from (-1=last)")
    p.add_argument("--k", type=int, default=32, help="Top-k sparsity")
    p.add_argument("--expansion", type=int, default=16, help="SAE expansion factor")
    p.add_argument("--positions", choices=["answer", "all"], default="answer")
    p.add_argument("--n_batches", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--sae_epochs", type=int, default=10)
    p.add_argument("--sae_batch", type=int, default=256)
    p.add_argument("--ops", type=str, default="add")
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from arithmetic.hub import load_model
    from transformers import AutoTokenizer

    print(f"Loading model: {args.model_subfolder}")
    model, config, metrics = load_model(args.model_subfolder, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    hidden_size = config.get("n_embd", 512)

    print(f"Collecting activations (layer={args.layer}, positions={args.positions})...")
    acts = collect_activations(
        model, tokenizer, layer=args.layer,
        n_digits=config.get("n_digits", 6), ops=args.ops,
        n_batches=args.n_batches, batch_size=args.batch_size,
        positions=args.positions, device=args.device,
    )
    acts = acts.to(args.device)
    print(f"Collected {acts.shape[0]} activation vectors, dim={acts.shape[1]}")

    print(f"Training SAE (k={args.k}, expansion={args.expansion})...")
    sae_cfg = SAETrainerConfig(
        d_in=hidden_size, k=args.k, expansion_factor=args.expansion,
        device=args.device,
    )
    trainer = SAETrainer(sae_cfg)

    for epoch in range(args.sae_epochs):
        indices = torch.randperm(acts.shape[0])
        epoch_loss, n_steps = 0, 0
        for i in range(0, acts.shape[0] - args.sae_batch, args.sae_batch):
            batch = acts[indices[i:i + args.sae_batch]]
            info = trainer.step(batch)
            epoch_loss += info["loss"]
            n_steps += 1
        print(f"  epoch {epoch + 1:2d} | fvu: {epoch_loss / n_steps:.4f} | dead: {info['num_dead']}")

    save_dir = args.save_dir or f"sae_{args.model_subfolder}_L{args.layer}_k{args.k}"
    trainer.save(save_dir)
    print(f"Saved to {save_dir}")


if __name__ == "__main__":
    main()
