#!/usr/bin/env python3
"""
Train SAEs on all models in thoughtworks/arithmetic-sorl.

Structure on HF (thoughtworks/arithmetic-sorl-saes):
    {model_name}/L{layer}_k{k}/
        sae.pt
        sae_config.json

Usage:
    python arithmetic/scripts/train_saes.py                    # all models, default k
    python arithmetic/scripts/train_saes.py --k 8 16 32 64    # k sweep
    python arithmetic/scripts/train_saes.py --layers 0 1 2     # all layers
    python arithmetic/scripts/train_saes.py --models add_baseline_500K
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoTokenizer
from arithmetic.hub import list_models, load_model
from arithmetic.interp_utils.sae_trainer import SAETrainer, SAETrainerConfig, collect_activations


def train_sae_for_model(model_name: str, tokenizer, layer: int, k: int,
                        expansion: int, n_batches: int, sae_epochs: int,
                        sae_batch: int, device: str):
    """Train one SAE and upload to HF."""
    subfolder = f"{model_name}/L{layer}_k{k}"
    print(f"\n{'─' * 50}")
    print(f"  {subfolder}")
    print(f"{'─' * 50}")

    model, config, metrics = load_model(model_name, device=device)
    hidden_size = config.get("n_embd", 512)
    ops = config.get("ops", "add")

    print(f"  Collecting activations (layer={layer}, {n_batches} batches)...")
    acts = collect_activations(
        model, tokenizer, layer=layer,
        n_digits=config.get("n_digits", 6), ops=ops,
        n_batches=n_batches, batch_size=64,
        positions="answer", device=device,
    )
    acts = acts.to(device)
    print(f"  {acts.shape[0]:,} vectors, dim={acts.shape[1]}")

    sae_cfg = SAETrainerConfig(
        d_in=hidden_size, k=k, expansion_factor=expansion, device=device,
    )
    trainer = SAETrainer(sae_cfg)

    print(f"  Training SAE (k={k}, expansion={expansion}, {sae_epochs} epochs, batch={sae_batch})...")
    final_fvu = 0
    for epoch in range(sae_epochs):
        indices = torch.randperm(acts.shape[0])
        epoch_loss, n_steps = 0, 0
        for i in range(0, acts.shape[0] - sae_batch, sae_batch):
            info = trainer.step(acts[indices[i:i + sae_batch]])
            epoch_loss += info["fvu"]
            n_steps += 1
        final_fvu = epoch_loss / max(n_steps, 1)
        print(f"    epoch {epoch + 1:2d}/{sae_epochs} | fvu: {final_fvu:.4f} | dead: {info['num_dead']}")

    train_info = {
        "model_name": model_name,
        "model_repo": "thoughtworks/arithmetic-sorl",
        "layer": layer,
        "k": k,
        "expansion_factor": expansion,
        "sae_epochs": sae_epochs,
        "n_vectors": acts.shape[0],
        "final_fvu": final_fvu,
        "num_dead": info["num_dead"],
        "positions": "answer",
        "model_config": config,
    }
    trainer.save_to_hub(subfolder, train_info)

    del model, acts, trainer
    torch.cuda.empty_cache()
    return final_fvu


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--k", nargs="*", type=int, default=[8, 16, 32, 64])
    p.add_argument("--layers", nargs="*", type=int, default=[-1])
    p.add_argument("--expansion", type=int, default=16)
    p.add_argument("--n_batches", type=int, default=16000,
                   help="batches of 64 to collect. 16000 = 1M examples = 7M answer vectors")
    p.add_argument("--sae_epochs", type=int, default=10)
    p.add_argument("--sae_batch", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    models = args.models or list_models()
    if not models:
        print("No models found in thoughtworks/arithmetic-sorl")
        return

    total = len(models) * len(args.layers) * len(args.k)
    print(f"Training {total} SAEs: {len(models)} models x {len(args.layers)} layers x {len(args.k)} k values")
    print(f"Data: {args.n_batches} batches x 64 = {args.n_batches * 64:,} examples")

    results = []
    for model_name in models:
        for layer in args.layers:
            for k in args.k:
                try:
                    fvu = train_sae_for_model(
                        model_name, tokenizer, layer, k,
                        args.expansion, args.n_batches, args.sae_epochs,
                        args.sae_batch, args.device,
                    )
                    results.append((model_name, layer, k, fvu, "OK"))
                except Exception as e:
                    print(f"  FAILED: {e}")
                    results.append((model_name, layer, k, -1, str(e)))

    print(f"\n{'═' * 60}")
    print(f"  {sum(1 for r in results if r[4] == 'OK')}/{len(results)} SAEs trained")
    print(f"{'═' * 60}")
    for model_name, layer, k, fvu, status in results:
        print(f"  {model_name}/L{layer}_k{k}: FVU={fvu:.4f} {status}")


if __name__ == "__main__":
    main()
