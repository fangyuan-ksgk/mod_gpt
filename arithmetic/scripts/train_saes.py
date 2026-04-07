#!/usr/bin/env python3
"""
Train SAEs on arithmetic models. Collects activations once per (model, layer),
trains all k values on it, uploads each to HF, then discards activations.

No local disk usage beyond temp files for HF upload.

Usage:
    python arithmetic/scripts/train_saes.py
    python arithmetic/scripts/train_saes.py --models add_baseline_500K --k 8 16 32 64
    python arithmetic/scripts/train_saes.py --layers 0 1 2 --k 32
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoTokenizer
from arithmetic.hub import list_models, load_model
from arithmetic.interp_utils.sae_trainer import SAETrainer, SAETrainerConfig, collect_activations


def train_saes_for_model_layer(model_name: str, model, config, tokenizer,
                                layer: int, k_values: list, expansion: int,
                                n_batches: int, sae_epochs: int, sae_batch: int,
                                device: str):
    """Collect activations once, train all k values, upload each."""
    hidden_size = config.get("n_embd", 512)
    ops = config.get("ops", "add")

    print(f"\n  Collecting activations layer={layer} ({n_batches} batches)...")
    acts = collect_activations(
        model, tokenizer, layer=layer,
        n_digits=config.get("n_digits", 6), ops=ops,
        n_batches=n_batches, batch_size=64,
        positions="answer", device=device,
    )
    acts = acts.to(device)
    print(f"  {acts.shape[0]:,} vectors, dim={acts.shape[1]}, {acts.nbytes / 1e9:.1f}GB")

    for k in k_values:
        subfolder = f"{model_name}/L{layer}_k{k}"
        print(f"\n    Training {subfolder}...")

        sae_cfg = SAETrainerConfig(
            d_in=hidden_size, k=k, expansion_factor=expansion, device=device,
        )
        trainer = SAETrainer(sae_cfg)

        final_fvu = 0
        for epoch in range(sae_epochs):
            indices = torch.randperm(acts.shape[0], device=device)
            epoch_loss, n_steps = 0, 0
            for i in range(0, acts.shape[0] - sae_batch, sae_batch):
                info = trainer.step(acts[indices[i:i + sae_batch]])
                epoch_loss += info["fvu"]
                n_steps += 1
            final_fvu = epoch_loss / max(n_steps, 1)
        print(f"    fvu={final_fvu:.4f} | dead={info['num_dead']}")

        train_info = {
            "model_name": model_name,
            "model_repo": "thoughtworks/arithmetic-sorl",
            "layer": layer,
            "k": k,
            "expansion_factor": expansion,
            "sae_epochs": sae_epochs,
            "sae_batch": sae_batch,
            "n_vectors": acts.shape[0],
            "n_examples": n_batches * 64,
            "positions": "answer",
            "final_fvu": final_fvu,
            "num_dead": info["num_dead"],
            "model_config": config,
        }
        trainer.save_to_hub(subfolder, train_info)
        del trainer

    del acts
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--k", nargs="*", type=int, default=[32])
    p.add_argument("--layers", nargs="*", type=int, default=[0, 1, 2])
    p.add_argument("--expansion", type=int, default=16)
    p.add_argument("--n_batches", type=int, default=2000,
                   help="2000 x 512 = 1M examples = 7M answer vectors")
    p.add_argument("--sae_epochs", type=int, default=10)
    p.add_argument("--sae_batch", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    models = args.models or list_models()
    if not models:
        print("No models found")
        return

    total = len(models) * len(args.layers) * len(args.k)
    n_collections = len(models) * len(args.layers)
    print(f"Plan: {len(models)} models x {len(args.layers)} layers x {len(args.k)} k values = {total} SAEs")
    print(f"Activation collections: {n_collections} (reused across k values)")
    print(f"Data: {args.n_batches * 64:,} examples per collection")
    print()

    results = []
    for model_name in models:
        print(f"\n{'═' * 50}")
        print(f"  {model_name}")
        print(f"{'═' * 50}")

        try:
            model, config, metrics = load_model(model_name, device=args.device)
        except Exception as e:
            print(f"  FAILED to load: {e}")
            for layer in args.layers:
                for k in args.k:
                    results.append((model_name, layer, k, -1, str(e)))
            continue

        for layer in args.layers:
            try:
                train_saes_for_model_layer(
                    model_name, model, config, tokenizer,
                    layer, args.k, args.expansion,
                    args.n_batches, args.sae_epochs, args.sae_batch,
                    args.device,
                )
                for k in args.k:
                    results.append((model_name, layer, k, 0, "OK"))
            except Exception as e:
                print(f"  FAILED layer {layer}: {e}")
                for k in args.k:
                    results.append((model_name, layer, k, -1, str(e)))

        del model
        torch.cuda.empty_cache()

    print(f"\n{'═' * 60}")
    ok = sum(1 for r in results if r[4] == "OK")
    print(f"  {ok}/{len(results)} SAEs trained")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
