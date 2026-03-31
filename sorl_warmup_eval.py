#!/usr/bin/env python3
"""
SoRL Warmup SFT → Evaluate abstraction-conditioned generation accuracy.

Pipeline:
  1. Load pretrained model + dataset
  2. Run WarmupSFTTrainer (clustering → centroid init → SFT warmup)
  3. Evaluate: K=args.K (with abstractions) vs K=None (NL-only)
  4. Print comparison results (+ optional JSON dump)

Usage:
  python sorl_warmup_eval.py --model Qwen/Qwen3-0.6B --dataset gsm8k --sft_steps 500
  python sorl_warmup_eval.py --model Qwen/Qwen3-1.7B --dataset gsm8k --sft_steps 1000 --alpha_jacobi 0.0
  python sorl_warmup_eval.py --model Qwen/Qwen3-0.6B --dataset gsm8k --sft_steps 500 --output results.json
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import json
import time
import torch
from transformers import AutoTokenizer
from sorl.sorl_wrapper import SorlModelWrapper
from sorl.trainer_ablate import WarmupSFTTrainer, WarmupSFTConfig
from data.pt_dataset import get_dataset, evaluate_accuracy, collate_fn


def parse_args():
    p = argparse.ArgumentParser(description="SoRL Warmup SFT → Eval")

    # Model / Data
    p.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--dataset", type=str, default="gsm8k")
    p.add_argument("--max_length", type=int, default=256)

    # Abstract vocab / chunking
    p.add_argument("--abs_vocab", type=int, default=128)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--n_chunks", type=int, default=50000)

    # SFT warmup training
    p.add_argument("--sft_steps", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--emb_lr_mult", type=float, default=10.0)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # Loss weights
    p.add_argument("--alpha_abs", type=float, default=0.5)
    p.add_argument("--alpha_traj", type=float, default=1.0)
    p.add_argument("--alpha_masked_traj", type=float, default=1.0)
    p.add_argument("--alpha_hinge", type=float, default=0.0)
    p.add_argument("--gamma_hinge", type=float, default=0.5)
    p.add_argument("--alpha_jacobi", type=float, default=0.5)

    # Corruption / masking
    p.add_argument("--corrupt_method", type=str, default="noise")
    p.add_argument("--corrupt_ratio", type=float, default=1.0)
    p.add_argument("--mask_nl_ratio", type=float, default=0.3)
    p.add_argument("--mask_nl_mode", type=str, default="random")
    p.add_argument("--mask_nl_fixed_id", type=int, default=0)

    # Centroid init
    p.add_argument("--skip_centroid_init", action="store_true",
                   help="Skip initializing abstract embeddings with K-means centroids (use random init)")

    # Memory spans
    p.add_argument("--memory_span_abs", type=int, default=1792)
    p.add_argument("--memory_span_traj", type=int, default=1792)

    # Eval
    p.add_argument("--eval_samples", type=int, default=100)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=256)

    # Logging
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--output", type=str, default=None, help="Path to save JSON results")
    p.add_argument("--tag", type=str, default=None, help="Tag for this run (appears in output)")

    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model:  {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"K={args.K} | abs_vocab={args.abs_vocab} | sft_steps={args.sft_steps}")
    print(f"Losses: abs={args.alpha_abs} traj={args.alpha_traj} m_traj={args.alpha_masked_traj} "
          f"hinge={args.alpha_hinge} jacobi={args.alpha_jacobi}")
    print(f"skip_centroid_init={args.skip_centroid_init}")
    print("=" * 70)

    # ---- Load model + data ----
    t0 = time.time()
    model = SorlModelWrapper.from_pretrained(args.model, abstract_vocab_size_list=[args.abs_vocab])
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    train_ds = get_dataset(args.dataset, split="train", tokenizer=tokenizer, max_length=args.max_length)
    val_ds = get_dataset(args.dataset, split="test", tokenizer=tokenizer, max_length=args.max_length)
    print(f"Loaded in {time.time()-t0:.1f}s | Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ---- Build WarmupSFTConfig ----
    warmup_cfg = WarmupSFTConfig(
        K=args.K,
        abs_vocab=args.abs_vocab,
        n_chunks_for_clustering=args.n_chunks,
        skip_centroid_init=args.skip_centroid_init,
        alpha_abs=args.alpha_abs,
        alpha_traj=args.alpha_traj,
        alpha_masked_traj=args.alpha_masked_traj,
        alpha_hinge=args.alpha_hinge,
        gamma_hinge=args.gamma_hinge,
        alpha_jacobi=args.alpha_jacobi,
        corrupt_method=args.corrupt_method,
        corrupt_ratio=args.corrupt_ratio,
        mask_nl_ratio=args.mask_nl_ratio,
        mask_nl_mode=args.mask_nl_mode,
        mask_nl_fixed_id=args.mask_nl_fixed_id,
        lr=args.lr,
        emb_lr_mult=args.emb_lr_mult,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        num_steps=args.sft_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        memory_span_abs=args.memory_span_abs,
        memory_span_traj=args.memory_span_traj,
        log_every=args.log_every,
    )

    # ---- Run SFT warmup ----
    trainer = WarmupSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        val_dataset=val_ds,
        compute_accuracy=evaluate_accuracy,
        collate_fn=collate_fn,
        config=warmup_cfg,
        device=device,
    )

    print("\n--- SFT Warmup ---")
    t_train = time.time()
    history = trainer.train()
    train_time = time.time() - t_train
    print(f"SFT warmup completed in {train_time:.1f}s")

    # ---- Evaluate: K vs None ----
    print(f"\n--- Evaluation ({args.eval_samples} samples) ---")
    model.eval()

    t_eval = time.time()
    result_k = evaluate_accuracy(
        model, tokenizer, val_ds, device,
        num_samples=args.eval_samples,
        eval_batch_size=args.eval_batch_size,
        max_new_tokens=args.max_new_tokens,
        eval_K=args.K,
    )
    result_none = evaluate_accuracy(
        model, tokenizer, val_ds, device,
        num_samples=args.eval_samples,
        eval_batch_size=args.eval_batch_size,
        max_new_tokens=args.max_new_tokens,
        eval_K=None,
    )
    eval_time = time.time() - t_eval

    # ---- Print results ----
    acc_k = result_k["strict_accuracy"] * 100
    acc_none = result_none["strict_accuracy"] * 100
    gap = acc_k - acc_none

    print(f"\n{'=' * 70}")
    print(f"  K={args.K}:   {acc_k:.1f}%  ({result_k['correct']}/{result_k['attempted']})"
          f"  parse_rate={result_k['parse_rate']*100:.1f}%")
    print(f"  K=None: {acc_none:.1f}%  ({result_none['correct']}/{result_none['attempted']})"
          f"  parse_rate={result_none['parse_rate']*100:.1f}%")
    print(f"  Gap (K-None): {gap:+.1f}%")
    print(f"{'=' * 70}")
    print(f"  Train time: {train_time:.1f}s | Eval time: {eval_time:.1f}s")

    # ---- Save JSON ----
    output = {
        "tag": args.tag,
        "model": args.model,
        "dataset": args.dataset,
        "K": args.K,
        "abs_vocab": args.abs_vocab,
        "sft_steps": args.sft_steps,
        "lr": args.lr,
        "emb_lr_mult": args.emb_lr_mult,
        "alpha_abs": args.alpha_abs,
        "alpha_traj": args.alpha_traj,
        "alpha_masked_traj": args.alpha_masked_traj,
        "alpha_hinge": args.alpha_hinge,
        "alpha_jacobi": args.alpha_jacobi,
        "skip_centroid_init": args.skip_centroid_init,
        "mask_nl_ratio": args.mask_nl_ratio,
        "mask_nl_mode": args.mask_nl_mode,
        "corrupt_method": args.corrupt_method,
        "corrupt_ratio": args.corrupt_ratio,
        "eval_samples": args.eval_samples,
        "acc_k": acc_k,
        "acc_none": acc_none,
        "gap": gap,
        "parse_rate_k": result_k["parse_rate"] * 100,
        "parse_rate_none": result_none["parse_rate"] * 100,
        "train_time_s": round(train_time, 1),
        "eval_time_s": round(eval_time, 1),
        "final_loss": history["loss"][-1] if history["loss"] else None,
        "final_abs_loss": history["abs_loss"][-1] if history["abs_loss"] else None,
        "final_traj_loss": history["traj_loss"][-1] if history["traj_loss"] else None,
    }

    if args.output:
        # Append to JSON lines file for easy comparison
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        with open(args.output, "a") as f:
            f.write(json.dumps(output) + "\n")
        print(f"Results appended to {args.output}")
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
