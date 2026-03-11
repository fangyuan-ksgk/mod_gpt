"""
SoRL Ablate Sanity Check — trains with trainer_ablate.SoRLTrainer using
only base_traj_loss (all aux losses zeroed). Should match SFT performance.

Usage:
    # Single GPU
    python train_ablate_sanity.py --model_name Qwen/Qwen3-0.6B

    # DDP (4 GPUs)
    torchrun --nproc_per_node=4 train_ablate_sanity.py --model_name Qwen/Qwen3-0.6B
"""

import os
import sys
import json
import time
import argparse

import torch
from transformers import AutoTokenizer

from sorl.sorl_wrapper import SorlModelWrapper
from sorl.trainer_ablate import SoRLTrainer, SoRLConfig
from data.pt_dataset import get_dataset, evaluate_accuracy, _filter_traj_tokens


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="SoRL Ablate Sanity Check")

    # Model
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--abstract_vocab_size", type=int, default=128)

    # Data
    p.add_argument("--dataset", type=str, default="gsm8k",
                   choices=["gsm8k", "math_qa", "arc", "hellaswag",
                            "winogrande", "boolq", "openbookqa",
                            "commonsenseqa", "mmlu",
                            "aqua", "math", "scienceqa"])
    p.add_argument("--max_length", type=int, default=512)

    # Optimizer
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--cooldown_frac", type=float, default=0.4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # Training
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--num_epochs", type=int, default=3)

    # SoRL search (needed by trainer but aux losses are zeroed)
    p.add_argument("--num_rollouts", type=int, default=4)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--max_iterations", type=int, default=2)
    p.add_argument("--memory_span_abs", type=int, default=1792)
    p.add_argument("--memory_span_traj", type=int, default=1792)
    p.add_argument("--temperature", type=float, default=1.0)

    # Logging / Eval / Checkpoint
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--eval_samples", type=int, default=50)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--output_dir", type=str, default="./ckpt/ablate_sanity")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Accuracy evaluator (same as SFT)
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_accuracy_fn_factory(tokenizer, max_new_tokens, log_fn):
    """Returns a compute_accuracy function compatible with SoRLTrainer."""

    def compute_accuracy_fn(model, tokenizer_, dataset, device, num_samples):
        model.eval()
        base_vocab_size = model.vocab_sizes[0].item()
        extract_fn = dataset.extract_answer if hasattr(dataset, "extract_answer") else None
        if extract_fn is None:
            return {"accuracy": 0.0, "correct": 0, "total": 0}

        correct, total = 0, 0
        indices = list(range(min(num_samples, len(dataset))))

        for i in indices:
            sample = dataset[i]
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
            prompt_len = sample["prompt_len"]

            generated = model.generate(
                input_ids=input_ids[:, :prompt_len],
                attention_mask=attention_mask[:, :prompt_len],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )

            # Filter out abstract tokens for evaluation
            traj_tokens = _filter_traj_tokens(generated, base_vocab_size)
            full_text = tokenizer.decode(traj_tokens[0], skip_special_tokens=True)

            ref_ids = input_ids[0][input_ids[0] < base_vocab_size]
            ref_text = tokenizer.decode(ref_ids, skip_special_tokens=True)

            pred_answer = extract_fn(full_text)
            gold_answer = extract_fn(ref_text)

            if gold_answer is not None:
                total += 1
                is_correct = (
                    pred_answer is not None
                    and pred_answer.strip() == gold_answer.strip()
                )
                if is_correct:
                    correct += 1

        acc = correct / max(total, 1)
        result = {"accuracy": acc, "correct": correct, "total": total}
        log_fn(f"Eval: {correct}/{total} = {acc*100:.1f}%")
        model.train()
        return result

    return compute_accuracy_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Detect DDP
    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    rank = int(os.environ.get("RANK", 0)) if ddp else 0
    is_master = rank == 0

    # Logging
    os.makedirs(args.output_dir, exist_ok=True)
    logfile = os.path.join(args.output_dir, "train.log")

    def log(msg):
        if is_master:
            print(msg)
            with open(logfile, "a") as f:
                f.write(msg + "\n")

    log(f"=== SoRL Ablate Sanity Check ===")
    log(f"Args: {json.dumps(vars(args), indent=2)}")
    log(f"DDP: {ddp} | World size: {os.environ.get('WORLD_SIZE', 1)}")

    # ---- Model ----
    log(f"Loading model: {args.model_name}")
    model = SorlModelWrapper.from_pretrained(
        args.model_name,
        abstract_vocab_size_list=[args.abstract_vocab_size],
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    log(f"Full fine-tuning: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # ---- Datasets ----
    log(f"Loading dataset: {args.dataset} (max_length={args.max_length})")
    train_ds = get_dataset(args.dataset, split="train", tokenizer=tokenizer, max_length=args.max_length)
    val_ds = get_dataset(args.dataset, split="test", tokenizer=tokenizer, max_length=args.max_length)
    log(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ---- Config ----
    # All aux loss alphas zeroed — only base_traj_loss contributes
    config = SoRLConfig(
        num_rollouts=args.num_rollouts,
        K=args.K,
        max_iterations=args.max_iterations,
        memory_span_abs=args.memory_span_abs,
        memory_span_traj=args.memory_span_traj,
        temperature=args.temperature,
        alpha_info_gain=0.0,
        alpha_abs=0.0,
        alpha_soft_zipf=0.0,
        ortho_reg=0.0,
        lr=args.lr,
        emb_lr_mult=1.0,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        cooldown_frac=args.cooldown_frac,
        max_grad_norm=args.max_grad_norm,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        emb_warmup_steps=0,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
        eval_samples=args.eval_samples,
        output_dir=args.output_dir,
    )

    # ---- Accuracy evaluator ----
    accuracy_fn = compute_accuracy_fn_factory(tokenizer, args.max_new_tokens, log)

    # ---- Trainer ----
    trainer = SoRLTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        val_dataset=val_ds,
        compute_accuracy=accuracy_fn,
        config=config,
        ddp=ddp,
    )
    log(f"Trainer: SoRLTrainer (ablate mode — base_traj_loss only)")

    # ---- Initial eval ----
    if is_master:
        log("--- Initial evaluation ---")
        result = trainer.evaluate()
        if result:
            log(f"Pre-train accuracy: {result['accuracy']*100:.1f}% "
                f"({result['correct']}/{result['total']})")

    # ---- Train ----
    history = trainer.train()

    # ---- Final eval ----
    if is_master:
        log("--- Final evaluation ---")
        result = trainer.evaluate()
        if result:
            log(f"Final accuracy: {result['accuracy']*100:.1f}% "
                f"({result['correct']}/{result['total']})")

        # Save history
        hist_path = os.path.join(args.output_dir, "history.json")
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
        log(f"History saved to {hist_path}")
        log("Training complete!")


if __name__ == "__main__":
    main()
