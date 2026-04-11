"""
Arithmetic training with Qwen3 + SorlModelWrapper.

Baseline (SFT): python -m arithmetic.train --mode baseline --ops add
SoRL v1:        python -m arithmetic.train --mode sorl --ops add --abs_vocab 16
"""
import os
import sys
import json
import argparse
import time
import math
import torch
import torch.nn as nn
import wandb
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import Qwen3Config, AutoTokenizer
from sorl.sorl_wrapper import SorlModelWrapper
from sorl.trainer_ablate import SoRLTrainer, SoRLConfig
from arithmetic.datasets.addition import (
    generate_batch, NUM_TOKENS, ALL_LABELS,
)

torch.set_float32_matmul_precision('high')

TOKENIZER_NAME = "Qwen/Qwen3-0.6B"
WANDB_PROJECT = "sorl-arithmetic"
WANDB_ENTITY = "nlp_and_interpretability"

QWEN3_TOKEN_MAP = {
    0: 15, 1: 16, 2: 17, 3: 18, 4: 19,
    5: 20, 6: 21, 7: 22, 8: 23, 9: 24,
    10: 10, 11: 28, 12: 12,
}
QWEN3_INV_MAP = {v: k for k, v in QWEN3_TOKEN_MAP.items()}


class Qwen3ArithmeticDataset(Dataset):
    """On-the-fly arithmetic dataset producing Qwen3 token IDs."""
    def __init__(self, tokenizer, n_digits=6, ops="add", size=100_000):
        assert "qwen" in type(tokenizer).__module__.lower() or "qwen" in getattr(tokenizer, 'name_or_path', '').lower()
        self.tokenizer = tokenizer
        self.n_digits = n_digits
        self.ops = ops
        self.size = size
        self.prompt_len = 2 * n_digits + 2
        self.seq_len = 3 * n_digits + 3

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        tokens, _, _ = generate_batch(1, self.n_digits, ops=self.ops,
                                      use_enrichment=True, device="cpu")
        token_ids = torch.tensor([QWEN3_TOKEN_MAP[t.item()] for t in tokens[0]], dtype=torch.long)
        return {
            "input_ids": token_ids,
            "attention_mask": torch.ones(len(token_ids), dtype=torch.long),
            "prompt_len": torch.tensor(self.prompt_len, dtype=torch.long),
        }


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "prompt_len": torch.stack([b["prompt_len"] for b in batch]),
    }


# ── Eval: generates with abstraction tokens, scores trajectory only ──

def eval_with_recursion(model, dataset, device, K=4, num_samples=200):
    """
    SoRL eval using recursion (same procedure as training):
    insert placeholders, denoise via recursion, teacher-force trajectory predictions.
    """
    from sorl.sorl_trainer import infer_insert_mask, insert_tokens_with_padding, expand_prompt_len

    model.eval()
    base_v = model.vocab_sizes[0].item()
    prompt_len = dataset.prompt_len
    pad_id = dataset.tokenizer.pad_token_id
    n_correct = 0

    for _ in range(num_samples):
        item = dataset[0]
        ids = item["input_ids"].unsqueeze(0).to(device)
        attn = item["attention_mask"].unsqueeze(0).to(device)
        pl_t = item["prompt_len"].unsqueeze(0).to(device)
        true_answer = ids[0, prompt_len:]

        with torch.no_grad():
            im = infer_insert_mask(ids, K, attn)
            ep = expand_prompt_len(pl_t, im)
            ed, ea = insert_tokens_with_padding(ids, attn, im, model.vocab_sizes[0], pad_id)
            data, ppt, logits = model.recursion(
                ed, ea, max_iterations=2,
                memory_span_abs=1792, memory_span_traj=1792,
                temperature=0.0, prompt_len=ep,
            )

            # Extract trajectory predictions at answer positions
            is_traj = data[0, 1:] < base_v
            pred_logits = logits[0, :-1][is_traj][:, :base_v].argmax(dim=-1)
            answer_len = dataset.n_digits + 1  # n_digits + overflow digit
            pred_answer = pred_logits[-answer_len:]

        if (pred_answer == true_answer).all():
            n_correct += 1

    model.train()
    return n_correct / max(num_samples, 1)


def eval_sft(model, dataset, device, num_samples=200):
    """SFT eval: no abstraction tokens, just forward pass."""
    model.eval()
    prompt_len = dataset.prompt_len
    base_v = model.vocab_sizes[0].item()
    n_correct = 0

    for _ in range(num_samples):
        item = dataset[0]
        ids = item["input_ids"].unsqueeze(0).to(device)
        attn = item["attention_mask"].unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(ids, attention_mask=attn, memory_span_abs=1792, memory_span_traj=1792)
        pred = out.logits[0, prompt_len - 1:-1, :base_v].argmax(dim=-1)
        target = ids[0, prompt_len:]
        if (pred == target).all():
            n_correct += 1

    model.train()
    return n_correct / max(num_samples, 1)


def compute_accuracy_for_trainer(model, tokenizer, dataset, device, num_samples, **kwargs):
    """Callback for SoRLTrainer eval — uses recursion (matches training)."""
    K = kwargs.get("eval_K") or 4  # eval_K can be None
    acc = eval_with_recursion(model, dataset, device, K=K, num_samples=num_samples)
    return {"accuracy": acc}


# ── Model factory ──────────────────────────────────────────────────

def make_model(args, tokenizer):
    config = Qwen3Config(
        hidden_size=args.n_embd, num_hidden_layers=args.n_layer,
        num_attention_heads=args.n_head, num_key_value_heads=args.n_head,
        intermediate_size=args.n_embd * 4, vocab_size=tokenizer.vocab_size,
        max_position_embeddings=128,
    )
    abs_vocab = args.abs_vocab if args.abs_vocab > 0 else 1
    return SorlModelWrapper.from_scratch(config, [tokenizer.vocab_size, abs_vocab], tokenizer.pad_token_id)


# ── SFT baseline trainer ──────────────────────────────────────────

def train_sft(model, train_ds, val_ds, args, run_name):
    device = args.device
    model = model.to(device)
    model.train()

    prompt_len = train_ds.prompt_len
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_fn, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
    total_steps = len(loader) * args.num_epochs
    warmup_steps = int(total_steps * 0.2)

    history = {"step": [], "loss": [], "base_loss": [], "lr": []}
    global_step = 0
    t_start = time.time()

    for epoch in range(args.num_epochs):
        for batch in loader:
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)

            out = model(ids, attention_mask=attn, memory_span_abs=1792, memory_span_traj=1792)
            labels = ids.clone()
            labels[:, :prompt_len] = -100
            shift_logits = out.logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            if global_step < warmup_steps:
                lr = args.lr * global_step / max(warmup_steps, 1)
            else:
                progress = (global_step - warmup_steps) / max(total_steps - warmup_steps, 1)
                lr = args.lr * 0.5 * (1 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1

            if global_step % 50 == 0:
                print(f"step {global_step} | loss={loss.item():.4f} | lr={lr:.2e}")
                history["step"].append(global_step)
                history["loss"].append(loss.item())
                history["base_loss"].append(loss.item())
                history["lr"].append(lr)
                if wandb.run is not None:
                    wandb.log({"loss": loss.item(), "lr": lr}, step=global_step)

            steps_per_epoch = max(1, len(train_ds) // args.batch_size)
            if global_step % steps_per_epoch == 0:
                acc = eval_sft(model, val_ds, device, 100)
                print(f"  --- Eval step {global_step}: accuracy={acc:.3f} ---")
                history.setdefault("eval_step", []).append(global_step)
                history.setdefault("eval_accuracy", []).append(acc)
                if wandb.run is not None:
                    wandb.log({"eval/accuracy": acc}, step=global_step)

    return history


# ── Wandb-aware v1 trainer ─────────────────────────────────────────

class WandbSoRLTrainer(SoRLTrainer):
    def _log(self, msg):
        super()._log(msg)
        if self.is_master and self.history["step"] and wandb.run is not None:
            wandb.log({
                "loss": self.history["loss"][-1],
                "base_loss": self.history["base_loss"][-1],
                "info_loss": self.history["info_loss"][-1],
                "abs_loss": self.history["abs_loss"][-1],
                "zipf_loss": self.history["zipf_loss"][-1],
                "lr": self.history["lr"][-1],
            }, step=self.history["step"][-1])

    def evaluate(self, eval_K=None):
        result = super().evaluate(eval_K=eval_K)
        if result and self.is_master:
            self.history.setdefault("eval_step", []).append(
                self.history["step"][-1] if self.history["step"] else 0)
            self.history.setdefault("eval_accuracy", []).append(result["accuracy"])
            if wandb.run is not None:
                wandb.log({"eval/accuracy": result["accuracy"]})
        return result


# ── Main ────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["baseline", "sorl"], default="baseline")
    p.add_argument("--ops", choices=["add", "add_sub"], default="add")
    p.add_argument("--n_digits", type=int, default=6)
    p.add_argument("--n_layer", type=int, default=2)
    p.add_argument("--n_head", type=int, default=3)
    p.add_argument("--n_embd", type=int, default=510)
    p.add_argument("--abs_vocab", type=int, default=0)
    p.add_argument("--K", type=int, default=4)
    # v1 SoRL hyperparams (Fangyuan's recommended defaults)
    p.add_argument("--alpha_info_gain", type=float, default=10.0)
    p.add_argument("--alpha_abs", type=float, default=0.1)
    p.add_argument("--alpha_soft_zipf", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_epochs", type=int, default=5)
    p.add_argument("--dataset_size", type=int, default=500_000)
    p.add_argument("--lr", type=float, default=8e-5)
    p.add_argument("--output_dir", type=str, default="ckpt/arithmetic")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--no_wandb", action="store_true")
    args = p.parse_args()

    if args.mode == "baseline":
        args.abs_vocab = 0

    run_name = f"{args.ops}_{args.mode}"
    if args.abs_vocab > 0:
        run_name += f"_abs{args.abs_vocab}"
    if args.K != 4 and args.mode == "sorl":
        run_name += f"_K{args.K}"
    run_name += f"_{args.dataset_size // 1000}K"
    if args.n_layer != 2 or args.n_head != 3 or args.n_embd != 510:
        run_name += f"_{args.n_layer}L{args.n_head}H{args.n_embd}d"

    if not args.no_wandb:
        wandb.init(
            project=WANDB_PROJECT, entity=WANDB_ENTITY,
            name=run_name, config=vars(args),
            tags=[args.ops, args.mode, f"abs{args.abs_vocab}", f"{args.dataset_size // 1000}K"],
        )

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model = make_model(args, tokenizer)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"{'═' * 60}")
    print(f"  {run_name}")
    print(f"  arch: {args.n_layer}L/{args.n_head}H/{args.n_embd}d | params: {n_params:,}")
    print(f"  train: {args.num_epochs} epochs x {args.dataset_size} samples, batch={args.batch_size}")
    if args.mode == "sorl":
        print(f"  SoRL v1: abs={args.abs_vocab} K={args.K} ig={args.alpha_info_gain} abs={args.alpha_abs} zipf={args.alpha_soft_zipf}")
    else:
        print(f"  pure SFT")
    print(f"{'═' * 60}")

    if wandb.run is not None:
        wandb.config.update({"n_params": n_params})

    train_ds = Qwen3ArithmeticDataset(tokenizer, args.n_digits, args.ops, args.dataset_size)
    val_ds = Qwen3ArithmeticDataset(tokenizer, args.n_digits, args.ops, 1000)

    import subprocess, datetime
    git_hash = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    manifest = {
        **vars(args),
        "n_params": n_params,
        "run_name": run_name,
        "git_commit": git_hash,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "tokenizer": TOKENIZER_NAME,
        "dataset_repo": "thoughtworks/arithmetic-sorl-data",
        "dataset_config": "add_sub_6digit" if args.ops == "add_sub" else "add_6digit",
        "model_repo": "thoughtworks/arithmetic-sorl",
        "trainer_version": "sft" if args.mode == "baseline" else "v1",
        "wandb_run_id": wandb.run.id if wandb.run is not None else None,
        "wandb_url": wandb.run.url if wandb.run is not None else None,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    if args.mode == "baseline":
        history = train_sft(model, train_ds, val_ds, args, run_name)
        final_acc = eval_sft(model, val_ds, args.device, 200)
    else:
        cfg = SoRLConfig(
            K=args.K, batch_size=args.batch_size,
            num_epochs=args.num_epochs, lr=args.lr,
            output_dir=args.output_dir,
            log_every=50,
            eval_every=max(1, args.dataset_size // args.batch_size),  # every epoch
            save_every=999999, eval_samples=100,
            alpha_info_gain=args.alpha_info_gain,
            alpha_abs=args.alpha_abs,
            alpha_soft_zipf=args.alpha_soft_zipf,
            alpha_traj=0.0,  # v1 uses info_gain, not traj
        )
        trainer = WandbSoRLTrainer(
            model, tokenizer, train_ds, val_ds,
            compute_accuracy=compute_accuracy_for_trainer,
            collate_fn=collate_fn, config=cfg, device=args.device,
        )
        trainer.train()
        history = trainer.history
        final_acc = eval_with_recursion(model, val_ds, args.device, K=args.K, num_samples=200)

    print(f"Final accuracy (random): {final_acc:.3f}")

    # Full per-split eval with ArithmeticEvaluator
    from arithmetic.evaluate import ArithmeticEvaluator
    evaluator = ArithmeticEvaluator(model, tokenizer, device=args.device, n_digits=args.n_digits)
    K_eval = args.K if args.mode == "sorl" else None
    eval_results_sft = evaluator.run(ops=args.ops, K=None, n_per_split=50)
    print(f"\nSFT eval (no abs):")
    evaluator.print_table(eval_results_sft)
    if args.mode == "sorl":
        eval_results_sorl = evaluator.run(ops=args.ops, K=args.K, n_per_split=50)
        print(f"\nSoRL eval (K={args.K}):")
        evaluator.print_table(eval_results_sorl)
    else:
        eval_results_sorl = None

    if wandb.run is not None:
        # Log per-split accuracy to both history and summary
        summary_metrics = {"eval/final_accuracy": final_acc}
        for eval_name, eval_result in [("sft", eval_results_sft), ("sorl", eval_results_sorl)]:
            if eval_result is None:
                continue
            summary_metrics[f"eval/{eval_name}/overall"] = eval_result["summary"]["overall_accuracy"]
            for split_name, split_data in eval_result.get("splits", {}).items():
                summary_metrics[f"eval/{eval_name}/{split_name}"] = split_data["full_accuracy"]
        wandb.log(summary_metrics)
        # Also set as summary so it shows in the runs table
        for k, v in summary_metrics.items():
            wandb.run.summary[k] = v
        wandb.finish()

    if args.push_to_hub:
        from arithmetic.hub import save_model
        manifest["final_accuracy"] = eval_results_sorl["summary"]["overall_accuracy"] if eval_results_sorl else eval_results_sft["summary"]["overall_accuracy"]
        manifest["sft_accuracy"] = eval_results_sft["summary"]["overall_accuracy"]
        manifest["eval_method"] = "ArithmeticEvaluator"
        metrics = {
            "history": history,
            "final_accuracy": manifest["final_accuracy"],
            "sft_eval": eval_results_sft,
        }
        if eval_results_sorl:
            metrics["sorl_eval"] = eval_results_sorl
            metrics["sorl_overall_accuracy"] = eval_results_sorl["summary"]["overall_accuracy"]
            metrics["sft_overall_accuracy"] = eval_results_sft["summary"]["overall_accuracy"]
        save_model(model, manifest, metrics, subfolder=run_name)

        # Per-job validation: verify upload is complete and correct
        print(f"\nValidating upload: {run_name}")
        from arithmetic.job_manager.post_sweep import validate_uploaded_model
        issues = validate_uploaded_model(run_name)
        if issues:
            print(f"  VALIDATION FAILED:")
            for issue in issues:
                print(f"    - {issue}")
            sys.exit(1)  # non-zero exit → queue will retry
        else:
            print(f"  Validation passed ✓")

        import shutil
        shutil.rmtree(args.output_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
