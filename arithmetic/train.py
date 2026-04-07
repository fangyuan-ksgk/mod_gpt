"""
Arithmetic training with Qwen3 + SorlModelWrapper.

Baseline (SFT): python -m arithmetic.train --mode baseline --ops add
SoRL v6:        python -m arithmetic.train --mode sorl --ops add --abs_vocab 16
"""
import os
import sys
import json
import argparse
import time
import torch
import torch.nn as nn
import wandb
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import Qwen3Config, AutoTokenizer
from sorl.sorl_wrapper import SorlModelWrapper
from sorl.selfroute import SoRLTrainerv6
from sorl.trainer_ablate import SoRLConfig
from arithmetic.datasets.addition import (
    generate_batch, eval_accuracy, NUM_TOKENS, ALL_LABELS,
)

torch.set_float32_matmul_precision('high')

TOKENIZER_NAME = "Qwen/Qwen3-0.6B"
WANDB_PROJECT = "sorl-arithmetic"
WANDB_ENTITY = "nlp_and_interpretability"

# Qwen3 token IDs for arithmetic characters (verified against Qwen3-0.6B tokenizer).
# WARNING: These IDs are specific to Qwen3. Do NOT use with other tokenizers.
QWEN3_TOKEN_MAP = {
    0: 15, 1: 16, 2: 17, 3: 18, 4: 19,
    5: 20, 6: 21, 7: 22, 8: 23, 9: 24,
    10: 10, 11: 28, 12: 12,
}


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
                                      use_sum9_aug=True, device="cpu")
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


# ── Accuracy ────────────────────────────────────────────────────────

def compute_accuracy(model, dataset, device, num_samples=200):
    """Evaluate accuracy: argmax prediction on answer tokens."""
    model.eval()
    n_correct, n_total = 0, 0
    prompt_len = dataset.prompt_len

    for _ in range(num_samples):
        item = dataset[0]
        ids = item["input_ids"].unsqueeze(0).to(device)
        attn = item["attention_mask"].unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(ids, attention_mask=attn,
                        memory_span_abs=1792, memory_span_traj=1792)
        pred = out.logits[0, prompt_len - 1:-1].argmax(dim=-1)
        target = ids[0, prompt_len:]
        if (pred == target).all():
            n_correct += 1
        n_total += 1

    model.train()
    return n_correct / max(n_total, 1)


def compute_accuracy_for_trainer(model, tokenizer, dataset, device, num_samples, **kwargs):
    """Wrapper matching SoRLTrainer's callback signature."""
    return {"accuracy": compute_accuracy(model, dataset, device, num_samples)}


# ── Model factory ──────────────────────────────────────────────────

def make_model(args, tokenizer):
    config = Qwen3Config(
        hidden_size=args.n_embd,
        num_hidden_layers=args.n_layer,
        num_attention_heads=args.n_head,
        num_key_value_heads=args.n_head,
        intermediate_size=args.n_embd * 4,
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=128,
    )
    abs_vocab = args.abs_vocab if args.abs_vocab > 0 else 1
    return SorlModelWrapper.from_scratch(
        config,
        full_vocab_size_list=[tokenizer.vocab_size, abs_vocab],
        pad_token_id=tokenizer.pad_token_id,
    )


# ── SFT baseline trainer ──────────────────────────────────────────

def train_sft(model, train_ds, val_ds, args, run_name):
    """
    Plain SFT: cross-entropy on answer tokens. No SoRL, no abstraction tokens.
    """
    device = args.device
    model = model.to(device)
    model.train()

    prompt_len = train_ds.prompt_len
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_fn, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                                   betas=(0.9, 0.98))
    total_steps = len(loader) * args.num_epochs
    warmup_steps = int(total_steps * 0.2)

    history = {"step": [], "loss": [], "base_loss": [], "lr": []}
    global_step = 0
    t_start = time.time()

    for epoch in range(args.num_epochs):
        for batch in loader:
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)

            # Forward
            out = model(ids, attention_mask=attn,
                        memory_span_abs=1792, memory_span_traj=1792)
            logits = out.logits

            # Loss on answer tokens only
            labels = ids.clone()
            labels[:, :prompt_len] = -100  # mask question
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            # LR schedule: linear warmup + cosine decay
            if global_step < warmup_steps:
                lr = args.lr * global_step / max(warmup_steps, 1)
            else:
                import math
                progress = (global_step - warmup_steps) / max(total_steps - warmup_steps, 1)
                lr = args.lr * 0.5 * (1 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1

            # Logging
            if global_step % 50 == 0:
                elapsed = time.time() - t_start
                print(f"epoch {epoch + global_step/len(loader)/args.num_epochs:.3f}/{args.num_epochs} | "
                      f"loss={loss.item():.4f} | lr={lr:.2e} | {elapsed:.0f}s")
                history["step"].append(global_step)
                history["loss"].append(loss.item())
                history["base_loss"].append(loss.item())
                history["lr"].append(lr)
                if not args.no_wandb:
                    wandb.log({"loss": loss.item(), "base_loss": loss.item(), "lr": lr,
                               "step": global_step}, step=global_step)

            # Eval
            if global_step % 500 == 0:
                acc = compute_accuracy(model, val_ds, device, 100)
                print(f"  --- Eval step {global_step}: accuracy={acc:.3f} ---")
                if not args.no_wandb:
                    wandb.log({"eval/accuracy": acc}, step=global_step)

        print(f"=== Epoch {epoch + 1} complete ===")

    return history


# ── Wandb-aware v6 trainer ─────────────────────────────────────────

class WandbSoRLTrainerv6(SoRLTrainerv6):
    def _log(self, msg):
        super()._log(msg)
        if self.is_master and self.history["step"]:
            wandb.log({
                "loss": self.history["loss"][-1],
                "base_loss": self.history["base_loss"][-1],
                "traj_loss": self.history["traj_loss"][-1],
                "lr": self.history["lr"][-1],
                "step": self.history["step"][-1],
            }, step=self.history["step"][-1])

    def evaluate(self, eval_K=None):
        result = super().evaluate(eval_K=eval_K)
        if result and self.is_master:
            wandb.log({"eval/accuracy": result["accuracy"]})
        return result


# ── Main ────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["baseline", "sorl"], default="baseline")
    p.add_argument("--ops", choices=["add", "add_sub"], default="add")
    p.add_argument("--n_digits", type=int, default=6)
    p.add_argument("--n_layer", type=int, default=3)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=512)
    p.add_argument("--abs_vocab", type=int, default=0)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--dataset_size", type=int, default=100_000)
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
    if args.n_layer != 3 or args.n_head != 4 or args.n_embd != 512:
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
    print(f"  mode: {args.mode}" + (f" | abs_vocab={args.abs_vocab} K={args.K}" if args.mode == "sorl" else " | pure SFT"))
    print(f"{'═' * 60}")

    if not args.no_wandb:
        wandb.config.update({"n_params": n_params})

    train_ds = Qwen3ArithmeticDataset(tokenizer, args.n_digits, args.ops, args.dataset_size)
    val_ds = Qwen3ArithmeticDataset(tokenizer, args.n_digits, args.ops, 1000)

    # Build manifest
    import subprocess, datetime
    git_hash = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    manifest = {
        **vars(args),
        "n_params": n_params,
        "run_name": run_name,
        "git_commit": git_hash,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "tokenizer": TOKENIZER_NAME,
        "dataset_repo": "thoughtworks/arithmetic-sorl-data",
        "dataset_config": "add_sub_6digit" if args.ops == "add_sub" else "add_6digit",
        "model_repo": "thoughtworks/arithmetic-sorl",
        "trainer_version": "sft" if args.mode == "baseline" else "v6",
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # ── Train ───────────────────────────────────────────────────
    if args.mode == "baseline":
        history = train_sft(model, train_ds, val_ds, args, run_name)
    else:
        cfg = SoRLConfig(
            K=args.K,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            lr=args.lr,
            output_dir=args.output_dir,
            log_every=50, eval_every=500,
            save_every=999999,
            eval_samples=100,
            alpha_traj=1.0,
        )
        trainer = WandbSoRLTrainerv6(
            model, tokenizer, train_ds, val_ds,
            compute_accuracy=compute_accuracy_for_trainer,
            collate_fn=collate_fn,
            config=cfg, device=args.device,
        )
        trainer.train()
        history = trainer.history

    # Final eval
    final_acc = compute_accuracy(model, val_ds, args.device, 200)
    print(f"Final accuracy: {final_acc:.3f}")
    if not args.no_wandb:
        wandb.log({"eval/final_accuracy": final_acc})
        wandb.finish()

    # Upload to HF Hub + clean up local
    if args.push_to_hub:
        from arithmetic.hub import save_model
        manifest["final_accuracy"] = final_acc
        metrics = {"history": history, "final_accuracy": final_acc}
        save_model(model, manifest, metrics, subfolder=run_name)
        import shutil
        shutil.rmtree(args.output_dir, ignore_errors=True)
        print(f"Cleaned up local: {args.output_dir}")


if __name__ == "__main__":
    main()
