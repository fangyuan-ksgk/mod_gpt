"""
Arithmetic training with Qwen3 + SorlModelWrapper.

Baseline:  python -m arithmetic.train --mode baseline --ops add
SoRL v6:   python -m arithmetic.train --mode sorl --ops add --abs_vocab 16
"""
import os
import sys
import json
import argparse
import torch
import random
import wandb
from pathlib import Path
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import Qwen3Config, AutoTokenizer
from sorl.sorl_wrapper import SorlModelWrapper
from sorl.selfroute import SoRLTrainerv6
from sorl.trainer_ablate import SoRLTrainer, SoRLConfig
from arithmetic.datasets.addition import (
    generate_batch, eval_accuracy, NUM_TOKENS, ALL_LABELS,
)

torch.set_float32_matmul_precision('high')

TOKENIZER_NAME = "Qwen/Qwen3-0.6B"
WANDB_PROJECT = "sorl-arithmetic"
WANDB_ENTITY = "nlp_and_interpretability"


# ── Dataset adapter ─────────────────────────────────────────────────

class ArithmeticDataset(Dataset):
    def __init__(self, tokenizer, n_digits=6, ops="add", size=100_000):
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
        token_ids = self._map_to_qwen_ids(tokens[0])
        return {
            "input_ids": token_ids,
            "attention_mask": torch.ones(len(token_ids), dtype=torch.long),
            "prompt_len": torch.tensor(self.prompt_len, dtype=torch.long),
        }

    def _map_to_qwen_ids(self, digit_tokens):
        mapping = {i: 15 + i for i in range(10)}
        mapping[10] = 10   # +
        mapping[11] = 28   # =
        mapping[12] = 12   # -
        return torch.tensor([mapping[t.item()] for t in digit_tokens], dtype=torch.long)


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "prompt_len": torch.stack([b["prompt_len"] for b in batch]),
    }


# ── Accuracy callback ──────────────────────────────────────────────

def compute_accuracy(model, tokenizer, dataset, device, num_samples, **kwargs):
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
    return {"accuracy": n_correct / max(n_total, 1)}


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
    model = SorlModelWrapper.from_scratch(
        config,
        full_vocab_size_list=[tokenizer.vocab_size, abs_vocab],
        pad_token_id=tokenizer.pad_token_id,
    )
    return model


# ── Wandb-aware trainer subclasses ─────────────────────────────────

class WandbSoRLTrainer(SoRLTrainer):
    """SoRLTrainer with wandb logging."""
    def _log(self, msg):
        super()._log(msg)
        if self.is_master and self.history["step"]:
            wandb.log({
                "loss": self.history["loss"][-1],
                "base_loss": self.history["base_loss"][-1],
                "lr": self.history["lr"][-1],
                "step": self.history["step"][-1],
            }, step=self.history["step"][-1])

    def evaluate(self, eval_K=None):
        result = super().evaluate(eval_K=eval_K)
        if result and self.is_master:
            wandb.log({"eval/accuracy": result["accuracy"]})
        return result


class WandbSoRLTrainerv6(SoRLTrainerv6):
    """SoRLTrainerv6 with wandb logging."""
    def _log(self, msg):
        super()._log(msg)
        if self.is_master and self.history["step"]:
            log_dict = {
                "loss": self.history["loss"][-1],
                "base_loss": self.history["base_loss"][-1],
                "traj_loss": self.history["traj_loss"][-1],
                "hinge_loss": self.history["hinge_loss"][-1],
                "lr": self.history["lr"][-1],
                "step": self.history["step"][-1],
            }
            wandb.log(log_dict, step=self.history["step"][-1])

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
    p.add_argument("--trainer", choices=["v1", "v3", "v6"], default="v6")
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

    # Wandb
    run_name = f"{args.ops}_{args.mode}"
    if args.abs_vocab > 0:
        run_name += f"_abs{args.abs_vocab}"
    run_name += f"_{args.dataset_size // 1000}K"

    if not args.no_wandb:
        wandb.init(
            project=WANDB_PROJECT, entity=WANDB_ENTITY,
            name=run_name,
            config=vars(args),
            tags=[args.ops, args.mode, f"abs{args.abs_vocab}", f"{args.dataset_size // 1000}K"],
        )

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model = make_model(args, tokenizer)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"{'═' * 60}")
    print(f"  {run_name}")
    print(f"  arch: {args.n_layer}L/{args.n_head}H/{args.n_embd}d | params: {n_params:,}")
    print(f"  train: {args.num_epochs} epochs x {args.dataset_size} samples, batch={args.batch_size}")
    print(f"{'═' * 60}")

    if not args.no_wandb:
        wandb.config.update({"n_params": n_params})

    train_ds = ArithmeticDataset(tokenizer, args.n_digits, args.ops, args.dataset_size)
    val_ds = ArithmeticDataset(tokenizer, args.n_digits, args.ops, 1000)

    cfg = SoRLConfig(
        K=args.K,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        output_dir=args.output_dir,
        log_every=50,
        eval_every=500,
        save_every=999999,  # don't save intermediate checkpoints (disk space)
        eval_samples=100,
        alpha_info_gain=0.0 if args.mode == "baseline" else 10.0,
        alpha_abs=0.0 if args.mode == "baseline" else 0.1,
        alpha_soft_zipf=0.0 if args.mode == "baseline" else 1.0,
        alpha_traj=1.0,
    )

    TrainerCls = WandbSoRLTrainer if (args.mode == "baseline" or args.trainer == "v1") else WandbSoRLTrainerv6
    if args.trainer == "v3":
        from sorl.trainer_ablate import SoRLTrainerv3
        TrainerCls = SoRLTrainerv3  # no wandb wrapper for v3 yet

    trainer = TrainerCls(
        model, tokenizer, train_ds, val_ds,
        compute_accuracy=compute_accuracy,
        collate_fn=collate_fn,
        config=cfg,
        device=args.device,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    trainer.train()

    # Final eval
    final_acc = compute_accuracy(model, tokenizer, val_ds, args.device, 200)
    print(f"Final accuracy: {final_acc['accuracy']:.3f}")
    if not args.no_wandb:
        wandb.log({"eval/final_accuracy": final_acc["accuracy"]})
        wandb.finish()

    # Upload to HF Hub + clean up local
    if args.push_to_hub:
        from arithmetic.hub import save_model
        metrics = {"history": trainer.history, "final_accuracy": final_acc["accuracy"]}
        save_model(model, vars(args), metrics, subfolder=run_name)
        # Delete local checkpoint to save disk
        import shutil
        shutil.rmtree(args.output_dir, ignore_errors=True)
        print(f"Cleaned up local: {args.output_dir}")


if __name__ == "__main__":
    main()
