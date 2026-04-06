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


# ── Dataset adapter ─────────────────────────────────────────────────

class ArithmeticDataset(Dataset):
    """
    On-the-fly arithmetic dataset formatted for SoRL trainers.
    Returns {"input_ids": (seq_len,), "attention_mask": (seq_len,), "prompt_len": int}
    """

    def __init__(self, tokenizer, n_digits=6, ops="add", size=100_000):
        self.tokenizer = tokenizer
        self.n_digits = n_digits
        self.ops = ops
        self.size = size
        self.prompt_len = 2 * n_digits + 2  # XXXXXX+YYYYYY=
        self.seq_len = 3 * n_digits + 3     # + ZZZZZZZ

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
        """Map our 0-12 token IDs to Qwen3 tokenizer IDs."""
        # Our tokens: 0-9=digits, 10=+, 11==, 12=-
        # Qwen3:      15-24=digits, 10=+, 28==, 12=-
        mapping = {i: 15 + i for i in range(10)}  # digits
        mapping[10] = 10   # +
        mapping[11] = 28   # =
        mapping[12] = 12   # -
        return torch.tensor([mapping[t.item()] for t in digit_tokens], dtype=torch.long)


def collate_fn(batch):
    """Collate for uniform-length arithmetic sequences."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "prompt_len": torch.stack([b["prompt_len"] for b in batch]),
    }


# ── Accuracy callback ──────────────────────────────────────────────

def compute_accuracy(model, tokenizer, dataset, device, num_samples, **kwargs):
    """Accuracy callback for the trainer's eval loop."""
    model.eval()
    n_correct, n_total = 0, 0
    prompt_len = dataset.prompt_len
    seq_len = dataset.seq_len

    for _ in range(num_samples):
        item = dataset[0]  # random each time (online gen)
        ids = item["input_ids"].unsqueeze(0).to(device)
        attn = item["attention_mask"].unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(ids, attention_mask=attn,
                        memory_span_abs=1792, memory_span_traj=1792)
            logits = out.logits

        # Check answer digits
        pred = logits[0, prompt_len - 1:-1].argmax(dim=-1)
        target = ids[0, prompt_len:]
        if (pred == target).all():
            n_correct += 1
        n_total += 1

    model.train()
    return {"accuracy": n_correct / max(n_total, 1)}


# ── Model factory ──────────────────────────────────────────────────

def make_model(args, tokenizer):
    """Create tiny Qwen3 wrapped in SorlModelWrapper."""
    config = Qwen3Config(
        hidden_size=args.n_embd,
        num_hidden_layers=args.n_layer,
        num_attention_heads=args.n_head,
        num_key_value_heads=args.n_head,
        intermediate_size=args.n_embd * 4,
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=128,
    )

    # Always include abstract vocab (even baseline) to avoid edge case in
    # _setup_vocabulary. For baseline, abs tokens exist but are never used.
    abs_vocab = args.abs_vocab if args.abs_vocab > 0 else 1
    model = SorlModelWrapper.from_scratch(
        config,
        full_vocab_size_list=[tokenizer.vocab_size, abs_vocab],
        pad_token_id=tokenizer.pad_token_id,
    )
    return model


# ── Main ────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["baseline", "sorl"], default="baseline")
    p.add_argument("--ops", choices=["add", "add_sub"], default="add")
    p.add_argument("--n_digits", type=int, default=6)
    # Architecture
    p.add_argument("--n_layer", type=int, default=3)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=512)
    # SoRL
    p.add_argument("--abs_vocab", type=int, default=0, help="0=baseline, >0=SoRL")
    p.add_argument("--K", type=int, default=4, help="insert abstract token every K tokens")
    p.add_argument("--trainer", choices=["v1", "v3", "v6"], default="v6")
    # Training
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--dataset_size", type=int, default=100_000)
    p.add_argument("--lr", type=float, default=8e-5)
    # System
    p.add_argument("--output_dir", type=str, default="ckpt/arithmetic")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    # Enforce: baseline means abs_vocab=0
    if args.mode == "baseline":
        args.abs_vocab = 0

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model = make_model(args, tokenizer)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"{'═' * 60}")
    print(f"  mode={args.mode} | ops={args.ops} | abs_vocab={args.abs_vocab}")
    print(f"  arch: {args.n_layer}L/{args.n_head}H/{args.n_embd}d | params: {n_params:,}")
    print(f"  train: {args.num_epochs} epochs × {args.dataset_size} samples, batch={args.batch_size}")
    print(f"{'═' * 60}")

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
        save_every=1000,
        eval_samples=100,
        # For baseline: all alpha=0 means loss = base_traj_loss
        alpha_info_gain=0.0 if args.mode == "baseline" else 10.0,
        alpha_abs=0.0 if args.mode == "baseline" else 0.1,
        alpha_soft_zipf=0.0 if args.mode == "baseline" else 1.0,
        alpha_traj=1.0,
    )

    if args.mode == "baseline" or args.trainer == "v1":
        trainer = SoRLTrainer(
            model, tokenizer, train_ds, val_ds,
            compute_accuracy=compute_accuracy,
            collate_fn=collate_fn,
            config=cfg,
            device=args.device,
        )
    elif args.trainer == "v6":
        trainer = SoRLTrainerv6(
            model, tokenizer, train_ds, val_ds,
            compute_accuracy=compute_accuracy,
            collate_fn=collate_fn,
            config=cfg,
            device=args.device,
        )
    elif args.trainer == "v3":
        from sorl.trainer_ablate import SoRLTrainerv3
        trainer = SoRLTrainerv3(
            model, tokenizer, train_ds, val_ds,
            compute_accuracy=compute_accuracy,
            collate_fn=collate_fn,
            config=cfg,
            device=args.device,
        )

    # Save config
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    trainer.train()


if __name__ == "__main__":
    main()
