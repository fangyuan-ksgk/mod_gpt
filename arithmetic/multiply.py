"""
GAT Multiplication — trains baseline vs SoRL on single-digit multiplication.
Converted from notebook/gat_arithmatic.ipynb
"""
import torch
from sorl.gat_sim import GAT, GATConfig
from arithmetic.datasets.multiplication import (
    DigitTokenizer, MultiplicationDataLoader, process_query, check_answer,
)
from arithmetic.trainer import BaselineTrainer, SoRLTrainer

torch.set_float32_matmul_precision('high')

# ── Config ──────────────────────────────────────────────────────────
BOS_TOKEN_ID = 20
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 4
num_steps = 500

gat_config = GATConfig(
    vocab_sizes=[BOS_TOKEN_ID + 1, 16],  # 16 abstract tokens
    n_layer=4,
    n_head=4,
    n_embd=128,
    device=device,
)

# ── Data ────────────────────────────────────────────────────────────
tokenizer = DigitTokenizer(BOS_TOKEN_ID)
train_loader = MultiplicationDataLoader(min_digits=1, max_digits=1, num_examples=1000, pad_digits=2, device=device)
val_loader = MultiplicationDataLoader(min_digits=1, max_digits=1, num_examples=100, pad_digits=2, device=device)

# ── Phase 1: Baseline ──────────────────────────────────────────────
print("=" * 60)
print("Phase 1: Baseline training (no abstraction tokens)")
print("=" * 60)

model = GAT(gat_config)
baseline = BaselineTrainer(model, train_loader, val_loader)
baseline.train(num_steps, batch_size)

# ── Phase 2: SoRL ──────────────────────────────────────────────────
print()
print("=" * 60)
print("Phase 2: SoRL training (with abstraction tokens)")
print("=" * 60)

model = GAT(gat_config)
sorl = SoRLTrainer(model, train_loader, val_loader)
sorl.train(num_steps, batch_size)

# ── Phase 3: Generation test ───────────────────────────────────────
print()
print("=" * 60)
print("Phase 3: Generation test")
print("=" * 60)

sorl.generate_answer(val_loader, tokenizer)
