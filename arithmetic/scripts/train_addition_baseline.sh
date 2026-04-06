#!/bin/bash
# ──────────────────────────────────────────────────────────────
# Train addition baseline (paper reproduction)
# Mirrors: Nanda et al., "Progress measures for grokking via
#          mechanistic interpretability" (ICLR 2024)
#
# Paper: 1 layer, 3 heads, d=510, 5000 steps, batch=64, lr=8e-5
# Ours:  2 layers (GAT U-net minimum), rest matches paper
# ──────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/../.."

SAVE_DIR=${1:-ckpt/addition_baseline_$(date +%Y%m%d_%H%M)}

python -m arithmetic.reference.addition_6digit \
    --n_digits 6 \
    --n_layer 2 \
    --n_head 3 \
    --n_embd 510 \
    --batch_size 64 \
    --num_steps 5000 \
    --lr 8e-5 \
    --weight_decay 0.1 \
    --log_every 50 \
    --eval_every 500 \
    --save_dir "$SAVE_DIR" \
    --device cuda

echo "Done. Model saved to $SAVE_DIR"
