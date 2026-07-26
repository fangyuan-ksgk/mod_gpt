#!/usr/bin/env bash
# Reproduction script for the yrxa Q5 interpretability replication.
#
# Two studies, two measurements each:
#   R1  subtask-pure codes      P(label | code), scored by lift over the marginal
#   R2  surgical-swap repairs   label-matched code vs matched random control
#
# Both use DLR v9 (residual-stream steering) on a real pretrained Qwen3-0.6B —
# the same mechanism as the main results table, not the from-scratch token variant.
set -euo pipefail

PY=/lambda/nfs/Amir-steering/codes/dlr/bin/python3
MODEL=Qwen/Qwen3-0.6B
LOGS=amir_interp_rebuttal/logs
mkdir -p "$LOGS" amir_interp_rebuttal/results

# Injection layer is FIXED a priori at the network midpoint (14 of 28). It is not
# swept. This keeps the claim clean and avoids the layer-selection question
# entirely rather than inviting it.
LAYER=14

# ── Study 1: six-digit addition + subtraction ────────────────────────
# L=1 -> one steering code per answer digit, so digit d_i maps to decode step i.
# C=30 matches the codebook size used in the original study.
arith_train() {
  $PY -u train_steer_pt.py \
    --mode v9 --dataset arithmetic --model_name "$MODEL" \
    --L 1 --C_SIZE 30 --scale 0.1 --inject_layers "$LAYER" \
    --max_length 64 --batch_size 32 --gradient_accumulation_steps 1 \
    --num_epochs 1 --lr 1e-5 --steer_lr 1e-3 \
    --num_rollouts 4 --search_temp 1.0 \
    --eval_samples 64 --eval_batch_size 32 --eval_every 30 --max_new_tokens 8 \
    --log_every 5 --output_dir ckpt/arith_v9 > "$LOGS/arith_train.log" 2>&1
}

arith_analyze() {
  $PY -u -W ignore -m amir_interp_rebuttal.analyze \
    --study arithmetic --ckpt ckpt/arith_v9 --model_name "$MODEL" \
    --eval_n 2600 --max_new_tokens 8 --max_swap_examples 150 \
    > "$LOGS/arith_analyze.log" 2>&1
}

# ── Study 2: CodeNet (Python) ────────────────────────────────────────
# L=8 -> a chunk is roughly a line, matching the granularity of the AST labels.
# Requires data_cache/Project_CodeNet_Python800 (see codenet_dataset.py header).
codenet_train() {
  $PY -u train_steer_pt.py \
    --mode v9 --dataset codenet --model_name "$MODEL" \
    --L 8 --C_SIZE 30 --scale 0.1 --inject_layers "$LAYER" \
    --max_length 256 --batch_size 8 --gradient_accumulation_steps 4 \
    --num_epochs 1 --lr 1e-5 --steer_lr 1e-3 \
    --num_rollouts 4 --search_temp 1.0 \
    --eval_samples 200 --eval_batch_size 16 --eval_every 100 --max_new_tokens 32 \
    --log_every 10 --output_dir ckpt/codenet_v9 > "$LOGS/codenet_train.log" 2>&1
}

codenet_analyze() {
  $PY -u -W ignore -m amir_interp_rebuttal.analyze \
    --study codenet --ckpt ckpt/codenet_v9 --model_name "$MODEL" \
    --eval_n 800 --max_new_tokens 32 --max_swap_examples 150 \
    > "$LOGS/codenet_analyze.log" 2>&1
}

case "${1:-all}" in
  arith)          arith_train; arith_analyze ;;
  arith-analyze)  arith_analyze ;;
  codenet)        codenet_train; codenet_analyze ;;
  codenet-analyze) codenet_analyze ;;
  all)            arith_train; arith_analyze; codenet_train; codenet_analyze ;;
  *) echo "usage: $0 {arith|arith-analyze|codenet|codenet-analyze|all}"; exit 1 ;;
esac
