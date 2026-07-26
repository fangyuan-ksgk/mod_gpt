#!/usr/bin/env bash
# Unattended chain: finish arithmetic (corrected hyperparameters), then run the
# CodeNet study end to end. Each stage waits for the previous one and logs to
# amir_interp_rebuttal/logs/.
#
# Deliberately does NOT retry or tune on failure — if a stage produces a weak or
# negative result, that is the result. The only hyperparameter change made in
# this whole effort is aligning to the published values
# (alpha_info=10.0, alpha_abs=0.1, alpha_zipf=1.0), which was a config-mismatch
# fix, not a search for a better number.
set -uo pipefail

PY=/lambda/nfs/Amir-steering/codes/dlr/bin/python3
MODEL=Qwen/Qwen3-0.6B
LOGS=amir_interp_rebuttal/logs
mkdir -p "$LOGS" amir_interp_rebuttal/results

wait_for_pattern() {   # $1 = pattern, $2 = file, $3 = label
  while pgrep -f "$1" >/dev/null; do sleep 30; done
  echo "[chain] $3 finished at $(date -u +%H:%M:%S)"
}

# ── 1. arithmetic @ published hyperparameters ───────────────────────
wait_for_pattern "output_dir ckpt/arith_v9_paperhp" "" "arith paperhp training"

if [ -f ckpt/arith_v9_paperhp/final.pt ]; then
  echo "[chain] analysing arith_v9_paperhp"
  $PY -u -W ignore -m amir_interp_rebuttal.analyze \
      --study arithmetic --ckpt ckpt/arith_v9_paperhp --model_name "$MODEL" \
      --eval_n 2600 --max_new_tokens 8 --max_swap_examples 150 \
      > "$LOGS/arith_paperhp_analyze.log" 2>&1
  echo "[chain] arith analysis done"
else
  echo "[chain] ERROR: no checkpoint at ckpt/arith_v9_paperhp/final.pt" >&2
fi

# ── 2. CodeNet ──────────────────────────────────────────────────────
# Wait for the tar to finish so the problem set is stable. The train/test split
# is hash-based so a partial extract would still be *correct*, but waiting keeps
# the reported problem counts meaningful.
while pgrep -f "tar xzf Python800" >/dev/null; do sleep 60; done
echo "[chain] codenet extraction complete: $(ls data_cache/Project_CodeNet_Python800 | wc -l) problems"

echo "[chain] training codenet"
$PY -u train_steer_pt.py \
    --mode v9 --dataset codenet --model_name "$MODEL" \
    --L 8 --C_SIZE 30 --scale 0.1 --inject_layers 14 \
    --alpha_info 10.0 --alpha_abs 0.1 --alpha_zipf 1.0 \
    --max_length 256 --batch_size 8 --gradient_accumulation_steps 4 \
    --num_epochs 1 --lr 1e-5 --steer_lr 1e-3 \
    --num_rollouts 4 --search_temp 1.0 \
    --eval_samples 200 --eval_batch_size 16 --eval_every 200 --max_new_tokens 32 \
    --eval_decode_scale 0.1 \
    --log_every 20 --output_dir ckpt/codenet_v9 > "$LOGS/codenet_train.log" 2>&1
echo "[chain] codenet training done"

if [ -f ckpt/codenet_v9/final.pt ]; then
  echo "[chain] analysing codenet_v9"
  $PY -u -W ignore -m amir_interp_rebuttal.analyze \
      --study codenet --ckpt ckpt/codenet_v9 --model_name "$MODEL" \
      --eval_n 800 --max_new_tokens 32 --max_swap_examples 150 \
      > "$LOGS/codenet_analyze.log" 2>&1
  echo "[chain] codenet analysis done"
else
  echo "[chain] ERROR: no checkpoint at ckpt/codenet_v9/final.pt" >&2
fi

echo "[chain] ALL STAGES COMPLETE $(date -u +%H:%M:%S)"
