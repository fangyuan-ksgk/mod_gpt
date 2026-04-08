#!/bin/bash
set -e

# Ablation on v7 "deep supervision"

# --- nvidia pod  specifics ------
DUMMY_CONFIG_PATH="/workspace/mod_gpt/dummy_tuner_config.txt"
rm -f "$DUMMY_CONFIG_PATH"
touch "$DUMMY_CONFIG_PATH"

export NCCL_TUNER_CONFIG_PATH="$DUMMY_CONFIG_PATH"
export NCCL_TUNER_PLUGIN=""
export NCCL_NET_PLUGIN=""
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

# ============================================================================
# Shared configuration
# ============================================================================
MASTER_ADDR=127.0.0.1
BASE_PORT=29501
N_GPUS=2

MODEL_NAME="Qwen/Qwen3-0.6B"
DATASET="gsm8k"
MAX_LENGTH=512

LR=1e-5
WARMUP_STEPS=50
BATCH_SIZE=2
NUM_EPOCHS=1

LOG_EVERY=10
EVAL_EVERY=99999
SAVE_EVERY=99999
EVAL_SAMPLES=1300
EVAL_BATCH_SIZE=64
MAX_NEW_TOKENS=256

# ===========================================================================
# Deep Supervision Sweep: v6 (standard) vs v7 (deep supervision)
#
# Core question: does per-iteration backward+step (v7) beat
#                N Jacobi iterations + final loss (v6)?
#
# Design: paired v6/v7 on same config, 2 GPUs per batch.
#   iter=4 for most runs (fair compute; v7 doesn't need 8 to shine).
#   Axes: dataset, model scale, prefix length, emb_lr, iteration depth.
# ===========================================================================

TIMESTAMP=$(date +%Y%m%d_%H%M)
EXP_IDX=0

# Model + dataset shorthands
M06="Qwen/Qwen3-0.6B"
M17="Qwen/Qwen3-1.7B"
DS_GSM="gsm8k"
DS_SCI="scienceqa"

# Common SoRL flags
ABS="--abstract_vocab_size 32 --prefix_abs --alpha_traj 1.0"

run_bg() {
  EXP_IDX=$((EXP_IDX + 1))
  local idx=$EXP_IDX
  local gpu=$(( (idx - 1) % N_GPUS ))
  local port=$((BASE_PORT + idx))
  local tag=$1; shift
  local model=$1; shift
  local dataset=$1; shift
  local grad_accum=$((8 / BATCH_SIZE))
  local output_dir="./ckpt/sweep_${TIMESTAMP}/exp${idx}_${tag}"

  echo "  Exp ${idx}: ${tag}  model=$(basename $model)  dataset=${dataset}  [GPU=${gpu}]"

  CUDA_VISIBLE_DEVICES=$gpu torchrun \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$port \
    train_sorl_post.py \
    --model_name $model \
    --dataset $dataset \
    --max_length $MAX_LENGTH \
    --lr $LR \
    --warmup_steps $WARMUP_STEPS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $grad_accum \
    --num_epochs $NUM_EPOCHS \
    --log_every $LOG_EVERY \
    --eval_every $EVAL_EVERY \
    --save_every $SAVE_EVERY \
    --eval_samples $EVAL_SAMPLES \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --max_new_tokens $MAX_NEW_TOKENS \
    --output_dir $output_dir \
    "$@" &
}

# ===========================================================================
# Batch 1 — Baseline pair: 0.6B, GSM8K, pfx8, iter=4
# ===========================================================================
echo ""
echo "Batch 1: v6 vs v7 — 0.6B, GSM8K, pfx8, iter=4"

run_bg "v6_06b_gsm_pfx8" $M06 $DS_GSM \
  --use_v6 --K 8 $ABS --abs_prefix_max 8 \
  --max_iterations 4

run_bg "v7_06b_gsm_pfx8" $M06 $DS_GSM \
  --use_v7 --K 8 $ABS --abs_prefix_max 8 \
  --max_iterations 4

wait

# ===========================================================================
# Batch 2 — Dataset transfer: 0.6B, ScienceQA, pfx8, iter=4
# ===========================================================================
echo ""
echo "Batch 2: v6 vs v7 — 0.6B, ScienceQA, pfx8, iter=4"

run_bg "v6_06b_sci_pfx8" $M06 $DS_SCI \
  --use_v6 --K 8 $ABS --abs_prefix_max 8 \
  --max_iterations 4

run_bg "v7_06b_sci_pfx8" $M06 $DS_SCI \
  --use_v7 --K 8 $ABS --abs_prefix_max 8 \
  --max_iterations 4

wait

# ===========================================================================
# Batch 3 — Scale up: 1.7B, GSM8K, pfx8, iter=4
# ===========================================================================
echo ""
echo "Batch 3: v6 vs v7 — 1.7B, GSM8K, pfx8, iter=4"

run_bg "v6_17b_gsm_pfx8" $M17 $DS_GSM \
  --use_v6 --K 8 $ABS --abs_prefix_max 8 \
  --max_iterations 4

run_bg "v7_17b_gsm_pfx8" $M17 $DS_GSM \
  --use_v7 --K 8 $ABS --abs_prefix_max 8 \
  --max_iterations 4

wait

# ===========================================================================
# Batch 4 — Scale + dataset: 1.7B, ScienceQA, pfx8, iter=4
# ===========================================================================
echo ""
echo "Batch 4: v6 vs v7 — 1.7B, ScienceQA, pfx8, iter=4"

run_bg "v6_17b_sci_pfx8" $M17 $DS_SCI \
  --use_v6 --K 8 $ABS --abs_prefix_max 8 \
  --max_iterations 4

run_bg "v7_17b_sci_pfx8" $M17 $DS_SCI \
  --use_v7 --K 8 $ABS --abs_prefix_max 8 \
  --max_iterations 4

wait

# ===========================================================================
# Batch 5 — v7 iteration depth: iter=2 vs iter=8 (0.6B, GSM8K, pfx8)
#   How many deep-supervision steps does v7 need?
# ===========================================================================
echo ""
echo "Batch 5: v7 iteration depth — iter=2 vs iter=8"

run_bg "v7_06b_gsm_i2" $M06 $DS_GSM \
  --use_v7 --K 8 $ABS --abs_prefix_max 8 \
  --max_iterations 2

run_bg "v7_06b_gsm_i8" $M06 $DS_GSM \
  --use_v7 --K 8 $ABS --abs_prefix_max 8 \
  --max_iterations 8

wait

echo ""
echo "All done. 8 batches × 2 = 16 experiments."