#!/bin/bash
set -e

# v8 self-distill sweep — ablate loss weight combos (alpha_traj × alpha_kd)
# 4 runs × 2 GPUs → 2 runs/GPU in parallel
#
# Acc[None]: free NL generation (no abs forced)   — baseline capability
# Acc[K]:    abs tokens FORCED at K prefix slots  — tests if abstractions carry info
# Goal: Acc[K] ≥ Acc[None] (positive gap = true dependency established)

# --- nvidia pod specifics ------
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
# Shared config
# ============================================================================
MASTER_ADDR=127.0.0.1
BASE_PORT=29600
N_GPUS=2

LR=1e-5
EMB_LR_MULT=10.0
WARMUP_STEPS=50
BATCH_SIZE=2
NUM_EPOCHS=3

LOG_EVERY=10
EVAL_EVERY=99999
SAVE_EVERY=99999
EVAL_SAMPLES=1300
EVAL_BATCH_SIZE=64
MAX_NEW_TOKENS=256
MAX_LENGTH=512

TIMESTAMP=$(date +%Y%m%d_%H%M)
EXP_IDX=0

M06="Qwen/Qwen3-0.6B"
DS_GSM="gsm8k"

# Common v8 flags: prefix-ABS layout, K=8, iter=4, GSM8K answer delimiter
V8_BASE="--use_v8 --prefix_abs --abs_prefix_max 8 --K 8 --max_iterations 4 \
  --abstract_vocab_size 32 --answer_token_id 820"

run_bg() {
  EXP_IDX=$((EXP_IDX + 1))
  local idx=$EXP_IDX
  local gpu=$(( (idx - 1) % N_GPUS ))
  local port=$((BASE_PORT + idx))
  local tag=$1; shift
  local model=$1; shift
  local dataset=$1; shift
  local grad_accum=$((8 / BATCH_SIZE))
  local output_dir="./ckpt/sweep_v8_${TIMESTAMP}/exp${idx}_${tag}"

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
    --emb_lr_mult $EMB_LR_MULT \
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
# Batch 1 — v8 loss weight ablation on Qwen3-0.6B × GSM8K
#
# Axes: alpha_traj (L_compress weight) × alpha_kd (KD weight)
#   (a) traj=1.0 kd=0.0 — no KD, pure compress loss + base (compression only)
#   (b) traj=1.0 kd=1.0 — balanced combo (default)
#   (c) traj=1.0 kd=0.1 — light KD nudge
#   (d) traj=0.0 kd=1.0 — KD only, no compress loss (KD alone sufficient?)
# ===========================================================================
echo ""
echo "Batch 1: v8 loss weight ablation — Qwen3-0.6B × GSM8K"

# (a) traj=1.0 kd=0.0 — compress only (no KD), upper bound w/o distillation
run_bg "v8_t1_k0" $M06 $DS_GSM \
  $V8_BASE \
  --alpha_traj 1.0 --alpha_kd 0.0

# (b) traj=1.0 kd=1.0 — balanced default
run_bg "v8_t1_k1" $M06 $DS_GSM \
  $V8_BASE \
  --alpha_traj 1.0 --alpha_kd 1.0

# (c) traj=1.0 kd=0.1 — light KD
run_bg "v8_t1_k01" $M06 $DS_GSM \
  $V8_BASE \
  --alpha_traj 1.0 --alpha_kd 0.1

# (d) traj=0.0 kd=1.0 — KD only (no compress signal)
run_bg "v8_t0_k1" $M06 $DS_GSM \
  $V8_BASE \
  --alpha_traj 0.0 --alpha_kd 1.0

echo ""
echo "All 4 runs launched. Waiting..."
wait
echo ""
echo "=== Batch 1 complete. Checkpoints: ./ckpt/sweep_v8_${TIMESTAMP}/ ==="
