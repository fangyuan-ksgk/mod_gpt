#!/bin/bash
# SoRL Ablation Experiments
#
# Usage:
#   bash run_ablate_sanity.sh
#
# Each run_experiment call takes: GPUS  TAG  [extra args...]
# GPUS is a CUDA_VISIBLE_DEVICES string, e.g. "0", "1", "0,1"

set -e

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
# Shared configuration
# ============================================================================
MASTER_ADDR=127.0.0.1
BASE_PORT=29501

MODEL_NAME="Qwen/Qwen3-0.6B"
DATASET="gsm8k"
MAX_LENGTH=512

LR=1e-5
WARMUP_STEPS=50
BATCH_SIZE=2
NUM_EPOCHS=3

LOG_EVERY=10
EVAL_EVERY=99999
SAVE_EVERY=99999
EVAL_SAMPLES=1000
EVAL_BATCH_SIZE=64
MAX_NEW_TOKENS=256

TIMESTAMP=$(date +%Y%m%d_%H%M)
EXP_IDX=0

# ---- Parallel scheduling: 2 x H200 (120GB each, ~6GB/run = ~20 runs/GPU) ----
# All 19 single-GPU experiments fit in ONE batch (~10 per GPU, ~60GB each).
# GPU assignment alternates: exp1→GPU0, exp2→GPU1, exp3→GPU0, ...

run_bg() {
  EXP_IDX=$((EXP_IDX + 1))
  local idx=$EXP_IDX
  local gpu=$(( (idx - 1) % 2 ))
  local port=$((BASE_PORT + idx))
  local tag=$1; shift
  local grad_accum=$((8 / (BATCH_SIZE * 1)))
  local output_dir="./ckpt/ablate_${TIMESTAMP}/exp${idx}_${tag}"

  echo "  Exp ${idx}: ${tag}  [GPU=${gpu}]  port=${port}"

  CUDA_VISIBLE_DEVICES=$gpu torchrun \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$port \
    train_ablate_sanity.py \
    --model_name $MODEL_NAME \
    --dataset $DATASET \
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

# ============================================================================
# Batch 1: 18 single-GPU experiments (9 per GPU, ~13GB each → ~117GB/GPU)
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 1: 18 experiments across 2 GPUs (${TIMESTAMP})"
echo "  Model: ${MODEL_NAME} | ~13GB/run (train) | 120GB/GPU → 9/GPU"
echo "============================================================"

# 1. Baseline (no aux losses)
run_bg "1gpu_bs8_aux0"

# 2-6. Info gain sweep
run_bg "1gpu_bs8_info1.0" --alpha_info_gain 1.0
run_bg "1gpu_bs8_info3.0" --alpha_info_gain 3.0
run_bg "1gpu_bs8_info5.0" --alpha_info_gain 5.0
run_bg "1gpu_bs8_info7.0" --alpha_info_gain 7.0
run_bg "1gpu_bs8_info9.0" --alpha_info_gain 9.0

# 7-11. Info gain + abs=0.5 sweep
run_bg "1gpu_bs8_info1.0_abs0.5" --alpha_info_gain 1.0 --alpha_abs 0.5
run_bg "1gpu_bs8_info3.0_abs0.5" --alpha_info_gain 3.0 --alpha_abs 0.5
run_bg "1gpu_bs8_info5.0_abs0.5" --alpha_info_gain 5.0 --alpha_abs 0.5
run_bg "1gpu_bs8_info7.0_abs0.5" --alpha_info_gain 7.0 --alpha_abs 0.5
run_bg "1gpu_bs8_info9.0_abs0.5" --alpha_info_gain 9.0 --alpha_abs 0.5

# 12-15. Abs sweep (info=9 fixed)
run_bg "1gpu_bs8_info9_abs0.5"  --alpha_info_gain 9.0 --alpha_abs 0.5
run_bg "1gpu_bs8_info9_abs1.0"  --alpha_info_gain 9.0 --alpha_abs 1.0
run_bg "1gpu_bs8_info9_abs1.5"  --alpha_info_gain 9.0 --alpha_abs 1.5
run_bg "1gpu_bs8_info9_abs2.0"  --alpha_info_gain 9.0 --alpha_abs 2.0

# 16-18. Zipf sweep (first 3)
run_bg "1gpu_bs8_info9_abs0.5_zipf0.5" --alpha_info_gain 9.0 --alpha_abs 0.5 --alpha_soft_zipf 0.5
run_bg "1gpu_bs8_info9_abs0.5_zipf1.0" --alpha_info_gain 9.0 --alpha_abs 0.5 --alpha_soft_zipf 1.0
run_bg "1gpu_bs8_info9_abs0.5_zipf1.5" --alpha_info_gain 9.0 --alpha_abs 0.5 --alpha_soft_zipf 1.5

echo "  18 experiments launched. Waiting..."
wait

# ============================================================================
# Batch 2: last zipf + DDP validation
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 2: remaining zipf experiment + DDP validation"
echo "============================================================"

# 19. Last zipf experiment (GPU 0)
run_bg "1gpu_bs8_info9_abs0.5_zipf2.0" --alpha_info_gain 9.0 --alpha_abs 0.5 --alpha_soft_zipf 2.0

# 20. DDP validation (GPU 0,1)
EXP_IDX=$((EXP_IDX + 1))
DDP_IDX=$EXP_IDX
DDP_PORT=$((BASE_PORT + DDP_IDX))
DDP_OUT="./ckpt/ablate_${TIMESTAMP}/exp${DDP_IDX}_2gpu_bs8_aux0"
echo ""
echo "============================================================"
echo "Exp ${DDP_IDX}: DDP validation  [GPU=0,1]  port=${DDP_PORT}"
echo "============================================================"
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nproc_per_node=2 \
  --master_addr=$MASTER_ADDR \
  --master_port=$DDP_PORT \
  train_ablate_sanity.py \
  --model_name $MODEL_NAME \
  --dataset $DATASET \
  --max_length $MAX_LENGTH \
  --lr $LR \
  --warmup_steps $WARMUP_STEPS \
  --batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $((8 / (BATCH_SIZE * 2))) \
  --num_epochs $NUM_EPOCHS \
  --log_every $LOG_EVERY \
  --eval_every $EVAL_EVERY \
  --save_every $SAVE_EVERY \
  --eval_samples $EVAL_SAMPLES \
  --eval_batch_size $EVAL_BATCH_SIZE \
  --max_new_tokens $MAX_NEW_TOKENS \
  --output_dir $DDP_OUT

echo ""
echo "============================================================"
echo "All experiments complete. Results in ./ckpt/ablate_${TIMESTAMP}/"
echo "============================================================"
