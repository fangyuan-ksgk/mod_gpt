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

# ============================================================================
# Shared configuration
# ============================================================================
MASTER_ADDR=127.0.0.1
BASE_PORT=29501

MODEL_NAME="Qwen/Qwen3-1.7B"
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

# ---- run_experiment GPUS TAG [extra flags...] ----
# GPUS: CUDA_VISIBLE_DEVICES string, e.g. "0", "0,1"
# n_gpus is inferred from the comma count
run_experiment() {
  local gpus=$1; shift
  local tag=$1; shift

  # Count GPUs from the comma-separated string
  local n_gpus=$(echo "$gpus" | awk -F',' '{print NF}')

  EXP_IDX=$((EXP_IDX + 1))
  local port=$((BASE_PORT + EXP_IDX))
  local grad_accum=$((8 / (BATCH_SIZE * n_gpus)))
  local output_dir="./ckpt/ablate_${TIMESTAMP}/exp${EXP_IDX}_${tag}"

  echo ""
  echo "============================================================"
  echo "Exp ${EXP_IDX}: ${tag}  [CUDA_VISIBLE_DEVICES=${gpus}]"
  echo "  GPUs=${n_gpus}  BS=${BATCH_SIZE}x${grad_accum}x${n_gpus}=8  extra: $@"
  echo "  Output: ${output_dir}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES=$gpus torchrun \
    --nproc_per_node=$n_gpus \
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
    --max_new_tokens $MAX_NEW_TOKENS \
    --output_dir $output_dir \
    "$@"

  echo "  -> Done: ${tag}"
}

# ---- Parallel scheduling across 4 GPUs ----
# EXP_IDX must be incremented in the PARENT shell (not inside backgrounded
# subshells) so that ports and output dirs stay unique.
GPU_SLOTS=(0 1 2 3)
SLOT=0

run_bg() {
  # Assign next GPU slot, pre-increment EXP_IDX in parent, launch in background
  local gpu=${GPU_SLOTS[$SLOT]}
  EXP_IDX=$((EXP_IDX + 1))
  local idx=$EXP_IDX
  local port=$((BASE_PORT + idx))
  local tag=$1; shift
  local n_gpus=1
  local grad_accum=$((8 / (BATCH_SIZE * n_gpus)))
  local output_dir="./ckpt/ablate_${TIMESTAMP}/exp${idx}_${tag}"

  echo ""
  echo "============================================================"
  echo "Exp ${idx}: ${tag}  [GPU=${gpu}]  port=${port}"
  echo "  Output: ${output_dir}"
  echo "============================================================"

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

  SLOT=$(( (SLOT + 1) % ${#GPU_SLOTS[@]} ))
}

wait_batch() {
  echo "  ... waiting for current batch to finish ..."
  wait
  SLOT=0
}

# ============================================================================
# Batch 1: baseline + first 3 info sweeps
# ============================================================================
run_bg "1gpu_bs8_aux0"
run_bg "1gpu_bs8_info1.0" --alpha_info_gain 1.0
run_bg "1gpu_bs8_info3.0" --alpha_info_gain 3.0
run_bg "1gpu_bs8_info5.0" --alpha_info_gain 5.0
wait_batch

# ============================================================================
# Batch 2: remaining info sweeps + DDP validation (uses 2 GPUs)
# ============================================================================
run_bg "1gpu_bs8_info7.0" --alpha_info_gain 7.0
run_bg "1gpu_bs8_info9.0" --alpha_info_gain 9.0
# DDP validation: 2 GPUs — runs on GPUs 2,3 (next 2 slots)
EXP_IDX=$((EXP_IDX + 1))
DDP_IDX=$EXP_IDX
DDP_PORT=$((BASE_PORT + DDP_IDX))
DDP_OUT="./ckpt/ablate_${TIMESTAMP}/exp${DDP_IDX}_2gpu_bs8_aux0"
echo "Exp ${DDP_IDX}: 2gpu_bs8_aux0  [GPU=2,3]  port=${DDP_PORT}"
CUDA_VISIBLE_DEVICES=2,3 torchrun \
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
  --output_dir $DDP_OUT &
wait_batch

# ============================================================================
# Batch 3: info + abs sweep
# ============================================================================
run_bg "1gpu_bs8_info1.0_abs0.5" --alpha_info_gain 1.0 --alpha_abs 0.5
run_bg "1gpu_bs8_info3.0_abs0.5" --alpha_info_gain 3.0 --alpha_abs 0.5
run_bg "1gpu_bs8_info5.0_abs0.5" --alpha_info_gain 5.0 --alpha_abs 0.5
run_bg "1gpu_bs8_info7.0_abs0.5" --alpha_info_gain 7.0 --alpha_abs 0.5
wait_batch

# ============================================================================
# Batch 4: last info+abs + abs sweep (info=9 fixed)
# ============================================================================
run_bg "1gpu_bs8_info9.0_abs0.5" --alpha_info_gain 9.0 --alpha_abs 0.5
run_bg "1gpu_bs8_info9_abs1.0"   --alpha_info_gain 9.0 --alpha_abs 1.0
run_bg "1gpu_bs8_info9_abs1.5"   --alpha_info_gain 9.0 --alpha_abs 1.5
run_bg "1gpu_bs8_info9_abs2.0"   --alpha_info_gain 9.0 --alpha_abs 2.0
wait_batch

# ============================================================================
# Batch 5: zipf sweep (info=9, abs=0.5 fixed)
# ============================================================================
run_bg "1gpu_bs8_info9_abs0.5_zipf0.5"  --alpha_info_gain 9.0 --alpha_abs 0.5 --alpha_soft_zipf 0.5
run_bg "1gpu_bs8_info9_abs0.5_zipf1.0"  --alpha_info_gain 9.0 --alpha_abs 0.5 --alpha_soft_zipf 1.0
run_bg "1gpu_bs8_info9_abs0.5_zipf1.5"  --alpha_info_gain 9.0 --alpha_abs 0.5 --alpha_soft_zipf 1.5
run_bg "1gpu_bs8_info9_abs0.5_zipf2.0"  --alpha_info_gain 9.0 --alpha_abs 0.5 --alpha_soft_zipf 2.0
wait_batch

echo ""
echo "============================================================"
echo "All experiments complete. Results in ./ckpt/ablate_${TIMESTAMP}/"
echo "============================================================"
