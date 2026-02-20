#!/bin/bash
# SoRL Post-Training Script (HuggingFace model + GSM8K/MathQA)
#
# Usage:
#   bash run_sorl_pt.sh

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
# Configuration
# ============================================================================
N_GPUS=4
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500

# Model
MODEL_NAME="Qwen/Qwen2.5-0.5B"
ABSTRACT_VOCAB_SIZE=128

# Data
DATASET="gsm8k"
MAX_LENGTH=512

# SoRL search
NUM_ROLLOUTS=4
K=4
MAX_ITERATIONS=2
MEMORY_SPAN_ABS=1792
MEMORY_SPAN_TRJ=1792
TEMPERATURE=1.0

# Loss function
DECAY=0.8
TARGET_VOCAB_UTIL=0.8

# Optimizer
LR=1e-5
WARMUP_STEPS=50

# Training
BATCH_SIZE=2
GRAD_ACCUM=$((8 / (BATCH_SIZE * N_GPUS)))
NUM_EPOCHS=3

# Logging / Eval / Checkpoint
LOG_EVERY=10
EVAL_EVERY=500
SAVE_EVERY=500
EVAL_SAMPLES=50
LOG_SAMPLES_EVERY=100
NUM_LOG_SAMPLES=3
MAX_NEW_TOKENS=256
BASE_OUTPUT_DIR="./ckpt/sorl_pt"

# ============================================================================
# Sweep: loss weight configurations
# Format: ALPHA_INFO_GAIN ALPHA_ABS ALPHA_SOFT_ZIPF
# ============================================================================
CONFIGS=(
  "10.0 0.1 1.0"    # baseline
  "10.0 0.1 5.0"    # 5x zipf (anti-stutter)
  "10.0 0.1 10.0"   # 10x zipf (strong anti-stutter)
  "10.0 0.5 5.0"    # higher abs + 5x zipf
  "5.0  0.1 10.0"   # lower info + 10x zipf
  "10.0 0.5 10.0"   # higher abs + 10x zipf
)

for CONFIG in "${CONFIGS[@]}"; do
  read -r ALPHA_INFO_GAIN ALPHA_ABS ALPHA_SOFT_ZIPF <<< "$CONFIG"
  RUN_NAME="info${ALPHA_INFO_GAIN}_abs${ALPHA_ABS}_zipf${ALPHA_SOFT_ZIPF}"
  OUTPUT_DIR="${BASE_OUTPUT_DIR}/${RUN_NAME}"

  echo "============================================================"
  echo "Running: $RUN_NAME"
  echo "  alpha_info_gain=$ALPHA_INFO_GAIN  alpha_abs=$ALPHA_ABS  alpha_soft_zipf=$ALPHA_SOFT_ZIPF"
  echo "  output_dir=$OUTPUT_DIR"
  echo "============================================================"

  torchrun \
    --nproc_per_node=$N_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_sorl_pt.py \
    --model_name $MODEL_NAME \
    --abstract_vocab_size $ABSTRACT_VOCAB_SIZE \
    --dataset $DATASET \
    --max_length $MAX_LENGTH \
    --num_rollouts $NUM_ROLLOUTS \
    --K $K \
    --max_iterations $MAX_ITERATIONS \
    --memory_span_abs $MEMORY_SPAN_ABS \
    --memory_span_traj $MEMORY_SPAN_TRJ \
    --temperature $TEMPERATURE \
    --alpha_info_gain $ALPHA_INFO_GAIN \
    --alpha_abs $ALPHA_ABS \
    --alpha_soft_zipf $ALPHA_SOFT_ZIPF \
    --decay $DECAY \
    --target_vocab_util $TARGET_VOCAB_UTIL \
    --lr $LR \
    --warmup_steps $WARMUP_STEPS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --num_epochs $NUM_EPOCHS \
    --log_every $LOG_EVERY \
    --eval_every $EVAL_EVERY \
    --save_every $SAVE_EVERY \
    --eval_samples $EVAL_SAMPLES \
    --log_samples_every $LOG_SAMPLES_EVERY \
    --num_log_samples $NUM_LOG_SAMPLES \
    --max_new_tokens $MAX_NEW_TOKENS \
    --output_dir $OUTPUT_DIR

done
