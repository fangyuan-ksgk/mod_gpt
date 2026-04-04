#!/bin/bash
# SFT Baseline Post-Training Script (comparable to run_sorl_pt.sh)
#
# Usage:
#   bash run_sft_pt.sh

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
N_GPUS=1
MASTER_ADDR=127.0.0.1
MASTER_PORT=29501

# Model
MODEL_NAME="Qwen/Qwen3-0.6B"

# Data | science-qa | mbpp
DATASET="gsm8k"
MAX_LENGTH=512

# Optimizer
LR=1e-5
WARMUP_STEPS=50

# Training (effective batch size of 8)
BATCH_SIZE=2
GRAD_ACCUM=$((8 / (BATCH_SIZE * N_GPUS)))
NUM_EPOCHS=3

# Logging / Eval / Checkpoint
LOG_EVERY=10
EVAL_EVERY=99999
SAVE_EVERY=99999
EVAL_SAMPLES=1270
LOG_SAMPLES_EVERY=100
NUM_LOG_SAMPLES=3
MAX_NEW_TOKENS=256
EVAL_BATCH_SIZE=64
OUTPUT_DIR="./ckpt/sft_pt"

# ============================================================================
# Run
# ============================================================================
torchrun \
  --nproc_per_node=$N_GPUS \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train_sft_pt.py \
  --model_name $MODEL_NAME \
  --dataset $DATASET \
  --max_length $MAX_LENGTH \
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
  --eval_batch_size $EVAL_BATCH_SIZE \
  --output_dir $OUTPUT_DIR
