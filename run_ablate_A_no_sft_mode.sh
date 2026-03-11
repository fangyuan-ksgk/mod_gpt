#!/bin/bash
# Ablation A: NO sft_mode, zero aux weights, full SoRL path (block_mask + full vocab CE)
# Tests whether sft_mode is necessary or if zeroing aux weights is sufficient.
# Uses 1 GPU to keep DDP out of the equation.
#
# Compare against: run_ablate_sanity.sh (sft_mode=True, 1 GPU)

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
# Configuration — match SFT baseline exactly
# ============================================================================
N_GPUS=1
MASTER_ADDR=127.0.0.1
MASTER_PORT=29502

MODEL_NAME="Qwen/Qwen3-0.6B"
DATASET="gsm8k"
MAX_LENGTH=512

LR=1e-5
WARMUP_STEPS=50
BATCH_SIZE=2
GRAD_ACCUM=$((8 / (BATCH_SIZE * N_GPUS)))  # =4, effective batch=8
NUM_EPOCHS=3

LOG_EVERY=10
EVAL_EVERY=500
SAVE_EVERY=500
EVAL_SAMPLES=50
MAX_NEW_TOKENS=256

TIMESTAMP=$(date +%Y%m%d_%H%M)
OUTPUT_DIR="./ckpt/ablate_A_no_sft_mode_${TIMESTAMP}"

echo "============================================================"
echo "Ablation A: NO sft_mode, zero aux weights: ${MODEL_NAME}"
echo "  - Full SoRL path (block_mask + full vocab CE)"
echo "  - All aux loss weights = 0"
echo "  - NL-only eval (eval_K=None)"
echo "  - 1 GPU (no DDP)"
echo "  Output: ${OUTPUT_DIR}"
echo "============================================================"

torchrun \
  --nproc_per_node=$N_GPUS \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train_ablate_sanity.py \
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
  --max_new_tokens $MAX_NEW_TOKENS \
  --output_dir $OUTPUT_DIR

echo "============================================================"
echo "Ablation A complete: ${MODEL_NAME}"
echo "  Output: ${OUTPUT_DIR}"
echo "============================================================"
