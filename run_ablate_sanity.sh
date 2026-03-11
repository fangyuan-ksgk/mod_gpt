#!/bin/bash
# SoRL Ablate Sanity Check — should match SFT baseline performance
#
# Uses trainer_ablate.SoRLTrainer in sft_mode:
#   - Abstract logits masked to -inf in CE loss (no softmax dilution)
#   - No SoRL search (skipped entirely)
#   - NL-only greedy generation for eval (no block_mask, no abstract tokens)
#
# If this doesn't match SFT, there's a bug in the model wrapper plumbing.
#
# Usage:
#   bash run_ablate_sanity.sh

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
N_GPUS=4
MASTER_ADDR=127.0.0.1
MASTER_PORT=29501

MODEL_NAME="Qwen/Qwen3-0.6B"
DATASET="gsm8k"
MAX_LENGTH=512

LR=1e-5
WARMUP_STEPS=50
BATCH_SIZE=2
GRAD_ACCUM=$((8 / (BATCH_SIZE * N_GPUS)))
NUM_EPOCHS=3

LOG_EVERY=10
EVAL_EVERY=500
SAVE_EVERY=500
EVAL_SAMPLES=50
MAX_NEW_TOKENS=256

TIMESTAMP=$(date +%Y%m%d_%H%M)
OUTPUT_DIR="./ckpt/ablate_sanity_${TIMESTAMP}"

echo "============================================================"
echo "SoRL Ablate Sanity Check (sft_mode): ${MODEL_NAME}"
echo "  - Abstract logits masked from CE loss"
echo "  - No SoRL search"
echo "  - NL-only greedy generation for eval"
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
echo "Ablate Sanity Check complete: ${MODEL_NAME}"
echo "  Output: ${OUTPUT_DIR}"
echo "============================================================"
