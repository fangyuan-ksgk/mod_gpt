#!/bin/bash
# Sweep: SFT baseline + SoRL configs on Qwen3-1.7B
#
# Layout:
#   1. SFT baseline (no abstract tokens)
#   2. SoRL sweep: emb_warmup_steps, emb_lr_mult, loss alphas
#      with lr=1e-5
#
# Usage:
#   bash run_sweep_qwen3_1.7B.sh

set -euo pipefail

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
# Shared Configuration
# ============================================================================
N_GPUS=4
MASTER_ADDR=127.0.0.1
MASTER_PORT=29600

# Model
MODEL_NAME="Qwen/Qwen3-1.7B"
ABSTRACT_VOCAB_SIZE=128

# Data
DATASET="gsm8k"
MAX_LENGTH=512

# SoRL search
NUM_ROLLOUTS=4
K=4
MAX_ITERATIONS=2
MEMORY_SPAN_ABS=1792
MEMORY_SPAN_TRAJ=1792
TEMPERATURE=1.0

# Loss function
DECAY=0.8
TARGET_VOCAB_UTIL=0.8

# Optimizer
LR=1e-5
EMB_LR_MULT=10.0      # embedding LR = LR * EMB_LR_MULT = 1e-4
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

TIMESTAMP=$(date +%Y%m%d_%H%M)
BASE_OUTPUT_DIR="./ckpt/sweep_qwen3_1.7B_${TIMESTAMP}"
mkdir -p "${BASE_OUTPUT_DIR}"

echo "============================================================"
echo "Sweep: Qwen3-1.7B  |  ${TIMESTAMP}"
echo "  LR=${LR}  EMB_LR_MULT=${EMB_LR_MULT}  (emb LR = 1e-4)"
echo "  Output: ${BASE_OUTPUT_DIR}"
echo "============================================================"

# ============================================================================
# 1. SFT Baseline
# ============================================================================
echo ""
echo ">>>>>>>>>> [1/N] SFT Baseline <<<<<<<<<<"
SFT_OUTPUT="${BASE_OUTPUT_DIR}/sft_baseline"

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
  --output_dir $SFT_OUTPUT

# Increment port to avoid "port already in use"
MASTER_PORT=$((MASTER_PORT + 1))

# ============================================================================
# 2. SoRL Sweep
# ============================================================================
# Format: ALPHA_INFO_GAIN  ALPHA_ABS  ALPHA_SOFT_ZIPF  ORTHO_REG  EMB_LR_M  EMB_WARMUP
#
# Priority 1: emb_warmup_steps (does warmup alone diversify embeddings?)
# Priority 2: emb_lr_mult (does high emb LR alone suffice?)
# Priority 3: loss alpha sweep (info/abs/zipf)
# Low priority: ortho_reg (just 1 run for comparison)
CONFIGS=(
  # --- A. emb_warmup_steps sweep (emb_lr_mult=10, no ortho) ---
  "10.0  1.0  5.0   0.0   10.0   500"     # warmup=500
  "10.0  1.0  5.0   0.0   10.0   1000"    # warmup=1000
  "10.0  1.0  5.0   0.0   10.0   1500"    # warmup=1500

  # --- B. emb_lr_mult sweep (no warmup, no ortho) ---
  "10.0  1.0  5.0   0.0   1.0    0"       # baseline: emb_lr_mult=1 (no boost)
  "10.0  1.0  5.0   0.0   5.0    0"       # emb_lr_mult=5
  "10.0  1.0  5.0   0.0   10.0   0"       # emb_lr_mult=10

  # --- C. Loss alpha sweep (emb_lr_mult=10, no warmup, no ortho) ---
  "10.0  0.1  1.0   0.0   10.0   0"       # low abs, low zipf
  "10.0  5.0  5.0   0.0   10.0   0"       # high abs, high zipf
  "5.0   1.0  10.0  0.0   10.0   0"       # lower info, strong anti-stutter

  # --- D. Ortho comparison (1 run only) ---
  "10.0  1.0  5.0   1.0   10.0   0"       # ortho=1.0 for reference
)

RUN_IDX=2
for CONFIG in "${CONFIGS[@]}"; do
  read -r ALPHA_INFO_GAIN ALPHA_ABS ALPHA_SOFT_ZIPF ORTHO_REG THIS_EMB_LR_MULT EMB_WARMUP <<< "$CONFIG"
  RUN_NAME="info${ALPHA_INFO_GAIN}_abs${ALPHA_ABS}_zipf${ALPHA_SOFT_ZIPF}_orth${ORTHO_REG}_emblr${THIS_EMB_LR_MULT}_wu${EMB_WARMUP}"
  OUTPUT_DIR="${BASE_OUTPUT_DIR}/${RUN_NAME}"

  echo ""
  echo ">>>>>>>>>> [${RUN_IDX}/N] ${RUN_NAME} <<<<<<<<<<"
  echo "  alpha_info_gain=${ALPHA_INFO_GAIN}  alpha_abs=${ALPHA_ABS}"
  echo "  alpha_soft_zipf=${ALPHA_SOFT_ZIPF}  ortho_reg=${ORTHO_REG}"
  echo "  emb_lr_mult=${THIS_EMB_LR_MULT}  emb_warmup_steps=${EMB_WARMUP}"
  echo "  output_dir=${OUTPUT_DIR}"

  torchrun \
    --nproc_per_node=$N_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_sorl_pt.py \
    --model_name $MODEL_NAME \
    --mode sorl_v2 \
    --abstract_vocab_size $ABSTRACT_VOCAB_SIZE \
    --dataset $DATASET \
    --max_length $MAX_LENGTH \
    --num_rollouts $NUM_ROLLOUTS \
    --K $K \
    --max_iterations $MAX_ITERATIONS \
    --memory_span_abs $MEMORY_SPAN_ABS \
    --memory_span_traj $MEMORY_SPAN_TRAJ \
    --temperature $TEMPERATURE \
    --alpha_info_gain $ALPHA_INFO_GAIN \
    --alpha_abs $ALPHA_ABS \
    --alpha_soft_zipf $ALPHA_SOFT_ZIPF \
    --ortho_reg $ORTHO_REG \
    --decay $DECAY \
    --target_vocab_util $TARGET_VOCAB_UTIL \
    --lr $LR \
    --emb_lr_mult $THIS_EMB_LR_MULT \
    --emb_warmup_steps $EMB_WARMUP \
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

  # Increment port for next run
  MASTER_PORT=$((MASTER_PORT + 1))
  RUN_IDX=$((RUN_IDX + 1))

done

echo ""
echo "============================================================"
echo "Sweep complete! Results in: ${BASE_OUTPUT_DIR}"
echo "============================================================"
