#!/bin/bash
# SoRL Post-Training: Qwen2.5-1.5B on GSM8K (2 GPU)
#
# Goal: Fix abstract vocab collapse via zipf_alpha (Zipfian prior shape).
#
# Previous findings (1.5B, all collapsed):
#   - low alpha_abs (0.01) → best accuracy
#   - higher zipf weight → lower accuracy
#   - all configs collapsed regardless of zipf weight
#
# Hypothesis: flattening the Zipfian prior (zipf_alpha → 0) will resolve
# collapse by pushing the regularization target toward uniform.
#
# Sweep axes:
#   zipf_alpha ∈ {0.0, 0.3, 0.5}     (uniform → mild Zipf)
#   alpha_soft_zipf ∈ {5.0, 10.0}     (moderate → strong regularization)
#   alpha_info_gain ∈ {20.0}          (best from previous sweep)
#   alpha_abs = 0.01                   (best from previous sweep)
#
# Usage:
#   bash run_sorl_pt_1.5B.sh

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
N_GPUS=2
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500

# Model
MODEL_NAME="Qwen/Qwen2.5-1.5B"
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

# Loss function (fixed across sweep)
DECAY=0.8
TARGET_VOCAB_UTIL=0.8
ALPHA_ABS=0.01

# Optimizer
LR=1e-5
WARMUP_STEPS=50

# Training
BATCH_SIZE=2
GRAD_ACCUM=2  # 2 GPUs × BS 2 × 2 accum = effective BS 8 (matches 4-GPU setting)
NUM_EPOCHS=3

# Logging / Eval / Checkpoint
LOG_EVERY=10
EVAL_EVERY=500
SAVE_EVERY=500
EVAL_SAMPLES=50
LOG_SAMPLES_EVERY=100
NUM_LOG_SAMPLES=3
MAX_NEW_TOKENS=256
BASE_OUTPUT_DIR="./ckpt/sorl_pt_1.5B"

# ============================================================================
# Sweep: zipf_alpha × zipf_weight × info_gain
# Format: ALPHA_INFO_GAIN ALPHA_SOFT_ZIPF ZIPF_ALPHA
# ============================================================================
CONFIGS=(
  # --- Attempt 3: Uniform prior (zipf_alpha=0.0) ---
  "20.0 5.0  0.0"   # uniform prior, moderate zipf weight (best info from prev)
  "20.0 10.0 0.0"   # uniform prior, strong zipf weight
  "20.0 20.0 0.0"   # uniform prior, very strong zipf weight

  # --- Middle ground: flatter Zipf (zipf_alpha=0.3) ---
  "20.0 5.0  0.3"   # mild Zipf, moderate weight
  "20.0 10.0 0.3"   # mild Zipf, strong weight

  # --- Attempt 2 revisit: standard Zipf but stronger push ---
  "20.0 5.0  0.5"   # half-Zipf, moderate weight
  "20.0 20.0 0.5"   # half-Zipf, very strong weight
)

for CONFIG in "${CONFIGS[@]}"; do
  read -r ALPHA_INFO_GAIN ALPHA_SOFT_ZIPF ZIPF_ALPHA <<< "$CONFIG"
  RUN_NAME="info${ALPHA_INFO_GAIN}_abs${ALPHA_ABS}_zipf${ALPHA_SOFT_ZIPF}_za${ZIPF_ALPHA}"
  OUTPUT_DIR="${BASE_OUTPUT_DIR}/${RUN_NAME}"

  echo "============================================================"
  echo "Running: $RUN_NAME"
  echo "  alpha_info_gain=$ALPHA_INFO_GAIN  alpha_abs=$ALPHA_ABS"
  echo "  alpha_soft_zipf=$ALPHA_SOFT_ZIPF  zipf_alpha=$ZIPF_ALPHA"
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
    --zipf_alpha $ZIPF_ALPHA \
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
