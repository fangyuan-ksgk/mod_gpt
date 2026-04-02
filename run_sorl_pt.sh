#!/bin/bash
# SoRL Post-Training — run best config for a specific model
#
# Usage:
#   bash run_sorl_pt.sh <model>
#
# Supported models:
#   qwen3-0.6b   — Qwen/Qwen3-0.6B
#   qwen3-1.7b   — Qwen/Qwen3-1.7B
#
# Best configs determined from sweep results (see log/warmup->sorl.md):
#   Both models: v3 trainer, jacobi=0.5 + mtraj=1.0
#   0.6B: NL=47.6%, K=4=45.5%, gap=2.1pp
#   1.7B: NL=64.0%, K=4=60.6%, gap=3.4pp

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
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4

# ============================================================================
# Model selection
# ============================================================================
MODEL_KEY="${1:-qwen3-0.6b}"

case "$MODEL_KEY" in
  qwen3-0.6b)
    # Best: exp14 in log/warmup->sorl.md — NL=47.6%, K=4=45.5%, gap=2.1pp
    MODEL_NAME="Qwen/Qwen3-0.6B"
    EMB_LR_MULT=1.0
    ALPHA_TRAJ=1.0
    ALPHA_ABS=0.5
    ALPHA_JACOBI=0.5
    ALPHA_MASKED_TRAJ=1.0
    MASK_NL_RATIO=0.3
    MASK_NL_MODE="fixed"
    CORRUPT_METHOD="shuffle"
    CORRUPT_RATIO=1.0
    GAMMA_CONTRASTIVE=0.5
    ;;
  qwen3-1.7b)
    # Best: exp10 in log/warmup->sorl.md — NL=64.0%, K=4=60.6%, gap=3.4pp
    MODEL_NAME="Qwen/Qwen3-1.7B"
    EMB_LR_MULT=1.0
    ALPHA_TRAJ=1.0
    ALPHA_ABS=0.5
    ALPHA_JACOBI=0.5
    ALPHA_MASKED_TRAJ=1.0
    MASK_NL_RATIO=0.3
    MASK_NL_MODE="fixed"
    CORRUPT_METHOD="shuffle"
    CORRUPT_RATIO=1.0
    GAMMA_CONTRASTIVE=0.5
    ;;
  *)
    echo "Unknown model: $MODEL_KEY"
    echo "Supported: qwen3-0.6b, qwen3-1.7b"
    exit 1
    ;;
esac

# ============================================================================
# Shared configuration (same across models)
# ============================================================================
N_GPUS=4
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500

# Data
DATASET="gsm8k"
MAX_LENGTH=512

# SoRL search
NUM_ROLLOUTS=4
K=4
MAX_ITERATIONS=2
TEMPERATURE=1.0

# Optimizer
LR=1e-5
WARMUP_STEPS=50
BATCH_SIZE=2
GRAD_ACCUM=4
NUM_EPOCHS=3

# Logging / Eval / Checkpoint
LOG_EVERY=10
EVAL_EVERY=500
SAVE_EVERY=500
EVAL_SAMPLES=1300
EVAL_BATCH_SIZE=64
MAX_NEW_TOKENS=256
NUM_LOG_SAMPLES=3

TIMESTAMP=$(date +%Y%m%d_%H%M)
OUTPUT_DIR="./ckpt/sorl_pt_${MODEL_KEY}_${TIMESTAMP}"

echo "============================================================"
echo "SoRL Post-Training — Best Config"
echo "  model:       $MODEL_NAME"
echo "  trainer:     v3 + jacobi=${ALPHA_JACOBI} + mtraj=${ALPHA_MASKED_TRAJ}"
echo "  output_dir:  $OUTPUT_DIR"
echo "============================================================"

torchrun \
  --nproc_per_node=$N_GPUS \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train_sorl_post.py \
  --model_name $MODEL_NAME \
  --dataset $DATASET \
  --max_length $MAX_LENGTH \
  --use_v3 \
  --num_rollouts $NUM_ROLLOUTS \
  --K $K \
  --max_iterations $MAX_ITERATIONS \
  --temperature $TEMPERATURE \
  --alpha_traj $ALPHA_TRAJ \
  --alpha_abs $ALPHA_ABS \
  --alpha_jacobi $ALPHA_JACOBI \
  --alpha_masked_traj $ALPHA_MASKED_TRAJ \
  --mask_nl_ratio $MASK_NL_RATIO \
  --mask_nl_mode $MASK_NL_MODE \
  --corrupt_method $CORRUPT_METHOD \
  --corrupt_ratio $CORRUPT_RATIO \
  --gamma_contrastive $GAMMA_CONTRASTIVE \
  --emb_lr_mult $EMB_LR_MULT \
  --lr $LR \
  --warmup_steps $WARMUP_STEPS \
  --batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $GRAD_ACCUM \
  --num_epochs $NUM_EPOCHS \
  --log_every $LOG_EVERY \
  --eval_every $EVAL_EVERY \
  --save_every $SAVE_EVERY \
  --eval_samples $EVAL_SAMPLES \
  --eval_batch_size $EVAL_BATCH_SIZE \
  --max_new_tokens $MAX_NEW_TOKENS \
  --num_log_samples $NUM_LOG_SAMPLES \
  --output_dir $OUTPUT_DIR
