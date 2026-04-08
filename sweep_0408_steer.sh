#!/bin/bash
set -e

# Representation Steering sweep — V6 self-routed diagonal steering
# 1 GPU per experiment, sequential batches of 1.

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
# Shared configuration
# ============================================================================
MASTER_ADDR=127.0.0.1
BASE_PORT=29600

MODEL="Qwen/Qwen3-0.6B"
DATASET="gsm8k"
MAX_LENGTH=512

BATCH_SIZE=2
GRAD_ACCUM=4   # effective batch = 8
WARMUP=50
MAX_GRAD_NORM=1.0

LOG_EVERY=10
EVAL_SAMPLES=1300
EVAL_BATCH_SIZE=64
MAX_NEW_TOKENS=256

TIMESTAMP=$(date +%Y%m%d_%H%M)
EXP_IDX=0

run_exp() {
  EXP_IDX=$((EXP_IDX + 1))
  local idx=$EXP_IDX
  local port=$((BASE_PORT + idx))
  local tag=$1; shift
  local output_dir="./ckpt/steer_${TIMESTAMP}/exp${idx}_${tag}"

  echo ""
  echo "=== Exp ${idx}: ${tag} ==="

  CUDA_VISIBLE_DEVICES=0 torchrun \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$port \
    train_steer_pt.py \
    --model_name $MODEL \
    --dataset $DATASET \
    --max_length $MAX_LENGTH \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --warmup_steps $WARMUP \
    --max_grad_norm $MAX_GRAD_NORM \
    --log_every $LOG_EVERY \
    --eval_samples $EVAL_SAMPLES \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --max_new_tokens $MAX_NEW_TOKENS \
    --output_dir $output_dir \
    "$@"
}

# ===========================================================================
# Batch 1 — Scale sweep (C=32, L=16, steer_lr=1e-3, 1 epoch)
#   Baseline was scale=0.1 → 44.2% (= no-steer baseline).
#   Hypothesis: larger scale forces model to depend on steering.
# ===========================================================================
echo ""
echo "====== Batch 1: Scale sweep ======"

run_exp "scale01" --mode v6 --C_SIZE 32 --L 16 --scale 0.1 \
  --lr 1e-5 --steer_lr 1e-3 --num_epochs 1

run_exp "scale05" --mode v6 --C_SIZE 32 --L 16 --scale 0.5 \
  --lr 1e-5 --steer_lr 1e-3 --num_epochs 1

run_exp "scale10" --mode v6 --C_SIZE 32 --L 16 --scale 1.0 \
  --lr 1e-5 --steer_lr 1e-3 --num_epochs 1

# ===========================================================================
# Batch 2 — steer_lr sweep (scale=0.5, C=32, L=16, 1 epoch)
#   Hypothesis: steering needs much higher LR to break out of near-zero.
# ===========================================================================
echo ""
echo "====== Batch 2: steer_lr sweep ======"

run_exp "slr1e2_s05" --mode v6 --C_SIZE 32 --L 16 --scale 0.5 \
  --lr 1e-5 --steer_lr 1e-2 --num_epochs 1

run_exp "slr1e1_s05" --mode v6 --C_SIZE 32 --L 16 --scale 0.5 \
  --lr 1e-5 --steer_lr 1e-1 --num_epochs 1

# ===========================================================================
# Batch 3 — C_SIZE sweep (scale=0.5, steer_lr=1e-2, L=16, 1 epoch)
#   Fewer codes → each code covers more data → easier to specialize.
# ===========================================================================
echo ""
echo "====== Batch 3: C_SIZE sweep ======"

run_exp "c4_s05" --mode v6 --C_SIZE 4 --L 16 --scale 0.5 \
  --lr 1e-5 --steer_lr 1e-2 --num_epochs 1

run_exp "c8_s05" --mode v6 --C_SIZE 8 --L 16 --scale 0.5 \
  --lr 1e-5 --steer_lr 1e-2 --num_epochs 1

run_exp "c128_s05" --mode v6 --C_SIZE 128 --L 16 --scale 0.5 \
  --lr 1e-5 --steer_lr 1e-2 --num_epochs 1

# ===========================================================================
# Batch 4 — Freeze model (only train steering, scale=1.0)
#   Tests if steering alone can improve over frozen baseline.
# ===========================================================================
echo ""
echo "====== Batch 4: Freeze model ======"

run_exp "freeze_c32_s10" --mode v6 --C_SIZE 32 --L 16 --scale 1.0 \
  --lr 1e-5 --steer_lr 1e-2 --num_epochs 1 --freeze_model

run_exp "freeze_c8_s10" --mode v6 --C_SIZE 8 --L 16 --scale 1.0 \
  --lr 1e-5 --steer_lr 1e-2 --num_epochs 1 --freeze_model

# ===========================================================================
# Batch 5 — More epochs (scale=0.5, steer_lr=1e-2, C=32)
#   1 epoch may be insufficient for steering to differentiate.
# ===========================================================================
echo ""
echo "====== Batch 5: Epoch sweep ======"

run_exp "ep3_c32_s05" --mode v6 --C_SIZE 32 --L 16 --scale 0.5 \
  --lr 1e-5 --steer_lr 1e-2 --num_epochs 3

echo ""
echo "All done. ${EXP_IDX} experiments completed."
