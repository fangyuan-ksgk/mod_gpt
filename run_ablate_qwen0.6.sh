#!/bin/bash
# SoRL Ablation — Qwen3-0.6B — response_only_abs probe
#
# 16 experiments (4 batches × 4 parallel)
# Mirrors prior batches 1-4 with --response_only_abs.
#
# Batch 1 — Baselines + resp_only (3 epochs):
#   v1, v1+zipf+ortho, v2, v3
#
# Batch 2 — v2 sweep + resp_only (3 epochs):
#   v2+ortho, v2+zipf, v2+zipf+ortho, v2 traj=0.5
#
# Batch 3 — v3 sweep + resp_only (3 epochs):
#   v3 r=0.3, v3 noise, v3 γ=0.1, v3+ortho
#
# Batch 4 — v4 i2 + resp_only (2 epochs):
#   v4 baseline, v4 r=0.3, v4 γ=0.1, v4 no hinge
#
# All use emb_lr_mult=10. V3/V4 default: r=1.0.
#
# Usage:
#   bash run_ablate_qwen0.6.sh

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
N_GPUS=2

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
EVAL_SAMPLES=1300
EVAL_BATCH_SIZE=64
MAX_NEW_TOKENS=256

# shared defaults — all use emb_lr_mult=10
# NOTE: 0.6B best v3 config uses r=1.0 (full corruption), not r=0.3
EMB="--emb_lr_mult 10.0"
V1="--alpha_info_gain 1.0 --alpha_abs 0.5 $EMB"
V2="--use_v2 --alpha_traj 1.0 --alpha_abs 0.5 $EMB"
V3="--use_v3 --alpha_traj 1.0 --alpha_abs 0.5 --corrupt_method shuffle --corrupt_ratio 1.0 --gamma_contrastive 0.5 $EMB"
V4="--use_v4 --alpha_traj 1.0 --alpha_abs 0.5 --corrupt_method shuffle --corrupt_ratio 1.0 --gamma_contrastive 0.5 $EMB"

TIMESTAMP=$(date +%Y%m%d_%H%M)
EXP_IDX=0

# ---- Parallel scheduling: 4 x H100 — 1 run/GPU ----

run_bg() {
  EXP_IDX=$((EXP_IDX + 1))
  local idx=$EXP_IDX
  local gpu=$(( (idx - 1) % N_GPUS ))
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
# Batch 1 — Baselines + response_only_abs (3 epochs, 4 parallel)
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 1: resp_only baselines — v1/v1+reg/v2/v3 (${TIMESTAMP})"
echo "============================================================"

run_bg "v1_ro"                 --num_epochs 3 $V1 --response_only_abs
run_bg "v1_zipf_ortho_ro"      --num_epochs 3 $V1 --alpha_soft_zipf 1.0 --alpha_ortho 1.0 --response_only_abs
run_bg "v2_ro"                 --num_epochs 3 $V2 --response_only_abs
run_bg "v3_ro"                 --num_epochs 3 $V3 --response_only_abs

echo "  4 experiments launched. Waiting..."
wait

# ============================================================================
# Batch 2 — v2 sweep + response_only_abs (3 epochs, 4 parallel)
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 2: resp_only v2 sweep (${TIMESTAMP})"
echo "============================================================"

run_bg "v2_ortho_ro"           --num_epochs 3 $V2 --alpha_ortho 1.0 --response_only_abs
run_bg "v2_zipf_ro"            --num_epochs 3 $V2 --alpha_soft_zipf 1.0 --response_only_abs
run_bg "v2_zipf_ortho_ro"      --num_epochs 3 $V2 --alpha_soft_zipf 1.0 --alpha_ortho 1.0 --response_only_abs
run_bg "v2_traj0.5_ro"         --num_epochs 3 $V2 --alpha_traj 0.5 --response_only_abs

echo "  4 experiments launched. Waiting..."
wait

# ============================================================================
# Batch 3 — v3 sweep + response_only_abs (3 epochs, 4 parallel)
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 3: resp_only v3 sweep (${TIMESTAMP})"
echo "============================================================"

run_bg "v3_r0.3_ro"            --num_epochs 3 $V3 --corrupt_ratio 0.3 --response_only_abs
run_bg "v3_noise_ro"           --num_epochs 3 $V3 --corrupt_method noise --response_only_abs
run_bg "v3_g0.1_ro"            --num_epochs 3 $V3 --gamma_contrastive 0.1 --response_only_abs
run_bg "v3_ortho_ro"           --num_epochs 3 $V3 --alpha_ortho 1.0 --response_only_abs

echo "  4 experiments launched. Waiting..."
wait

# ============================================================================
# Batch 4 — v4 i2 + response_only_abs (2 epochs, 4 parallel)
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 4: resp_only v4 i2 (${TIMESTAMP})"
echo "============================================================"

run_bg "v4_i2_ro"              --num_epochs 2 --n_inner 2 $V4 --response_only_abs
run_bg "v4_i2_r0.3_ro"         --num_epochs 2 --n_inner 2 $V4 --corrupt_ratio 0.3 --response_only_abs
run_bg "v4_i2_g0.1_ro"         --num_epochs 2 --n_inner 2 $V4 --gamma_contrastive 0.1 --response_only_abs
run_bg "v4_i2_nohinge_ro"      --num_epochs 2 --n_inner 2 $V4 --alpha_contrastive 0.0 --response_only_abs

echo "  4 experiments launched. Waiting..."
wait

echo ""
echo "============================================================"
echo "All 16 experiments complete. Results in ./ckpt/ablate_${TIMESTAMP}/"
echo "============================================================"