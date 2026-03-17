#!/bin/bash
# SoRL Ablation Experiments — Qwen3-0.6B on 4×H100
#
# 20 single-GPU experiments (5 batches × 4 parallel) ≈ 5 hours
#
# Prior best configs (0.6B):
#   v3: shuffle r=1.0 γ=0.5 (full corruption works best for 0.6B)
#   v1: info=1 abs=0.5
#   ortho=1.0 is helpful
#
# Batch 1 (≈1h) — Validate prior conclusions (3 epochs):
#   exp1   v1 baseline        info_gain + abs              → no dependency
#   exp2   v1 + regularizers  + zipf + ortho               → doesn't help
#   exp3   v2 baseline        traj + abs                   → dependency not established
#   exp4   v3 baseline        traj + abs + hinge (r=1.0)   → v3 >> v1 cond accuracy
#
# Batch 2 (≈1h) — v2 sweep (3 epochs):
#   exp5   v2 + ortho         does ortho help v2?
#   exp6   v2 + zipf          does zipf help v2?
#   exp7   v2 + zipf + ortho  full regularization for v2
#   exp8   v2 lower traj      alpha_traj=0.5
#
# Batch 3 (≈1h) — v3 sweep (3 epochs):
#   exp9   v3 r=0.3           weaker corruption (compare to r=1.0 baseline)
#   exp10  v3 noise           noise instead of shuffle
#   exp11  v3 γ=0.1           lower margin (easier to satisfy)
#   exp12  v3 + ortho         does ortho help v3?
#
# Batch 4 (≈1.5h) — v4 inner=4, 1 epoch:
#   exp13  v4 baseline        shuffle r=1.0 γ=0.5
#   exp14  v4 r=0.3           weaker corruption
#   exp15  v4 γ=0.1           lower margin
#   exp16  v4 no hinge        alpha_contrastive=0 (pure inner-loop ablation)
#
# Batch 5 (≈1.5h) — v4 inner=2, 2 epochs:
#   exp17  v4 baseline        shuffle r=1.0 γ=0.5
#   exp18  v4 r=0.3           weaker corruption
#   exp19  v4 γ=0.1           lower margin
#   exp20  v4 no hinge        alpha_contrastive=0
#
# All use emb_lr_mult=10.
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
N_GPUS=4

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
# Batch 1/5 — Validate prior conclusions (3 epochs, 4 parallel) ≈ 1h
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 1/5: Validate — v1/v1+reg/v2/v3 baselines (${TIMESTAMP})"
echo "  Model: ${MODEL_NAME} | 4xH100 | 3 epochs each"
echo "============================================================"

run_bg "v1_baseline"           --num_epochs 3 $V1
run_bg "v1_zipf_ortho"         --num_epochs 3 $V1 --alpha_soft_zipf 1.0 --alpha_ortho 1.0
run_bg "v2_baseline"           --num_epochs 3 $V2
run_bg "v3_baseline"           --num_epochs 3 $V3

echo "  4 experiments launched. Waiting..."
wait

# ============================================================================
# Batch 2/5 — v2 sweep (3 epochs, 4 parallel) ≈ 1h
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 2/5: v2 sweep — regularization variants (${TIMESTAMP})"
echo "============================================================"

run_bg "v2_ortho"              --num_epochs 3 $V2 --alpha_ortho 1.0
run_bg "v2_zipf"               --num_epochs 3 $V2 --alpha_soft_zipf 1.0
run_bg "v2_zipf_ortho"         --num_epochs 3 $V2 --alpha_soft_zipf 1.0 --alpha_ortho 1.0
run_bg "v2_traj0.5"            --num_epochs 3 $V2 --alpha_traj 0.5

echo "  4 experiments launched. Waiting..."
wait

# ============================================================================
# Batch 3/5 — v3 sweep (3 epochs, 4 parallel) ≈ 1h
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 3/5: v3 sweep — corruption & margin variants (${TIMESTAMP})"
echo "============================================================"

run_bg "v3_r0.3"               --num_epochs 3 $V3 --corrupt_ratio 0.3
run_bg "v3_noise"              --num_epochs 3 $V3 --corrupt_method noise
run_bg "v3_g0.1"               --num_epochs 3 $V3 --gamma_contrastive 0.1
run_bg "v3_ortho"              --num_epochs 3 $V3 --alpha_ortho 1.0

echo "  4 experiments launched. Waiting..."
wait

# ============================================================================
# Batch 4/5 — v4 inner=4, 1 epoch (4 parallel) ≈ 1.5h
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 4/5: v4 inner=4, 1 epoch — config sweep (${TIMESTAMP})"
echo "============================================================"

run_bg "v4_i4_baseline"        --num_epochs 1 --n_inner 4 $V4
run_bg "v4_i4_r0.3"            --num_epochs 1 --n_inner 4 $V4 --corrupt_ratio 0.3
run_bg "v4_i4_g0.1"            --num_epochs 1 --n_inner 4 $V4 --gamma_contrastive 0.1
run_bg "v4_i4_nohinge"         --num_epochs 1 --n_inner 4 $V4 --alpha_contrastive 0.0

echo "  4 experiments launched. Waiting..."
wait

# ============================================================================
# Batch 5/5 — v4 inner=2, 2 epochs (4 parallel) ≈ 1.5h
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 5/5: v4 inner=2, 2 epochs — config sweep (${TIMESTAMP})"
echo "============================================================"

run_bg "v4_i2_baseline"        --num_epochs 2 --n_inner 2 $V4
run_bg "v4_i2_r0.3"            --num_epochs 2 --n_inner 2 $V4 --corrupt_ratio 0.3
run_bg "v4_i2_g0.1"            --num_epochs 2 --n_inner 2 $V4 --gamma_contrastive 0.1
run_bg "v4_i2_nohinge"         --num_epochs 2 --n_inner 2 $V4 --alpha_contrastive 0.0

echo "  4 experiments launched. Waiting..."
wait

echo ""
echo "============================================================"
echo "All 20 experiments complete. Results in ./ckpt/ablate_${TIMESTAMP}/"
echo "============================================================"