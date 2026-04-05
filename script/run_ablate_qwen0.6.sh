#!/bin/bash
# SoRL Ablation Experiments — Qwen3-0.6B on 4×H100
#
# 24 single-GPU experiments (6 batches × 4 parallel)
#
# Batch 1 — Baselines (3 epochs):
#   v1 baseline, v1+zipf+ortho, v2 baseline, v3 baseline
#
# Batch 2 — v2 sweep (3 epochs):
#   v2+ortho, v2+zipf, v2+zipf+ortho, v2 traj=0.5
#
# Batch 3 — v3 sweep (3 epochs):
#   v3 r=0.3, v3 noise, v3 γ=0.1, v3+ortho
#
# Batch 4 — v4 inner=2 (2 epochs):
#   v4 baseline, v4 r=0.3, v4 γ=0.1, v4 no hinge
#
# Batch 5 — response_only_abs baselines (3 epochs):
#   v1, v1+zipf+ortho, v2, v3
#
# Batch 6 — response_only_abs sweep (3 epochs):
#   v2+ortho, v3+ortho, v3 r=0.3, v4 i2
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

# =============================
# Base configs WITHOUT emb_lr baked in (used in Batches 7-9)
R_V1="--alpha_info_gain 1.0 --alpha_abs 0.5"
R_V2="--use_v2 --alpha_traj 1.0 --alpha_abs 0.5"
R_V3_R03="--use_v3 --alpha_traj 1.0 --alpha_abs 0.5 --corrupt_method shuffle --corrupt_ratio 0.3 --gamma_contrastive 0.5"
R_V3_R05="--use_v3 --alpha_traj 1.0 --alpha_abs 0.5 --corrupt_method shuffle --corrupt_ratio 0.5 --gamma_contrastive 0.5"
R_V3_R10="--use_v3 --alpha_traj 1.0 --alpha_abs 0.5 --corrupt_method shuffle --corrupt_ratio 1.0 --gamma_contrastive 0.5"
R_V3_NOISE="--use_v3 --alpha_traj 1.0 --alpha_abs 0.5 --corrupt_method noise --corrupt_ratio 0.3 --gamma_contrastive 0.5"


# ============================================================================
# Batch 5 — response_only_abs baselines (3 epochs, 4 parallel)
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 5: response_only_abs — baselines (${TIMESTAMP})"
echo "============================================================"

run_bg "v1_resp_only"          --num_epochs 3 $V1 --response_only_abs
run_bg "v1_resp_only_reg"      --num_epochs 3 $V1 --alpha_soft_zipf 1.0 --alpha_ortho 1.0 --response_only_abs
run_bg "v2_resp_only"          --num_epochs 3 $V2 --response_only_abs
run_bg "v3_resp_only"          --num_epochs 3 $V3 --response_only_abs

echo "  4 experiments launched. Waiting..."
wait

# ============================================================================
# Batch 6 — response_only_abs sweep (3 epochs, 4 parallel)
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 6: response_only_abs — sweep (${TIMESTAMP})"
echo "============================================================"

run_bg "v2_resp_only_ortho"    --num_epochs 3 $V2 --alpha_ortho 1.0 --response_only_abs
run_bg "v3_resp_only_ortho"    --num_epochs 3 $V3 --alpha_ortho 1.0 --response_only_abs
run_bg "v3_resp_only_r0.3"     --num_epochs 3 $V3 --corrupt_ratio 0.3 --response_only_abs
run_bg "v4_i2_resp_only"       --num_epochs 2 --n_inner 2 $V4 --response_only_abs

echo "  4 experiments launched. Waiting..."
wait

# ============================================================================
# Batch 7 — emb_lr=1.0: v1, v2, v3 r=0.3, v3 r=0.5  (4 parallel)
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 7: emb_lr=1.0 — v1/v2/v3_r0.3/v3_r0.5 (${TIMESTAMP})"
echo "============================================================"

run_bg "v1_e1"       --num_epochs 3 $R_V1      --emb_lr_mult 1.0
run_bg "v2_e1"       --num_epochs 3 $R_V2      --emb_lr_mult 1.0
run_bg "v3_r03_e1"   --num_epochs 3 $R_V3_R03  --emb_lr_mult 1.0
run_bg "v3_r05_e1"   --num_epochs 3 $R_V3_R05  --emb_lr_mult 1.0

echo "  4 experiments launched. Waiting..."
wait

# ============================================================================
# Batch 8 — emb_lr=1.0: v3 r=1.0, v3 noise  |  emb_lr=10.0: v1, v2  (4 parallel)
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 8: emb_lr=1.0 v3_r1.0/noise + emb_lr=10.0 v1/v2 (${TIMESTAMP})"
echo "============================================================"

run_bg "v3_r10_e1"    --num_epochs 3 $R_V3_R10    --emb_lr_mult 1.0
run_bg "v3_noise_e1"  --num_epochs 3 $R_V3_NOISE  --emb_lr_mult 1.0
run_bg "v1_e10"       --num_epochs 3 $R_V1        --emb_lr_mult 10.0
run_bg "v2_e10"       --num_epochs 3 $R_V2        --emb_lr_mult 10.0

echo "  4 experiments launched. Waiting..."
wait

# ============================================================================
# Batch 9 — emb_lr=10.0: v3 r=0.3, r=0.5, r=1.0, noise  (4 parallel)
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 9: emb_lr=10.0 — v3 sweep (${TIMESTAMP})"
echo "============================================================"

run_bg "v3_r03_e10"   --num_epochs 3 $R_V3_R03   --emb_lr_mult 10.0
run_bg "v3_r05_e10"   --num_epochs 3 $R_V3_R05   --emb_lr_mult 10.0
run_bg "v3_r10_e10"   --num_epochs 3 $R_V3_R10   --emb_lr_mult 10.0
run_bg "v3_noise_e10" --num_epochs 3 $R_V3_NOISE --emb_lr_mult 10.0

echo "  4 experiments launched. Waiting..."
wait

echo ""
echo "============================================================"
echo "All 36 experiments complete. Results in ./ckpt/ablate_${TIMESTAMP}/"
echo "============================================================"