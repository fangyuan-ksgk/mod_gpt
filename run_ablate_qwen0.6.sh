#!/bin/bash
# SoRL Ablation Experiments v3 — Qwen3-0.6B on 4×H100
#
# 12 single-GPU experiments (3 batches × 4 parallel) ≈ 3 hours
#
# Previous findings:
#   - info=1.0, abs=0.5 is best accuracy config
#   - v2 (no info-gain) → abstract tokens collapse (effective_vocab=7), K=4 < K=None
#   - ortho=1.0 + emb_lr_mult=10 helps diversity
#
# This round: test v3 contrastive trainer (hinge loss against corrupted abstractions)
#
# Research questions:
#   Q5 (4 runs): Does v3 build dependency? (K=4 ≥ K=None?)
#   Q6 (4 runs): Gamma & corruption settings — margin, ratio, method
#   Q7 (4 runs): v1 vs v2 validation — head-to-head with matched configs
#
# All v3 runs share: --use_v3 --alpha_info_gain 1.0 --alpha_abs 0.5
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
EVAL_SAMPLES=1000
EVAL_BATCH_SIZE=64
MAX_NEW_TOKENS=256

# v3 shared defaults
V3="--use_v3 --alpha_info_gain 1.0 --alpha_abs 0.5"

TIMESTAMP=$(date +%Y%m%d_%H%M)
EXP_IDX=0

# ---- Parallel scheduling: 4 x H100 — 1 run/GPU, 4 parallel per batch ----

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
    --num_epochs $NUM_EPOCHS \
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
# Batch 1/3 — Q5: v3 core — does contrastive build dependency?
#   Sweep corruption ratio with shuffle method, gamma=0.5, alpha_contrastive=1.0
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 1/3: Q5 — v3 core contrastive (${TIMESTAMP})"
echo "  Model: ${MODEL_NAME} | 4xH100 | 1 run/GPU"
echo "============================================================"

run_bg "v3_shuf_r0.3_g0.5"   $V3 --corrupt_method shuffle --corrupt_ratio 0.3 --gamma_contrastive 0.5
run_bg "v3_shuf_r0.5_g0.5"   $V3 --corrupt_method shuffle --corrupt_ratio 0.5 --gamma_contrastive 0.5
run_bg "v3_shuf_r1.0_g0.5"   $V3 --corrupt_method shuffle --corrupt_ratio 1.0 --gamma_contrastive 0.5
run_bg "v3_noise_r0.3_g0.5"  $V3 --corrupt_method noise   --corrupt_ratio 0.3 --gamma_contrastive 0.5

echo "  4 experiments launched. Waiting..."
wait

# ============================================================================
# Batch 2/3 — Q6: gamma & alpha_contrastive sweep
#   Fix shuffle, ratio=0.3, vary gamma and weight
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 2/3: Q6 — gamma & alpha_contrastive sweep (${TIMESTAMP})"
echo "============================================================"

run_bg "v3_shuf_g0.1"        $V3 --corrupt_method shuffle --corrupt_ratio 0.3 --gamma_contrastive 0.1
run_bg "v3_shuf_g1.0"        $V3 --corrupt_method shuffle --corrupt_ratio 0.3 --gamma_contrastive 1.0
run_bg "v3_shuf_g2.0"        $V3 --corrupt_method shuffle --corrupt_ratio 0.3 --gamma_contrastive 2.0
run_bg "v3_acontr3.0"        $V3 --corrupt_method shuffle --corrupt_ratio 0.3 --gamma_contrastive 0.5 --alpha_contrastive 3.0

echo "  4 experiments launched. Waiting..."
wait

# ============================================================================
# Batch 3/3 — Q7: v1 vs v2 validation (same configs, head-to-head)
#   v2 now has torch.no_grad() fix on base_traj_loss — compare against v1
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 3/3: Q7 — v1 vs v2 validation + SFT baseline (${TIMESTAMP})"
echo "============================================================"

# v1 vs v2: info=1, abs=0.5 (matched config)
run_bg "v1_info1_abs0.5"               --alpha_info_gain 1.0 --alpha_abs 0.5
run_bg "v2_info1_abs0.5"               --use_v2 --alpha_info_gain 1.0 --alpha_abs 0.5
# v1 vs v2: info=1, abs=0.5, ortho=1.0, emb10x (matched config + diversity)
run_bg "v1_info1_ortho1_emb10x"        --alpha_info_gain 1.0 --alpha_abs 0.5 --alpha_ortho 1.0 --emb_lr_mult 10.0
run_bg "v2_info1_ortho1_emb10x"        --use_v2 --alpha_info_gain 1.0 --alpha_abs 0.5 --alpha_ortho 1.0 --emb_lr_mult 10.0

echo "  4 experiments launched. Waiting..."
wait

echo ""
echo "============================================================"
echo "All 12 experiments complete. Results in ./ckpt/ablate_${TIMESTAMP}/"
echo "============================================================"