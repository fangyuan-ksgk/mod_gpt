#!/bin/bash
# SoRL Ablation Experiments v2 — Qwen3-0.6B on 4×H100
#
# 16 single-GPU experiments (4 batches × 4 parallel) ≈ 8 hours
#
# Research questions:
#   Q1 (2 runs): Bug-fix validation — re-run prev best configs with fixed CE logits slicing
#   Q2 (2 runs): v2 ablation — does optimizing p(s|a) directly suffice vs info-gain?
#   Q3 (2 runs): emb_lr_mult — does 10× embedding LR help ortho_loss converge?
#   Q4 (8 runs): Diversity — does ortho alone suffice, or is zipf needed?
#                 2×2 factorial (ortho × zipf) at info=1 and info=3
#   Extended (2 runs): v2 + ortho combos
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
# Batch 1/4: Q1 (bug-fix re-run) + Q2 (v2 ablation)
#   Q1: re-run prev best configs (info+abs) with fixed CE logits slicing
#   Q2: --use_v2 optimizes p(s|a) directly, no info-gain formulation
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 1/4: Q1 bug-fix re-run + Q2 v2 ablation (${TIMESTAMP})"
echo "  Model: ${MODEL_NAME} | 4xH100 | 1 run/GPU"
echo "============================================================"

# Q1: prev best configs (with abs>0) — compare to old runs under fixed CE
run_bg "Q1_fix_info1_abs0.5"       --alpha_info_gain 1.0 --alpha_abs 0.5
run_bg "Q1_fix_info3_abs0.5"       --alpha_info_gain 3.0 --alpha_abs 0.5
# Q2: v2 trainer — p(s|a) directly (alpha_info_gain controls weight on cond_traj_loss)
run_bg "Q2_v2_info1_abs0.5"        --use_v2 --alpha_info_gain 1.0 --alpha_abs 0.5
run_bg "Q2_v2_info3_abs0.5"        --use_v2 --alpha_info_gain 3.0 --alpha_abs 0.5

echo "  4 experiments launched. Waiting..."
wait

# ============================================================================
# Batch 2/4: Q3 (emb_lr_mult) + Q4 start (diversity at info=1)
#   Q3: same config, emb_lr_mult=1 vs 10 — does higher emb LR help ortho converge?
#   Q4: zipf-only baseline at info=1 + ortho+zipf combo at info=1
#   (ortho-only at info=1 = Q3 emb10x run; neither = Q1 fix_info1 run)
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 2/4: Q3 emb_lr_mult + Q4 diversity at info=1 (${TIMESTAMP})"
echo "============================================================"

# Q3: emb_lr_mult effect (both have ortho=1.0)
run_bg "Q3_ortho1.0_emb1x"         --alpha_info_gain 1.0 --alpha_abs 0.5 --alpha_ortho 1.0 --emb_lr_mult 1.0
run_bg "Q3_ortho1.0_emb10x"        --alpha_info_gain 1.0 --alpha_abs 0.5 --alpha_ortho 1.0 --emb_lr_mult 10.0
# Q4 at info=1: zipf-only, ortho+zipf  (ortho-only = Q3_emb10x, neither = Q1_fix_info1)
run_bg "Q4_i1_zipf1.0"             --alpha_info_gain 1.0 --alpha_abs 0.5 --alpha_soft_zipf 1.0
run_bg "Q4_i1_ortho1.0_zipf1.0"    --alpha_info_gain 1.0 --alpha_abs 0.5 --alpha_ortho 1.0 --alpha_soft_zipf 1.0 --emb_lr_mult 10.0

echo "  4 experiments launched. Waiting..."
wait

# ============================================================================
# Batch 3/4: Q4 continued (diversity factorial at info=3)
#   Full 2×2: ortho-only, zipf-only, both, neither=Q1_fix_info3
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 3/4: Q4 diversity factorial at info=3 (${TIMESTAMP})"
echo "============================================================"

# Q4 at info=3: ortho-only (neither = Q1_fix_info3)
run_bg "Q4_i3_ortho1.0"            --alpha_info_gain 3.0 --alpha_abs 0.5 --alpha_ortho 1.0 --emb_lr_mult 10.0
run_bg "Q4_i3_zipf1.0"             --alpha_info_gain 3.0 --alpha_abs 0.5 --alpha_soft_zipf 1.0
run_bg "Q4_i3_ortho1.0_zipf1.0"    --alpha_info_gain 3.0 --alpha_abs 0.5 --alpha_ortho 1.0 --alpha_soft_zipf 1.0 --emb_lr_mult 10.0
# Extended: v2 + ortho (does v2 benefit from diversity regularization?)
run_bg "v2_info1_ortho1.0_emb10x"  --use_v2 --alpha_info_gain 1.0 --alpha_abs 0.5 --alpha_ortho 1.0 --emb_lr_mult 10.0

echo "  4 experiments launched. Waiting..."
wait

# ============================================================================
# Batch 4/4: Extended combos + SFT baseline
#   v2+ortho at info=3, v2+zipf, stronger ortho, fresh SFT reference
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 4/4: Extended combos + baseline (${TIMESTAMP})"
echo "============================================================"

run_bg "v2_info3_ortho1.0_emb10x"  --use_v2 --alpha_info_gain 3.0 --alpha_abs 0.5 --alpha_ortho 1.0 --emb_lr_mult 10.0
run_bg "v2_info1_zipf1.0"          --use_v2 --alpha_info_gain 1.0 --alpha_abs 0.5 --alpha_soft_zipf 1.0
run_bg "ortho0.5_emb10x"           --alpha_info_gain 1.0 --alpha_abs 0.5 --alpha_ortho 0.5 --emb_lr_mult 10.0
run_bg "baseline_sft"

echo "  4 experiments launched. Waiting..."
wait

echo ""
echo "============================================================"
echo "All 16 experiments complete. Results in ./ckpt/ablate_${TIMESTAMP}/"
echo "============================================================"