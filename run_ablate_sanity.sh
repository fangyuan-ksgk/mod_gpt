#!/bin/bash
# SoRL Ablation Experiments — Qwen3-1.7B on 4×H100
#
# Usage:
#   bash run_ablate_sanity.sh
#
# 36 single-GPU experiments (1 run/GPU, 4 parallel) → 9 batches + 1 DDP validation.

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

MODEL_NAME="Qwen/Qwen3-1.7B"
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
# Batch 1/9: Baseline + info sweep (4 runs, 1/GPU)
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 1/9: Baseline + info sweep (${TIMESTAMP})"
echo "  Model: ${MODEL_NAME} | 4xA100 | 1 run/GPU"
echo "============================================================"

run_bg "baseline_aux0"
run_bg "info1.0"           --alpha_info_gain 1.0
run_bg "info3.0"           --alpha_info_gain 3.0
run_bg "info5.0"           --alpha_info_gain 5.0

echo "  4 experiments launched. Waiting..."
wait

# ============================================================================
# Batch 2/9: Info sweep tail + info+abs start (4 runs, 1/GPU)
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 2/9: Info sweep tail + info+abs start (${TIMESTAMP})"
echo "============================================================"

run_bg "info7.0"           --alpha_info_gain 7.0
run_bg "info9.0"           --alpha_info_gain 9.0
run_bg "info1.0_abs0.5"   --alpha_info_gain 1.0 --alpha_abs 0.5
run_bg "info3.0_abs0.5"   --alpha_info_gain 3.0 --alpha_abs 0.5

echo "  4 experiments launched. Waiting..."
wait

# ============================================================================
# Batch 3/9: Info+abs continued (4 runs, 1/GPU)
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 3/9: Info+abs continued (${TIMESTAMP})"
echo "============================================================"

run_bg "info5.0_abs0.5"           --alpha_info_gain 5.0 --alpha_abs 0.5
run_bg "info7.0_abs0.5"           --alpha_info_gain 7.0 --alpha_abs 0.5
run_bg "info9.0_abs0.5"           --alpha_info_gain 9.0 --alpha_abs 0.5
run_bg "info9.0_abs1.0"           --alpha_info_gain 9.0 --alpha_abs 1.0

echo "  4 experiments launched. Waiting..."
wait

# ============================================================================
# Batch 4/9: Abs sweep at info=9 + zipf start (4 runs, 1/GPU)
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 4/9: Abs sweep at info=9 + zipf start (${TIMESTAMP})"
echo "============================================================"

run_bg "info9.0_abs1.5"           --alpha_info_gain 9.0 --alpha_abs 1.5
run_bg "info9.0_abs2.0"           --alpha_info_gain 9.0 --alpha_abs 2.0
run_bg "info9.0_abs0.5_zipf0.5"  --alpha_info_gain 9.0 --alpha_abs 0.5 --alpha_soft_zipf 0.5
run_bg "info9.0_abs0.5_zipf1.0"  --alpha_info_gain 9.0 --alpha_abs 0.5 --alpha_soft_zipf 1.0

echo "  4 experiments launched. Waiting..."
wait

# ============================================================================
# Batch 5/9: Zipf at info=9 tail + fine-grid abs at info=1.0 (4 runs, 1/GPU)
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 5/9: Zipf tail + fine-grid abs at info=1.0 (${TIMESTAMP})"
echo "============================================================"

run_bg "info9.0_abs0.5_zipf1.5"  --alpha_info_gain 9.0 --alpha_abs 0.5 --alpha_soft_zipf 1.5
run_bg "info9.0_abs0.5_zipf2.0"  --alpha_info_gain 9.0 --alpha_abs 0.5 --alpha_soft_zipf 2.0
run_bg "info1.0_abs0.1"   --alpha_info_gain 1.0 --alpha_abs 0.1
run_bg "info1.0_abs0.3"   --alpha_info_gain 1.0 --alpha_abs 0.3

echo "  4 experiments launched. Waiting..."
wait

# ============================================================================
# Batch 6/9: Fine-grid abs at info=1.0 + fine info at abs=0.5 (4 runs, 1/GPU)
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 6/9: Fine-grid abs + fine info (${TIMESTAMP})"
echo "============================================================"

run_bg "info1.0_abs0.7"   --alpha_info_gain 1.0 --alpha_abs 0.7
run_bg "info1.0_abs1.0"   --alpha_info_gain 1.0 --alpha_abs 1.0
run_bg "info0.5_abs0.5"   --alpha_info_gain 0.5 --alpha_abs 0.5
run_bg "info1.5_abs0.5"   --alpha_info_gain 1.5 --alpha_abs 0.5

echo "  4 experiments launched. Waiting..."
wait

# ============================================================================
# Batch 7/9: Fine info continued + info=3.0 abs sweep (4 runs, 1/GPU)
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 7/9: Fine info + info=3.0 abs sweep (${TIMESTAMP})"
echo "============================================================"

run_bg "info2.0_abs0.5"            --alpha_info_gain 2.0 --alpha_abs 0.5
run_bg "info2.5_abs0.5"            --alpha_info_gain 2.5 --alpha_abs 0.5
run_bg "info3.0_abs0.1"            --alpha_info_gain 3.0 --alpha_abs 0.1
run_bg "info3.0_abs0.3"            --alpha_info_gain 3.0 --alpha_abs 0.3

echo "  4 experiments launched. Waiting..."
wait

# ============================================================================
# Batch 8/9: Info=3.0 abs tail + zipf at winner (4 runs, 1/GPU)
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 8/9: Info=3.0 abs tail + zipf at winner (${TIMESTAMP})"
echo "============================================================"

run_bg "info3.0_abs0.7"            --alpha_info_gain 3.0 --alpha_abs 0.7
run_bg "info3.0_abs1.0"            --alpha_info_gain 3.0 --alpha_abs 1.0
run_bg "info1.0_abs0.5_zipf0.5"   --alpha_info_gain 1.0 --alpha_abs 0.5 --alpha_soft_zipf 0.5
run_bg "info1.0_abs0.5_zipf1.0"   --alpha_info_gain 1.0 --alpha_abs 0.5 --alpha_soft_zipf 1.0

echo "  4 experiments launched. Waiting..."
wait

# ============================================================================
# Batch 9/9: Zipf at winner + K sweep (4 runs, 1/GPU)
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 9/9: Zipf at winner + K sweep (${TIMESTAMP})"
echo "============================================================"

run_bg "info1.0_abs0.5_zipf1.5"   --alpha_info_gain 1.0 --alpha_abs 0.5 --alpha_soft_zipf 1.5
run_bg "info1.0_abs0.5_zipf2.0"   --alpha_info_gain 1.0 --alpha_abs 0.5 --alpha_soft_zipf 2.0
run_bg "info1.0_abs0.5_K2"        --alpha_info_gain 1.0 --alpha_abs 0.5 --K 2
run_bg "info1.0_abs0.5_K8"        --alpha_info_gain 1.0 --alpha_abs 0.5 --K 8

echo "  4 experiments launched. Waiting..."
wait

# ============================================================================
# Batch 10: DDP validation (uses all 4 GPUs)
# ============================================================================
EXP_IDX=$((EXP_IDX + 1))
DDP_IDX=$EXP_IDX
DDP_PORT=$((BASE_PORT + DDP_IDX))
DDP_OUT="./ckpt/ablate_${TIMESTAMP}/exp${DDP_IDX}_4gpu_ddp_baseline"
echo ""
echo "============================================================"
echo "Exp ${DDP_IDX}: DDP validation  [GPU=0,1,2,3]  port=${DDP_PORT}"
echo "============================================================"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc_per_node=$N_GPUS \
  --master_addr=$MASTER_ADDR \
  --master_port=$DDP_PORT \
  train_ablate_sanity.py \
  --model_name $MODEL_NAME \
  --dataset $DATASET \
  --max_length $MAX_LENGTH \
  --lr $LR \
  --warmup_steps $WARMUP_STEPS \
  --batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $((8 / (BATCH_SIZE * N_GPUS))) \
  --num_epochs $NUM_EPOCHS \
  --log_every $LOG_EVERY \
  --eval_every $EVAL_EVERY \
  --save_every $SAVE_EVERY \
  --eval_samples $EVAL_SAMPLES \
  --eval_batch_size $EVAL_BATCH_SIZE \
  --max_new_tokens $MAX_NEW_TOKENS \
  --output_dir $DDP_OUT

echo ""
echo "============================================================"
echo "All experiments complete. Results in ./ckpt/ablate_${TIMESTAMP}/"
echo "============================================================"
