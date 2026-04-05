#!/bin/bash
# SFT Baseline Post-Training — 4 runs in parallel (1 per GPU)
#
# Matches sweep_20260403_1539 config: max_length=512, lr=1e-5, epochs=3, effective_batch=8
#
# Runs:
#   GPU 0: Qwen3-0.6B  GSM8K      (eval=1300)
#   GPU 1: Qwen3-1.7B  GSM8K      (eval=1300)
#   GPU 2: Qwen3-0.6B  ScienceQA  (eval=1270)
#   GPU 3: Qwen3-1.7B  ScienceQA  (eval=1270)
#
# Usage:
#   bash run_sft_pt.sh

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
# Shared configuration
# ============================================================================
MASTER_ADDR=127.0.0.1
BASE_PORT=29601

MAX_LENGTH=512
LR=1e-5
WARMUP_STEPS=50
BATCH_SIZE=2
GRAD_ACCUM=4       # effective batch = 2 * 4 = 8 (single GPU, no DDP)
NUM_EPOCHS=3

LOG_EVERY=10
EVAL_EVERY=99999
SAVE_EVERY=99999
LOG_SAMPLES_EVERY=99999
NUM_LOG_SAMPLES=3
MAX_NEW_TOKENS=256
EVAL_BATCH_SIZE=64

TIMESTAMP=$(date +%Y%m%d_%H%M)
EXP_DIR="./ckpt/sft_pt_${TIMESTAMP}"

# ============================================================================
# Helper: launch one SFT run on a single GPU
# ============================================================================
run_bg() {
  local gpu=$1
  local model=$2
  local dataset=$3
  local eval_samples=$4
  local tag=$5
  local port=$((BASE_PORT + gpu))
  local output_dir="${EXP_DIR}/${tag}"

  echo "  ${tag}: model=${model} dataset=${dataset}  [GPU=${gpu}]  port=${port}"

  CUDA_VISIBLE_DEVICES=$gpu torchrun \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$port \
    train_sft_pt.py \
    --model_name $model \
    --dataset $dataset \
    --max_length $MAX_LENGTH \
    --lr $LR \
    --warmup_steps $WARMUP_STEPS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --num_epochs $NUM_EPOCHS \
    --log_every $LOG_EVERY \
    --eval_every $EVAL_EVERY \
    --save_every $SAVE_EVERY \
    --eval_samples $eval_samples \
    --log_samples_every $LOG_SAMPLES_EVERY \
    --num_log_samples $NUM_LOG_SAMPLES \
    --max_new_tokens $MAX_NEW_TOKENS \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --output_dir $output_dir &
}

# ============================================================================
# Launch all 4 runs in parallel
# ============================================================================
echo ""
echo "============================================================"
echo "SFT Baselines — ${TIMESTAMP}"
echo "============================================================"

run_bg 0 "Qwen/Qwen3-0.6B" "gsm8k"     1300 "06b_gsm8k"
run_bg 1 "Qwen/Qwen3-1.7B" "gsm8k"     1300 "17b_gsm8k"
run_bg 2 "Qwen/Qwen3-0.6B" "scienceqa"  1270 "06b_sciqa"
run_bg 3 "Qwen/Qwen3-1.7B" "scienceqa"  1270 "17b_sciqa"

echo "  4 runs launched. Waiting..."
wait
echo "Done! Results in ${EXP_DIR}"