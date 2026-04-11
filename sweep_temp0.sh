#!/bin/bash
set -e

# Temperature=0.0 ablation on q06/q17 + GSM8K/ScienceQA
# Testing whether greedy training (t=0.0) vs stochastic training (t=1.0 default) matters

# --- nvidia pod  specifics ------
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
NUM_EPOCHS=1

LOG_EVERY=10
EVAL_EVERY=99999
SAVE_EVERY=99999
EVAL_SAMPLES=1300
EVAL_BATCH_SIZE=256

# ============================================================================================
# Temperature=0.0 sweep on q06/q17 + GSM8K/ScienceQA
# Inherits all hyper-params from sweep_0411.sh (v7, LoRA, similar_magnitude, V=128, pfx=4, iter=2)
# Only change: --temperature 0.0 (greedy training vs default t=1.0 stochastic)
# ============================================================================================

TIMESTAMP=$(date +%Y%m%d_%H%M)
EXP_IDX=0

# Model shorthands
M06="Qwen/Qwen3-0.6B"
M17="Qwen/Qwen3-1.7B"

# Dataset shorthands
DS_GSM="gsm8k"
DS_SCI="scienceqa"

# Common SoRL flags (inherited from sweep_0411.sh)
LORA="--use_lora --lora_rank 16 --lora_alpha 32"
BASE="--use_v7 --abs_routing_mode similar_magnitude \
  --prefix_abs --abs_prefix_max 4 --K 4 \
  --max_iterations 2 --eval_K 4 $LORA"

SORL="$BASE --abstract_vocab_size 128 --emb_lr_mult 1.0"

eval_samples_for_dataset() {
  local dataset=$1
  case "$dataset" in
    gsm8k)                  echo 1319 ;;
    scienceqa)              echo 2224 ;;
    *)                      echo 1000 ;;
  esac
}

run_bg() {
  EXP_IDX=$((EXP_IDX + 1))
  local idx=$EXP_IDX
  local gpu=$(( (idx - 1) % N_GPUS ))
  local port=$((BASE_PORT + idx))
  local tag=$1; shift
  local model=$1; shift
  local dataset=$1; shift
  local grad_accum=$((8 / BATCH_SIZE))
  local output_dir="./ckpt/sweep_${TIMESTAMP}/exp${idx}_${tag}"
  local eval_samples; eval_samples=$(eval_samples_for_dataset "$dataset")

  echo "  Exp ${idx}: ${tag}  model=$(basename $model)  dataset=${dataset}  [GPU=${gpu}]"

  CUDA_VISIBLE_DEVICES=$gpu torchrun \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$port \
    train_sorl_post.py \
    --model_name $model \
    --dataset $dataset \
    --lr $LR \
    --warmup_steps $WARMUP_STEPS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $grad_accum \
    --num_epochs $NUM_EPOCHS \
    --log_every $LOG_EVERY \
    --eval_every $EVAL_EVERY \
    --save_every $SAVE_EVERY \
    --eval_samples $eval_samples \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --output_dir $output_dir \
    --untie_embedding \
    --temperature 0.0 \
    "$@" &
}

echo "=== Temperature=0.0 Sweep === $(date)"

# =============================================================================
# Temperature=0.0 on q06/q17 + GSM8K/ScienceQA
# 4 experiments total: 2 models × 2 datasets
# All other params identical to sweep_0411.sh baseline
# =============================================================================

# ---- GSM8K ----
echo "GSM8K: t=0.0 × {q06, q17}"
run_bg "gsm_t0_q06"  $M06 $DS_GSM $SORL --eval_batch_size 8
run_bg "gsm_t0_q17"  $M17 $DS_GSM $SORL --eval_batch_size 8
wait

# ---- ScienceQA ----
echo "ScienceQA: t=0.0 × {q06, q17}"
run_bg "sci_t0_q06"  $M06 $DS_SCI $SORL --eval_batch_size 8
run_bg "sci_t0_q17"  $M17 $DS_SCI $SORL --eval_batch_size 8
wait


echo ""
echo "=== Temperature=0.0 experiments complete. $(date) ==="
