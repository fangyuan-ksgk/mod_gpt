#!/bin/bash
set -e

# --- nvidia pod specifics ------
DUMMY_CONFIG_PATH="./dummy_tuner_config.txt"
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

LR=1e-5
WARMUP_STEPS=50
BATCH_SIZE=2
NUM_EPOCHS=3
MAX_LENGTH=512

LOG_EVERY=10
EVAL_EVERY=99999
SAVE_EVERY=99999
EVAL_BATCH_SIZE=64
MAX_NEW_TOKENS=256

# dataset: (gsm8k, scienceqa, math, arc)
# model: (Qwen3-4B, Qwen3-8B, Qwen3-14B)
# sft baseline (no SoRL, just LoRA)
DS_MMLU="mmlu"
DS_CSQA="commonsenseqa"

eval_samples_for_dataset() {
  local dataset=$1
  case "$dataset" in
    gsm8k) echo 1319 ;;
    scienceqa) echo 2224 ;;
    math) echo 5000 ;;
    arc) echo 1172 ;;
    mmlu) echo 2000 ;;
    commonsenseqa) echo 1221 ;;
    *) echo 1000 ;;
  esac
}

TIMESTAMP=$(date +%Y%m%d_%H%M)
EXP_IDX=0

# Model + dataset shorthands
M4="Qwen/Qwen3-4B"
M8="Qwen/Qwen3-8B"
M14="Qwen/Qwen3-14B"

DS_GSM="gsm8k"
DS_SCI="scienceqa"
DS_MATH="math"
DS_ARC="arc"

# ---- Parallel scheduling: 4 x H100 — 1 run/GPU ----
# Usage: run_bg <tag> <model> <dataset> [sorl flags...]
run_bg() {
  EXP_IDX=$((EXP_IDX + 1))
  local idx=$EXP_IDX
  local gpu=$(( (idx - 1) % N_GPUS ))
  local port=$((BASE_PORT + idx))
  local tag=$1; shift
  local model=$1; shift
  local dataset=$1; shift
  local grad_accum=$((8 / BATCH_SIZE))
  local eval_samples=$(eval_samples_for_dataset "$dataset")
  local output_dir="./ckpt/sweep_${TIMESTAMP}/exp${idx}_${tag}"

  echo "  Exp ${idx}: ${tag}  model=$(basename $model)  dataset=${dataset}  [GPU=${gpu}]"

  CUDA_VISIBLE_DEVICES=$gpu torchrun \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$port \
    train_sorl_post.py \
    --model_name $model \
    --dataset $dataset \
    --max_length $MAX_LENGTH \
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
    --max_new_tokens $MAX_NEW_TOKENS \
    --output_dir $output_dir \
    --use_lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    "$@" &
}

# ============================================================================
# Batch 1: SFT Baseline on Qwen3-4B
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 1: SFT Baseline on Qwen3-4B (${TIMESTAMP})"
echo "============================================================"

run_bg "sft_gsm_4B"   $M4  $DS_GSM
run_bg "sft_sci_4B"   $M4  $DS_SCI
run_bg "sft_math_4B"  $M4  $DS_MATH
run_bg "sft_arc_4B"   $M4  $DS_ARC
run_bg "sft_mmlu_4B"  $M4  $DS_MMLU
run_bg "sft_csqa_4B"  $M4  $DS_CSQA
wait

# ============================================================================
# Batch 2: SFT Baseline on Qwen3-8B
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 2: SFT Baseline on Qwen3-8B (${TIMESTAMP})"
echo "============================================================"

run_bg "sft_gsm_8B"   $M8  $DS_GSM
run_bg "sft_sci_8B"   $M8  $DS_SCI
run_bg "sft_math_8B"  $M8  $DS_MATH
run_bg "sft_arc_8B"   $M8  $DS_ARC
run_bg "sft_mmlu_8B"  $M8  $DS_MMLU
run_bg "sft_csqa_8B"  $M8  $DS_CSQA
wait

# ============================================================================
# Batch 3: SFT Baseline on Qwen3-14B
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 3: SFT Baseline on Qwen3-14B (${TIMESTAMP})"
echo "============================================================"

run_bg "sft_gsm_14B"   $M14  $DS_GSM
run_bg "sft_sci_14B"   $M14  $DS_SCI
run_bg "sft_math_14B"  $M14  $DS_MATH
run_bg "sft_arc_14B"   $M14  $DS_ARC
run_bg "sft_mmlu_14B"  $M14  $DS_MMLU
run_bg "sft_csqa_14B"  $M14  $DS_CSQA
wait

echo ""
echo "============================================================"
echo "All experiments complete. Results in ./ckpt/sweep_${TIMESTAMP}/"
echo "============================================================"