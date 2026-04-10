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
N_GPUS=2

LR=1e-5
WARMUP_STEPS=50
BATCH_SIZE=2
NUM_EPOCHS=1
MAX_LENGTH=512

LOG_EVERY=10
LOG_SAMPLES_EVERY=999999
NUM_LOG_SAMPLES=3
EVAL_EVERY=99999
SAVE_EVERY=99999
EVAL_BATCH_SIZE=64
MAX_NEW_TOKENS=256

TIMESTAMP=$(date +%Y%m%d_%H%M)
EXP_IDX=0

# Model + dataset shorthands
M06="Qwen/Qwen3-0.6B"
M16="Qwen/Qwen3-1.7B"
M4="Qwen/Qwen3-4B"
ML1="meta-llama/Llama-3.2-1B"
ML3="meta-llama/Llama-3.2-3B"

M8="Qwen/Qwen3-8B"
DS_GSM="gsm8k"
DS_SCI="scienceqa"
DS_ARC="arc"
DS_MMLU="mmlu"
DS_CSQA="commonsenseqa"

DS_MATH="math"
DS_CODE="deepmind_code_contests"

DS_BOOLQ="boolq"
DS_OBQA="openbookqa"
DS_AQUA="aqua"
DS_HPQA="hotpotqa"

eval_samples_for_dataset() {
  local dataset=$1
  case "$dataset" in
    gsm8k) echo 1319 ;;
    scienceqa) echo 2224 ;;
    arc) echo 1172 ;;
    mmlu) echo 2000 ;;
    commonsenseqa) echo 1221 ;;
    boolq) echo 3270 ;;
    openbookqa) echo 1000 ;;
    aqua) echo 254 ;;
    hotpotqa) echo 4000 ;;
    deepmind_code_contests) echo 282 ;;
    *) echo 1000 ;;
  esac
}

max_length_for_dataset() {
  local dataset=$1
  case "$dataset" in
    boolq) echo 1024 ;;
    openbookqa) echo 768 ;;
    aqua) echo 1024 ;;
    hotpotqa) echo 512 ;;
    deepmind_code_contests) echo 2048 ;;
    *) echo 512 ;;
  esac
}

max_new_tokens_for_dataset() {
  local dataset=$1
  case "$dataset" in
    boolq) echo 32 ;;
    openbookqa) echo 128 ;;
    aqua) echo 768 ;;
    hotpotqa) echo 64 ;;
    deepmind_code_contests) echo 1024 ;;
    *) echo 256 ;;
  esac
}

# ---- Parallel scheduling: 2 GPUs — 2 runs/batch ----
# Usage: run_bg <tag> <model> <dataset>
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
  local max_len=$(max_length_for_dataset "$dataset")
  local max_new=$(max_new_tokens_for_dataset "$dataset")
  local output_dir="./ckpt/sweep_${TIMESTAMP}/exp${idx}_${tag}"

  echo "  Exp ${idx}: ${tag}  model=$(basename $model)  dataset=${dataset}  [GPU=${gpu}]  max_len=${max_len}"

  CUDA_VISIBLE_DEVICES=$gpu torchrun \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$port \
    train_sft_pt.py \
    --model_name $model \
    --dataset $dataset \
    --max_length $max_len \
    --lr $LR \
    --warmup_steps $WARMUP_STEPS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $grad_accum \
    --num_epochs $NUM_EPOCHS \
    --log_every $LOG_EVERY \
    --eval_every $EVAL_EVERY \
    --save_every $SAVE_EVERY \
    --eval_samples $eval_samples \
    --log_samples_every $LOG_SAMPLES_EVERY \
    --num_log_samples $NUM_LOG_SAMPLES \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --max_new_tokens $max_new \
    --output_dir $output_dir \
    "$@" &
}

# --use_lora \
# --lora_r 16 \
# --lora_alpha 32 \



# ============================================================================================
# SFT sensitivity sweep — 2 GPUs, 2 runs/batch
#
# All 5 models × 3 axes, done one axis at a time:
#   A. Batch size: bs=8            × ALL models × GSM8K, ep=1
#   B. Num epochs: {1, 4}          × ALL models × GSM8K, bs=8
#   C. Dataset breadth: GSM8K, SciQA, ARC, MMLU, CSQA, BoolQ, OpenBookQA, AQuA, HotpotQA
#                             × ALL models, bs=8, ep=1
#
# Models: Qwen 0.6B, 1.7B, 4B, Llama 1B, Llama 3B
# No LoRA — full fine-tune.
# ============================================================================================

ALL_MODELS=("$M06" "$M16" "$ML1" "$ML3" "$M4")
ALL_TAGS=("06b" "17b" "l1b" "l3b" "4b")

# ═══════════════════════════════════════════════════════════════════════════
# A. Batch size: bs=8 × ALL models (GSM8K, ep=1)
#    5 runs = 5 models × 1 bs value
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "A: batch size sweep — all models × bs=8 on GSM8K"

for i in "${!ALL_MODELS[@]}"; do
  m="${ALL_MODELS[$i]}"
  t="${ALL_TAGS[$i]}"
  run_bg "${t}_gsm_bs8"  $m $DS_GSM  # default: bs=2 × ga=4 = 8

  if (( (i + 1) % N_GPUS == 0 )); then
    wait
  fi
done
wait

# ═══════════════════════════════════════════════════════════════════════════
# B. Epoch sweep: ep={1, 4} × ALL models (GSM8K, bs=8)
#    ep=1 already covered in A (bs=8). Only run ep=4.
#    5 runs = 5 models × 1 new ep value
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "B: epoch sweep — all models × ep=4 on GSM8K"

for i in "${!ALL_MODELS[@]}"; do
  m="${ALL_MODELS[$i]}"
  t="${ALL_TAGS[$i]}"
  run_bg "${t}_gsm_ep4" $m $DS_GSM --num_epochs 4

  # 2 at a time
  if (( (i + 1) % N_GPUS == 0 )); then
    wait
  fi
done
wait

# ═══════════════════════════════════════════════════════════════════════════
# C. Dataset breadth: 9 datasets × ALL models (bs=8, ep=1)
#    GSM8K + SciQA included for full-FT SFT baseline (no LoRA).
#    45 runs = 5 models × 9 datasets
# ═══════════════════════════════════════════════════════════════════════════
SWEEP_DS=("$DS_GSM" "$DS_SCI" "$DS_ARC" "$DS_MMLU" "$DS_CSQA" "$DS_BOOLQ" "$DS_OBQA" "$DS_AQUA" "$DS_HPQA")

echo ""
echo "C: dataset breadth — all models × 9 datasets"

for ds in "${SWEEP_DS[@]}"; do
  echo "  --- dataset: $ds ---"
  for i in "${!ALL_MODELS[@]}"; do
    m="${ALL_MODELS[$i]}"
    t="${ALL_TAGS[$i]}"
    run_bg "${t}_${ds}" $m $ds

    if (( (i + 1) % N_GPUS == 0 )); then
      wait
    fi
  done
  wait
done

echo ""
echo "============================================================"
echo "All done."
echo "  A (batch size):    5 runs  (5 models × bs=8)"
echo "  B (epochs):        5 runs  (5 models × ep=4; ep=1 from A)"
echo "  C (datasets):     45 runs  (5 models × 9 datasets)"
echo "  Total:            55 experiments"
echo "  Results in ./ckpt/sweep_${TIMESTAMP}/"
echo "============================================================"
