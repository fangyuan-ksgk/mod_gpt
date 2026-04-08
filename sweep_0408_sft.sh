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

eval_samples_for_dataset() {
  local dataset=$1
  case "$dataset" in
    gsm8k) echo 1319 ;;
    scienceqa) echo 2224 ;;
    arc) echo 1172 ;;
    mmlu) echo 2000 ;;
    commonsenseqa) echo 1221 ;;
    deepmind_code_contests) echo 282 ;;
    *) echo 1000 ;;
  esac
}

max_length_for_dataset() {
  local dataset=$1
  case "$dataset" in
    deepmind_code_contests) echo 2048 ;;
    *) echo 512 ;;
  esac
}

max_new_tokens_for_dataset() {
  local dataset=$1
  case "$dataset" in
    deepmind_code_contests) echo 1024 ;;
    *) echo 256 ;;
  esac
}

# ---- Parallel scheduling: 4 x H100 — 1 run/GPU ----
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
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    "$@" &
}



# (V). How sensitive is SFT to different configurations?
#
# Axes:
#   A. Effective batch size: {1, 2, 4, 8, 16, 32}  (0.6B, GSM, ep=1)
#   B. Num epochs:           {1, 2, 3, 4, 5, 6}     (0.6B, GSM, bs=8)
#   C. Model scale:          {1.7B, 4B}              (GSM, ep={1,3}, bs={2,8,32})
#
# 5 batches × 4 GPUs (≤4 runs/batch), 19 experiments total.
# All runs use LoRA (r=16, α=32) on GSM8K.
# argparse uses last value, so extra --batch_size/--num_epochs override run_bg defaults.

# ===========================================================================
# Batch 1 — 0.6B batch size sweep pt1: eff_bs={1,2,4,8}, ep=1
# ===========================================================================
echo ""
echo "Batch 1: 0.6B bs sweep — eff_bs=1,2,4,8"

run_bg "06b_gsm_bs1"  $M06 $DS_GSM \
  --batch_size 1 --gradient_accumulation_steps 1

run_bg "06b_gsm_bs2"  $M06 $DS_GSM \
  --batch_size 2 --gradient_accumulation_steps 1

run_bg "06b_gsm_bs4"  $M06 $DS_GSM \
  --batch_size 2 --gradient_accumulation_steps 2

run_bg "06b_gsm_bs8"  $M06 $DS_GSM
# bs=8 is the default (bs=2 × ga=4)

wait

# ===========================================================================
# Batch 2 — 0.6B bs sweep pt2 + epoch sweep pt1
# ===========================================================================
echo ""
echo "Batch 2: 0.6B bs=16,32 + ep=2,3"

run_bg "06b_gsm_bs16" $M06 $DS_GSM \
  --batch_size 2 --gradient_accumulation_steps 8

run_bg "06b_gsm_bs32" $M06 $DS_GSM \
  --batch_size 2 --gradient_accumulation_steps 16

run_bg "06b_gsm_ep2"  $M06 $DS_GSM \
  --num_epochs 2

run_bg "06b_gsm_ep3"  $M06 $DS_GSM \
  --num_epochs 3

wait

# ===========================================================================
# Batch 3 — 0.6B epoch sweep pt2 + 1.7B anchor
# ===========================================================================
echo ""
echo "Batch 3: 0.6B ep=4,5,6 + 1.7B ep=1"

run_bg "06b_gsm_ep4"  $M06 $DS_GSM \
  --num_epochs 4

run_bg "06b_gsm_ep5"  $M06 $DS_GSM \
  --num_epochs 5

run_bg "06b_gsm_ep6"  $M06 $DS_GSM \
  --num_epochs 6

run_bg "17b_gsm_bs8_ep1" $M16 $DS_GSM

wait

# ===========================================================================
# Batch 4 — 1.7B sweep + 4B anchor
# ===========================================================================
echo ""
echo "Batch 4: 1.7B ep=3 + bs=2,32 + 4B ep=1"

run_bg "17b_gsm_ep3"    $M16 $DS_GSM \
  --num_epochs 3

run_bg "17b_gsm_bs2"    $M16 $DS_GSM \
  --batch_size 2 --gradient_accumulation_steps 1

run_bg "17b_gsm_bs32"   $M16 $DS_GSM \
  --batch_size 2 --gradient_accumulation_steps 16

run_bg "4b_gsm_bs8_ep1" $M4 $DS_GSM

wait

# ===========================================================================
# Batch 5 — 4B sweep
# ===========================================================================
echo ""
echo "Batch 5: 4B ep=3 + bs=2,32"

run_bg "4b_gsm_ep3"  $M4 $DS_GSM \
  --num_epochs 3

run_bg "4b_gsm_bs2"  $M4 $DS_GSM \
  --batch_size 2 --gradient_accumulation_steps 1

run_bg "4b_gsm_bs32" $M4 $DS_GSM \
  --batch_size 2 --gradient_accumulation_steps 16

wait

echo ""
echo "============================================================"
echo "All done. 5 batches, 19 experiments. Results in ./ckpt/sweep_${TIMESTAMP}/"
echo "============================================================"
