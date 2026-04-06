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
EVAL_EVERY=99999
SAVE_EVERY=99999
EVAL_BATCH_SIZE=64
MAX_NEW_TOKENS=256

REG_WEIGHT=1.0
PATCH_SIZE=4

TIMESTAMP=$(date +%Y%m%d_%H%M)
EXP_IDX=0

# Model + dataset shorthands
M16="Qwen/Qwen3-1.7B"
M4="Qwen/Qwen3-4B"
M8="Qwen/Qwen3-8B"
ML1="meta-llama/Llama-3.2-1B"
ML3="meta-llama/Llama-3.2-3B"

DS_GSM="gsm8k"
DS_SCI="scienceqa"
DS_MATH="math"
DS_ARC="arc"
DS_MMLU="mmlu"
DS_CSQA="commonsenseqa"
DS_CODE="deepmind_code_contests"

# Regularization variants: SFT (mbe_val tracked) vs MBE-regularized
R_NONE="--reg_type none"
R_MBE="--reg_type mbe   --reg_weight $REG_WEIGHT"

eval_samples_for_dataset() {
  local dataset=$1
  case "$dataset" in
    gsm8k) echo 1319 ;;
    scienceqa) echo 2224 ;;
    math) echo 2500 ;;
    arc) echo 1172 ;;
    mmlu) echo 2000 ;;
    commonsenseqa) echo 1221 ;;
    deepmind_code_contests) echo 282 ;;
    *) echo 1000 ;;
  esac
}

# ---- Parallel scheduling: 4 x GPU — 1 run/GPU ----
# Usage: run_bg <tag> <model> <dataset> [reg flags...]
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
    train_iblm_post.py \
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
    --patch_size $PATCH_SIZE \
    "$@" &
}

# Per model: 7 datasets × 2 variants = 14 runs
# Packed 2 datasets per wait-block → 4 jobs/block, all 4 GPUs used

# ============================================================================
# Batch 1: IBLM on Qwen3-1.7B
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 1: IBLM on Qwen3-1.7B (${TIMESTAMP})"
echo "============================================================"

run_bg "none_gsm_1.7B"  $M16 $DS_GSM  $R_NONE
run_bg "mbe_gsm_1.7B"   $M16 $DS_GSM  $R_MBE
run_bg "none_sci_1.7B"  $M16 $DS_SCI  $R_NONE
run_bg "mbe_sci_1.7B"   $M16 $DS_SCI  $R_MBE
wait
run_bg "none_math_1.7B" $M16 $DS_MATH $R_NONE
run_bg "mbe_math_1.7B"  $M16 $DS_MATH $R_MBE
run_bg "none_arc_1.7B"  $M16 $DS_ARC  $R_NONE
run_bg "mbe_arc_1.7B"   $M16 $DS_ARC  $R_MBE
wait
run_bg "none_mmlu_1.7B" $M16 $DS_MMLU $R_NONE
run_bg "mbe_mmlu_1.7B"  $M16 $DS_MMLU $R_MBE
run_bg "none_csqa_1.7B" $M16 $DS_CSQA $R_NONE
run_bg "mbe_csqa_1.7B"  $M16 $DS_CSQA $R_MBE
wait
run_bg "none_code_1.7B" $M16 $DS_CODE $R_NONE
run_bg "mbe_code_1.7B"  $M16 $DS_CODE $R_MBE
wait

# ============================================================================
# Batch 2: IBLM on Llama-3.2-1B
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 2: IBLM on Llama-3.2-1B (${TIMESTAMP})"
echo "============================================================"

run_bg "none_gsm_L1"  $ML1 $DS_GSM  $R_NONE
run_bg "mbe_gsm_L1"   $ML1 $DS_GSM  $R_MBE
run_bg "none_sci_L1"  $ML1 $DS_SCI  $R_NONE
run_bg "mbe_sci_L1"   $ML1 $DS_SCI  $R_MBE
wait
run_bg "none_math_L1" $ML1 $DS_MATH $R_NONE
run_bg "mbe_math_L1"  $ML1 $DS_MATH $R_MBE
run_bg "none_arc_L1"  $ML1 $DS_ARC  $R_NONE
run_bg "mbe_arc_L1"   $ML1 $DS_ARC  $R_MBE
wait
run_bg "none_mmlu_L1" $ML1 $DS_MMLU $R_NONE
run_bg "mbe_mmlu_L1"  $ML1 $DS_MMLU $R_MBE
run_bg "none_csqa_L1" $ML1 $DS_CSQA $R_NONE
run_bg "mbe_csqa_L1"  $ML1 $DS_CSQA $R_MBE
wait
run_bg "none_code_L1" $ML1 $DS_CODE $R_NONE
run_bg "mbe_code_L1"  $ML1 $DS_CODE $R_MBE
wait

echo ""
echo "============================================================"
echo "All experiments complete. Results in ./ckpt/sweep_${TIMESTAMP}/"
echo "============================================================"
