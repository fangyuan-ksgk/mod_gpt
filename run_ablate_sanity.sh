#!/bin/bash
# SoRL Ablation Experiments
#
# Usage:
#   bash run_ablate_sanity.sh
#
# Each run_experiment call takes: GPUS  TAG  [extra args...]
# GPUS is a CUDA_VISIBLE_DEVICES string, e.g. "0", "1", "0,1"

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

# ============================================================================
# Shared configuration
# ============================================================================
MASTER_ADDR=127.0.0.1
BASE_PORT=29501

MODEL_NAME="Qwen/Qwen3-0.6B"
DATASET="gsm8k"
MAX_LENGTH=512

LR=1e-5
WARMUP_STEPS=50
BATCH_SIZE=2
NUM_EPOCHS=3

LOG_EVERY=10
EVAL_EVERY=500
SAVE_EVERY=500
EVAL_SAMPLES=50
MAX_NEW_TOKENS=256

TIMESTAMP=$(date +%Y%m%d_%H%M)
EXP_IDX=0

# ---- run_experiment GPUS TAG [extra flags...] ----
# GPUS: CUDA_VISIBLE_DEVICES string, e.g. "0", "0,1"
# n_gpus is inferred from the comma count
run_experiment() {
  local gpus=$1; shift
  local tag=$1; shift

  # Count GPUs from the comma-separated string
  local n_gpus=$(echo "$gpus" | awk -F',' '{print NF}')

  EXP_IDX=$((EXP_IDX + 1))
  local port=$((BASE_PORT + EXP_IDX))
  local grad_accum=$((8 / (BATCH_SIZE * n_gpus)))
  local output_dir="./ckpt/ablate_${TIMESTAMP}/exp${EXP_IDX}_${tag}"

  echo ""
  echo "============================================================"
  echo "Exp ${EXP_IDX}: ${tag}  [CUDA_VISIBLE_DEVICES=${gpus}]"
  echo "  GPUs=${n_gpus}  BS=${BATCH_SIZE}x${grad_accum}x${n_gpus}=8  extra: $@"
  echo "  Output: ${output_dir}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES=$gpus torchrun \
    --nproc_per_node=$n_gpus \
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
    --max_new_tokens $MAX_NEW_TOKENS \
    --output_dir $output_dir \
    "$@"

  echo "  -> Done: ${tag}"
}

# ============================================================================
# Exp 1: 1 GPU, BS=8, aux=0 (baseline)
# ============================================================================
run_experiment 0 "1gpu_bs8_aux0"

# ============================================================================
# Exp 2: 2 GPU, BS=8, aux=0 (DDP validation)
# ============================================================================
run_experiment 0,1 "2gpu_bs8_aux0"

# ============================================================================
# Exp 3: 1 GPU, BS=8, info_gain sweep (1.0 -> 3.0 -> 5.0 -> 7.0 -> 9.0)
# ============================================================================
for info in 1.0 3.0 5.0 7.0 9.0; do
  run_experiment 0 "1gpu_bs8_info${info}" \
    --alpha_info_gain $info
done

# ============================================================================
# Exp 4: 1 GPU, BS=8, info_gain sweep + abs=0.5
# ============================================================================
for info in 1.0 3.0 5.0 7.0 9.0; do
  run_experiment 0 "1gpu_bs8_info${info}_abs0.5" \
    --alpha_info_gain $info --alpha_abs 0.5
done

# ============================================================================
# Exp 5: 1 GPU, BS=8, info=9, abs sweep (0.5 -> 1.0 -> 1.5 -> 2.0)
# ============================================================================
for abs_w in 0.5 1.0 1.5 2.0; do
  run_experiment 0 "1gpu_bs8_info9_abs${abs_w}" \
    --alpha_info_gain 9.0 --alpha_abs $abs_w
done

# ============================================================================
# Exp 6: 1 GPU, BS=8, info=9, abs=0.5, zipf sweep (0.5 -> 1.0 -> 1.5 -> 2.0)
# ============================================================================
for zipf in 0.5 1.0 1.5 2.0; do
  run_experiment 0 "1gpu_bs8_info9_abs0.5_zipf${zipf}" \
    --alpha_info_gain 9.0 --alpha_abs 0.5 --alpha_soft_zipf $zipf
done

echo ""
echo "============================================================"
echo "All experiments complete. Results in ./ckpt/ablate_${TIMESTAMP}/"
echo "============================================================"
