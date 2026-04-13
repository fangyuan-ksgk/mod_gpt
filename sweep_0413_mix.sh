#!/bin/bash
# ===========================================================================
# Mixed-dataset steering: does C>1 matter when training on diverse data?
#
# Train on: gsm8k,scienceqa,commonsenseqa
# Eval on: same mix (test splits)
# Sweep: C ∈ {1, 4, 32}, layers, slr
# Model: Qwen3-0.6B (28 layers)
#
# Hypothesis: single-dataset tasks are too simple for multiple codes,
#             but a mix should force C>1 specialization.
# ===========================================================================
set -euo pipefail

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

MASTER_ADDR=127.0.0.1
BASE_PORT=29600
N_GPUS=4

MODEL="Qwen/Qwen3-0.6B"
DATASET="gsm8k,scienceqa,commonsenseqa"
LR=1e-5
SCALE=0.5
EPOCHS=1
BATCH_SIZE=2
GRAD_ACCUM=4
WARMUP=50
EVAL_SAMPLES=1000
EVAL_BATCH=128
NUM_LOG=5
SLR="1e-1"
L=4

TIMESTAMP=$(date +%Y%m%d_%H%M)
OUT_ROOT="./ckpt/mix3_q06_${TIMESTAMP}"
mkdir -p "$OUT_ROOT"

JOB_IDX=0

run_exp() {
  local tag=$1
  local C=$2
  local layers=$3
  local sc=${4:-$SCALE}

  JOB_IDX=$((JOB_IDX + 1))
  local gpu=$(( (JOB_IDX - 1) % N_GPUS ))
  local port=$((BASE_PORT + JOB_IDX))
  local out="${OUT_ROOT}/${tag}"

  echo "  [GPU ${gpu}] ${tag}"

  CUDA_VISIBLE_DEVICES=$gpu torchrun \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$port \
    train_steer_pt.py \
    --mode v6 \
    --model_name $MODEL \
    --dataset "$DATASET" \
    --num_epochs $EPOCHS \
    --lr $LR \
    --steer_lr $SLR \
    --warmup_steps $WARMUP \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --C_SIZE $C \
    --L $L \
    --scale $sc \
    --inject_layers $layers \
    --code_position first \
    --routing_mode similar_magnitude \
    --eval_every 99999 \
    --save_every 99999 \
    --eval_samples $EVAL_SAMPLES \
    --eval_batch_size $EVAL_BATCH \
    --num_log_samples $NUM_LOG \
    --log_every 10 \
    --output_dir $out &

  if (( JOB_IDX % N_GPUS == 0 )); then wait; fi
}

echo "=== Mixed-dataset C sweep: Qwen3-0.6B ==="
echo "Train/eval on: $DATASET"
echo "Output: $OUT_ROOT"

# SFT baseline (scale=0, steering disabled)
run_exp "sft"       1  "14"  0

# C=1 baseline
run_exp "C1_mid"    1  "14"
run_exp "C1_ml"     1  "14,24"

# C=4
run_exp "C4_mid"    4  "14"
run_exp "C4_ml"     4  "14,24"

wait

# C=32
run_exp "C32_mid"   32 "14"
run_exp "C32_ml"    32 "14,24"

# C=64
run_exp "C64_mid"   64 "14"
run_exp "C64_ml"    64 "14,24"

wait
echo "=== Done ==="
echo "grep 'Final accuracy' ${OUT_ROOT}/*/train.log | sort -t: -k2 -rn"
