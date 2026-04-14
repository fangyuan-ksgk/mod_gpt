#!/bin/bash
# ===========================================================================
# V6 routing temperature ablation — Hyp 2
#
# Question: Does temperature sampling during V6 training help?
#   - t=None (argmax, deterministic routing)
#   - t=1.0  (multinomial sampling from softmax logits)
#
# Fixed (from prior best):
#   Qwen3-0.6B, layer 14, L=4, scale=0.5, slr=5e-2, code_position=first
#
# Sweep axes:
#   C_SIZE   ∈ {4, 32}                  (2)
#   temp     ∈ {None, 1.0}              (2)
#   dataset  ∈ {scienceqa, gsm8k}       (2)
#
# Total: 2 × 2 × 2 = 8 runs
#
# Usage: ./sweep_0414_v6_temp.sh [PART]
#   PART=1   → scienceqa
#   PART=2   → gsm8k
#   PART=all → both (default)
# ===========================================================================
set -euo pipefail

PART=${1:-all}

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4

MASTER_ADDR=127.0.0.1
BASE_PORT=29800
N_GPUS=4

# ---- Fixed config ----
MODEL="Qwen/Qwen3-0.6B"
MTAG="q06"
LR=1e-5
SLR="5e-2"
SCALE=0.5
L=4
LAYERS="14"
EPOCHS=1
BATCH_SIZE=2
GRAD_ACCUM=4
WARMUP=50
EVAL_SAMPLES=1319
EVAL_BATCH=128
NUM_LOG=5

TIMESTAMP=$(date +%Y%m%d_%H%M)
OUT_ROOT="./ckpt/v6_temp_${TIMESTAMP}"
mkdir -p "$OUT_ROOT"

# ---- Sweep axes ----
C_SIZES=(4 32)
# temp=0 means no flag (argmax); temp=1 means --routing_temperature 1.0
TEMPS=("0" "1.0")

JOB_IDX=0

run_one() {
  local C=$1 temp=$2 dataset=$3 dtag=$4

  JOB_IDX=$((JOB_IDX + 1))
  local gpu=$(( (JOB_IDX - 1) % N_GPUS ))
  local port=$((BASE_PORT + JOB_IDX))

  local temp_tag="t${temp}"
  local tag="${MTAG}_${dtag}_C${C}_L${L}_${temp_tag}"
  local out="${OUT_ROOT}/${tag}"

  local temp_flag=""
  if [ "$temp" != "0" ]; then
    temp_flag="--routing_temperature $temp"
  fi

  echo "  [GPU ${gpu}] ${tag}"

  CUDA_VISIBLE_DEVICES=$gpu torchrun \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$port \
    train_steer_pt.py \
    --mode v6 \
    --model_name $MODEL \
    --dataset $dataset \
    --num_epochs $EPOCHS \
    --lr $LR \
    --steer_lr $SLR \
    --warmup_steps $WARMUP \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --C_SIZE $C \
    --L $L \
    --scale $SCALE \
    --inject_layers $LAYERS \
    --eval_every 99999 \
    --save_every 99999 \
    --eval_samples $EVAL_SAMPLES \
    --eval_batch_size $EVAL_BATCH \
    --num_log_samples $NUM_LOG \
    --log_every 10 \
    --output_dir $out $temp_flag &

  if (( JOB_IDX % N_GPUS == 0 )); then wait; fi
}

echo ""
echo "============================================================"
echo "V6 temperature ablation | ${TIMESTAMP}"
echo "============================================================"

if [ "$PART" = "all" ] || [ "$PART" = "1" ]; then
  echo ""
  echo "--- Part 1: ScienceQA ---"
  for C in "${C_SIZES[@]}"; do
    for temp in "${TEMPS[@]}"; do
      run_one $C $temp scienceqa sci
    done
  done
  wait
  echo "Part 1 complete."
fi

if [ "$PART" = "all" ] || [ "$PART" = "2" ]; then
  echo ""
  echo "--- Part 2: GSM8K ---"
  for C in "${C_SIZES[@]}"; do
    for temp in "${TEMPS[@]}"; do
      run_one $C $temp gsm8k gsm
    done
  done
  wait
  echo "Part 2 complete."
fi

echo ""
echo "============================================================"
echo "V6 temp ablation done. Results in ${OUT_ROOT}/"
echo "============================================================"
echo ""
echo "To gather results:"
echo "  grep 'Final accuracy' ${OUT_ROOT}/*/train.log"
