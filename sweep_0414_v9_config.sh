#!/bin/bash
# ===========================================================================
# V9 config sweep — find optimal hyperparams for search-based steering
#
# Fixed (from v6/v9 comparison):
#   - mode=v9, detach_routing=True (clear winner)
#   - Qwen3-0.6B, layer=14, scale=0.5, lr=1e-5, 1 epoch
#
# Sweep axes:
#   C_SIZE      ∈ {1, 4, 32}             (3)
#   L           ∈ {2, 4, 8}              (3)
#   slr         ∈ {5e-2, 1e-1}           (2)
#   alpha_zipf  ∈ {0.01, 0.1}            (2)
#   alpha_abs   ∈ {0.5, 1.0}             (2)
#   num_rollouts∈ {2, 4, 8}              (3)
#
# Per dataset: 3 × 3 × 2 × 2 × 2 × 3 = 216 runs
# Total: 216 × 2 datasets = 432 runs
#
# Parts:
#   1  scienceqa
#   2  gsm8k
#
# Usage: ./sweep_0414_v9_config.sh <PART>
#   PART=1    → scienceqa
#   PART=2    → gsm8k
#   PART=all  → both (default)
# ===========================================================================
set -euo pipefail

PART=${1:-all}

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4

MASTER_ADDR=127.0.0.1
BASE_PORT=29900
N_GPUS=4

# ---- Fixed config ----
MODEL="Qwen/Qwen3-0.6B"
MTAG="q06"
LR=1e-5
SCALE=0.5
LAYERS="14"
EPOCHS=1
BATCH_SIZE=2
GRAD_ACCUM=4
WARMUP=50
EVAL_SAMPLES=1319
EVAL_BATCH=128
NUM_LOG=5

# ---- Fixed V9 config ----
SEARCH_TEMP=1.0
ALPHA_INFO=1.0

# ---- Sweep axes ----
C_SIZES=(1 4 32)
LS=(2 4 8)
SLRS=("5e-2" "1e-1")
A_ZIPFS=("0.01" "0.1")
A_ABS=("0.5" "1.0")
ROLLOUTS=(2 4 8)

TIMESTAMP=$(date +%Y%m%d_%H%M)
OUT_ROOT="./ckpt/v9_config_${TIMESTAMP}"
mkdir -p "$OUT_ROOT"

JOB_IDX=0

run_one() {
  local C=$1 L=$2 slr=$3 az=$4 aa=$5 nr=$6 dataset=$7 dtag=$8

  JOB_IDX=$((JOB_IDX + 1))
  local gpu=$(( (JOB_IDX - 1) % N_GPUS ))
  local port=$((BASE_PORT + JOB_IDX))
  local tag="${MTAG}_${dtag}_C${C}_L${L}_slr${slr}_az${az}_aa${aa}_nr${nr}"
  local out="${OUT_ROOT}/${tag}"

  echo "  [${JOB_IDX}] GPU${gpu} ${tag}"

  CUDA_VISIBLE_DEVICES=$gpu torchrun \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$port \
    train_steer_pt.py \
    --mode v9 \
    --model_name $MODEL \
    --dataset $dataset \
    --num_epochs $EPOCHS \
    --lr $LR \
    --steer_lr $slr \
    --warmup_steps $WARMUP \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --C_SIZE $C \
    --L $L \
    --scale $SCALE \
    --inject_layers $LAYERS \
    --num_rollouts $nr \
    --search_temp $SEARCH_TEMP \
    --alpha_info $ALPHA_INFO \
    --alpha_abs $aa \
    --alpha_zipf $az \
    --detach_routing \
    --eval_every 99999 \
    --save_every 99999 \
    --eval_samples $EVAL_SAMPLES \
    --eval_batch_size $EVAL_BATCH \
    --num_log_samples $NUM_LOG \
    --log_every 10 \
    --output_dir $out &

  if (( JOB_IDX % N_GPUS == 0 )); then wait; fi
}

run_dataset() {
  local dataset=$1 dtag=$2
  JOB_IDX=0

  echo ""
  echo "--- ${dataset} (216 runs) ---"

  for C in "${C_SIZES[@]}"; do
    for L in "${LS[@]}"; do
      for slr in "${SLRS[@]}"; do
        for az in "${A_ZIPFS[@]}"; do
          for aa in "${A_ABS[@]}"; do
            for nr in "${ROLLOUTS[@]}"; do
              run_one $C $L $slr $az $aa $nr $dataset $dtag
            done
          done
        done
      done
    done
  done
  wait

  echo "${dataset} complete (${JOB_IDX} runs)."
}

echo ""
echo "============================================================"
echo "V9 config sweep | ${TIMESTAMP}"
echo "============================================================"

if [ "$PART" = "all" ] || [ "$PART" = "1" ]; then
  run_dataset scienceqa sci
fi

if [ "$PART" = "all" ] || [ "$PART" = "2" ]; then
  run_dataset gsm8k gsm
fi

echo ""
echo "============================================================"
echo "V9 config sweep done. Results in ${OUT_ROOT}/"
echo "============================================================"
echo ""
echo "To gather results:"
echo "  grep 'Final accuracy' ${OUT_ROOT}/*/train.log | sort -t= -k2 -rn"
