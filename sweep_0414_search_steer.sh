#!/bin/bash
# ===========================================================================
# V9 search-based steering sweep — Hyp 3
#
# Question: Does search-based routing (V9) improve over static routing (V6)?
#   - V9 detach_routing=False → joint policy & rep training
#   - V9 detach_routing=True  → decoupled (policy only learns from search signal)
#   - V6 baseline for comparison (same C, L, layer)
#
# Fixed (from prior sweeps):
#   - Qwen3-0.6B, layer 14, L=4, scale=0.5, code_position=first
#   - slr=5e-2, lr=1e-5, 1 epoch, LoRA off
#
# Sweep axes:
#   mode       ∈ {v6, v9}                     (2)
#   C_SIZE     ∈ {4, 32}                      (2)
#   detach     ∈ {on, off}  (v9 only)         (2 for v9, 1 for v6)
#   dataset    ∈ {scienceqa, gsm8k}           (2)
#
# Total: (2 v9 × 2 C × 2 ds) + (1 v6 × 2 C × 2 ds) = 8 + 4 = 12 runs
#
# Usage: ./sweep_0414_search_steer.sh [PART]
#   PART=1  → scienceqa
#   PART=2  → gsm8k
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
BASE_PORT=29700
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
OUT_ROOT="./ckpt/search_v9_${TIMESTAMP}"
mkdir -p "$OUT_ROOT"

# ---- Sweep axes ----
C_SIZES=(4 32)
DATASETS_1=("scienceqa")
DTAGS_1=("sci")
DATASETS_2=("gsm8k")
DTAGS_2=("gsm")

# ---- V9 search config ----
NUM_ROLLOUTS=4
SEARCH_TEMP=1.0
ALPHA_INFO=1.0
ALPHA_ABS=0.5
ALPHA_ZIPF=0.01

run_one() {
  local mode=$1 C=$2 dataset=$3 dtag=$4 detach_flag=$5 detach_tag=$6 job_idx=$7

  local gpu=$(( (job_idx - 1) % N_GPUS ))
  local port=$((BASE_PORT + job_idx))
  local tag="${MTAG}_${dtag}_${mode}_C${C}_L${L}_${detach_tag}"
  local out="${OUT_ROOT}/${tag}"

  echo "  [GPU ${gpu}] ${tag}"

  local extra_args=""
  if [ "$mode" = "v9" ]; then
    extra_args="--num_rollouts $NUM_ROLLOUTS --search_temp $SEARCH_TEMP"
    extra_args="$extra_args --alpha_info $ALPHA_INFO --alpha_abs $ALPHA_ABS --alpha_zipf $ALPHA_ZIPF"
    if [ "$detach_flag" = "1" ]; then
      extra_args="$extra_args --detach_routing"
    fi
  fi

  CUDA_VISIBLE_DEVICES=$gpu torchrun \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$port \
    train_steer_pt.py \
    --mode $mode \
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
    --output_dir $out $extra_args &
}

run_part() {
  local datasets=("$@")
  # Unpack: first half is dataset names, second half is dtags
  local n=$(( ${#datasets[@]} / 2 ))
  local ds_arr=("${datasets[@]:0:$n}")
  local dt_arr=("${datasets[@]:$n}")

  local JOB_IDX=0
  for i in $(seq 0 $((n-1))); do
    local dataset=${ds_arr[$i]}
    local dtag=${dt_arr[$i]}

    for C in "${C_SIZES[@]}"; do
      # V6 baseline (no detach concept)
      JOB_IDX=$((JOB_IDX + 1))
      run_one v6 $C $dataset $dtag 0 "base" $JOB_IDX
      if (( JOB_IDX % N_GPUS == 0 )); then wait; fi

      # V9 detach_routing=False (joint)
      JOB_IDX=$((JOB_IDX + 1))
      run_one v9 $C $dataset $dtag 0 "joint" $JOB_IDX
      if (( JOB_IDX % N_GPUS == 0 )); then wait; fi

      # V9 detach_routing=True (decoupled)
      JOB_IDX=$((JOB_IDX + 1))
      run_one v9 $C $dataset $dtag 1 "detach" $JOB_IDX
      if (( JOB_IDX % N_GPUS == 0 )); then wait; fi
    done
  done
  wait
}

echo ""
echo "============================================================"
echo "V9 search-steer sweep | ${TIMESTAMP}"
echo "============================================================"

if [ "$PART" = "all" ] || [ "$PART" = "1" ]; then
  echo ""
  echo "--- Part 1: ScienceQA ---"
  run_part "${DATASETS_1[@]}" "${DTAGS_1[@]}"
  echo "Part 1 complete."
fi

if [ "$PART" = "all" ] || [ "$PART" = "2" ]; then
  echo ""
  echo "--- Part 2: GSM8K ---"
  run_part "${DATASETS_2[@]}" "${DTAGS_2[@]}"
  echo "Part 2 complete."
fi

echo ""
echo "============================================================"
echo "V9 search sweep done. Results in ${OUT_ROOT}/"
echo "============================================================"
echo ""
echo "To gather results:"
echo "  grep 'Final accuracy' ${OUT_ROOT}/*/train.log"
