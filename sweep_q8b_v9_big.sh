#!/usr/bin/env bash
# ============================================================================
# Qwen3-8B × DLR (v9) × bigger sweep
#
# Three independent blocks (toggle via $1). sci is dropped (L22 already known).
#   A) Layer fine-grain sweep   : layers {15,16,17,19,20,21} × {gsm,stra,csqa} = 18 runs
#   B) Steering-LR sweep        : best (layer,dataset) × slr ∈ {1e-1, 5e-2}    = 6 runs
#   C) Chunk-size (L) sweep     : best (layer,dataset) × L   ∈ {1, 8}          = 6 runs
#
# Best (layer, dataset) combos from the prior 3-layer sweep:
#   csqa@L22, gsm@L18, stra@L14
#
# Fairness fixes vs sweep_q8b_v9.sh:
#   - num_rollouts: 1 -> 4   (matches default in train_steer_pt.py)
#   - search_temp:  1.0      (default; explicit for clarity)
#   - steer_lr:     5e-2     (was already set; documented here)
#
# Hardware: 2× 80G; 8B+LoRA+seq_len=1024 fits 1 GPU per job, 2 jobs in parallel.
#
# Usage:
#   bash sweep_q8b_v9_big.sh A         # layer fine-grain (18 runs)
#   bash sweep_q8b_v9_big.sh B         # steer-lr sweep    (6 runs)
#   bash sweep_q8b_v9_big.sh C         # L sweep           (6 runs)
#   bash sweep_q8b_v9_big.sh all       # A + B + C        (30 runs)
#   DRY=dry bash sweep_q8b_v9_big.sh A # print-only
# ============================================================================
set -euo pipefail

BLOCK=${1:-all}        # A | B | C | all
DRY=${DRY:-}

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4

MASTER_ADDR=127.0.0.1
BASE_PORT=29800
N_GPUS=2

# ---- Model fixed hyper-params (8B-specific) -------------------------------
MODEL_NAME="Qwen/Qwen3-8B"
MTAG="q8b"
SCALE=0.5
SLR_DEFAULT=5e-2
AZIPF=0.1
AABS=0.1
AINFO=1.0
LORA_FLAGS="--use_lora --lora_rank 16 --lora_alpha 32"

# ---- Per-dataset context / decode budgets ---------------------------------
declare -A DTAG MAX_LEN MAX_NEW EVAL_BSZ
DTAG[gsm8k]="gsm";              MAX_LEN[gsm8k]=1024;         MAX_NEW[gsm8k]=768;       EVAL_BSZ[gsm8k]=32
DTAG[scienceqa]="sci";          MAX_LEN[scienceqa]=1024;     MAX_NEW[scienceqa]=768;   EVAL_BSZ[scienceqa]=32
DTAG[strategyqa]="stra";        MAX_LEN[strategyqa]=512;     MAX_NEW[strategyqa]=512;  EVAL_BSZ[strategyqa]=64
DTAG[commonsenseqa]="csqa";     MAX_LEN[commonsenseqa]=512;  MAX_NEW[commonsenseqa]=512; EVAL_BSZ[commonsenseqa]=64

# sci dropped: L22 already established as best in prior sweep; focusing on gsm/csqa/stra.
DATASETS_ORDER=(gsm8k strategyqa commonsenseqa)

# ---- Best (layer, dataset) combos from prior sweep (block B, C) -----------
declare -A BEST_LAYER
BEST_LAYER[commonsenseqa]=22
BEST_LAYER[gsm8k]=18
BEST_LAYER[strategyqa]=14

# ---- Shared training hyper-params -----------------------------------------
LR=1e-5
EPOCHS=1
BATCH_SIZE=1
GRAD_ACCUM=8
WARMUP=50
EVAL_SAMPLES=-1
NUM_LOG=5

# ---- v9 (DLR) config (defaults; overridden per block) ---------------------
MODE=v9
C_SIZE=32
L_CHUNK_DEFAULT=4
DETACH_FLAG="--detach_routing"
NUM_ROLLOUTS=4              # FIX: default in train_steer_pt.py is 4
SEARCH_TEMP=1.0

# ---- Build job list -------------------------------------------------------
# Each job = "layer ds slr L_chunk extra_tag"
JOBS=()

if [ "$BLOCK" = "A" ] || [ "$BLOCK" = "all" ]; then
  for layer in 15 16 17 19 20 21; do
    for ds in "${DATASETS_ORDER[@]}"; do
      JOBS+=("$layer $ds $SLR_DEFAULT $L_CHUNK_DEFAULT layer")
    done
  done
fi

if [ "$BLOCK" = "B" ] || [ "$BLOCK" = "all" ]; then
  for ds in "${DATASETS_ORDER[@]}"; do
    for slr in 1e-1 5e-2; do
      JOBS+=("${BEST_LAYER[$ds]} $ds $slr $L_CHUNK_DEFAULT slr${slr}")
    done
  done
fi

if [ "$BLOCK" = "C" ] || [ "$BLOCK" = "all" ]; then
  for ds in "${DATASETS_ORDER[@]}"; do
    for Lc in 1 8; do
      JOBS+=("${BEST_LAYER[$ds]} $ds $SLR_DEFAULT $Lc L${Lc}")
    done
  done
fi

N_JOBS=${#JOBS[@]}

TS=$(date +%Y%m%d_%H%M)
OUT_ROOT="./ckpt/q8b_v9_big_${TS}"
mkdir -p "$OUT_ROOT"

echo "============================================================"
echo "Qwen3-8B × DLR (v9) BIG sweep"
echo "  block:  ${BLOCK}"
echo "  jobs:   ${N_JOBS}"
echo "  N(rollouts)=${NUM_ROLLOUTS}, search_T=${SEARCH_TEMP}"
echo "  out:    ${OUT_ROOT}"
echo "============================================================"

# ---- Launcher -------------------------------------------------------------
run_one() {
  local layer=$1
  local ds=$2
  local slr=$3
  local L_c=$4
  local extra=$5
  local job_idx=$6

  local dtag=${DTAG[$ds]}
  local max_len=${MAX_LEN[$ds]}
  local max_new=${MAX_NEW[$ds]}
  local eval_bsz=${EVAL_BSZ[$ds]}

  local gpu=$(( (job_idx - 1) % N_GPUS ))
  local port=$((BASE_PORT + job_idx))

  local tag="${MTAG}_${dtag}_v9_C${C_SIZE}_L${L_c}_N${NUM_ROLLOUTS}_layer${layer}_slr${slr}_${extra}"
  local out="${OUT_ROOT}/${tag}"

  echo ""
  echo "  [GPU ${gpu}] job ${job_idx}/${N_JOBS}  ${tag}"

  local -a CMD=(
    env CUDA_VISIBLE_DEVICES=$gpu
    torchrun
      --nproc_per_node=1
      --master_addr=$MASTER_ADDR
      --master_port=$port
      train_steer_pt.py
      --model_name $MODEL_NAME
      --dataset $ds
      --max_length $max_len
      --max_new_tokens $max_new
      --num_epochs $EPOCHS
      --lr $LR
      --warmup_steps $WARMUP
      --batch_size $BATCH_SIZE
      --gradient_accumulation_steps $GRAD_ACCUM
      --eval_every 99999
      --save_every 99999
      --eval_samples $EVAL_SAMPLES
      --eval_batch_size $eval_bsz
      --num_log_samples $NUM_LOG
      --log_every 10
      --output_dir $out
      $LORA_FLAGS
      --mode $MODE
      --steer_lr $slr
      --scale $SCALE
      --C_SIZE $C_SIZE
      --L $L_c
      --inject_layers $layer
      --alpha_info $AINFO
      --alpha_zipf $AZIPF
      --alpha_abs $AABS
      --num_rollouts $NUM_ROLLOUTS
      --search_temp $SEARCH_TEMP
      $DETACH_FLAG
  )

  if [ "$DRY" = "dry" ]; then
    printf '%q ' "${CMD[@]}"; echo
  else
    "${CMD[@]}" &
    if (( job_idx % N_GPUS == 0 )); then wait; fi
  fi
}

# ---- Main loop ------------------------------------------------------------
job_idx=0
for J in "${JOBS[@]}"; do
  job_idx=$((job_idx + 1))
  read -r layer ds slr Lc extra <<<"$J"
  run_one "$layer" "$ds" "$slr" "$Lc" "$extra" "$job_idx"
done
wait

echo ""
echo "============================================================"
echo "Done. Results in ${OUT_ROOT}/"
echo "  grep -E 'accuracy|eval' ${OUT_ROOT}/*/train.log"
echo "============================================================"
