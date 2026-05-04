#!/usr/bin/env bash
# ============================================================================
# Qwen3-8B × DLR (v9) × 4 benchmarks × 3 layers
#
# Purpose: extend the DLR evaluation to an 8B base model on GSM8K, ScienceQA,
#          StrategyQA, and CommonsenseQA, with longer context than the 4B
#          sweeps used (Qwen3-8B CoT runs noticeably longer).
#
# Matrix: 3 layers × 4 datasets = 12 DLR runs (+ 4 matched-SFT baselines if
#         --with_sft is passed).
#
# Hardware: assumes 2× A100/H100-80G. 8B+LoRA+seq_len=1024 fits on
#           one GPU per job; runs 2 jobs in parallel.
#
# Usage:
#   bash sweep_q8b_v9.sh            # full sweep (12 DLR runs)
#   bash sweep_q8b_v9.sh dry        # print commands, don't execute
#   bash sweep_q8b_v9.sh all sft    # also run matched-length SFT baselines
#   bash sweep_q8b_v9.sh 5          # run only part 5 (1..12)
#
# Output: ./ckpt/q8b_v9_<TS>/<layer_tag>_<dataset_tag>/
# ============================================================================
set -euo pipefail

PART=${1:-all}
WITH_SFT=${2:-}        # pass "sft" to also run matched SFT baselines
DRY=${DRY:-}           # set DRY=dry to print-only

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4

MASTER_ADDR=127.0.0.1
BASE_PORT=29700
N_GPUS=2

# ---- Model fixed hyper-params (8B-specific) -------------------------------
MODEL_NAME="Qwen/Qwen3-8B"
MTAG="q8b"
SCALE=0.5
SLR=5e-2
AZIPF=0.1
AABS=0.1
AINFO=1.0
LORA_FLAGS="--use_lora --lora_rank 16 --lora_alpha 32"

# ---- Layers to sweep ------------------------------------------------------
# Priors: L14 = ablation-best for Qwen3-8B SciQA (71.3%); L18 = middle;
#         L22 = post-peak sanity.
LAYERS=(14 18 22)

# ---- Per-dataset context / decode budgets ---------------------------------
# Extended vs the 4B sweeps: +2x for short (strategyqa/csqa), +2x for long
# (gsm8k/sciqa), keeping max_new_tokens generous for 8B verbosity.
declare -A DTAG MAX_LEN MAX_NEW EVAL_BSZ
DTAG[gsm8k]="gsm";              MAX_LEN[gsm8k]=1024;         MAX_NEW[gsm8k]=768;   EVAL_BSZ[gsm8k]=32
DTAG[scienceqa]="sci";          MAX_LEN[scienceqa]=1024;     MAX_NEW[scienceqa]=768;   EVAL_BSZ[scienceqa]=32
DTAG[strategyqa]="stra";        MAX_LEN[strategyqa]=512;     MAX_NEW[strategyqa]=512;  EVAL_BSZ[strategyqa]=64
DTAG[commonsenseqa]="csqa";     MAX_LEN[commonsenseqa]=512;  MAX_NEW[commonsenseqa]=512; EVAL_BSZ[commonsenseqa]=64

DATASETS_ORDER=(gsm8k scienceqa strategyqa commonsenseqa)

# ---- Shared training hyper-params -----------------------------------------
LR=1e-5
EPOCHS=1
BATCH_SIZE=1          # 8B + LoRA + seq=1024 → drop from 2 to 1
GRAD_ACCUM=8          # keep effective batch = 8 (was 2*4)
WARMUP=50
EVAL_SAMPLES=-1
NUM_LOG=5

# ---- v9 (DLR) config ------------------------------------------------------
MODE=v9
C_SIZE=32
L_CHUNK=4
DETACH_FLAG="--detach_routing"
NUM_ROLLOUTS=1

# ---- Build part list: (layer, dataset) -----------------------------------
declare -A PART_LAYER PART_DATASET
P=0
for layer in "${LAYERS[@]}"; do
  for ds in "${DATASETS_ORDER[@]}"; do
    P=$((P + 1))
    PART_LAYER[$P]=$layer
    PART_DATASET[$P]=$ds
  done
done
N_PARTS=$P

if [ "$PART" = "all" ]; then
  PARTS=()
  for ((i=1; i<=N_PARTS; i++)); do PARTS+=($i); done
else
  PARTS=($PART)
fi

TS=$(date +%Y%m%d_%H%M)
OUT_ROOT="./ckpt/q8b_v9_${TS}"
mkdir -p "$OUT_ROOT"

echo "============================================================"
echo "Qwen3-8B × DLR (v9) sweep"
echo "  layers: ${LAYERS[*]}"
echo "  datasets: ${DATASETS_ORDER[*]}"
echo "  parts: ${PARTS[*]} / ${N_PARTS}"
echo "  out:   ${OUT_ROOT}"
[ "$WITH_SFT" = "sft" ] && echo "  + matched-length SFT baselines"
echo "============================================================"

# ---- Launcher for a single run -------------------------------------------
run_one() {
  local mode_override=$1     # "v9" or "sft" (sft = run without DLR)
  local layer=$2
  local ds=$3
  local job_idx=$4

  local dtag=${DTAG[$ds]}
  local max_len=${MAX_LEN[$ds]}
  local max_new=${MAX_NEW[$ds]}
  local eval_bsz=${EVAL_BSZ[$ds]}

  local gpu=$(( (job_idx - 1) % N_GPUS ))
  local port=$((BASE_PORT + job_idx))

  local tag
  if [ "$mode_override" = "sft" ]; then
    tag="${MTAG}_${dtag}_sft_maxlen${max_len}"
  else
    tag="${MTAG}_${dtag}_v9_C${C_SIZE}_L${L_CHUNK}_N${NUM_ROLLOUTS}_layer${layer}"
  fi
  local out="${OUT_ROOT}/${tag}"

  echo ""
  echo "  [GPU ${gpu}] part ${job_idx}  ${tag}"

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
  )

  if [ "$mode_override" = "sft" ]; then
    # SFT baseline: use v6 with scale=0 and zero steer_lr → effectively vanilla
    # LoRA SFT at matched max_length. (If you have a cleaner --mode sft flag,
    # substitute here.)
    CMD+=(
      --mode v6
      --scale 0.0
      --steer_lr 0.0
      --C_SIZE 1
      --L $L_CHUNK
      --inject_layers $layer
    )
  else
    CMD+=(
      --mode $MODE
      --steer_lr $SLR
      --scale $SCALE
      --C_SIZE $C_SIZE
      --L $L_CHUNK
      --inject_layers $layer
      --alpha_info $AINFO
      --alpha_zipf $AZIPF
      --alpha_abs $AABS
      --num_rollouts $NUM_ROLLOUTS
      $DETACH_FLAG
    )
  fi

  if [ "$DRY" = "dry" ]; then
    printf '%q ' "${CMD[@]}"; echo
  else
    "${CMD[@]}" &
    if (( job_idx % N_GPUS == 0 )); then wait; fi
  fi
}

# ---- Main loop: DLR runs --------------------------------------------------
echo ""
echo "############################################################"
echo "# DLR runs"
echo "############################################################"

job_idx=0
for P in "${PARTS[@]}"; do
  job_idx=$((job_idx + 1))
  run_one v9 "${PART_LAYER[$P]}" "${PART_DATASET[$P]}" $job_idx
done
wait

# ---- Optional: matched SFT baselines (1 per dataset, layer doesn't matter) --
if [ "$WITH_SFT" = "sft" ]; then
  echo ""
  echo "############################################################"
  echo "# Matched-length SFT baselines"
  echo "############################################################"

  job_idx=0
  for ds in "${DATASETS_ORDER[@]}"; do
    job_idx=$((job_idx + 1))
    run_one sft 18 "$ds" $job_idx       # layer 18 is inert when scale=0
  done
  wait
fi

echo ""
echo "============================================================"
echo "Done. Results in ${OUT_ROOT}/"
echo "  grep -E 'accuracy|eval' ${OUT_ROOT}/*/train.log"
echo "============================================================"
