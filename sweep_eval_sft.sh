#!/bin/bash
# ===========================================================================
# SFT baseline matched to sweep_eval_steered.sh.
#
# Same 5 models × 4 datasets matrix, same per-dataset max_length /
# max_new_tokens, same LR / epochs / batch / eval-samples policy as the SoRL
# sweep — only the steering machinery is removed. Use this as the head-to-
# head comparison group for SoRL's eval-steered numbers.
#
# Datasets × max_new_tokens (matches sweep_eval_steered.sh — previous +256):
#   gsm8k          : 512
#   commonsenseqa  : 320
#   scienceqa      : 512
#   strategyqa     : 320
#
# Per-model: only LoRA flag differs (Qwen3-4B uses LoRA, others full FT).
#
# Usage:  ./sweep_eval_sft.sh <PART>      (PART = 1-20 | all)
#         ./sweep_eval_sft.sh <PART> dry  (just print commands)
# ===========================================================================
set -euo pipefail

PART=${1:-all}
DRY=${2:-}

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4

MASTER_ADDR=127.0.0.1
BASE_PORT=29600
N_GPUS=4

# ---- Per-model config (LoRA flag only — matches sweep_eval_steered.sh) ----
declare -A MODEL_NAME MTAG LORA
MODEL_NAME[ll1]="meta-llama/Llama-3.2-1B"; MTAG[ll1]="ll1"; LORA[ll1]=""
MODEL_NAME[l3b]="meta-llama/Llama-3.2-3B"; MTAG[l3b]="l3b"; LORA[l3b]=""
MODEL_NAME[q06]="Qwen/Qwen3-0.6B";        MTAG[q06]="q06"; LORA[q06]=""
MODEL_NAME[q17]="Qwen/Qwen3-1.7B";        MTAG[q17]="q17"; LORA[q17]=""
MODEL_NAME[q4b]="Qwen/Qwen3-4B";          MTAG[q4b]="q4b"; LORA[q4b]="--use_lora"

# ---- Per-dataset eval / data settings (must match sweep_eval_steered.sh) ----
declare -A DTAG MAX_LEN MAX_NEW
DTAG[gsm8k]="gsm";              MAX_LEN[gsm8k]=512;          MAX_NEW[gsm8k]=512
DTAG[commonsenseqa]="csqa";     MAX_LEN[commonsenseqa]=256;  MAX_NEW[commonsenseqa]=320
DTAG[scienceqa]="sci";          MAX_LEN[scienceqa]=512;      MAX_NEW[scienceqa]=512
DTAG[strategyqa]="stra";        MAX_LEN[strategyqa]=256;     MAX_NEW[strategyqa]=320

# ---- Shared training hyper-params (must match SoRL sweep) ----
LR=1e-5
EPOCHS=1
BATCH_SIZE=2
GRAD_ACCUM=4
WARMUP=50
EVAL_SAMPLES=-1   # -1 / 0 → eval full test set per dataset
EVAL_BATCH=64
NUM_LOG=5

TIMESTAMP=$(date +%Y%m%d_%H%M)
OUT_ROOT="./ckpt/sft_eval_${TIMESTAMP}"
mkdir -p "$OUT_ROOT"

# ---- Part → (model, dataset) mapping (model-major, matches SoRL sweep) ----
MODELS_ORDER=(q06 q17 q4b ll1 l3b)
DATASETS_ORDER=(gsm8k commonsenseqa scienceqa strategyqa)

declare -A PART_MODEL_KEY PART_DATASET
P=0
for mkey in "${MODELS_ORDER[@]}"; do
  for ds in "${DATASETS_ORDER[@]}"; do
    P=$((P + 1))
    PART_MODEL_KEY[$P]=$mkey
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

# ---- Run loop ----
JOB_IDX=0
for P in "${PARTS[@]}"; do
  mkey=${PART_MODEL_KEY[$P]}
  ds=${PART_DATASET[$P]}

  model=${MODEL_NAME[$mkey]}
  mtag=${MTAG[$mkey]}
  lora_flag=${LORA[$mkey]}
  dtag=${DTAG[$ds]}
  max_len=${MAX_LEN[$ds]}
  max_new=${MAX_NEW[$ds]}

  JOB_IDX=$((JOB_IDX + 1))
  gpu=$(( (JOB_IDX - 1) % N_GPUS ))
  port=$((BASE_PORT + P))

  tag="${mtag}_${dtag}_sft"
  out="${OUT_ROOT}/${tag}"

  echo ""
  echo "============================================================"
  echo "Part ${P}/${N_PARTS}: ${mtag} + ${ds} [SFT]  [GPU ${gpu}]"
  echo "  ${lora_flag:+(+lora)}  max_length=${max_len}  max_new_tokens=${max_new}"
  echo "  out=${out}"
  echo "============================================================"

  CMD=(
    env CUDA_VISIBLE_DEVICES=$gpu
    torchrun
      --nproc_per_node=1
      --master_addr=$MASTER_ADDR
      --master_port=$port
      train_sft_pt.py
      --model_name $model
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
      --eval_batch_size $EVAL_BATCH
      --num_log_samples $NUM_LOG
      --log_every 10
      --output_dir $out
      $lora_flag
  )

  if [ "$DRY" = "dry" ]; then
    printf '%q ' "${CMD[@]}"; echo
  else
    "${CMD[@]}" &
    if (( JOB_IDX % N_GPUS == 0 )); then wait; fi
  fi
done
wait

echo ""
echo "============================================================"
echo "SFT baseline sweep done. Results in ${OUT_ROOT}/"
echo "============================================================"
echo ""
echo "Inspect:"
echo "  grep -E 'accuracy|eval' ${OUT_ROOT}/*/train.log"
echo ""
echo "Compare to SoRL: pair runs by (model, dataset) tag. Example:"
echo "  diff <(grep accuracy ./ckpt/sorl_eval_steered_<TS>/q06_gsm_*/train.log | tail -1) \\"
echo "       <(grep accuracy ./ckpt/sft_eval_<TS>/q06_gsm_sft/train.log | tail -1)"
