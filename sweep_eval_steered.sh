#!/bin/bash
# ===========================================================================
# SoRL re-run on 4 datasets with corrected eval (decode-time steering ON).
#
# Motivation:
#   The previous V9 eval pipeline used `_decode_scale_override = 0.0` by
#   default — meaning generation was steered only during prompt prefill, not
#   during autoregressive decoding. This understated SoRL's advantage over
#   SFT (most visible on ScienceQA). Fix: pass `--eval_decode_scale` (handled
#   in train_steer_pt.py:evaluate_accuracy → wrapper.generate(decode_scale)).
#   Default is now `wrapper.scale`, i.e. decode-time steering matches train.
#
# Datasets × max_new_tokens (previous default + 256):
#   gsm8k          : 256 → 512
#   commonsenseqa  :  64 → 320
#   scienceqa      : 256 → 512
#   strategyqa     :  64 → 320
#
# Optimal per-model config (from log/compare0412.md):
#   model         layer  scale  steer_lr  alpha_zipf  alpha_abs
#   Llama3.2-1B    10     0.1   1e-2      0.1         0.5
#   Llama3.2-3B    16     0.1   1e-2      0.1         0.1
#   Qwen3-0.6B     14     0.5   5e-2      0.1         0.5
#   Qwen3-1.7B     14     0.5   5e-2      0.1         0.5
#   Qwen3-4B       19     0.5   5e-2      0.1         0.1     (+ LoRA)
#
# Common: V9, C=32, L=4, detach_routing.
#
# Usage:  ./sweep_eval_steered.sh <PART>      (PART = 1-20 | all)
#         ./sweep_eval_steered.sh <PART> dry  (just print commands)
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
BASE_PORT=29500
N_GPUS=4

# ---- Per-model optimal config (from log/compare0412.md) ----
declare -A MODEL_NAME MTAG LAYER SCALE SLR AZIPF AABS LORA
MODEL_NAME[ll1]="meta-llama/Llama-3.2-1B"; MTAG[ll1]="ll1"; LAYER[ll1]=10; SCALE[ll1]=0.1; SLR[ll1]=1e-2; AZIPF[ll1]=0.1; AABS[ll1]=0.5; LORA[ll1]=""
MODEL_NAME[l3b]="meta-llama/Llama-3.2-3B"; MTAG[l3b]="l3b"; LAYER[l3b]=16; SCALE[l3b]=0.1; SLR[l3b]=1e-2; AZIPF[l3b]=0.1; AABS[l3b]=0.1; LORA[l3b]=""
MODEL_NAME[q06]="Qwen/Qwen3-0.6B";        MTAG[q06]="q06"; LAYER[q06]=14; SCALE[q06]=0.5; SLR[q06]=5e-2; AZIPF[q06]=0.1; AABS[q06]=0.5; LORA[q06]=""
MODEL_NAME[q17]="Qwen/Qwen3-1.7B";        MTAG[q17]="q17"; LAYER[q17]=14; SCALE[q17]=0.5; SLR[q17]=5e-2; AZIPF[q17]=0.1; AABS[q17]=0.5; LORA[q17]=""
MODEL_NAME[q4b]="Qwen/Qwen3-4B";          MTAG[q4b]="q4b"; LAYER[q4b]=19; SCALE[q4b]=0.5; SLR[q4b]=5e-2; AZIPF[q4b]=0.1; AABS[q4b]=0.1; LORA[q4b]="--use_lora"

# ---- Per-dataset eval / data settings ----
declare -A DTAG MAX_LEN MAX_NEW
DTAG[gsm8k]="gsm";              MAX_LEN[gsm8k]=512;          MAX_NEW[gsm8k]=512
DTAG[commonsenseqa]="csqa";     MAX_LEN[commonsenseqa]=256;  MAX_NEW[commonsenseqa]=320
DTAG[scienceqa]="sci";          MAX_LEN[scienceqa]=512;      MAX_NEW[scienceqa]=512
DTAG[strategyqa]="stra";        MAX_LEN[strategyqa]=256;     MAX_NEW[strategyqa]=320

# ---- Shared training hyper-params ----
LR=1e-5
EPOCHS=1
BATCH_SIZE=2
GRAD_ACCUM=4
WARMUP=50
EVAL_SAMPLES=-1   # -1 / 0 → eval full test set per dataset (1319 / 1221 / 2017 / 229)
EVAL_BATCH=64
NUM_LOG=5

# V9 / steering config (shared)
MODE=v9
C_SIZE=32
L_CHUNK=4
DETACH_FLAG="--detach_routing"

# Eval-time decode steering: empty → use wrapper.scale (i.e. trained scale,
# decode steering ON). Set to "0.0" to recover legacy prefill-only eval.
EVAL_DECODE_SCALE=""

TIMESTAMP=$(date +%Y%m%d_%H%M)
OUT_ROOT="./ckpt/sorl_eval_steered_${TIMESTAMP}"
mkdir -p "$OUT_ROOT"

# ---- Part → (model, dataset) mapping ----
# 5 models × 4 datasets = 20 parts; ordered model-major.
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
  layer=${LAYER[$mkey]}
  scale=${SCALE[$mkey]}
  slr=${SLR[$mkey]}
  azipf=${AZIPF[$mkey]}
  aabs=${AABS[$mkey]}
  lora_flag=${LORA[$mkey]}
  dtag=${DTAG[$ds]}
  max_len=${MAX_LEN[$ds]}
  max_new=${MAX_NEW[$ds]}

  JOB_IDX=$((JOB_IDX + 1))
  gpu=$(( (JOB_IDX - 1) % N_GPUS ))
  port=$((BASE_PORT + P))

  tag="${mtag}_${dtag}_v9_C${C_SIZE}_L${L_CHUNK}_layer${layer}"
  out="${OUT_ROOT}/${tag}"

  echo ""
  echo "============================================================"
  echo "Part ${P}/${N_PARTS}: ${mtag} + ${ds}  [GPU ${gpu}]"
  echo "  layer=${layer} scale=${scale} slr=${slr} azipf=${azipf} aabs=${aabs} ${lora_flag:+(+lora)}"
  echo "  max_length=${max_len} max_new_tokens=${max_new}"
  echo "  out=${out}"
  echo "============================================================"

  CMD=(
    env CUDA_VISIBLE_DEVICES=$gpu
    torchrun
      --nproc_per_node=1
      --master_addr=$MASTER_ADDR
      --master_port=$port
      train_steer_pt.py
      --mode $MODE
      --model_name $model
      --dataset $ds
      --max_length $max_len
      --max_new_tokens $max_new
      --num_epochs $EPOCHS
      --lr $LR
      --steer_lr $slr
      --warmup_steps $WARMUP
      --batch_size $BATCH_SIZE
      --gradient_accumulation_steps $GRAD_ACCUM
      --C_SIZE $C_SIZE
      --L $L_CHUNK
      --scale $scale
      --inject_layers $layer
      --alpha_zipf $azipf
      --alpha_abs $aabs
      $DETACH_FLAG
      --eval_every 99999
      --save_every 99999
      --eval_samples $EVAL_SAMPLES
      --eval_batch_size $EVAL_BATCH
      --num_log_samples $NUM_LOG
      --log_every 10
      --output_dir $out
      $lora_flag
  )

  if [ -n "$EVAL_DECODE_SCALE" ]; then
    CMD+=(--eval_decode_scale "$EVAL_DECODE_SCALE")
  fi

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
echo "SoRL eval-steered sweep done. Results in ${OUT_ROOT}/"
echo "============================================================"
echo ""
echo "Inspect:"
echo "  grep -E 'accuracy|eval' ${OUT_ROOT}/*/train.log"
