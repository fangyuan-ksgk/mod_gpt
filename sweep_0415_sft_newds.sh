#!/bin/bash
# ===========================================================================
# SFT sweep — new datasets: mmlupro, strategyqa, bbhlogic
#
# Models:
#   q06 = Qwen3-0.6B  (28L)
#   q17 = Qwen3-1.7B  (28L)
#   l1  = Llama-3.2-1B (16L)
#   l3  = Llama-3.2-3B (28L)
#   q4b = Qwen3-4B     (36L, LoRA r=16)
#
# Epochs: 1 and 2
# Datasets: mmlupro, strategyqa, bbhlogic
#
# Total: 5 models × 3 datasets × 2 epochs = 30 runs
#
# Parts (1 per model):
#   1=q06  2=q17  3=l1  4=l3  5=q4b
#
# Usage: ./sweep_0415_sft_newds.sh [PART]
#   PART=1..5  → single model
#   PART=all   → all models (default)
# ===========================================================================
set -euo pipefail

PART=${1:-all}

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4

MASTER_ADDR=127.0.0.1
BASE_PORT=30100
N_GPUS=4

# ---- Shared config ----
LR=1e-5
BATCH_SIZE=2
GRAD_ACCUM=4
WARMUP=50
EVAL_BATCH=64
NUM_LOG=3

TIMESTAMP=$(date +%Y%m%d_%H%M)
OUT_ROOT="./ckpt/sft_newds_${TIMESTAMP}"
mkdir -p "$OUT_ROOT"

# ---- Dataset config ----
DATASETS=(strategyqa race logiqa medqa drop triviaqa)
DTAGS=(sqqa race lqa medqa drop tqa)
EVAL_SAMPLES=(687 4000 651 1273 4000 4000)

# ---- Model config ----
declare -A M_MODEL M_MTAG M_EXTRA
M_MODEL[1]="Qwen/Qwen3-0.6B";    M_MTAG[1]="q06"; M_EXTRA[1]=""
M_MODEL[2]="Qwen/Qwen3-1.7B";    M_MTAG[2]="q17"; M_EXTRA[2]=""
M_MODEL[3]="meta-llama/Llama-3.2-1B"; M_MTAG[3]="l1"; M_EXTRA[3]=""
# M_MODEL[4]="meta-llama/Llama-3.2-3B"; M_MTAG[4]="l3"; M_EXTRA[4]=""
# M_MODEL[5]="Qwen/Qwen3-4B";      M_MTAG[5]="q4b"; M_EXTRA[5]="--use_lora --lora_r 16 --lora_alpha 32"

N_MODELS=3

JOB_IDX=0

run_one() {
  local model=$1 mtag=$2 dataset=$3 dtag=$4 ep=$5 eval_s=$6 model_extra="$7"

  JOB_IDX=$((JOB_IDX + 1))
  local gpu=$(( (JOB_IDX - 1) % N_GPUS ))
  local port=$((BASE_PORT + JOB_IDX))
  local tag="${mtag}_${dtag}_ep${ep}"
  local out="${OUT_ROOT}/${tag}"

  echo "  [${JOB_IDX}] GPU${gpu} ${tag}"

  CUDA_VISIBLE_DEVICES=$gpu torchrun \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$port \
    train_sft_pt.py \
    --model_name $model \
    --dataset $dataset \
    --num_epochs $ep \
    --lr $LR \
    --warmup_steps $WARMUP \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --eval_every 99999 \
    --save_every 99999 \
    --eval_samples $eval_s \
    --eval_batch_size $EVAL_BATCH \
    --num_log_samples $NUM_LOG \
    --log_every 10 \
    --output_dir $out $model_extra &

  if (( JOB_IDX % N_GPUS == 0 )); then wait; fi
}

run_model() {
  local p=$1
  local model=${M_MODEL[$p]}
  local mtag=${M_MTAG[$p]}
  local extra="${M_EXTRA[$p]}"

  echo ""
  echo "--- Model: ${mtag} (${model}) ---"
  JOB_IDX=0

  for ep in 1 2; do
    for i in "${!DATASETS[@]}"; do
      run_one "$model" "$mtag" "${DATASETS[$i]}" "${DTAGS[$i]}" "$ep" "${EVAL_SAMPLES[$i]}" "$extra"
    done
  done
  wait

  echo "Model ${mtag} complete."
}

echo ""
echo "============================================================"
echo "SFT sweep — mmlupro / strategyqa / bbhlogic | ${TIMESTAMP}"
echo "============================================================"

if [ "$PART" = "all" ]; then
  for p in $(seq 1 $N_MODELS); do
    run_model $p
  done
else
  for p in $PART; do
    run_model $p
  done
fi

echo ""
echo "============================================================"
echo "SFT sweep done. Results in ${OUT_ROOT}/"
echo "============================================================"
echo ""
echo "To gather results:"
echo "  grep 'Final accuracy' ${OUT_ROOT}/*/train.log | sort"
