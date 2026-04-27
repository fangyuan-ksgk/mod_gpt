#!/bin/bash
# ===========================================================================
# Pause Token + Token Assorted baselines on StrategyQA.
#
# Fills the missing StrategyQA cells in the Pause / TA tables of
# log/compare0412.md. Uses the trainers' built-in defaults for max_length
# (64) and max_new_tokens (64) — the "original" values, NOT the bumped
# +256 values used in sweep_eval_steered.sh / sweep_eval_sft.sh.
#
# 5 models × 2 methods (pause, ta) = 10 runs.
#
# Parts (1 per model):
#   1=q06  2=q17  3=l1  4=l3  5=q4b
#
# Usage: ./sweep_eval_pause_ta_strategyqa.sh [PART]
#   PART=1..5 → single model (runs both methods)
#   PART=all  → all 5 models (default)
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
BASE_PORT=30300
N_GPUS=1

# ---- Shared training config (matches sweep_0415_sft_newds.sh / log 0411) ----
LR=1e-5
EPOCHS=1
BATCH_SIZE=2
GRAD_ACCUM=4
WARMUP=50
EVAL_BATCH=64
NUM_LOG=3
EVAL_SAMPLES=687     # full StrategyQA test split (ChilleD/StrategyQA)

# Method-specific
K_PAUSE=8            # pause: <pause> tokens between query and response
TA_VQVAE_STEPS=20000 # TA: VQ-VAE Phase-1 steps (default)

TIMESTAMP=$(date +%Y%m%d_%H%M)
OUT_ROOT="./ckpt/pause_ta_strategyqa_${TIMESTAMP}"
mkdir -p "$OUT_ROOT"

# ---- Model config (matches sweep_eval_sft.sh) ----
declare -A M_MODEL M_MTAG M_EXTRA
# NOTE: pause-token and token-assorted baselines use FULL fine-tuning on the
# four smaller models (matches the SFT/SoRL main-table setup). Only Qwen3-4B
# uses LoRA, for memory reasons. Do NOT add --use_lora to the smaller models
# here, or the comparison vs. SFT/SoRL becomes unfair.
M_MODEL[1]="Qwen/Qwen3-0.6B";          M_MTAG[1]="q06"; M_EXTRA[1]=""
M_MODEL[2]="Qwen/Qwen3-1.7B";          M_MTAG[2]="q17"; M_EXTRA[2]=""
M_MODEL[3]="meta-llama/Llama-3.2-1B";  M_MTAG[3]="l1";  M_EXTRA[3]=""
M_MODEL[4]="meta-llama/Llama-3.2-3B";  M_MTAG[4]="l3";  M_EXTRA[4]=""
# M_MODEL[5]="Qwen/Qwen3-4B";            M_MTAG[5]="q4b"; M_EXTRA[5]="--use_lora --lora_r 16 --lora_alpha 32"

N_MODELS=4

JOB_IDX=0

run_pause() {
  local model=$1 mtag=$2 extra="$3"
  JOB_IDX=$((JOB_IDX + 1))
  local gpu=$(( (JOB_IDX - 1) % N_GPUS ))
  local port=$((BASE_PORT + JOB_IDX))
  local tag="${mtag}_stra_pause_ep${EPOCHS}"
  local out="${OUT_ROOT}/${tag}"

  echo "  [${JOB_IDX}] GPU${gpu} ${tag}"

  CUDA_VISIBLE_DEVICES=$gpu torchrun \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$port \
    train_pause_pt.py \
    --model_name $model \
    --dataset strategyqa \
    --num_epochs $EPOCHS \
    --lr $LR \
    --warmup_steps $WARMUP \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --k_pause $K_PAUSE \
    --eval_every 99999 \
    --save_every 99999 \
    --eval_samples $EVAL_SAMPLES \
    --eval_batch_size $EVAL_BATCH \
    --num_log_samples $NUM_LOG \
    --log_every 10 \
    --output_dir $out $extra &

  if (( JOB_IDX % N_GPUS == 0 )); then wait; fi
}

run_ta() {
  local model=$1 mtag=$2 extra="$3"
  JOB_IDX=$((JOB_IDX + 1))
  local gpu=$(( (JOB_IDX - 1) % N_GPUS ))
  local port=$((BASE_PORT + JOB_IDX))
  local tag="${mtag}_stra_ta_ep${EPOCHS}"
  local out="${OUT_ROOT}/${tag}"

  echo "  [${JOB_IDX}] GPU${gpu} ${tag}"

  CUDA_VISIBLE_DEVICES=$gpu torchrun \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$port \
    train_ta_pt.py \
    --model_name $model \
    --dataset strategyqa \
    --num_epochs $EPOCHS \
    --lr $LR \
    --warmup_steps $WARMUP \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --vqvae_steps $TA_VQVAE_STEPS \
    --eval_every 99999 \
    --save_every 99999 \
    --eval_samples $EVAL_SAMPLES \
    --eval_batch_size $EVAL_BATCH \
    --num_log_samples $NUM_LOG \
    --log_every 10 \
    --output_dir $out $extra &

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

  run_pause "$model" "$mtag" "$extra"
  run_ta    "$model" "$mtag" "$extra"
  wait

  echo "Model ${mtag} complete."
}

echo ""
echo "============================================================"
echo "Pause + TA on StrategyQA (1 ep, original max_new=64) | ${TIMESTAMP}"
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
echo "Sweep done. Results in ${OUT_ROOT}/"
echo "============================================================"
echo ""
echo "Gather:"
echo "  grep 'Final accuracy' ${OUT_ROOT}/*/train.log | sort"
