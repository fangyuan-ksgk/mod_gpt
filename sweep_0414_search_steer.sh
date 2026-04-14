#!/bin/bash
# ===========================================================================
# V9 search-based steering sweep — Hyp 3 (expanded)
#
# Question: Does search-based routing (V9) improve over static routing (V6)?
#   - V9 detach_routing=True  → decoupled (policy only learns from search signal)
#   - V9 detach_routing=False → joint policy & rep training
#   - V6 baseline for comparison (same C, L, layer)
#
# Models (best mid-layer from prior sweep):
#   q06 = Qwen3-0.6B  (28L, layer 14)
#   q17 = Qwen3-1.7B  (28L, layer 14)
#   l1  = Llama-3.2-1B (16L, layer 8)
#   l3  = Llama-3.2-3B (28L, layer 14)
#   q4b = Qwen3-4B     (36L, layer 18, LoRA r=16)
#
# Datasets (9):
#   scienceqa, gsm8k, arc, commonsenseqa, mmlu, boolq, openbookqa, aqua, hotpotqa
#
# Per (model, dataset): 6 runs
#   v6 C=4, v6 C=32, v9 joint C=4, v9 detach C=4, v9 joint C=32, v9 detach C=32
#
# Parts (1 per model×dataset):
#   q06:  1-9     q17: 10-18    l1: 19-27    l3: 28-36    q4b: 37-45
#
# Total: 45 parts × 6 runs = 270 runs
#
# Usage: ./sweep_0414_search_steer.sh <PART>
#   PART=1..45  → specific model+dataset combo
#   PART=all    → everything (default)
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

# ---- Shared hyper-params ----
LR=1e-5
SLR="5e-2"
SCALE=0.5
L=4
EPOCHS=1
BATCH_SIZE=2
GRAD_ACCUM=4
WARMUP=50
EVAL_SAMPLES=1319
EVAL_BATCH=128
NUM_LOG=5

# ---- V9 search config ----
NUM_ROLLOUTS=4
SEARCH_TEMP=1.0
ALPHA_INFO=1.0
ALPHA_ABS=0.5
ALPHA_ZIPF=0.01

# ---- Sweep axes ----
C_SIZES=(4 32)

TIMESTAMP=$(date +%Y%m%d_%H%M)
OUT_ROOT="./ckpt/search_v9_${TIMESTAMP}"
mkdir -p "$OUT_ROOT"

# ---- Part → (model, mtag, dataset, dtag, layer, extra_args) mapping ----
QWEN06="Qwen/Qwen3-0.6B"
QWEN17="Qwen/Qwen3-1.7B"
LLAMA1="meta-llama/Llama-3.2-1B"
LLAMA3="meta-llama/Llama-3.2-3B"
QWEN4B="Qwen/Qwen3-4B"

# 9 datasets in order
DS_NAMES=(scienceqa gsm8k arc commonsenseqa mmlu boolq openbookqa aqua hotpotqa)
DS_TAGS=(sci gsm arc csqa mmlu boolq oqa aqua hpqa)
N_DS=${#DS_NAMES[@]}

declare -A P_MODEL P_MTAG P_DS P_DTAG P_LAYER P_EXTRA

# Helper to fill 9 consecutive parts for a model
fill_model() {
  local start=$1 model=$2 mtag=$3 layer=$4 extra="$5"
  for i in $(seq 0 $((N_DS-1))); do
    local p=$((start + i))
    P_MODEL[$p]="$model"
    P_MTAG[$p]="$mtag"
    P_DS[$p]="${DS_NAMES[$i]}"
    P_DTAG[$p]="${DS_TAGS[$i]}"
    P_LAYER[$p]="$layer"
    P_EXTRA[$p]="$extra"
  done
}

fill_model  1 "$QWEN06" "q06"  "14" ""
fill_model 10 "$QWEN17" "q17"  "14" ""
fill_model 19 "$LLAMA1" "l1"   "8"  ""
fill_model 28 "$LLAMA3" "l3"   "14" ""
fill_model 37 "$QWEN4B" "q4b"  "18" "--use_lora --lora_rank 16 --lora_alpha 32"

N_PARTS=45

# ---- Runner ----
run_one() {
  local mode=$1 C=$2 dataset=$3 dtag=$4 detach_flag=$5 detach_tag=$6 job_idx=$7 \
        cur_model=$8 cur_mtag=$9 cur_layer=${10} model_extra="${11:-}"

  local gpu=$(( (job_idx - 1) % N_GPUS ))
  local port=$((BASE_PORT + PART_NUM * 100 + job_idx))
  local tag="${cur_mtag}_${dtag}_${mode}_C${C}_${detach_tag}"
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
    --model_name $cur_model \
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
    --inject_layers $cur_layer \
    --eval_every 99999 \
    --save_every 99999 \
    --eval_samples $EVAL_SAMPLES \
    --eval_batch_size $EVAL_BATCH \
    --num_log_samples $NUM_LOG \
    --log_every 10 \
    --output_dir $out $extra_args $model_extra &
}

run_part() {
  local p=$1
  local cur_model=${P_MODEL[$p]}
  local cur_mtag=${P_MTAG[$p]}
  local dataset=${P_DS[$p]}
  local dtag=${P_DTAG[$p]}
  local cur_layer=${P_LAYER[$p]}
  PART_NUM=$p

  echo ""
  echo "--- Part ${p}/${N_PARTS}: ${cur_mtag} + ${dataset} (layer ${cur_layer}) ---"

  local cur_extra="${P_EXTRA[$p]}"

  local JOB_IDX=0
  for C in "${C_SIZES[@]}"; do
    # V6 baseline
    JOB_IDX=$((JOB_IDX + 1))
    run_one v6 $C $dataset $dtag 0 "base" $JOB_IDX $cur_model $cur_mtag $cur_layer "$cur_extra"
    if (( JOB_IDX % N_GPUS == 0 )); then wait; fi

    # V9 joint
    JOB_IDX=$((JOB_IDX + 1))
    run_one v9 $C $dataset $dtag 0 "joint" $JOB_IDX $cur_model $cur_mtag $cur_layer "$cur_extra"
    if (( JOB_IDX % N_GPUS == 0 )); then wait; fi

    # V9 detach
    JOB_IDX=$((JOB_IDX + 1))
    run_one v9 $C $dataset $dtag 1 "detach" $JOB_IDX $cur_model $cur_mtag $cur_layer "$cur_extra"
    if (( JOB_IDX % N_GPUS == 0 )); then wait; fi
  done
  wait

  echo "Part ${p} complete."
}

# ---- Dispatch ----
echo ""
echo "============================================================"
echo "V9 search-steer sweep (expanded) | ${TIMESTAMP}"
echo "============================================================"

if [ "$PART" = "all" ]; then
  for p in $(seq 1 $N_PARTS); do
    run_part $p
  done
else
  for p in $PART; do
    run_part $p
  done
fi

echo ""
echo "============================================================"
echo "V9 search sweep done. Results in ${OUT_ROOT}/"
echo "============================================================"
echo ""
echo "To gather results:"
echo "  grep 'Final accuracy' ${OUT_ROOT}/*/train.log"
