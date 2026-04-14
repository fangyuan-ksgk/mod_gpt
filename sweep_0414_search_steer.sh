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
#
# Per (model, dataset): 6 runs
#   v6 C=4, v6 C=32, v9 joint C=4, v9 detach C=4, v9 joint C=32, v9 detach C=32
#
# Parts (1 per model×dataset):
#   1  q06+scienceqa     5  q17+scienceqa     9  l1+scienceqa     13 l3+scienceqa
#   2  q06+gsm8k         6  q17+gsm8k        10  l1+gsm8k        14 l3+gsm8k
#   3  q06+arc           7  q17+arc           11  l1+arc          15 l3+arc
#   4  q06+commonsenseqa 8  q17+commonsenseqa 12  l1+commonsenseqa16 l3+commonsenseqa
#
# Total: 16 parts × 6 runs = 96 runs
#
# Usage: ./sweep_0414_search_steer.sh <PART>
#   PART=1..16  → specific model+dataset combo
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

# ---- Part → (model, mtag, dataset, dtag, layer) mapping ----
QWEN06="Qwen/Qwen3-0.6B"
QWEN17="Qwen/Qwen3-1.7B"
LLAMA1="meta-llama/Llama-3.2-1B"
LLAMA3="meta-llama/Llama-3.2-3B"

declare -A P_MODEL P_MTAG P_DS P_DTAG P_LAYER

# q06
P_MODEL[1]="$QWEN06"; P_MTAG[1]="q06"; P_DS[1]="scienceqa";      P_DTAG[1]="sci";  P_LAYER[1]="14"
P_MODEL[2]="$QWEN06"; P_MTAG[2]="q06"; P_DS[2]="gsm8k";          P_DTAG[2]="gsm";  P_LAYER[2]="14"
P_MODEL[3]="$QWEN06"; P_MTAG[3]="q06"; P_DS[3]="arc";            P_DTAG[3]="arc";  P_LAYER[3]="14"
P_MODEL[4]="$QWEN06"; P_MTAG[4]="q06"; P_DS[4]="commonsenseqa";  P_DTAG[4]="csqa"; P_LAYER[4]="14"

# q17
P_MODEL[5]="$QWEN17"; P_MTAG[5]="q17"; P_DS[5]="scienceqa";      P_DTAG[5]="sci";  P_LAYER[5]="14"
P_MODEL[6]="$QWEN17"; P_MTAG[6]="q17"; P_DS[6]="gsm8k";          P_DTAG[6]="gsm";  P_LAYER[6]="14"
P_MODEL[7]="$QWEN17"; P_MTAG[7]="q17"; P_DS[7]="arc";            P_DTAG[7]="arc";  P_LAYER[7]="14"
P_MODEL[8]="$QWEN17"; P_MTAG[8]="q17"; P_DS[8]="commonsenseqa";  P_DTAG[8]="csqa"; P_LAYER[8]="14"

# l1
P_MODEL[9]="$LLAMA1";  P_MTAG[9]="l1";  P_DS[9]="scienceqa";      P_DTAG[9]="sci";  P_LAYER[9]="8"
P_MODEL[10]="$LLAMA1"; P_MTAG[10]="l1"; P_DS[10]="gsm8k";         P_DTAG[10]="gsm"; P_LAYER[10]="8"
P_MODEL[11]="$LLAMA1"; P_MTAG[11]="l1"; P_DS[11]="arc";           P_DTAG[11]="arc"; P_LAYER[11]="8"
P_MODEL[12]="$LLAMA1"; P_MTAG[12]="l1"; P_DS[12]="commonsenseqa"; P_DTAG[12]="csqa";P_LAYER[12]="8"

# l3
P_MODEL[13]="$LLAMA3"; P_MTAG[13]="l3"; P_DS[13]="scienceqa";      P_DTAG[13]="sci";  P_LAYER[13]="14"
P_MODEL[14]="$LLAMA3"; P_MTAG[14]="l3"; P_DS[14]="gsm8k";          P_DTAG[14]="gsm";  P_LAYER[14]="14"
P_MODEL[15]="$LLAMA3"; P_MTAG[15]="l3"; P_DS[15]="arc";            P_DTAG[15]="arc";  P_LAYER[15]="14"
P_MODEL[16]="$LLAMA3"; P_MTAG[16]="l3"; P_DS[16]="commonsenseqa";  P_DTAG[16]="csqa"; P_LAYER[16]="14"

N_PARTS=16

# ---- Runner ----
run_one() {
  local mode=$1 C=$2 dataset=$3 dtag=$4 detach_flag=$5 detach_tag=$6 job_idx=$7 \
        cur_model=$8 cur_mtag=$9 cur_layer=${10}

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
    --output_dir $out $extra_args &
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

  local JOB_IDX=0
  for C in "${C_SIZES[@]}"; do
    # V6 baseline
    JOB_IDX=$((JOB_IDX + 1))
    run_one v6 $C $dataset $dtag 0 "base" $JOB_IDX $cur_model $cur_mtag $cur_layer
    if (( JOB_IDX % N_GPUS == 0 )); then wait; fi

    # V9 joint
    JOB_IDX=$((JOB_IDX + 1))
    run_one v9 $C $dataset $dtag 0 "joint" $JOB_IDX $cur_model $cur_mtag $cur_layer
    if (( JOB_IDX % N_GPUS == 0 )); then wait; fi

    # V9 detach
    JOB_IDX=$((JOB_IDX + 1))
    run_one v9 $C $dataset $dtag 1 "detach" $JOB_IDX $cur_model $cur_mtag $cur_layer
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
