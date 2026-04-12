#!/bin/bash
# ===========================================================================
# V2 insight sweep — distributed across 4 nodes × 4 GPUs
#
# Usage:  bash sweep_0412_v2.sh <NODE_ID>    # NODE_ID ∈ {0, 1, 2, 3}
#
# Each node runs its assigned experiments on GPUs 0-3.
# All results go to ./ckpt/v2sweep_<TIMESTAMP>/
#
# Experiment allocation (57 total, ~15 per node):
#   Node 0: Stage 3 — gsm/sci/arc       × {v1,v2,v7o}  = 9 exps  + 6 from Stage 4
#   Node 1: Stage 3 — mmlu/csqa/obqa     × {v1,v2,v7o}  = 9 exps  + 6 from Stage 4
#   Node 2: Stage 3 — boolq              × {v1,v2,v7o}  = 3 exps  + 12 from Stage 4
#   Node 3: Stage 4b — V2 sweep on SciQA                = 18 exps
#
# ~25 min per experiment → each node finishes in ~2 hours
# ===========================================================================
set -euo pipefail

NODE_ID=${1:?Usage: bash sweep_0412_v2.sh <NODE_ID>  (0-3)}
if (( NODE_ID < 0 || NODE_ID > 3 )); then
  echo "ERROR: NODE_ID must be 0-3, got $NODE_ID"
  exit 1
fi

# ---- NCCL / env ----
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
MASTER_ADDR=127.0.0.1
BASE_PORT=29500
N_GPUS=4

# ---- Shared hyper-params ----
M06="Qwen/Qwen3-0.6B"
LR=1e-5
WARMUP_STEPS=50
BATCH_SIZE=2
GRAD_ACCUM=4
EVAL_SAMPLES=2000
EVAL_BATCH_SIZE=256
NUM_LOG_SAMPLES=5

TIMESTAMP=$(date +%Y%m%d_%H%M)
OUT_ROOT="./ckpt/v2sweep_${TIMESTAMP}"
mkdir -p "$OUT_ROOT"

# ---- Helper: launch one experiment ----
JOB_IDX=0
launch() {
  local tag=$1; shift
  local dataset=$1; shift
  # remaining args: extra flags

  JOB_IDX=$((JOB_IDX + 1))
  local gpu=$(( (JOB_IDX - 1) % N_GPUS ))
  local port=$((BASE_PORT + NODE_ID * 100 + JOB_IDX))
  local out="${OUT_ROOT}/${tag}"

  echo "  [Node ${NODE_ID} GPU ${gpu}] ${tag}  dataset=${dataset}"

  CUDA_VISIBLE_DEVICES=$gpu torchrun \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$port \
    train_sorl_post.py \
    --model_name $M06 \
    --dataset $dataset \
    --num_epochs 1 \
    --lr $LR \
    --warmup_steps $WARMUP_STEPS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --use_v7 \
    --abs_routing_mode similar_magnitude \
    --prefix_abs --abs_prefix_max 8 \
    --K 8 --eval_K 8 \
    --max_iterations 2 \
    --emb_lr_mult 1.0 \
    --abstract_vocab_size 128 \
    --eval_every 99999 \
    --save_every 99999 \
    --eval_samples $EVAL_SAMPLES \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --num_log_samples $NUM_LOG_SAMPLES \
    --log_every 10 \
    "$@" \
    --output_dir $out &

  if (( JOB_IDX % N_GPUS == 0 )); then wait; fi
}

# ---- Helper: launch V2 hyperparam sweep job ----
launch_v2sweep() {
  local dataset=$1
  local dtag=$2
  local elr=$3
  local v=$4
  local k=$5

  JOB_IDX=$((JOB_IDX + 1))
  local gpu=$(( (JOB_IDX - 1) % N_GPUS ))
  local port=$((BASE_PORT + NODE_ID * 100 + JOB_IDX))
  local tag="v2_${dtag}_elr${elr}_v${v}_k${k}"
  local out="${OUT_ROOT}/${tag}"

  echo "  [Node ${NODE_ID} GPU ${gpu}] ${tag}"

  CUDA_VISIBLE_DEVICES=$gpu torchrun \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$port \
    train_sorl_post.py \
    --model_name $M06 \
    --dataset $dataset \
    --num_epochs 1 \
    --lr $LR \
    --warmup_steps $WARMUP_STEPS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --use_v7 \
    --abs_routing_mode similar_magnitude \
    --prefix_abs --abs_prefix_max $k \
    --K $k --eval_K $k \
    --max_iterations 2 \
    --emb_lr_mult $elr \
    --abstract_vocab_size $v \
    --separate_abs_params \
    --eval_every 99999 \
    --save_every 99999 \
    --eval_samples $EVAL_SAMPLES \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --num_log_samples $NUM_LOG_SAMPLES \
    --log_every 10 \
    --output_dir $out &

  if (( JOB_IDX % N_GPUS == 0 )); then wait; fi
}

echo "============================================================"
echo "V2 insight sweep — Node ${NODE_ID} | ${TIMESTAMP}"
echo "============================================================"

# ===========================================================================
# STAGE 3: V1 vs V2 vs v7o — 7 datasets × 3 variants = 21 experiments
#
# Node 0: gsm8k, scienceqa, arc      (9 exps)
# Node 1: mmlu, commonsenseqa, obqa  (9 exps)
# Node 2: boolq                      (3 exps)
# ===========================================================================

S3_DATASETS_0=("gsm8k" "scienceqa" "arc")
S3_TAGS_0=("gsm" "sci" "arc")
S3_DATASETS_1=("mmlu" "commonsenseqa" "openbookqa")
S3_TAGS_1=("mmlu" "csqa" "obqa")
S3_DATASETS_2=("boolq")
S3_TAGS_2=("boolq")

run_stage3() {
  local -n ds_arr=$1
  local -n dt_arr=$2
  for di in "${!ds_arr[@]}"; do
    local ds=${ds_arr[$di]}
    local dt=${dt_arr[$di]}
    # V1
    launch "s3_${dt}_v1" "$ds"
    # V2
    launch "s3_${dt}_v2" "$ds" --separate_abs_params
    # v7o
    launch "s3_${dt}_v7o" "$ds" --v7_outer
  done
}

# ===========================================================================
# STAGE 4: V2 hyperparam sweep — elr × V × K
#
# 3 × 3 × 2 = 18 per dataset
# Node 0: GSM8K  exps 1-6   (fill remaining GPUs after Stage 3)
# Node 1: GSM8K  exps 7-12
# Node 2: GSM8K  exps 13-18 (fill remaining GPUs after Stage 3)
# Node 3: SciQA  exps 1-18  (full node)
# ===========================================================================

ELRS=(1.0 5.0 10.0)
VS=(64 128 256)
KS=(4 8)

# Flatten into indexed arrays for splitting
ALL_ELR=()
ALL_V=()
ALL_K=()
for elr in "${ELRS[@]}"; do
  for v in "${VS[@]}"; do
    for k in "${KS[@]}"; do
      ALL_ELR+=("$elr")
      ALL_V+=("$v")
      ALL_K+=("$k")
    done
  done
done
# ALL_ELR/ALL_V/ALL_K now have 18 entries each (indices 0-17)

run_v2sweep_range() {
  local dataset=$1
  local dtag=$2
  local start=$3
  local end=$4
  for (( i=start; i<end; i++ )); do
    launch_v2sweep "$dataset" "$dtag" "${ALL_ELR[$i]}" "${ALL_V[$i]}" "${ALL_K[$i]}"
  done
}

# ===========================================================================
# Dispatch by NODE_ID
# ===========================================================================

case $NODE_ID in
  0)
    echo "--- Stage 3: gsm8k, scienceqa, arc (9 exps) ---"
    run_stage3 S3_DATASETS_0 S3_TAGS_0
    wait
    echo "--- Stage 4: V2 sweep on GSM8K [1-6] (6 exps) ---"
    run_v2sweep_range gsm8k gsm 0 6
    wait
    ;;
  1)
    echo "--- Stage 3: mmlu, csqa, obqa (9 exps) ---"
    run_stage3 S3_DATASETS_1 S3_TAGS_1
    wait
    echo "--- Stage 4: V2 sweep on GSM8K [7-12] (6 exps) ---"
    run_v2sweep_range gsm8k gsm 6 12
    wait
    ;;
  2)
    echo "--- Stage 3: boolq (3 exps) ---"
    run_stage3 S3_DATASETS_2 S3_TAGS_2
    wait
    echo "--- Stage 4: V2 sweep on GSM8K [13-18] (6+6=12 exps) ---"
    run_v2sweep_range gsm8k gsm 12 18
    wait
    echo "--- Stage 4b: V2 sweep on ARC [1-6] (6 exps) ---"
    run_v2sweep_range arc arc 0 6
    wait
    ;;
  3)
    echo "--- Stage 4b: V2 sweep on SciQA (18 exps) ---"
    run_v2sweep_range scienceqa sci 0 18
    wait
    ;;
esac

wait

echo ""
echo "============================================================"
echo "Node ${NODE_ID} complete. Results in ${OUT_ROOT}/"
echo "============================================================"
