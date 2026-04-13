#!/bin/bash
# ===========================================================================
# V6 Steering sweep — 4 models × 8 datasets
#
# V6 = single-pass self-routed steering:
#   At inject_layer(s), extract codes via diagonal routing
#   Apply steering vectors to the same layer's hidden states
#
# Updated from 0412 observations:
#   - L=8 > L=16,32 → try even smaller: L=1, 2, 4
#   - ml > mid > late → add more multi-layer configs (sp3, sp5)
#   - slr=5e-2 & 1e-1 good → add 2e-1
#   - ep=1 sufficient for sweep (tune epochs later with best config)
#
# Axes:
#   C_SIZE    ∈ {4, 32}                                 (2)
#   steer_lr  ∈ {5e-2, 1e-1, 2e-1}                      (3)
#   L         ∈ {1, 2, 4}                               (3)
#   layers    ∈ {mid, ml, sp3, sp5}                     (4)
#     28-layer (Qwen06, Qwen17, Llama3):
#       mid  = [14]           ml   = [14,24]
#       sp3  = [3,14,24]      sp5  = [3,7,14,21,24]
#     16-layer (Llama1):
#       mid  = [8]            ml   = [8,14]
#       sp3  = [2,8,14]       sp5  = [2,4,8,12,14]
#   model     ∈ {Qwen06, Qwen17, Llama1, Llama3}       (4)
#   dataset   ∈ {gsm8k, sciqa, arc, mmlu,
#                 csqa, boolq, obqa, aqua}              (8)
#
# Total: 2 × 3 × 3 × 4 = 72 experiments per part
# Split into 32 parts by model × dataset:
#   Part  1-8:  Qwen3-0.6B
#   Part  9-16: Qwen3-1.7B
#   Part 17-24: Llama-1B
#   Part 25-32: Llama-3B
#
# Usage: ./sweep_0413_v6_steer.sh <PART>   (PART = 1-32|all)
# ===========================================================================
set -euo pipefail

PART=${1:-all}

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4

MASTER_ADDR=127.0.0.1
BASE_PORT=29500
N_GPUS=4

# ---- Models ----
QWEN06="Qwen/Qwen3-0.6B"
QWEN17="Qwen/Qwen3-1.7B"
LLAMA1="meta-llama/Llama-3.2-1B"
LLAMA3="meta-llama/Llama-3.2-3B"

# ---- Shared hyper-params ----
LR=1e-5
SCALE=0.5
EPOCHS=1
BATCH_SIZE=2
GRAD_ACCUM=4
WARMUP=50
EVAL_SAMPLES=1319
EVAL_BATCH=128
NUM_LOG=5

TIMESTAMP=$(date +%Y%m%d_%H%M)
OUT_ROOT="./ckpt/steer6_${TIMESTAMP}"
mkdir -p "$OUT_ROOT"

# ---- Sweep axes ----
C_SIZES=(4 32)
STEER_LRS=("5e-2" "1e-1" "2e-1")
LS=(1 2 4)
LAYER_NAMES=("mid" "ml" "sp3" "sp5")

# Layer configs: 28-layer models (Qwen06, Qwen17, Llama3)
declare -A LAYERS_28
LAYERS_28[mid]="14"
LAYERS_28[ml]="14,24"
LAYERS_28[sp3]="3,14,24"
LAYERS_28[sp5]="3,7,14,21,24"

# Layer configs: 16-layer models (Llama1)
declare -A LAYERS_16
LAYERS_16[mid]="8"
LAYERS_16[ml]="8,14"
LAYERS_16[sp3]="2,8,14"
LAYERS_16[sp5]="2,4,8,12,14"

# ---- Part → (model, dataset) mapping ----
DATASETS=("gsm8k" "scienceqa" "commonsenseqa" "openbookqa" "arc" "mmlu" "boolq" "aqua")
DTAGS=("gsm" "sci" "csqa" "obqa" "arc" "mmlu" "boolq" "aqua")

declare -A PART_MODEL PART_MTAG PART_DATASET PART_DTAG PART_NL

MODELS=("$QWEN06" "$QWEN17" "$LLAMA1" "$LLAMA3")
MTAGS=("q06" "q17" "l1" "l3")
NLAYERS=(28 28 16 28)

P=0
for mi in 0 1 2 3; do
  for di in 0 1 2 3 4 5 6 7; do
    P=$((P + 1))
    PART_MODEL[$P]="${MODELS[$mi]}"
    PART_MTAG[$P]="${MTAGS[$mi]}"
    PART_DATASET[$P]="${DATASETS[$di]}"
    PART_DTAG[$P]="${DTAGS[$di]}"
    PART_NL[$P]="${NLAYERS[$mi]}"
  done
done

if [ "$PART" = "all" ]; then
  PARTS=($(seq 1 32))
else
  PARTS=($PART)
fi

for P in "${PARTS[@]}"; do
  model=${PART_MODEL[$P]}
  mtag=${PART_MTAG[$P]}
  dataset=${PART_DATASET[$P]}
  dtag=${PART_DTAG[$P]}
  nl=${PART_NL[$P]}

  echo ""
  echo "============================================================"
  echo "Part ${P}/32: ${mtag} + ${dataset} (${nl} layers)"
  echo "72 experiments → ${TIMESTAMP}"
  echo "============================================================"

  JOB_IDX=0

  for C in "${C_SIZES[@]}"; do
    for slr in "${STEER_LRS[@]}"; do
      for L in "${LS[@]}"; do
        for lname in "${LAYER_NAMES[@]}"; do

          JOB_IDX=$((JOB_IDX + 1))
          gpu=$(( (JOB_IDX - 1) % N_GPUS ))
          port=$((BASE_PORT + P * 100 + JOB_IDX))

          # Select layer config based on model's layer count
          if [ "$nl" = "28" ]; then
            layers=${LAYERS_28[$lname]}
          else
            layers=${LAYERS_16[$lname]}
          fi

          # Use per-layer embeddings for multi-layer configs
          ple_flag=""
          if [ "$lname" != "mid" ]; then
            ple_flag="--per_layer_emb"
          fi

          tag="${mtag}_${dtag}_C${C}_slr${slr}_L${L}_${lname}"
          out="${OUT_ROOT}/${tag}"

          echo "  [GPU ${gpu}] ${tag}"

          CUDA_VISIBLE_DEVICES=$gpu torchrun \
            --nproc_per_node=1 \
            --master_addr=$MASTER_ADDR \
            --master_port=$port \
            train_steer_pt.py \
            --mode v6 \
            --model_name $model \
            --dataset $dataset \
            --num_epochs $EPOCHS \
            --lr $LR \
            --steer_lr $slr \
            --warmup_steps $WARMUP \
            --batch_size $BATCH_SIZE \
            --gradient_accumulation_steps $GRAD_ACCUM \
            --C_SIZE $C \
            --L $L \
            --scale $SCALE \
            --inject_layers $layers \
            $ple_flag \
            --eval_every 99999 \
            --save_every 99999 \
            --eval_samples $EVAL_SAMPLES \
            --eval_batch_size $EVAL_BATCH \
            --num_log_samples $NUM_LOG \
            --log_every 10 \
            --output_dir $out &

          if (( JOB_IDX % N_GPUS == 0 )); then wait; fi

        done
      done
    done
  done
  wait

  echo "Part ${P} complete."
done

echo ""
echo "============================================================"
echo "V6 steering sweep complete. Results in ${OUT_ROOT}/"
echo "============================================================"
echo ""
echo "To gather results:"
echo "  grep 'accuracy' ${OUT_ROOT}/*/train.log"
