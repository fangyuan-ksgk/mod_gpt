#!/bin/bash
# ===========================================================================
# Steering V7 sweep — Qwen3-0.6B × {GSM8K, SciQA}
#
# V7 = two-pass steering:
#   Pass 1: read codes from read_layer
#   Pass 2: steer at inject_layers using those codes (per-layer embeddings)
#
# Axes:
#   C_SIZE             ∈ {1, 4, 16, 32}                         (4)
#   code_position       ∈ {first, last}                          (2)
#   L                  ∈ {8, 16, 32}                             (3)
#   layers             ∈ {mid, early, late, mid3}               (4)
#     mid    = [14]          (middle layer only)
#     early  = [3]           (very early layer)
#     late   = [24]          (very late layer)
#     mid3   = [13,14,15]    (consecutive middle layers)
#   steering_direction ∈ {forward, backward}                     (2)
#     forward  = read from early layer, inject later
#     backward = read from late layer, inject earlier (default)
#   routing_mode       ∈ {similar_magnitude}                     (1)
#   steer_lr           ∈ {1e-1, 5e-2, 1e-2}                      (3)
#   model              ∈ {Qwen3-0.6B, Llama-1B}                   (2)
#   dataset            ∈ {gsm8k, sciqa, csqa, obqa}              (4)
#
# Total: 4 × 2 × 3 × 4 × 1 × 3 = 288 experiments per part
# Split into 16 parts by model × dataset × direction:
#   Qwen3-0.6B: gsm8k, sciqa, csqa, obqa × forward/backward = 8 parts
#   Llama-1B:  gsm8k, sciqa, csqa, obqa × forward/backward = 8 parts
#
# Usage: ./sweep_0413_steer.sh <PART>   (PART = 1-16|all)
#
# Best config from ablation-0408: scale=0.5, slr=1e-2, C=32, ep=1
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

# ---- Model & shared hyper-params ----
QWEN06="Qwen/Qwen3-0.6B"
LLAMA1="meta-llama/Llama-3.2-1B"
LR=1e-5
SCALE=0.5
# C_SIZE is now a sweep axis (1, 4, 16, 32)
EPOCHS=1
BATCH_SIZE=2
GRAD_ACCUM=4
WARMUP=50
EVAL_SAMPLES=1319    # GSM8K val size; SciQA has 2224 but we cap for speed
EVAL_BATCH=128
NUM_LOG=5

TIMESTAMP=$(date +%Y%m%d_%H%M)
OUT_ROOT="./ckpt/steer7_${TIMESTAMP}"
mkdir -p "$OUT_ROOT"

# ---- Sweep axes ----
C_SIZES=(1 4 16 32)
CODE_POSITIONS=("first" "last")
LS=(8 16 32)
ROUTING_MODES=("similar_magnitude")
STEER_LRS=("1e-1" "5e-2" "1e-2")

# Layer configs: name -> inject_layers string
declare -A LAYER_ARGS
LAYER_ARGS[mid]="14"
LAYER_ARGS[early]="3"
LAYER_ARGS[late]="24"
LAYER_ARGS[mid3]="13,14,15"
LAYER_NAMES=("mid" "early" "late" "mid3")

# ---- Part → (model, dataset, direction) mapping ----
# Qwen3-0.6B: Parts 1-8
# Llama-1B:  Parts 9-16
declare -A PART_MODEL PART_DATASET PART_DTAG PART_DIR PART_READ

# Qwen3-0.6B Parts 1-8
PART_MODEL[1]="$QWEN06"; PART_DATASET[1]="gsm8k";      PART_DTAG[1]="gsm"; PART_DIR[1]="forward";  PART_READ[1]=7
PART_MODEL[2]="$QWEN06"; PART_DATASET[2]="gsm8k";      PART_DTAG[2]="gsm"; PART_DIR[2]="backward"; PART_READ[2]=27
PART_MODEL[3]="$QWEN06"; PART_DATASET[3]="scienceqa";   PART_DTAG[3]="sci"; PART_DIR[3]="forward";  PART_READ[3]=7
PART_MODEL[4]="$QWEN06"; PART_DATASET[4]="scienceqa";   PART_DTAG[4]="sci"; PART_DIR[4]="backward"; PART_READ[4]=27
PART_MODEL[5]="$QWEN06"; PART_DATASET[5]="commonsenseqa"; PART_DTAG[5]="csqa"; PART_DIR[5]="forward";  PART_READ[5]=7
PART_MODEL[6]="$QWEN06"; PART_DATASET[6]="commonsenseqa"; PART_DTAG[6]="csqa"; PART_DIR[6]="backward"; PART_READ[6]=27
PART_MODEL[7]="$QWEN06"; PART_DATASET[7]="openbookqa";    PART_DTAG[7]="obqa"; PART_DIR[7]="forward";  PART_READ[7]=7
PART_MODEL[8]="$QWEN06"; PART_DATASET[8]="openbookqa";    PART_DTAG[8]="obqa"; PART_DIR[8]="backward"; PART_READ[8]=27

# Llama-1B Parts 9-16
PART_MODEL[9]="$LLAMA1";  PART_DATASET[9]="gsm8k";      PART_DTAG[9]="gsm"; PART_DIR[9]="forward";  PART_READ[9]=7
PART_MODEL[10]="$LLAMA1"; PART_DATASET[10]="gsm8k";      PART_DTAG[10]="gsm"; PART_DIR[10]="backward"; PART_READ[10]=27
PART_MODEL[11]="$LLAMA1"; PART_DATASET[11]="scienceqa";   PART_DTAG[11]="sci"; PART_DIR[11]="forward";  PART_READ[11]=7
PART_MODEL[12]="$LLAMA1"; PART_DATASET[12]="scienceqa";   PART_DTAG[12]="sci"; PART_DIR[12]="backward"; PART_READ[12]=27
PART_MODEL[13]="$LLAMA1"; PART_DATASET[13]="commonsenseqa"; PART_DTAG[13]="csqa"; PART_DIR[13]="forward";  PART_READ[13]=7
PART_MODEL[14]="$LLAMA1"; PART_DATASET[14]="commonsenseqa"; PART_DTAG[14]="csqa"; PART_DIR[14]="backward"; PART_READ[14]=27
PART_MODEL[15]="$LLAMA1"; PART_DATASET[15]="openbookqa";    PART_DTAG[15]="obqa"; PART_DIR[15]="forward";  PART_READ[15]=7
PART_MODEL[16]="$LLAMA1"; PART_DATASET[16]="openbookqa";    PART_DTAG[16]="obqa"; PART_DIR[16]="backward"; PART_READ[16]=27

if [ "$PART" = "all" ]; then
  PARTS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
else
  PARTS=($PART)
fi

for P in "${PARTS[@]}"; do
  model=${PART_MODEL[$P]}
  dataset=${PART_DATASET[$P]}
  dtag=${PART_DTAG[$P]}
  sdir=${PART_DIR[$P]}
  read_layer=${PART_READ[$P]}

  echo ""
  echo "============================================================"
  echo "Part ${P}/16: $(basename $model) + ${dataset} + ${sdir} (read_layer=${read_layer})"
  echo "288 experiments â ${TIMESTAMP}"
  echo "============================================================"

  JOB_IDX=0

  for C in "${C_SIZES[@]}"; do
    for cpos in "${CODE_POSITIONS[@]}"; do
      for L in "${LS[@]}"; do
        for lname in "${LAYER_NAMES[@]}"; do
          for routing in "${ROUTING_MODES[@]}"; do
            for slr in "${STEER_LRS[@]}"; do

              JOB_IDX=$((JOB_IDX + 1))
              gpu=$(( (JOB_IDX - 1) % N_GPUS ))
              port=$((BASE_PORT + P * 100 + JOB_IDX))

              layers=${LAYER_ARGS[$lname]}

              tag="${dtag}_C${C}_${cpos}_L${L}_${lname}_${sdir}_${routing}_slr${slr}"
              out="${OUT_ROOT}/${tag}"

              echo "  [GPU ${gpu}] ${tag}"

              CUDA_VISIBLE_DEVICES=$gpu torchrun \
                --nproc_per_node=1 \
                --master_addr=$MASTER_ADDR \
                --master_port=$port \
                train_steer_pt.py \
                --mode v7 \
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
                --read_layer $read_layer \
                --code_position $cpos \
                --routing_mode $routing \
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
    done
  done
  wait

  echo "Part ${P} complete."
done

echo ""
echo "============================================================"
echo "Steering V7 sweep complete. Results in ${OUT_ROOT}/"
echo "============================================================"
echo ""
echo "To gather results:"
echo "  grep 'accuracy' ${OUT_ROOT}/*/train.log"
