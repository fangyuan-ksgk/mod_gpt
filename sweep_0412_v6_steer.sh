#!/bin/bash
# ===========================================================================
# V6 Steering sweep - Qwen3-0.6B & Llama-1B × {GSM8K, SciQA, CSQA, OBQA}
#
# V6 = single-pass self-routed steering:
#   At inject_layer(s), extract codes via diagonal routing (argmax of last C_SIZE dims)
#   Apply steering vectors to the same layer's hidden states
#
# Best from ablation-0408:
#   - exp5: C=32, scale=0.5, slr=1e-1, ep=1, L=16, layer=[14] -> 47.2%
#   - exp11: C=32, scale=0.5, slr=1e-2, ep=3, L=16, layer=[14] -> 49.0%
#
# Axes:
#   C_SIZE             Î {4, 32}                                 (2)
#   scale              Î {0.5}                                   (1)
#   steer_lr           Î {1e-1, 5e-2, 1e-2}                      (3)
#   L                  Î {8, 16, 32}                             (3)
#   epochs             Î {1, 2, 3}                               (3)
#   layers             Î {mid, late, mid_late}                   (3)
#     mid     = [14]         (middle layer only)
#     late    = [24]         (very late layer)
#     mid_late= [14,24]       (middle + late layers)
#   model              Î {Qwen3-0.6B, Llama-1B}                   (2)
#   dataset            Î {gsm8k, sciqa, csqa, obqa}              (4)
#
# Total: 2 × 1 × 3 × 3 × 3 × 3 = 162 experiments per model-dataset pair
# Split into 8 parts by model × dataset:
#   Part 1-4: Qwen3-0.6B (gsm8k, sciqa, csqa, obqa)
#   Part 5-8: Llama-1B (gsm8k, sciqa, csqa, obqa)
#
# Usage: ./sweep_0412_v6_steer.sh <PART>   (PART = 1-8|all)
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
EPOCHS=1
BATCH_SIZE=2
GRAD_ACCUM=4
WARMUP=50
EVAL_SAMPLES=1319    # GSM8K val size; SciQA has 2224 but we cap for speed
EVAL_BATCH=128
NUM_LOG=5

TIMESTAMP=$(date +%Y%m%d_%H%M)
OUT_ROOT="./ckpt/steer6_${TIMESTAMP}"
mkdir -p "$OUT_ROOT"

# ---- Sweep axes ----
C_SIZES=(4 32)
SCALES=(0.5)
STEER_LRS=("1e-1" "5e-2" "1e-2")
LS=(8 16 32)
EPOCHS_LIST=(1 2 3)

# Layer configs: name -> inject_layers string
declare -A LAYER_ARGS
LAYER_ARGS[mid]="14"
LAYER_ARGS[late]="24"
LAYER_ARGS[mid_late]="14,24"
LAYER_NAMES=("mid" "late" "mid_late")

# ---- Part â (model, dataset) mapping ----
# Qwen3-0.6B: Parts 1-4
# Llama-1B:  Parts 5-8
declare -A PART_MODEL PART_DATASET PART_DTAG

# Qwen3-0.6B Parts 1-4
PART_MODEL[1]="$QWEN06"; PART_DATASET[1]="gsm8k";      PART_DTAG[1]="gsm"
PART_MODEL[2]="$QWEN06"; PART_DATASET[2]="scienceqa";   PART_DTAG[2]="sci"
PART_MODEL[3]="$QWEN06"; PART_DATASET[3]="commonsenseqa"; PART_DTAG[3]="csqa"
PART_MODEL[4]="$QWEN06"; PART_DATASET[4]="openbookqa";    PART_DTAG[4]="obqa"

# Llama-1B Parts 5-8
PART_MODEL[5]="$LLAMA1";  PART_DATASET[5]="gsm8k";      PART_DTAG[5]="gsm"
PART_MODEL[6]="$LLAMA1";  PART_DATASET[6]="scienceqa";   PART_DTAG[6]="sci"
PART_MODEL[7]="$LLAMA1";  PART_DATASET[7]="commonsenseqa"; PART_DTAG[7]="csqa"
PART_MODEL[8]="$LLAMA1";  PART_DATASET[8]="openbookqa";    PART_DTAG[8]="obqa"

if [ "$PART" = "all" ]; then
  PARTS=(1 2 3 4 5 6 7 8)
else
  PARTS=($PART)
fi

for P in "${PARTS[@]}"; do
  model=${PART_MODEL[$P]}
  dataset=${PART_DATASET[$P]}
  dtag=${PART_DTAG[$P]}

  echo ""
  echo "============================================================"
  echo "Part ${P}/8: $(basename $model) + ${dataset}"
  echo "162 experiments â ${TIMESTAMP}"
  echo "============================================================"

  JOB_IDX=0

  for C in "${C_SIZES[@]}"; do
    for scale in "${SCALES[@]}"; do
      for slr in "${STEER_LRS[@]}"; do
        for L in "${LS[@]}"; do
          for ep in "${EPOCHS_LIST[@]}"; do
            for lname in "${LAYER_NAMES[@]}"; do

              JOB_IDX=$((JOB_IDX + 1))
              gpu=$(( (JOB_IDX - 1) % N_GPUS ))
              port=$((BASE_PORT + P * 100 + JOB_IDX))

              layers=${LAYER_ARGS[$lname]}

              tag="${dtag}_C${C}_s${scale}_slr${slr}_L${L}_ep${ep}_${lname}"
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
                --num_epochs $ep \
                --lr $LR \
                --steer_lr $slr \
                --warmup_steps $WARMUP \
                --batch_size $BATCH_SIZE \
                --gradient_accumulation_steps $GRAD_ACCUM \
                --C_SIZE $C \
                --L $L \
                --scale $scale \
                --inject_layers $layers \
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
echo "V6 steering sweep complete. Results in ${OUT_ROOT}/"
echo "============================================================"
echo ""
echo "To gather results:"
echo "  grep 'accuracy' ${OUT_ROOT}/*/train.log"
