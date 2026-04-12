#!/bin/bash
# ===========================================================================
# Steering V7 sweep — Qwen3-0.6B × {GSM8K, SciQA}
#
# V7 = two-pass steering:
#   Pass 1: read codes from read_layer
#   Pass 2: steer at inject_layers using those codes (per-layer embeddings)
#
# Axes:
#   code_position       ∈ {first, last}                          (2)
#   L                  ∈ {4, 8, 16}                             (3)
#   layers             ∈ {mid, multi3}                          (2)
#     mid    = [14]        (middle layer only)
#     multi3 = [7,14,21]   (early + mid + late)
#   steering_direction ∈ {forward, backward}                     (2)
#     forward  = read from early layer, inject later
#     backward = read from late layer, inject earlier (default)
#   routing_mode       ∈ {diagonal, similar_magnitude}           (2)
#   steer_lr           ∈ {1e-3, 1e-2}                           (2)
#   dataset            ∈ {gsm8k, scienceqa}                     (2)
#
# Total: 2 × 3 × 2 × 2 × 2 × 2 × 2 = 96 experiments
# Each ~20 min on q06 → 24 batches of 4 GPUs → ~8 hours
#
# Best config from ablation-0408: scale=0.5, slr=1e-2, C=32, ep=1
# ===========================================================================
set -euo pipefail

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4

MASTER_ADDR=127.0.0.1
BASE_PORT=29500
N_GPUS=4

# ---- Model & shared hyper-params ----
MODEL="Qwen/Qwen3-0.6B"
LR=1e-5
SCALE=0.5
C_SIZE=32
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
CODE_POSITIONS=("first" "last")
LS=(4 8 16)
DATASETS=("gsm8k" "scienceqa")
DTAGS=("gsm" "sci")
STEERING_DIRS=("forward" "backward")
ROUTING_MODES=("diagonal" "similar_magnitude")
STEER_LRS=("1e-3" "1e-2")

# Layer configs: name -> inject_layers string
declare -A LAYER_ARGS
LAYER_ARGS[mid]="14"
LAYER_ARGS[multi3]="7,14,21"

LAYER_NAMES=("mid" "multi3")

echo "============================================================"
echo "Steering V7 sweep (two-pass) — ${TIMESTAMP}"
echo "96 experiments: 2 code_pos × 3 L × 2 layers × 2 direction × 2 routing × 2 lr × 2 datasets"
echo "============================================================"

JOB_IDX=0

for ds_idx in "${!DATASETS[@]}"; do
  dataset=${DATASETS[$ds_idx]}
  dtag=${DTAGS[$ds_idx]}

  for cpos in "${CODE_POSITIONS[@]}"; do
    for L in "${LS[@]}"; do
      for lname in "${LAYER_NAMES[@]}"; do
        for sdir in "${STEERING_DIRS[@]}"; do
          for routing in "${ROUTING_MODES[@]}"; do
            for slr in "${STEER_LRS[@]}"; do

              JOB_IDX=$((JOB_IDX + 1))
              gpu=$(( (JOB_IDX - 1) % N_GPUS ))
              port=$((BASE_PORT + JOB_IDX))

              layers=${LAYER_ARGS[$lname]}

              # Set read_layer based on steering direction
              if [ "$sdir" = "forward" ]; then
                read_layer=7  # early layer for forward steering
              else
                read_layer=27  # late layer for backward steering (default)
              fi

              tag="${dtag}_${cpos}_L${L}_${lname}_${sdir}_${routing}_slr${slr}"
              out="${OUT_ROOT}/${tag}"

              echo "  [GPU ${gpu}] ${tag}"

              CUDA_VISIBLE_DEVICES=$gpu torchrun \
                --nproc_per_node=1 \
                --master_addr=$MASTER_ADDR \
                --master_port=$port \
                train_steer_pt.py \
                --mode v7 \
                --model_name $MODEL \
                --dataset $dataset \
                --num_epochs $EPOCHS \
                --lr $LR \
                --steer_lr $slr \
                --warmup_steps $WARMUP \
                --batch_size $BATCH_SIZE \
                --gradient_accumulation_steps $GRAD_ACCUM \
                --C_SIZE $C_SIZE \
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
done
wait

echo ""
echo "============================================================"
echo "Steering V7 sweep complete. Results in ${OUT_ROOT}/"
echo "============================================================"
echo ""
echo "To gather results:"
echo "  grep 'accuracy' ${OUT_ROOT}/*/train.log"
