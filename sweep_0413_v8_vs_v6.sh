#!/bin/bash
# ===========================================================================
# V8 (STE) vs V6 comparison on Qwen3-0.6B + ScienceQA
#
# Same settings for both modes; sweep over the configs that matter:
#   - layers: mid(14), ml(14,24)
#   - C: 4, 32
#   - L: 2, 4
#   - slr: 5e-2, 1e-1
#
# Total: 2 modes × 2 layers × 2 C × 2 L × 2 slr = 32 runs
# ===========================================================================
set -euo pipefail

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4

MASTER_ADDR=127.0.0.1
BASE_PORT=29700
N_GPUS=4

MODEL="Qwen/Qwen3-0.6B"
DATASET="scienceqa"
LR=1e-5
SCALE=0.5
EPOCHS=1
BATCH_SIZE=2
GRAD_ACCUM=4
WARMUP=50
EVAL_SAMPLES=1319
EVAL_BATCH=128
NUM_LOG=5

MODES=("v6" "v8")
C_SIZES=(4 32)
LS=(4)
SLRS=("5e-2")
LAYER_NAMES=("mid" "ml")
LAYER_MID="14"
LAYER_ML="14,24"

TIMESTAMP=$(date +%Y%m%d_%H%M)
OUT_ROOT="./ckpt/v8v6_sci_${TIMESTAMP}"
mkdir -p "$OUT_ROOT"

JOB_IDX=0

run_exp() {
  local mode=$1
  local tag=$2
  local C=$3
  local layers=$4
  local slr=$5
  local chunk_L=$6
  local ple_flag=$7   # "" or "--per_layer_emb"

  JOB_IDX=$((JOB_IDX + 1))
  local gpu=$(( (JOB_IDX - 1) % N_GPUS ))
  local port=$((BASE_PORT + JOB_IDX))
  local out="${OUT_ROOT}/${tag}"

  echo "  [GPU ${gpu}] ${tag}"

  CUDA_VISIBLE_DEVICES=$gpu torchrun \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$port \
    train_steer_pt.py \
    --mode $mode \
    --model_name $MODEL \
    --dataset $DATASET \
    --num_epochs $EPOCHS \
    --lr $LR \
    --steer_lr $slr \
    --warmup_steps $WARMUP \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --C_SIZE $C \
    --L $chunk_L \
    --scale $SCALE \
    --inject_layers $layers \
    --code_position first \
    $ple_flag \
    --eval_every 99999 \
    --save_every 99999 \
    --eval_samples $EVAL_SAMPLES \
    --eval_batch_size $EVAL_BATCH \
    --num_log_samples $NUM_LOG \
    --log_every 10 \
    --output_dir $out &

  if (( JOB_IDX % N_GPUS == 0 )); then wait; fi
}

echo "=== V8 vs V6: Qwen3-0.6B + ScienceQA ==="
echo "Output: $OUT_ROOT"

for mode in "${MODES[@]}"; do
  for slr in "${SLRS[@]}"; do
    for C in "${C_SIZES[@]}"; do
      for chunk_L in "${LS[@]}"; do
        for ln in "${LAYER_NAMES[@]}"; do
          if [[ "$ln" == "mid" ]]; then
            layers="$LAYER_MID"
            ple=""
          else
            layers="$LAYER_ML"
            ple="--per_layer_emb"
          fi
          tag="${mode}_C${C}_L${chunk_L}_slr${slr}_${ln}"
          run_exp "$mode" "$tag" "$C" "$layers" "$slr" "$chunk_L" "$ple"
        done
      done
    done
  done
done

wait
echo "=== All done. Gather results: ==="
echo "grep 'eval_accuracy' ${OUT_ROOT}/*/log.txt | sort -t= -k2 -rn | head -20"
