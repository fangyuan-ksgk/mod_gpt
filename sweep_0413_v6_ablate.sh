#!/bin/bash
# ===========================================================================
# V6 ablation: test new features on Qwen3-0.6B + GSM8K
#
# Baseline: C=32, slr=5e-2, ep=1, scale=0.5, layer=[14], L=4
# (best region from 0412 sweep)
#
# Axes:
#   1. C_SIZE ∈ {1, 4, 32}                    — C=1 = unconditional baseline
#   2. routing_mode ∈ {diagonal, similar_magnitude}
#   3. per_layer_emb ∈ {false, true}           — only meaningful with ml layers
#   4. code_position ∈ {first, last}           — forward vs backward steering
#   5. layers ∈ {mid, ml}                      — single vs multi-layer
#
# Fixed: slr=5e-2, ep=1, scale=0.5, L=4, model=Qwen3-0.6B, dataset=gsm8k
#
# Experiments (see below):  22 runs
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
DATASET="gsm8k"
LR=1e-5
SLR="5e-2"
SCALE=0.5
EPOCHS=1
L=4
BATCH_SIZE=2
GRAD_ACCUM=4
WARMUP=50
EVAL_SAMPLES=1319
EVAL_BATCH=128
NUM_LOG=5

TIMESTAMP=$(date +%Y%m%d_%H%M)
OUT_ROOT="./ckpt/steer6_ablate_${TIMESTAMP}"
mkdir -p "$OUT_ROOT"

JOB_IDX=0

run_exp() {
  local tag=$1
  local C=$2
  local layers=$3
  local routing=$4
  local cpos=$5
  local ple_flag=$6   # "" or "--per_layer_emb"

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
    --mode v6 \
    --model_name $MODEL \
    --dataset $DATASET \
    --num_epochs $EPOCHS \
    --lr $LR \
    --steer_lr $SLR \
    --warmup_steps $WARMUP \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --C_SIZE $C \
    --L $L \
    --scale $SCALE \
    --inject_layers $layers \
    --routing_mode $routing \
    --code_position $cpos \
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

echo "============================================================"
echo "V6 ablation: Qwen3-0.6B + GSM8K | ${TIMESTAMP}"
echo "============================================================"

# ── Group 1: C=1 baseline (routing/cpos don't matter, only 1 code) ──
run_exp "C1_mid"           1  14     diagonal first ""
run_exp "C1_ml"            1  14,24  diagonal first ""

# ── Group 2: C=4, mid layer — routing × code_position ──
run_exp "C4_mid_diag_first"   4  14  diagonal           first ""
run_exp "C4_mid_diag_last"    4  14  diagonal           last  ""
run_exp "C4_mid_simmag_first" 4  14  similar_magnitude  first ""
run_exp "C4_mid_simmag_last"  4  14  similar_magnitude  last  ""

# ── Group 3: C=32, mid layer — routing × code_position ──
run_exp "C32_mid_diag_first"   32 14  diagonal           first ""
run_exp "C32_mid_diag_last"    32 14  diagonal           last  ""
run_exp "C32_mid_simmag_first" 32 14  similar_magnitude  first ""
run_exp "C32_mid_simmag_last"  32 14  similar_magnitude  last  ""

# ── Group 4: C=4, ml layers — routing × code_position × per_layer_emb ──
run_exp "C4_ml_diag_first"         4  14,24  diagonal           first ""
run_exp "C4_ml_diag_first_ple"     4  14,24  diagonal           first "--per_layer_emb"
run_exp "C4_ml_diag_last"          4  14,24  diagonal           last  ""
run_exp "C4_ml_diag_last_ple"      4  14,24  diagonal           last  "--per_layer_emb"
run_exp "C4_ml_simmag_first"       4  14,24  similar_magnitude  first ""
run_exp "C4_ml_simmag_first_ple"   4  14,24  similar_magnitude  first "--per_layer_emb"

# ── Group 5: C=32, ml layers — routing × code_position × per_layer_emb ──
run_exp "C32_ml_diag_first"        32 14,24  diagonal           first ""
run_exp "C32_ml_diag_first_ple"    32 14,24  diagonal           first "--per_layer_emb"
run_exp "C32_ml_diag_last"         32 14,24  diagonal           last  ""
run_exp "C32_ml_diag_last_ple"     32 14,24  diagonal           last  "--per_layer_emb"
run_exp "C32_ml_simmag_first"      32 14,24  similar_magnitude  first ""
run_exp "C32_ml_simmag_first_ple"  32 14,24  similar_magnitude  first "--per_layer_emb"

wait

echo ""
echo "============================================================"
echo "V6 ablation complete. ${JOB_IDX} experiments in ${OUT_ROOT}/"
echo "============================================================"
echo ""
echo "To gather results:"
echo "  grep 'accuracy' ${OUT_ROOT}/*/train.log"
