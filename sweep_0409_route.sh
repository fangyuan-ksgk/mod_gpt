#!/bin/bash
set -e

# Ablation on v7 "deep supervision"

# --- nvidia pod  specifics ------
DUMMY_CONFIG_PATH="/workspace/mod_gpt/dummy_tuner_config.txt"
rm -f "$DUMMY_CONFIG_PATH"
touch "$DUMMY_CONFIG_PATH"

export NCCL_TUNER_CONFIG_PATH="$DUMMY_CONFIG_PATH"
export NCCL_TUNER_PLUGIN=""
export NCCL_NET_PLUGIN=""
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

# ============================================================================
# Shared configuration
# ============================================================================
MASTER_ADDR=127.0.0.1
BASE_PORT=29501
N_GPUS=2

MODEL_NAME="Qwen/Qwen3-0.6B"
DATASET="gsm8k"
MAX_LENGTH=512

LR=1e-5
WARMUP_STEPS=50
BATCH_SIZE=2
NUM_EPOCHS=1

LOG_EVERY=10
EVAL_EVERY=99999
SAVE_EVERY=99999
EVAL_SAMPLES=1300
EVAL_BATCH_SIZE=64
MAX_NEW_TOKENS=256

# ============================================================================================
# similar_magnitude routing + v7 deep supervision sweep
#
# Structure:
#   Phase A — 0.6B baseline across ALL datasets (default config: pfx=8, iter=4, V=32)
#   Phase B — Config ablations (0.6B, GSM8K only)
#             B1: max_iterations
#             B2: abs_prefix_max
#             B3: abstract_vocab_size
#   Phase C — Other models across ALL datasets (default config)
#   Phase D — Outer-loop ablation
#             D1: 0.6B across ALL datasets
#             D2: Other models on GSM8K
# ============================================================================================

TIMESTAMP=$(date +%Y%m%d_%H%M)
EXP_IDX=0

# Model shorthands
M06="Qwen/Qwen3-0.6B"
M17="Qwen/Qwen3-1.7B"
ML1="meta-llama/Llama-3.2-1B"
ML3="meta-llama/Llama-3.2-3B"
M4B="Qwen/Qwen3-4B"

# Dataset shorthands
DS_GSM="gsm8k"
DS_ARC="arc"
DS_SCI="scienceqa"
DS_MMLU="mmlu"
DS_CSQA="commonsenseqa"

ALL_DS=("$DS_GSM" "$DS_ARC" "$DS_SCI" "$DS_MMLU" "$DS_CSQA")
OTHER_MODELS=("$M17" "$ML1" "$ML3" "$M4B")
OTHER_MODEL_TAGS=("17b" "l1b" "l3b" "q4b")

# Common SoRL flags (default config)
ABS="--abstract_vocab_size 32 --prefix_abs --alpha_traj 1.0 --abs_routing_mode similar_magnitude"
DEFAULT="--use_v7 $ABS --abs_prefix_max 8 --max_iterations 4"

run_bg() {
  EXP_IDX=$((EXP_IDX + 1))
  local idx=$EXP_IDX
  local gpu=$(( (idx - 1) % N_GPUS ))
  local port=$((BASE_PORT + idx))
  local tag=$1; shift
  local model=$1; shift
  local dataset=$1; shift
  local grad_accum=$((8 / BATCH_SIZE))
  local output_dir="./ckpt/sweep_${TIMESTAMP}/exp${idx}_${tag}"

  echo "  Exp ${idx}: ${tag}  model=$(basename $model)  dataset=${dataset}  [GPU=${gpu}]"

  CUDA_VISIBLE_DEVICES=$gpu torchrun \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$port \
    train_sorl_post.py \
    --model_name $model \
    --dataset $dataset \
    --max_length $MAX_LENGTH \
    --lr $LR \
    --warmup_steps $WARMUP_STEPS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $grad_accum \
    --num_epochs $NUM_EPOCHS \
    --log_every $LOG_EVERY \
    --eval_every $EVAL_EVERY \
    --save_every $SAVE_EVERY \
    --eval_samples $EVAL_SAMPLES \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --max_new_tokens $MAX_NEW_TOKENS \
    --output_dir $output_dir \
    "$@" &
}

# ═══════════════════════════════════════════════════════════════════════════
# Phase A — 0.6B baseline across ALL datasets (default config)
#   5 datasets × 1 model = 5 runs
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "Phase A: 0.6B v7 baseline across all datasets"

run_bg "v7_06b_gsm" $M06 $DS_GSM $DEFAULT
run_bg "v7_06b_arc" $M06 $DS_ARC $DEFAULT
wait
run_bg "v7_06b_sci" $M06 $DS_SCI $DEFAULT
run_bg "v7_06b_mmlu" $M06 $DS_MMLU $DEFAULT
wait
run_bg "v7_06b_csqa" $M06 $DS_CSQA $DEFAULT
wait

# ═══════════════════════════════════════════════════════════════════════════
# Phase B — Config ablations (0.6B, GSM8K)
# ═══════════════════════════════════════════════════════════════════════════

# --- B1: max_iterations (iter=4 already in Phase A) ----------------------
echo ""
echo "Phase B1: max_iterations ablation — i1, i2, i6, i8"

run_bg "v7_06b_gsm_i1" $M06 $DS_GSM \
  --use_v7 $ABS --abs_prefix_max 8 --max_iterations 1
run_bg "v7_06b_gsm_i2" $M06 $DS_GSM \
  --use_v7 $ABS --abs_prefix_max 8 --max_iterations 2
wait
run_bg "v7_06b_gsm_i6" $M06 $DS_GSM \
  --use_v7 $ABS --abs_prefix_max 8 --max_iterations 6
run_bg "v7_06b_gsm_i8" $M06 $DS_GSM \
  --use_v7 $ABS --abs_prefix_max 8 --max_iterations 8
wait

# --- B2: abs_prefix_max (pfx=8 already in Phase A) ----------------------
echo ""
echo "Phase B2: abs_prefix_max ablation — pfx1, pfx2, pfx4, pfx16"

run_bg "v7_06b_gsm_pfx1" $M06 $DS_GSM \
  --use_v7 $ABS --abs_prefix_max 1 --max_iterations 4
run_bg "v7_06b_gsm_pfx2" $M06 $DS_GSM \
  --use_v7 $ABS --abs_prefix_max 2 --max_iterations 4
wait
run_bg "v7_06b_gsm_pfx4" $M06 $DS_GSM \
  --use_v7 $ABS --abs_prefix_max 4 --max_iterations 4
run_bg "v7_06b_gsm_pfx16" $M06 $DS_GSM \
  --use_v7 $ABS --abs_prefix_max 16 --max_iterations 4
wait

# --- B3: abstract_vocab_size (V=32 already in Phase A) -------------------
#   --abstract_vocab_size after $ABS overrides the 32 baked into ABS.
echo ""
echo "Phase B3: vocab size ablation — V8, V16, V64, V128"

run_bg "v7_06b_gsm_V8" $M06 $DS_GSM \
  --use_v7 $ABS --abs_prefix_max 8 --max_iterations 4 --abstract_vocab_size 8
run_bg "v7_06b_gsm_V16" $M06 $DS_GSM \
  --use_v7 $ABS --abs_prefix_max 8 --max_iterations 4 --abstract_vocab_size 16
wait
run_bg "v7_06b_gsm_V64" $M06 $DS_GSM \
  --use_v7 $ABS --abs_prefix_max 8 --max_iterations 4 --abstract_vocab_size 64
run_bg "v7_06b_gsm_V128" $M06 $DS_GSM \
  --use_v7 $ABS --abs_prefix_max 8 --max_iterations 4 --abstract_vocab_size 128
wait

# ═══════════════════════════════════════════════════════════════════════════
# Phase C — Other models across ALL datasets (default config)
#   4 models × 5 datasets = 20 runs
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "Phase C: other models × all datasets"

for ds_idx in "${!ALL_DS[@]}"; do
  ds="${ALL_DS[$ds_idx]}"
  echo "  --- dataset: $ds ---"
  for m_idx in "${!OTHER_MODELS[@]}"; do
    m="${OTHER_MODELS[$m_idx]}"
    mt="${OTHER_MODEL_TAGS[$m_idx]}"
    run_bg "v7_${mt}_${ds}" $m $ds $DEFAULT

    # launch 2 at a time (N_GPUS=2), then wait
    if (( (m_idx + 1) % N_GPUS == 0 )); then
      wait
    fi
  done
  wait  # ensure all runs for this dataset finish before next
done

# ═══════════════════════════════════════════════════════════════════════════
# Phase D — Outer-loop ablation (--v7_outer)
# ═══════════════════════════════════════════════════════════════════════════

# --- D1: 0.6B across ALL datasets (outer-loop) --------------------------
echo ""
echo "Phase D1: v7 outer-loop — 0.6B across all datasets"

run_bg "v7o_06b_gsm" $M06 $DS_GSM \
  --use_v7 --v7_outer $ABS --abs_prefix_max 8 --max_iterations 4
run_bg "v7o_06b_arc" $M06 $DS_ARC \
  --use_v7 --v7_outer $ABS --abs_prefix_max 8 --max_iterations 4
wait
run_bg "v7o_06b_sci" $M06 $DS_SCI \
  --use_v7 --v7_outer $ABS --abs_prefix_max 8 --max_iterations 4
run_bg "v7o_06b_mmlu" $M06 $DS_MMLU \
  --use_v7 --v7_outer $ABS --abs_prefix_max 8 --max_iterations 4
wait
run_bg "v7o_06b_csqa" $M06 $DS_CSQA \
  --use_v7 --v7_outer $ABS --abs_prefix_max 8 --max_iterations 4
wait

# --- D2: Other models on GSM8K (outer-loop) ------------------------------
echo ""
echo "Phase D2: v7 outer-loop — other models on GSM8K"

run_bg "v7o_17b_gsm" $M17 $DS_GSM \
  --use_v7 --v7_outer $ABS --abs_prefix_max 8 --max_iterations 4
run_bg "v7o_l1b_gsm" $ML1 $DS_GSM \
  --use_v7 --v7_outer $ABS --abs_prefix_max 8 --max_iterations 4
wait
run_bg "v7o_l3b_gsm" $ML3 $DS_GSM \
  --use_v7 --v7_outer $ABS --abs_prefix_max 8 --max_iterations 4
run_bg "v7o_q4b_gsm" $M4B $DS_GSM \
  --use_v7 --v7_outer $ABS --abs_prefix_max 8 --max_iterations 4
wait

echo ""
echo "All done."
echo "  Phase A:  5 runs  (0.6B × 5 datasets)"
echo "  Phase B: 12 runs  (ablations: 4 iter + 4 pfx + 4 vocab)"
echo "  Phase C: 20 runs  (4 models × 5 datasets)"
echo "  Phase D:  9 runs  (outer: 5 datasets + 4 models)"
echo "  Total:   46 experiments"