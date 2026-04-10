#!/bin/bash
set -e

# Re-run v6 / v7 / v7o reference experiments with --untie_embeddings
# to test whether decoupling lm_head from embed_tokens improves SoRL.
#
# Reference table (tied weights):
#   model        | v6 NL_G K8_G NL_S K8_S | v7 ...  | v7o ...
#   Qwen3-0.6B   | 43.0 43.5 48.3 49.5    | 47.1 47.4 61.8 61.2 | 44.1 45.1 50.8 52.6
#   Qwen3-1.7B   | 61.0 59.5 60.0 54.4    | 63.2 63.2 63.2 61.0 | 61.1 60.8 56.4 57.2
#   Qwen3-4B     | 77.5 78.5 68.1 63.8    | 74.5 74.6 67.6 64.6 | 77.3 78.1 67.1 64.4
#   Llama-1B     |   —                     |   —                  | 19.2 19.8 55.2 54.5
#   Llama-3B     |   —                     |   —                  | 44.0 45.9 81.7 63.5

# --- nvidia pod specifics ------
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
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4

# ============================================================================
# Shared configuration
# ============================================================================
MASTER_ADDR=127.0.0.1
BASE_PORT=29600
N_GPUS=4       # logical slots: 2 GPUs × 2 parallel = 4 slots
GPU_OFFSET=0   # physical GPU index to start from

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

TIMESTAMP=$(date +%Y%m%d_%H%M)
EXP_IDX=0

# Model shorthands
M06="Qwen/Qwen3-0.6B"
M17="Qwen/Qwen3-1.7B"
M4B="Qwen/Qwen3-4B"
ML1="meta-llama/Llama-3.2-1B"
ML3="meta-llama/Llama-3.2-3B"

DS_GSM="gsm8k"
DS_SCI="scienceqa"

# ── Common flag blocks (match reference configs exactly, + untie_embeddings) ──
UNTIE="--untie_embeddings"
ABS_BASE="--abstract_vocab_size 32 --prefix_abs --K 8 --abs_prefix_max 8"

# v6: diagonal self-routing (lm_head abstract rows frozen to diagonal)
V6="--use_v6 $ABS_BASE"

# v7: deep supervision, similar_magnitude routing
V7="--use_v7 --abs_routing_mode similar_magnitude --alpha_traj 1.0 $ABS_BASE --max_iterations 4"

# v7o: outer-loop variant of v7
V7O="--use_v7 --v7_outer --abs_routing_mode similar_magnitude --alpha_traj 1.0 $ABS_BASE --max_iterations 4"

run_bg() {
  EXP_IDX=$((EXP_IDX + 1))
  local idx=$EXP_IDX
  local gpu=$(( (idx - 1) % 2 + GPU_OFFSET ))  # 2 physical GPUs, round-robin
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
# Phase A — v6 (diagonal routing) + untie_embeddings
#   Qwen3-0.6B, 1.7B, 4B × GSM8K + SciQA = 6 runs
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "Phase A: v6 + untie_embeddings"

run_bg "v6u_06b_gsm" $M06 $DS_GSM $V6 $UNTIE
run_bg "v6u_06b_sci" $M06 $DS_SCI $V6 $UNTIE
run_bg "v6u_17b_gsm" $M17 $DS_GSM $V6 $UNTIE
run_bg "v6u_17b_sci" $M17 $DS_SCI $V6 $UNTIE
wait
run_bg "v6u_4b_gsm"  $M4B $DS_GSM $V6 $UNTIE
run_bg "v6u_4b_sci"  $M4B $DS_SCI $V6 $UNTIE
wait

# ═══════════════════════════════════════════════════════════════════════════
# Phase B — v7 (deep supervision, similar_magnitude) + untie_embeddings
#   Qwen3-0.6B, 1.7B, 4B × GSM8K + SciQA = 6 runs
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "Phase B: v7 + untie_embeddings"

run_bg "v7u_06b_gsm" $M06 $DS_GSM $V7 $UNTIE
run_bg "v7u_06b_sci" $M06 $DS_SCI $V7 $UNTIE
run_bg "v7u_17b_gsm" $M17 $DS_GSM $V7 $UNTIE
run_bg "v7u_17b_sci" $M17 $DS_SCI $V7 $UNTIE
wait
run_bg "v7u_4b_gsm"  $M4B $DS_GSM $V7 $UNTIE
run_bg "v7u_4b_sci"  $M4B $DS_SCI $V7 $UNTIE
wait

# ═══════════════════════════════════════════════════════════════════════════
# Phase C — v7o (outer-loop) + untie_embeddings
#   Qwen3-0.6B, 1.7B, 4B, Llama-1B, Llama-3B × GSM8K + SciQA = 10 runs
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "Phase C: v7o + untie_embeddings"

run_bg "v7ou_06b_gsm" $M06 $DS_GSM $V7O $UNTIE
run_bg "v7ou_06b_sci" $M06 $DS_SCI $V7O $UNTIE
run_bg "v7ou_17b_gsm" $M17 $DS_GSM $V7O $UNTIE
run_bg "v7ou_17b_sci" $M17 $DS_SCI $V7O $UNTIE
wait
run_bg "v7ou_4b_gsm"  $M4B $DS_GSM $V7O $UNTIE
run_bg "v7ou_4b_sci"  $M4B $DS_SCI $V7O $UNTIE
run_bg "v7ou_l1b_gsm" $ML1 $DS_GSM $V7O $UNTIE
run_bg "v7ou_l1b_sci" $ML1 $DS_SCI $V7O $UNTIE
wait
run_bg "v7ou_l3b_gsm" $ML3 $DS_GSM $V7O $UNTIE
run_bg "v7ou_l3b_sci" $ML3 $DS_SCI $V7O $UNTIE
wait

echo ""
echo "============================================================"
echo "All done — untie_embeddings ablation"
echo "  Phase A (v6):  6 runs  (3 models × 2 datasets)"
echo "  Phase B (v7):  6 runs  (3 models × 2 datasets)"
echo "  Phase C (v7o): 10 runs (5 models × 2 datasets)"
echo "  Total:         22 runs"
echo "  Results in ./ckpt/sweep_${TIMESTAMP}/"
echo "============================================================"
