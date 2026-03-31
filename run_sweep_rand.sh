#!/bin/bash
# ============================================================================
# SoRL Randomization Sweep — Qwen3-0.6B / GSM8K / 4×A10
#
# 5 experiments: baseline + 4 randomization ablations (one-at-a-time)
# Key metric: K=4 vs K=None accuracy gap (goal: K=4 ≥ K=None)
#
# Exp 1: v3 baseline (no randomization)
# Exp 2: random_K = (2,4,6,8)        — vary chunk granularity
# Exp 3: strip_suffix = (0.3,1.0)    — abs in prefix only
# Exp 4: compress_prefix = (0.0,0.6) — drop NL in prefix
# Exp 5: random_mem_span = (128,1792) — vary memory span
#
# Schedule: 2 batches (batch 1: exp 1-4 on 4 GPUs, batch 2: exp 5 on GPU 0)
# ============================================================================

set -e

# --- Environment ---
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4

# Uncomment below for nvidia pod specifics if needed:
# DUMMY_CONFIG_PATH="/workspace/mod_gpt/dummy_tuner_config.txt"
# rm -f "$DUMMY_CONFIG_PATH" && touch "$DUMMY_CONFIG_PATH"
# export NCCL_TUNER_CONFIG_PATH="$DUMMY_CONFIG_PATH"
# export NCCL_TUNER_PLUGIN=""
# export NCCL_NET_PLUGIN=""
# export NCCL_SOCKET_IFNAME=lo
# export NCCL_IB_DISABLE=1
# export NCCL_DEBUG=WARN

# ============================================================================
# Shared configuration
# ============================================================================
MASTER_ADDR=127.0.0.1
BASE_PORT=29801
N_GPUS=4

MODEL_NAME="Qwen/Qwen3-0.6B"
DATASET="gsm8k"
MAX_LENGTH=512

LR=1e-5
LR_WARMUP=50
BATCH_SIZE=2
GRAD_ACCUM=4
NUM_EPOCHS=3

LOG_EVERY=10
EVAL_EVERY=99999
SAVE_EVERY=99999
EVAL_SAMPLES=1300
EVAL_BATCH_SIZE=64
MAX_NEW_TOKENS=256

# v3 baseline config (shuffle r=1.0 γ=0.5)
V3_BASE="--use_v3 --alpha_traj 1.0 --alpha_abs 0.5 --corrupt_method shuffle --corrupt_ratio 1.0 --gamma_contrastive 0.5 --emb_lr_mult 1.0"

TIMESTAMP=$(date +%Y%m%d_%H%M)
SWEEP_DIR="./ckpt/sweep_rand_${TIMESTAMP}"
mkdir -p "$SWEEP_DIR"
EXP_IDX=0

# ---- Helper: launch one experiment on a specific GPU ----
run_bg() {
  EXP_IDX=$((EXP_IDX + 1))
  local idx=$EXP_IDX
  local gpu=$(( (idx - 1) % N_GPUS ))
  local port=$((BASE_PORT + idx))
  local tag=$1; shift
  local output_dir="./ckpt/sweep_rand_${TIMESTAMP}/exp${idx}_${tag}"

  echo "  Exp ${idx}: ${tag}  [GPU=${gpu}]  port=${port}"

  CUDA_VISIBLE_DEVICES=$gpu torchrun \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$port \
    train_ablate_sanity.py \
    --model_name $MODEL_NAME \
    --dataset $DATASET \
    --max_length $MAX_LENGTH \
    --lr $LR \
    --warmup_steps $LR_WARMUP \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --num_epochs $NUM_EPOCHS \
    --log_every $LOG_EVERY \
    --eval_every $EVAL_EVERY \
    --save_every $SAVE_EVERY \
    --eval_samples $EVAL_SAMPLES \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --max_new_tokens $MAX_NEW_TOKENS \
    --output_dir $output_dir \
    "$@" > "${output_dir}.log" 2>&1 &
}

# ============================================================================
# Batch 1 (GPU 0-3): Baseline + 3 randomizations — all 4 in parallel
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 1: 4 experiments on 4 GPUs (${TIMESTAMP})"
echo "============================================================"

# Exp 1: v3 baseline — no randomization (GPU 0)
run_bg "v3_baseline" \
  $V3_BASE

# Exp 2: random_K = (2,4,6,8) (GPU 1)
run_bg "v3_randK" \
  $V3_BASE --random_K "2,4,6,8"

# Exp 3: strip_suffix = (0.3, 1.0) (GPU 2)
run_bg "v3_strip" \
  $V3_BASE --strip_suffix "0.3,1.0"

# Exp 4: compress_prefix = (0.0, 0.6) (GPU 3)
run_bg "v3_compress" \
  $V3_BASE --compress_prefix "0.0,0.6"

echo "  4 experiments launched. Waiting..."
wait

# ============================================================================
# Batch 2 (GPU 0): Random memory span
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 2: Random memory span (${TIMESTAMP})"
echo "============================================================"

# Exp 5: random_mem_span = (128, 1792) (GPU 0)
run_bg "v3_randmem" \
  $V3_BASE --random_mem_span "128,1792"

echo "  1 experiment launched. Waiting..."
wait

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================================"
echo "All 5 experiments complete. Results in ./ckpt/sweep_rand_${TIMESTAMP}/"
echo "============================================================"
echo ""
echo "Experiment matrix (all v3 shuffle r=1.0 γ=0.5, 3 epochs):"
echo "  1  v3_baseline  — no randomization (control)"
echo "  2  v3_randK     — random_K=(2,4,6,8)"
echo "  3  v3_strip     — strip_suffix=(0.3,1.0)"
echo "  4  v3_compress  — compress_prefix=(0.0,0.6)"
echo "  5  v3_randmem   — random_mem_span=(128,1792)"
echo ""
echo "Key metric: K=4 vs K=None accuracy gap in final eval."
echo "Logs: ./ckpt/sweep_rand_${TIMESTAMP}/exp*_.log"
