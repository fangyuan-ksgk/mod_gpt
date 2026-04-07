#!/bin/bash
set -e

# Ablation on v6 algorithm

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

# ============================================================================
# Shared configuration
# ============================================================================
MASTER_ADDR=127.0.0.1
BASE_PORT=29501
N_GPUS=4

MODEL_NAME="Qwen/Qwen3-0.6B"
DATASET="gsm8k"
MAX_LENGTH=512

LR=1e-5
WARMUP_STEPS=50
BATCH_SIZE=2
NUM_EPOCHS=3

LOG_EVERY=10
EVAL_EVERY=99999
SAVE_EVERY=99999
EVAL_SAMPLES=1300
EVAL_BATCH_SIZE=64
MAX_NEW_TOKENS=256

# Baseline (0405): K=16, abs=32, emb=1.0 → NL=47.1%, K=46.2%, gap=0.9 (0.6B gsm8k)
# Ablate 3 axes vs that baseline

R_V6="--use_v6 --K 16 --abstract_vocab_size 32"
R_V6_RSPAN="--use_v6 --K 16 --abstract_vocab_size 32 --random_mem_span 16,1024"

TIMESTAMP=$(date +%Y%m%d_%H%M)
EXP_IDX=0

# Model + dataset shorthands
M06="Qwen/Qwen3-0.6B"
M17="Qwen/Qwen3-1.7B"
DS_GSM="gsm8k"
DS_SCI="scienceqa"

# ---- Parallel scheduling: 4 x H100 — 1 run/GPU ----
# Usage: run_bg <tag> <model> <dataset> [sorl flags...]
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

# ============================================================================
# Batch 1 — V6 + random_mem_span (16,1024): KV-cache dropping robustness
# Baseline: fixed memory_span_abs=1792 (0405 result)
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 1: V6 random_mem_span [16,1024] — 4 combos (${TIMESTAMP})"
echo "============================================================"

run_bg "v6_rspan_gsm_06"  $M06 $DS_GSM $R_V6_RSPAN
run_bg "v6_rspan_gsm_17"  $M17 $DS_GSM $R_V6_RSPAN
run_bg "v6_rspan_sci_06"  $M06 $DS_SCI $R_V6_RSPAN
run_bg "v6_rspan_sci_17"  $M17 $DS_SCI $R_V6_RSPAN

echo "  4 experiments launched. Waiting..."
wait

# ============================================================================
# Batch 2 — V6 temperature ablation: 0.3 / 0.7 / 1.0 / 2.0 on 0.6B+gsm8k
# Baseline: default temperature=1.0 (0405 result)
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 2: V6 temperature ablation — 0.6B gsm8k (${TIMESTAMP})"
echo "============================================================"

run_bg "v6_temp03"  $M06 $DS_GSM $R_V6  --temperature 0.3
run_bg "v6_temp07"  $M06 $DS_GSM $R_V6  --temperature 0.7
run_bg "v6_temp10"  $M06 $DS_GSM $R_V6  --temperature 1.0
run_bg "v6_temp20"  $M06 $DS_GSM $R_V6  --temperature 2.0

echo "  4 experiments launched. Waiting..."
wait

# ============================================================================
# Batch 3 — V6 max_iterations ablation: iter=4 vs default iter=2
# Baseline: default max_iterations=2 (0405 result)
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 3: V6 max_iterations=4 — 4 combos (${TIMESTAMP})"
echo "============================================================"

run_bg "v6_iter4_gsm_06"  $M06 $DS_GSM $R_V6  --max_iterations 4
run_bg "v6_iter4_gsm_17"  $M17 $DS_GSM $R_V6  --max_iterations 4
run_bg "v6_iter4_sci_06"  $M06 $DS_SCI $R_V6  --max_iterations 4
run_bg "v6_iter4_sci_17"  $M17 $DS_SCI $R_V6  --max_iterations 4

echo "  4 experiments launched. Waiting..."
wait

echo ""
echo "============================================================"
echo "All 12 experiments complete. Results in ./ckpt/sweep_${TIMESTAMP}/"
echo "============================================================"