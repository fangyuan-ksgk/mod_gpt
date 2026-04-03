#!/bin/bash
# ============================================================================
# Self-routing ablation: K × abstract_vocab_size
# 4 experiments on 4 GPUs in parallel (1 batch)
# ============================================================================

set -e

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
MAX_ITERS=2

TIMESTAMP=$(date +%Y%m%d_%H%M)
EXP_IDX=0

# ---- Parallel scheduling: 4 x GPU — 1 run/GPU ----

run_bg() {
  EXP_IDX=$((EXP_IDX + 1))
  local idx=$EXP_IDX
  local gpu=$(( (idx - 1) % N_GPUS ))
  local port=$((BASE_PORT + idx))
  local tag=$1; shift
  local output_dir="./ckpt/selfroute_ablate_${TIMESTAMP}/exp${idx}_${tag}"

  echo "  Exp ${idx}: ${tag}  [GPU=${gpu}]  port=${port}"

  CUDA_VISIBLE_DEVICES=$gpu torchrun \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$port \
    train_sorl_post.py \
    --model_name $MODEL_NAME \
    --dataset $DATASET \
    --max_length $MAX_LENGTH \
    --use_v6 \
    --max_iterations $MAX_ITERS \
    --temperature 0.0 \
    --alpha_traj 1.0 \
    --emb_lr_mult 1.0 \
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
    "$@" &
}

# ============================================================================
# Batch 1 — K × abstract_vocab_size (4 experiments, 4 GPUs)
# ============================================================================
echo ""
echo "============================================================"
echo "Self-routing ablation: K × abs_vocab (${TIMESTAMP})"
echo "============================================================"

# 1. K=4, abs_vocab=64
run_bg "K4_abs64" \
  --K 4 --eval_K 4 --abstract_vocab_size 64

# 2. K=4, abs_vocab=128
run_bg "K4_abs128" \
  --K 4 --eval_K 4 --abstract_vocab_size 128

# 3. K=8, abs_vocab=64
run_bg "K8_abs64" \
  --K 8 --eval_K 8 --abstract_vocab_size 64

# 4. K=8, abs_vocab=128
run_bg "K8_abs128" \
  --K 8 --eval_K 8 --abstract_vocab_size 128

echo "  4 experiments launched. Waiting..."
wait

echo ""
echo "All self-routing ablations complete."
