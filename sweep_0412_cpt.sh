#!/bin/bash
set -e

# ============================================================================
# Mixed-dataset CPT experiment: SoRL v7 vs SFT
#
# Train on: gsm8k,scienceqa,arc,mmlu,commonsenseqa (mixed)
# Eval on:  each of the 5 datasets individually (auto at end of training)
#
# 2 runs per model: SoRL v7 + SFT baseline
# Start with q06, expand to other models if results are promising.
# ============================================================================

# --- nvidia pod specifics ------
DUMMY_CONFIG_PATH="./dummy_tuner_config.txt"
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
# Config
# ============================================================================
MASTER_ADDR=127.0.0.1
BASE_PORT=29501

LR=1e-5
WARMUP_STEPS=50
BATCH_SIZE=2
GRAD_ACCUM=4          # EBS = 2 * 4 = 8
NUM_EPOCHS=3
EVAL_SAMPLES=1000
EVAL_BATCH_SIZE=128
NUM_LOG_SAMPLES=3

TIMESTAMP=$(date +%Y%m%d_%H%M)

# Models
M06="Qwen/Qwen3-0.6B"
M16="Qwen/Qwen3-1.7B"

# Mixed dataset string
MIX_DS="gsm8k,scienceqa,arc,mmlu,commonsenseqa"

echo "============================================================"
echo "Mixed-CPT experiment | ${TIMESTAMP}"
echo "  Train: ${MIX_DS} | ep=${NUM_EPOCHS} | EBS=${BATCH_SIZE}x${GRAD_ACCUM}=8"
echo "  Eval:  all 5 datasets (auto)"
echo "============================================================"

# ============================================================================
# Run 1: SoRL v7 on mixed datasets (GPU 0)
# ============================================================================
SORL_DIR="./ckpt/cpt_${TIMESTAMP}/sorl_v7_q06_mix"

echo ""
echo ">>> [GPU 0] SoRL v7 — q06 on mixed datasets"

CUDA_VISIBLE_DEVICES=0 torchrun \
  --nproc_per_node=1 \
  --master_addr=$MASTER_ADDR \
  --master_port=$((BASE_PORT + 1)) \
  train_sorl_post.py \
  --model_name $M06 \
  --dataset $MIX_DS \
  --num_epochs $NUM_EPOCHS \
  --lr $LR \
  --warmup_steps $WARMUP_STEPS \
  --batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $GRAD_ACCUM \
  --use_v7 \
  --abs_routing_mode similar_magnitude \
  --prefix_abs --abs_prefix_max 8 \
  --K 8 --eval_K 8 \
  --max_iterations 2 \
  --emb_lr_mult 10.0 \
  --abstract_vocab_size 128 \
  --eval_every 99999 \
  --save_every 99999 \
  --eval_samples $EVAL_SAMPLES \
  --eval_batch_size $EVAL_BATCH_SIZE \
  --num_log_samples $NUM_LOG_SAMPLES \
  --log_every 10 \
  --output_dir $SORL_DIR &

# ============================================================================
# Run 2: SFT on mixed datasets (GPU 1)
# ============================================================================
SFT_DIR="./ckpt/cpt_${TIMESTAMP}/sft_q06_mix"

echo ">>> [GPU 1] SFT — q06 on mixed datasets"

CUDA_VISIBLE_DEVICES=1 torchrun \
  --nproc_per_node=1 \
  --master_addr=$MASTER_ADDR \
  --master_port=$((BASE_PORT + 2)) \
  train_sft_pt.py \
  --model_name $M06 \
  --dataset $MIX_DS \
  --num_epochs $NUM_EPOCHS \
  --lr $LR \
  --warmup_steps $WARMUP_STEPS \
  --batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $GRAD_ACCUM \
  --eval_every 99999 \
  --save_every 99999 \
  --eval_samples $EVAL_SAMPLES \
  --eval_batch_size $EVAL_BATCH_SIZE \
  --num_log_samples $NUM_LOG_SAMPLES \
  --log_every 10 \
  --log_samples_every 99999 \
  --output_dir $SFT_DIR &

wait

echo ""
echo "============================================================"
echo "Both runs complete. Results in ./ckpt/cpt_${TIMESTAMP}/"
echo "  SoRL: ${SORL_DIR}/train.log"
echo "  SFT:  ${SFT_DIR}/train.log"
echo "============================================================"
