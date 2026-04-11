#!/bin/bash
set -e

# SorlModelWrapperV2 ablation: separate abs_embed / abs_proj
# Untied abs_embed (trainable) + frozen abs_proj (similar_magnitude routing)
# NL embed_tokens ↔ lm_head remain tied
#
# Models:  Qwen3-0.6B, Qwen3-1.7B
# Dataset: GSM8K
# Sweep:   emb_lr_mult ∈ {1.0, 10.0}
# → 2 models × 2 emb_lr = 4 experiments (1 batch on 4 GPUs)

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
# Shared configuration (inherited from sweep_0411.sh)
# ============================================================================
MASTER_ADDR=127.0.0.1
BASE_PORT=29501
N_GPUS=4

LR=1e-5
WARMUP_STEPS=50
BATCH_SIZE=2
NUM_EPOCHS=1

LOG_EVERY=10
EVAL_EVERY=99999
SAVE_EVERY=99999
EVAL_BATCH_SIZE=256

TIMESTAMP=$(date +%Y%m%d_%H%M)
EXP_IDX=0

# Model shorthands
M06="Qwen/Qwen3-0.6B"
M17="Qwen/Qwen3-1.7B"

# Dataset
DS_GSM="gsm8k"

eval_samples_for_dataset() {
  local dataset=$1
  case "$dataset" in
    gsm8k) echo 1319 ;;
    *)     echo 1000 ;;
  esac
}

# Common SoRL flags (same as sweep_0411.sh baseline)
LORA="--use_lora --lora_rank 16 --lora_alpha 32"
BASE="--use_v7 --abs_routing_mode similar_magnitude \
  --prefix_abs --abs_prefix_max 4 --K 4 \
  --max_iterations 2 --eval_K 4 $LORA"

run_bg() {
  EXP_IDX=$((EXP_IDX + 1))
  local idx=$EXP_IDX
  local gpu=$(( (idx - 1) % N_GPUS ))
  local port=$((BASE_PORT + idx))
  local tag=$1; shift
  local model=$1; shift
  local dataset=$1; shift
  local grad_accum=$((8 / BATCH_SIZE))
  local output_dir="./ckpt/sweep_v2abs_${TIMESTAMP}/exp${idx}_${tag}"
  local eval_samples; eval_samples=$(eval_samples_for_dataset "$dataset")

  echo "  Exp ${idx}: ${tag}  model=$(basename $model)  dataset=${dataset}  [GPU=${gpu}]"

  CUDA_VISIBLE_DEVICES=$gpu torchrun \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$port \
    train_sorl_post.py \
    --model_name $model \
    --dataset $dataset \
    --lr $LR \
    --warmup_steps $WARMUP_STEPS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $grad_accum \
    --num_epochs $NUM_EPOCHS \
    --log_every $LOG_EVERY \
    --eval_every $EVAL_EVERY \
    --save_every $SAVE_EVERY \
    --eval_samples $eval_samples \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --output_dir $output_dir \
    --separate_abs_params \
    "$@" &
}

echo "=== V2 Separate abs_embed/abs_proj Sweep === $(date)"

# =============================================================================
# {q06, q17} × GSM8K × emb_lr_mult {1.0, 10.0}
# All with --separate_abs_params (SorlModelWrapperV2)
# 4 experiments, 1 batch
# =============================================================================

echo "GSM8K V2: {q06, q17} × emb_lr {1.0, 10.0}"
run_bg "gsm_v2_q06_emb1"   $M06 $DS_GSM $BASE --abstract_vocab_size 128 --emb_lr_mult 1.0  --eval_batch_size 8
run_bg "gsm_v2_q06_emb10"  $M06 $DS_GSM $BASE --abstract_vocab_size 128 --emb_lr_mult 10.0 --eval_batch_size 8
run_bg "gsm_v2_q17_emb1"   $M17 $DS_GSM $BASE --abstract_vocab_size 128 --emb_lr_mult 1.0  --eval_batch_size 8
run_bg "gsm_v2_q17_emb10"  $M17 $DS_GSM $BASE --abstract_vocab_size 128 --emb_lr_mult 10.0 --eval_batch_size 8
wait

echo ""
echo "=== V2 abs_embed/abs_proj sweep complete. $(date) ==="
