#!/bin/bash
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

# dataset: (gsm8k, scienceqa)
# model: (qwne0.6B, qwen1.7B)
# sorl config: v1, v2, v3 (shuffle & noise), v6

R_V1="--alpha_info_gain 1.0 --alpha_abs 0.5"
R_V1_E10="--alpha_info_gain 1.0 --alpha_abs 0.5 --emb_lr_mult 10.0"
R_V2="--use_v2 --alpha_traj 1.0 --alpha_abs 0.5"
R_V3_R10="--use_v3 --alpha_traj 1.0 --alpha_abs 0.5 --corrupt_method shuffle --corrupt_ratio 1.0 --gamma_contrastive 0.5"
R_V3_NOISE="--use_v3 --alpha_traj 1.0 --alpha_abs 0.5 --corrupt_method noise --corrupt_ratio 1.0 --gamma_contrastive 0.5"
R_V6="--use_v6"
R_V6_E10="--use_v6 --emb_lr_mult 10.0"

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

# VQ pretrain config shorthands (2000 steps, defaults match SoRLConfig)
VQ2K="--vq_abs_pretrain_steps 2000"

# ============================================================================
# Batch 1 — VQ-init vs baseline V6: K=8 abs=64, GSM8K, both model sizes
# Hypothesis: VQ-pretrained centroids (mNN≈0.82) → faster convergence + better gap
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 1: VQ-init V6 K=8 abs=64 GSM8K (${TIMESTAMP})"
echo "============================================================"

run_bg "v6_vq_K8_abs64_gsm_06" $M06 $DS_GSM --use_v6 --abstract_vocab_size 64 --K 8 --eval_K 8 $VQ2K
run_bg "v6_vq_K8_abs64_gsm_17" $M17 $DS_GSM --use_v6 --abstract_vocab_size 64 --K 8 --eval_K 8 $VQ2K
run_bg "v6_K8_abs64_gsm_06"    $M06 $DS_GSM --use_v6 --abstract_vocab_size 64 --K 8 --eval_K 8
run_bg "v6_K8_abs64_gsm_17"    $M17 $DS_GSM --use_v6 --abstract_vocab_size 64 --K 8 --eval_K 8

echo "  4 experiments launched. Waiting..."
wait

# ============================================================================
# Batch 2 — VQ-init vs baseline V6: K=8 abs=32, GSM8K, both model sizes
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 2: VQ-init V6 K=8 abs=32 GSM8K (${TIMESTAMP})"
echo "============================================================"

run_bg "v6_vq_K8_abs32_gsm_06" $M06 $DS_GSM --use_v6 --abstract_vocab_size 32 --K 8 --eval_K 8 $VQ2K
run_bg "v6_vq_K8_abs32_gsm_17" $M17 $DS_GSM --use_v6 --abstract_vocab_size 32 --K 8 --eval_K 8 $VQ2K
run_bg "v6_K8_abs32_gsm_06"    $M06 $DS_GSM --use_v6 --abstract_vocab_size 32 --K 8 --eval_K 8
run_bg "v6_K8_abs32_gsm_17"    $M17 $DS_GSM --use_v6 --abstract_vocab_size 32 --K 8 --eval_K 8

echo "  4 experiments launched. Waiting..."
wait

echo ""
echo "============================================================"
echo "All experiments complete. Results in ./ckpt/sweep_${TIMESTAMP}/"
echo "============================================================"