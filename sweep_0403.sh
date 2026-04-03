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
    train_ablate_sanity.py \
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
# Batch 1 — V1 (emb_lr=1.0) × all combos
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 1: V1 e1.0 — 4 combos (${TIMESTAMP})"
echo "============================================================"

run_bg "v1_e1_gsm_06"  $M06 $DS_GSM $R_V1
run_bg "v1_e1_gsm_17"  $M17 $DS_GSM $R_V1
run_bg "v1_e1_sci_06"  $M06 $DS_SCI $R_V1
run_bg "v1_e1_sci_17"  $M17 $DS_SCI $R_V1

echo "  4 experiments launched. Waiting..."
wait

# ============================================================================
# Batch 2 — V1 (emb_lr=10.0) × all combos
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 2: V1 e10.0 — 4 combos (${TIMESTAMP})"
echo "============================================================"

run_bg "v1_e10_gsm_06" $M06 $DS_GSM $R_V1_E10
run_bg "v1_e10_gsm_17" $M17 $DS_GSM $R_V1_E10
run_bg "v1_e10_sci_06" $M06 $DS_SCI $R_V1_E10
run_bg "v1_e10_sci_17" $M17 $DS_SCI $R_V1_E10

echo "  4 experiments launched. Waiting..."
wait

# ============================================================================
# Batch 3 — V2 (emb_lr=1.0) × all combos
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 3: V2 e1.0 — 4 combos (${TIMESTAMP})"
echo "============================================================"

run_bg "v2_e1_gsm_06"  $M06 $DS_GSM $R_V2
run_bg "v2_e1_gsm_17"  $M17 $DS_GSM $R_V2
run_bg "v2_e1_sci_06"  $M06 $DS_SCI $R_V2
run_bg "v2_e1_sci_17"  $M17 $DS_SCI $R_V2

echo "  4 experiments launched. Waiting..."
wait

# ============================================================================
# Batch 4 — V3 shuffle r=1.0 (emb_lr=1.0) × all combos
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 4: V3 shuffle r=1.0 — 4 combos (${TIMESTAMP})"
echo "============================================================"

run_bg "v3_shuf_gsm_06" $M06 $DS_GSM $R_V3_R10
run_bg "v3_shuf_gsm_17" $M17 $DS_GSM $R_V3_R10
run_bg "v3_shuf_sci_06" $M06 $DS_SCI $R_V3_R10
run_bg "v3_shuf_sci_17" $M17 $DS_SCI $R_V3_R10

echo "  4 experiments launched. Waiting..."
wait

# ============================================================================
# Batch 5 — V3 noise r=1.0 (emb_lr=1.0) × all combos
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 5: V3 noise r=1.0 — 4 combos (${TIMESTAMP})"
echo "============================================================"

run_bg "v3_noise_gsm_06" $M06 $DS_GSM $R_V3_NOISE
run_bg "v3_noise_gsm_17" $M17 $DS_GSM $R_V3_NOISE
run_bg "v3_noise_sci_06" $M06 $DS_SCI $R_V3_NOISE
run_bg "v3_noise_sci_17" $M17 $DS_SCI $R_V3_NOISE

echo "  4 experiments launched. Waiting..."
wait

# ============================================================================
# Batch 6 — V6 (emb_lr=1.0) × all combos
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 6: V6 e1.0 — 4 combos (${TIMESTAMP})"
echo "============================================================"

run_bg "v6_e1_gsm_06"  $M06 $DS_GSM $R_V6
run_bg "v6_e1_gsm_17"  $M17 $DS_GSM $R_V6
run_bg "v6_e1_sci_06"  $M06 $DS_SCI $R_V6
run_bg "v6_e1_sci_17"  $M17 $DS_SCI $R_V6

echo "  4 experiments launched. Waiting..."
wait

# ============================================================================
# Batch 7 — V6 (emb_lr=10.0) × all combos
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 7: V6 e10.0 — 4 combos (${TIMESTAMP})"
echo "============================================================"

run_bg "v6_e10_gsm_06" $M06 $DS_GSM $R_V6_E10
run_bg "v6_e10_gsm_17" $M17 $DS_GSM $R_V6_E10
run_bg "v6_e10_sci_06" $M06 $DS_SCI $R_V6_E10
run_bg "v6_e10_sci_17" $M17 $DS_SCI $R_V6_E10

echo "  4 experiments launched. Waiting..."
wait

echo ""
echo "============================================================"
echo "All 28 experiments complete. Results in ./ckpt/sweep_${TIMESTAMP}/"
echo "============================================================"