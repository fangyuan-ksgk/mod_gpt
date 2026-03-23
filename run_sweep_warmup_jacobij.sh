#!/bin/bash
# ============================================================================
# SoRL Sweep — Qwen3-1.7B / GSM8K
#
# Baseline: v3 noise r=0.3 γ=0.5 (exp10 config)
#
# Questions:
#   Q1: Does SFT warmup help SoRL? Which warmup config?
#   Q2: Does Jacobi loss help pure SoRL?
#   Q3: Does masked_traj_loss help pure SoRL?
#
# Step matching:
#   - Pure SoRL runs: 3 epochs (~2800 steps)
#   - Warmup runs: 500 warmup steps + 2 epochs SoRL (~2370 total)
#     This keeps total forward passes roughly comparable.
#
# 20 experiments in 5 batches × 4 GPUs
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

# ============================================================================
# Shared configuration
# ============================================================================
MASTER_ADDR=127.0.0.1
BASE_PORT=29601
N_GPUS=4

MODEL_NAME="Qwen/Qwen3-1.7B"
DATASET="gsm8k"
MAX_LENGTH=512

LR=1e-5
LR_WARMUP=50
BATCH_SIZE=2
GRAD_ACCUM=4

LOG_EVERY=10
EVAL_EVERY=99999
SAVE_EVERY=99999
EVAL_SAMPLES=1300
EVAL_BATCH_SIZE=64
MAX_NEW_TOKENS=256

# Baseline v3 config (exp10-style: noise r=0.3 γ=0.5)
V3_BASE="--use_v3 --alpha_traj 1.0 --alpha_abs 0.5 --corrupt_method noise --corrupt_ratio 0.3 --gamma_contrastive 0.5 --emb_lr_mult 1.0"

# Warmup defaults
WU_BASE="--warmup_sft --warmup_sft_steps 500 --warmup_lr 1e-5 --warmup_emb_lr_mult 1.0 --warmup_grad_accum 4 --warmup_alpha_abs 0.5 --warmup_alpha_traj 1.0"

TIMESTAMP=$(date +%Y%m%d_%H%M)
EXP_IDX=0

# ---- Parallel scheduling: 4 x H100 — 1 run/GPU ----

run_bg() {
  EXP_IDX=$((EXP_IDX + 1))
  local idx=$EXP_IDX
  local gpu=$(( (idx - 1) % N_GPUS ))
  local port=$((BASE_PORT + idx))
  local tag=$1; shift
  local output_dir="./ckpt/sweep_${TIMESTAMP}/exp${idx}_${tag}"

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
# Batch 1 — Baselines: pure SoRL jacobi sweep (3 epochs, no warmup)
#   Q2: Does jacobi loss help pure SoRL?
#   (masked_traj experiments deferred — validating in notebook first)
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 1: Pure SoRL baselines + jacobi (${TIMESTAMP})"
echo "============================================================"

# 1. Baseline: v3 noise r=0.3 γ=0.5 (no jacobi, no mtraj)
run_bg "v3_baseline" \
  --num_epochs 3 $V3_BASE

# 2. v3 + jacobi=0.5
run_bg "v3_jacobi0.5" \
  --num_epochs 3 $V3_BASE --alpha_jacobi 0.5

# 3. v3 + jacobi=1.0 (stronger)
run_bg "v3_jacobi1.0" \
  --num_epochs 3 $V3_BASE --alpha_jacobi 1.0

# 4. v3 + jacobi=0.25 (lighter)
run_bg "v3_jacobi0.25" \
  --num_epochs 3 $V3_BASE --alpha_jacobi 0.25

echo "  4 experiments launched. Waiting..."
wait

# ============================================================================
# Batch 2 — Q1: SFT warmup → SoRL (step-matched: 500 wu + 2 ep SoRL)
#   NOTE: No masked_traj during warmup (degrades accuracy).
#   Warmup uses abs+traj ± jacobi only.
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 2: Warmup → SoRL (step-matched) (${TIMESTAMP})"
echo "============================================================"

# 5. Warmup (jacobi=0.5) → v3 baseline
run_bg "wu_jacobi_v3" \
  --num_epochs 2 $V3_BASE \
  $WU_BASE --warmup_alpha_jacobi 0.5

# 6. Warmup (vanilla: abs+traj only) → v3 baseline
run_bg "wu_vanilla_v3" \
  --num_epochs 2 $V3_BASE \
  $WU_BASE --warmup_alpha_jacobi 0.0

# 7. Warmup (jacobi=1.0, stronger) → v3 baseline
run_bg "wu_jacobi1.0_v3" \
  --num_epochs 2 $V3_BASE \
  $WU_BASE --warmup_alpha_jacobi 1.0

# 8. Warmup (jacobi=0.5) → v3 + jacobi carry-over
run_bg "wu_jacobi_v3jacobi" \
  --num_epochs 2 $V3_BASE --alpha_jacobi 0.5 \
  $WU_BASE --warmup_alpha_jacobi 0.5

echo "  4 experiments launched. Waiting..."
wait

# ============================================================================
# Batch 3 — Warmup length sweep
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 3: Warmup length sweep (${TIMESTAMP})"
echo "============================================================"

# 9. Warmup 250 steps (shorter) + jacobi → v3 (3 ep to compensate)
run_bg "wu_250_jacobi_v3" \
  --num_epochs 3 $V3_BASE \
  $WU_BASE --warmup_sft_steps 250 --warmup_alpha_jacobi 0.5

# 10. Warmup 1000 steps (longer) + jacobi → v3 (1 ep SoRL)
run_bg "wu_1000_jacobi_v3" \
  --num_epochs 1 $V3_BASE \
  $WU_BASE --warmup_sft_steps 1000 --warmup_alpha_jacobi 0.5

# 11. Warmup 250 steps (vanilla, no jacobi) → v3 3ep
run_bg "wu_250_vanilla_v3" \
  --num_epochs 3 $V3_BASE \
  $WU_BASE --warmup_sft_steps 250 --warmup_alpha_jacobi 0.0

# 12. Warmup 1000 steps (vanilla, no jacobi) → v3 1ep
run_bg "wu_1000_vanilla_v3" \
  --num_epochs 1 $V3_BASE \
  $WU_BASE --warmup_sft_steps 1000 --warmup_alpha_jacobi 0.0

echo "  4 experiments launched. Waiting..."
wait

echo ""
echo "============================================================"
echo "All 12 experiments complete. Results in ./ckpt/sweep_${TIMESTAMP}/"
echo "============================================================"
echo ""
echo "Experiment matrix:"
echo "  Batch 1 (pure SoRL jacobi sweep):"
echo "    1  v3_baseline          — exp10 config (no jacobi, no mtraj)"
echo "    2  v3_jacobi0.5         — + jacobi=0.5"
echo "    3  v3_jacobi1.0         — + jacobi=1.0"
echo "    4  v3_jacobi0.25        — + jacobi=0.25"
echo "  Batch 2 (warmup → SoRL, no warmup mtraj):"
echo "    5  wu_jacobi_v3         — warmup(jacobi=0.5) → v3"
echo "    6  wu_vanilla_v3        — warmup(abs+traj only) → v3"
echo "    7  wu_jacobi1.0_v3      — warmup(jacobi=1.0) → v3"
echo "    8  wu_jacobi_v3jacobi   — warmup(jacobi) → v3+jacobi carry-over"
echo "  Batch 3 (warmup length sweep):"
echo "    9  wu_250_jacobi_v3     — warmup 250 steps + jacobi → v3 3ep"
echo "    10 wu_1000_jacobi_v3    — warmup 1000 steps + jacobi → v3 1ep"
echo "    11 wu_250_vanilla_v3    — warmup 250 steps (vanilla) → v3 3ep"
echo "    12 wu_1000_vanilla_v3   — warmup 1000 steps (vanilla) → v3 1ep"
echo ""
echo "  masked_traj experiments deferred — validating in notebook first"
