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
# Batch 1 — Baselines: pure SoRL (3 epochs, no warmup)
#   Q2: Jacobi loss sweep
#   Q3: Masked traj loss sweep (basic)
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 1: Pure SoRL baselines + jacobi/mtraj (${TIMESTAMP})"
echo "============================================================"

# 1. Baseline: v3 noise r=0.3 γ=0.5 (no jacobi, no mtraj)
run_bg "v3_baseline" \
  --num_epochs 3 $V3_BASE

# 2. v3 + jacobi=0.5
run_bg "v3_jacobi0.5" \
  --num_epochs 3 $V3_BASE --alpha_jacobi 0.5

# 3. v3 + masked_traj=1.0 (ratio=0.3, random)
run_bg "v3_mtraj1.0" \
  --num_epochs 3 $V3_BASE --alpha_masked_traj 1.0 --mask_nl_ratio 0.3

# 4. v3 + jacobi=0.5 + mtraj=1.0
run_bg "v3_jacobi_mtraj" \
  --num_epochs 3 $V3_BASE --alpha_jacobi 0.5 --alpha_masked_traj 1.0 --mask_nl_ratio 0.3

echo "  4 experiments launched. Waiting..."
wait

# ============================================================================
# Batch 2 — Q3 continued: masked_traj_loss tuning
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 2: Masked traj loss tuning (${TIMESTAMP})"
echo "============================================================"

# 5. v3 + mtraj=0.5 (lighter weight)
run_bg "v3_mtraj0.5" \
  --num_epochs 3 $V3_BASE --alpha_masked_traj 0.5 --mask_nl_ratio 0.3

# 6. v3 + mtraj=1.0 + ratio=0.5 (more aggressive masking)
run_bg "v3_mtraj1.0_r0.5" \
  --num_epochs 3 $V3_BASE --alpha_masked_traj 1.0 --mask_nl_ratio 0.5

# 7. v3 + mtraj=1.0 + fixed mode (single rare token instead of random)
run_bg "v3_mtraj1.0_fixed" \
  --num_epochs 3 $V3_BASE --alpha_masked_traj 1.0 --mask_nl_ratio 0.3 --mask_nl_mode fixed

# 8. v3 + jacobi=1.0 (stronger jacobi)
run_bg "v3_jacobi1.0" \
  --num_epochs 3 $V3_BASE --alpha_jacobi 1.0

echo "  4 experiments launched. Waiting..."
wait

# ============================================================================
# Batch 3 — Q1: SFT warmup → SoRL (step-matched: 500 wu + 2 ep SoRL)
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 3: Warmup → SoRL (step-matched) (${TIMESTAMP})"
echo "============================================================"

# 9. Warmup (mtraj+jacobi) → v3 baseline
run_bg "wu_both_v3" \
  --num_epochs 2 $V3_BASE \
  $WU_BASE --warmup_alpha_masked_traj 1.0 --warmup_alpha_jacobi 0.5

# 10. Warmup (mtraj only, no jacobi) → v3 baseline
run_bg "wu_mtraj_v3" \
  --num_epochs 2 $V3_BASE \
  $WU_BASE --warmup_alpha_masked_traj 1.0 --warmup_alpha_jacobi 0.0

# 11. Warmup (jacobi only, no mtraj) → v3 baseline
run_bg "wu_jacobi_v3" \
  --num_epochs 2 $V3_BASE \
  $WU_BASE --warmup_alpha_masked_traj 0.0 --warmup_alpha_jacobi 0.5

# 12. Warmup (vanilla SFT: abs+traj only) → v3 baseline
run_bg "wu_vanilla_v3" \
  --num_epochs 2 $V3_BASE \
  $WU_BASE --warmup_alpha_masked_traj 0.0 --warmup_alpha_jacobi 0.0

echo "  4 experiments launched. Waiting..."
wait

# ============================================================================
# Batch 4 — Q1 continued: warmup config tuning
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 4: Warmup config tuning (${TIMESTAMP})"
echo "============================================================"

# 13. Warmup (mtraj+jacobi) → v3 + mtraj (warmup carries into SoRL)
run_bg "wu_both_v3mtraj" \
  --num_epochs 2 $V3_BASE --alpha_masked_traj 1.0 --mask_nl_ratio 0.3 \
  $WU_BASE --warmup_alpha_masked_traj 1.0 --warmup_alpha_jacobi 0.5

# 14. Warmup (mtraj+jacobi) → v3 + jacobi (warmup carries into SoRL)
run_bg "wu_both_v3jacobi" \
  --num_epochs 2 $V3_BASE --alpha_jacobi 0.5 \
  $WU_BASE --warmup_alpha_masked_traj 1.0 --warmup_alpha_jacobi 0.5

# 15. Warmup with higher mask ratio 0.5 → v3
run_bg "wu_ratio0.5_v3" \
  --num_epochs 2 $V3_BASE \
  $WU_BASE --warmup_alpha_masked_traj 1.0 --warmup_alpha_jacobi 0.5 --warmup_mask_nl_ratio 0.5

# 16. Warmup 250 steps (shorter) → v3 (2.5 epochs to balance)
run_bg "wu_250_v3" \
  --num_epochs 3 $V3_BASE \
  $WU_BASE --warmup_sft_steps 250 --warmup_alpha_masked_traj 1.0 --warmup_alpha_jacobi 0.5

echo "  4 experiments launched. Waiting..."
wait

# ============================================================================
# Batch 5 — Best combos & ablations
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 5: Combined effects & ablations (${TIMESTAMP})"
echo "============================================================"

# 17. Warmup (both) → v3 + jacobi + mtraj (full pipeline)
run_bg "wu_both_v3full" \
  --num_epochs 2 $V3_BASE --alpha_jacobi 0.5 --alpha_masked_traj 1.0 --mask_nl_ratio 0.3 \
  $WU_BASE --warmup_alpha_masked_traj 1.0 --warmup_alpha_jacobi 0.5

# 18. v3 + no hinge (alpha_contrastive=0) + mtraj=1.0 (mtraj replaces hinge?)
run_bg "v3_nohinge_mtraj" \
  --num_epochs 3 $V3_BASE --alpha_contrastive 0.0 --alpha_masked_traj 1.0 --mask_nl_ratio 0.3

# 19. v3 + no hinge + jacobi=0.5 + mtraj=1.0
run_bg "v3_nohinge_jacobi_mtraj" \
  --num_epochs 3 $V3_BASE --alpha_contrastive 0.0 --alpha_jacobi 0.5 --alpha_masked_traj 1.0 --mask_nl_ratio 0.3

# 20. Warmup (both) → v3 no hinge + mtraj (warmup establishes dep, SoRL maintains)
run_bg "wu_both_v3nohinge_mtraj" \
  --num_epochs 2 $V3_BASE --alpha_contrastive 0.0 --alpha_masked_traj 1.0 --mask_nl_ratio 0.3 \
  $WU_BASE --warmup_alpha_masked_traj 1.0 --warmup_alpha_jacobi 0.5

echo "  4 experiments launched. Waiting..."
wait

echo ""
echo "============================================================"
echo "All 20 experiments complete. Results in ./ckpt/sweep_${TIMESTAMP}/"
echo "============================================================"
echo ""
echo "Experiment matrix:"
echo "  Batch 1 (pure SoRL baselines):"
echo "    1  v3_baseline          — exp10 config (no jacobi, no mtraj)"
echo "    2  v3_jacobi0.5         — + jacobi=0.5"
echo "    3  v3_mtraj1.0          — + masked_traj=1.0"
echo "    4  v3_jacobi_mtraj      — + jacobi=0.5 + mtraj=1.0"
echo "  Batch 2 (mtraj tuning):"
echo "    5  v3_mtraj0.5          — mtraj=0.5 (lighter)"
echo "    6  v3_mtraj1.0_r0.5     — mtraj=1.0, mask ratio=0.5"
echo "    7  v3_mtraj1.0_fixed    — mtraj=1.0, fixed rare token mode"
echo "    8  v3_jacobi1.0         — jacobi=1.0 (stronger)"
echo "  Batch 3 (warmup → SoRL, step-matched):"
echo "    9  wu_both_v3           — warmup(mtraj+jacobi) → v3"
echo "    10 wu_mtraj_v3          — warmup(mtraj only) → v3"
echo "    11 wu_jacobi_v3         — warmup(jacobi only) → v3"
echo "    12 wu_vanilla_v3        — warmup(abs+traj only) → v3"
echo "  Batch 4 (warmup config):"
echo "    13 wu_both_v3mtraj      — warmup → v3+mtraj (carries over)"
echo "    14 wu_both_v3jacobi     — warmup → v3+jacobi (carries over)"
echo "    15 wu_ratio0.5_v3       — warmup mask_ratio=0.5 → v3"
echo "    16 wu_250_v3            — warmup 250 steps → v3 3ep"
echo "  Batch 5 (combos & hinge ablation):"
echo "    17 wu_both_v3full       — warmup → v3+jacobi+mtraj (full pipeline)"
echo "    18 v3_nohinge_mtraj     — no hinge, mtraj replaces it?"
echo "    19 v3_nohinge_j_mtraj   — no hinge, jacobi+mtraj"
echo "    20 wu_both_v3nohinge    — warmup → v3 no hinge + mtraj"
