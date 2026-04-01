#!/bin/bash
# Sweep: Compounding + Diversity
# Qwen3-1.7B & Qwen3-0.6B | GSM8K | 1.3K val | emb_lr_mult=1.0
#
# NOTE: v1/v2 trainers do NOT support random_K, alpha_masked_traj, or
#       alpha_jacobi — those are v3/v4 only. All experiments here use v3.
#       RO ablation excluded: best RO config (v1) can't use these losses.
#
# Experiment 1 — Compounding: Do best Q+R configs + mtraj + jacobi + randK compound?
#   Batch A (Q+R): v3+noise → +mtraj → +jacobi → +randK   (4 exps)
#
# Experiment 2 — Accurate Diversity: α_abs sweep on v3+noise + zipf + ortho
#   Batch B (Q+R): α_abs ∈ {0.5, 1.0, 2.0, 4.0}           (4 exps)
#
# 8 experiments per model × 2 models = 16 total
# 2 batches per model, 4 parallel each → 4 serial waits
#
# Usage:
#   bash run_sweep_compound_diversity.sh                  # both models
#   bash run_sweep_compound_diversity.sh 1.7B             # 1.7B only
#   bash run_sweep_compound_diversity.sh 0.6B             # 0.6B only

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
BASE_PORT=29501
N_GPUS=4

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

EMB="--emb_lr_mult 1.0"

# Per-model best v3 base configs (set inside loop)
# 1.7B: v3 + noise r=0.3 γ=0.5  (A.exp10: NL=63.7, K=4=59.7)
# 0.6B: v3 + shuffle r=1.0 γ=0.5 (exp3: NL=49.1, K=4=46.2)

TIMESTAMP=$(date +%Y%m%d_%H%M)

# ---- Parallel scheduling: 4 GPUs — 1 run/GPU ----
EXP_IDX=0

run_bg() {
  EXP_IDX=$((EXP_IDX + 1))
  local idx=$EXP_IDX
  local gpu=$(( (idx - 1) % N_GPUS ))
  local port=$((BASE_PORT + idx))
  local tag=$1; shift
  local grad_accum=$((8 / (BATCH_SIZE * 1)))
  local output_dir="./ckpt/sweep_${SWEEP_TAG}_${TIMESTAMP}/exp${idx}_${tag}"

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
    --warmup_steps $WARMUP_STEPS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $grad_accum \
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
# Model selection (default: both)
# ============================================================================
MODELS=("Qwen/Qwen3-1.7B" "Qwen/Qwen3-0.6B")
if [[ "$1" == "1.7B" ]]; then
  MODELS=("Qwen/Qwen3-1.7B")
elif [[ "$1" == "0.6B" ]]; then
  MODELS=("Qwen/Qwen3-0.6B")
fi

for MODEL_NAME in "${MODELS[@]}"; do

  MODEL_SHORT=$(echo $MODEL_NAME | sed 's/Qwen\/Qwen3-//')
  SWEEP_TAG="${MODEL_SHORT}"
  EXP_IDX=0

  # --- Per-model v3 base config ---
  if [[ "$MODEL_SHORT" == "0.6B" ]]; then
    V3_BASE="--use_v3 --alpha_traj 1.0 --alpha_abs 0.5 --corrupt_method shuffle --corrupt_ratio 1.0 --gamma_contrastive 0.5 $EMB"
  else
    V3_BASE="--use_v3 --alpha_traj 1.0 --alpha_abs 0.5 --corrupt_method noise --corrupt_ratio 0.3 --gamma_contrastive 0.5 $EMB"
  fi

  echo ""
  echo "############################################################"
  echo "# Model: ${MODEL_NAME}"
  echo "############################################################"

  # ========================================================================
  # Experiment 1 — Q+R Compounding (per-model v3 base)
  #
  # 1.7B best: v3+noise r=0.3 (A.exp10: NL=63.7, K=4=59.7)
  # 0.6B best: v3+shuffle r=1.0 (exp3: NL=49.1, K=4=46.2)
  # Progressively stack: +mtraj → +jacobi → +randK
  # ========================================================================
  echo ""
  echo "============================================================"
  echo "Exp1: Q+R Compounding — ${MODEL_SHORT} (${TIMESTAMP})"
  echo "============================================================"

  # 1. Control: per-model v3 best
  run_bg "qr_v3base" \
    --num_epochs 3 $V3_BASE

  # 2. + masked traj loss (mtraj=1.0, fixed mask, ratio=0.3)
  run_bg "qr_v3base_mtraj" \
    --num_epochs 3 $V3_BASE \
    --alpha_masked_traj 1.0

  # 3. + mtraj + jacobi
  run_bg "qr_v3base_mtraj_jacobi" \
    --num_epochs 3 $V3_BASE \
    --alpha_masked_traj 1.0 --alpha_jacobi 0.5

  # 4. + mtraj + jacobi + randK (full compound)
  run_bg "qr_v3base_mtraj_jacobi_rK" \
    --num_epochs 3 $V3_BASE \
    --alpha_masked_traj 1.0 --alpha_jacobi 0.5 --random_K 2,4,6,8

  echo "  4 experiments launched. Waiting..."
  wait

  # ========================================================================
  # Experiment 2 — Q+R α_abs sweep (per-model v3 base + zipf + ortho)
  #
  # Prior data: zipf+ortho with abs=0.5 → abs_loss 0.87-1.44, K=4 drops 5-8pp.
  # Hypothesis: higher α_abs forces p(a|s) to be sharp under zipf pressure,
  # giving diversity WITHOUT the abs_loss / accuracy penalty.
  #
  # Note: V3_BASE already has --alpha_abs 0.5; we override with last arg.
  # ========================================================================
  echo ""
  echo "============================================================"
  echo "Exp2: Q+R α_abs sweep (zipf+ortho) — ${MODEL_SHORT} (${TIMESTAMP})"
  echo "============================================================"

  run_bg "qr_zipf_abs0.5" \
    --num_epochs 3 $V3_BASE \
    --alpha_soft_zipf 1.0 --alpha_ortho 1.0 --alpha_abs 0.5

  run_bg "qr_zipf_abs1.0" \
    --num_epochs 3 $V3_BASE \
    --alpha_soft_zipf 1.0 --alpha_ortho 1.0 --alpha_abs 1.0

  run_bg "qr_zipf_abs2.0" \
    --num_epochs 3 $V3_BASE \
    --alpha_soft_zipf 1.0 --alpha_ortho 1.0 --alpha_abs 2.0

  run_bg "qr_zipf_abs4.0" \
    --num_epochs 3 $V3_BASE \
    --alpha_soft_zipf 1.0 --alpha_ortho 1.0 --alpha_abs 4.0

  echo "  4 experiments launched. Waiting..."
  wait

  echo ""
  echo "============================================================"
  echo "All 8 experiments complete for ${MODEL_SHORT}."
  echo "Results in ./ckpt/sweep_${SWEEP_TAG}_${TIMESTAMP}/"
  echo "============================================================"

done

echo ""
echo "############################################################"
echo "All sweeps complete. Timestamp: ${TIMESTAMP}"
echo "############################################################"
