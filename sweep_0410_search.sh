#!/bin/bash
# Two-phase sweep:
#
#   Phase 1 — v7 SoRL training (similar_magnitude, tied weights)
#             Models × Datasets:  {0.6B, 1.7B} × {gsm8k, scienceqa}  → 4 runs
#
#   Phase 2 — REINFORCE abstract-routing search on phase-1 checkpoints
#             Conditions × Checkpoints:  {tied, untied} × 4  → 8 runs
#
#   GPU layout (adjust N_GPUS):
#     N_GPUS=4  → Phase 1: all 4 in parallel (one per GPU)
#                 Phase 2: 2 batches of 4
#     N_GPUS=2  → Phase 1: 2 batches of 2
#                 Phase 2: 4 batches of 2
#
set -e

# ── Pod env (same as sweep_0410.sh) ──────────────────────────────────────────
DUMMY_CONFIG_PATH="/workspace/mod_gpt/dummy_tuner_config.txt"
rm -f "$DUMMY_CONFIG_PATH"; touch "$DUMMY_CONFIG_PATH"
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Shared config ─────────────────────────────────────────────────────────────
MASTER_ADDR=127.0.0.1
BASE_PORT=29700
N_GPUS=2          # set to 2 if only 2 GPUs available

M06="Qwen/Qwen3-0.6B"
M17="Qwen/Qwen3-1.7B"
ABS_VOCAB=128

TIMESTAMP=$(date +%Y%m%d_%H%M)
P1_ROOT="./ckpt/v7_${TIMESTAMP}"     # phase-1 output dirs
P2_ROOT="./ckpt/search_${TIMESTAMP}" # phase-2 output dirs
EXP_IDX=0

# ── Phase 1 hyperparams (v7, similar_magnitude, prefix_abs) ──────────────────
V7_FLAGS="--use_v7 --abs_routing_mode similar_magnitude \
  --alpha_traj 1.0 --alpha_contrastive 1.0 --gamma_contrastive 0.5 --n_inner 4"
P1_ABS="--abstract_vocab_size $ABS_VOCAB --prefix_abs --abs_prefix_max 8 --K 8 \
  --max_iterations 4 --eval_K 4"
P1_TRAIN="--lr 1e-5 --warmup_steps 50 --batch_size 2 --gradient_accumulation_steps 4 \
  --num_epochs 3 --max_length 512"
P1_EVAL="--eval_every 99999 --save_every 99999 \
  --eval_samples 1300 --eval_batch_size 8 --max_new_tokens 256"

# ── Phase 2 hyperparams (REINFORCE search) ────────────────────────────────────
K=8               # match phase-1 training (abs_prefix_max=8)
N=4
MAX_ITER=4
EVAL_ABS_PREFIX_MAX=8
P2_TRAIN="--lr 1e-5 --batch_size 2 --gradient_accumulation_steps 4 --max_steps 1000"
P2_EVAL="--eval_every 99999 --save_every 99999 \
  --eval_samples 1300 --baseline_eval_samples 200 \
  --eval_batch_size 8 --max_new_tokens 256"

# ─────────────────────────────────────────────────────────────────────────────
# run_p1 <tag> <model> <dataset>  — mirrors sweep_0410.sh run_bg
#   Saves to $P1_ROOT/exp{N}_{tag}/;  trainer writes final ckpt to .../final/
# ─────────────────────────────────────────────────────────────────────────────
run_p1() {
  EXP_IDX=$((EXP_IDX + 1))
  local idx=$EXP_IDX
  local gpu=$(( (idx - 1) % N_GPUS ))
  local port=$((BASE_PORT + idx))
  local tag=$1  model=$2  dataset=$3; shift 3
  local out="${P1_ROOT}/exp${idx}_${tag}"

  echo "  P1-Exp ${idx}: ${tag}  model=$(basename $model)  dataset=${dataset}  [GPU=${gpu}]"

  CUDA_VISIBLE_DEVICES=$gpu torchrun \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$port \
    train_sorl_post.py \
    --model_name $model \
    --dataset    $dataset \
    $V7_FLAGS $P1_ABS $P1_TRAIN $P1_EVAL \
    --output_dir $out \
    "$@" &
}

# ─────────────────────────────────────────────────────────────────────────────
# run_p2 <tag> <ckpt_dir> <model> <dataset> [extra_flags...]
#   Runs train_sorl_search.py for REINFORCE fine-tuning of abstract routing.
# ─────────────────────────────────────────────────────────────────────────────
run_p2() {
  EXP_IDX=$((EXP_IDX + 1))
  local idx=$EXP_IDX
  local gpu=$(( (idx - 1) % N_GPUS ))
  local tag=$1  ckpt=$2  model=$3  dataset=$4; shift 4
  local out="${P2_ROOT}/exp${idx}_${tag}"

  echo "  P2-Exp ${idx}: ${tag}  ckpt=$(basename $ckpt)  dataset=${dataset}  [GPU=${gpu}]"

  CUDA_VISIBLE_DEVICES=$gpu python train_sorl_search.py \
    --model_name          $model \
    --abstract_vocab_size $ABS_VOCAB \
    --ckpt_dir            $ckpt \
    --dataset             $dataset \
    --K $K --N $N --max_iterations $MAX_ITER \
    --eval_abs_prefix_max $EVAL_ABS_PREFIX_MAX \
    $P2_TRAIN $P2_EVAL \
    --output_dir $out \
    "$@" &
}

# =============================================================================
echo "=== Two-Phase REINFORCE Search Sweep === $(date)"
echo ""

# ── Phase 1: v7 SoRL training ─────────────────────────────────────────────────
echo "Phase 1: v7 SoRL training (similar_magnitude, tied, V=$ABS_VOCAB, pfx=8, iter=4)"
echo "  {0.6B, 1.7B} × {gsm8k, scienceqa}  — 4 runs"

run_p1 "06b_gsm" $M06 gsm8k
run_p1 "06b_sci" $M06 scienceqa
run_p1 "17b_gsm" $M17 gsm8k
run_p1 "17b_sci" $M17 scienceqa
wait
echo "Phase 1 complete."
echo ""

# Resolve phase-1 checkpoint dirs (trainer saves to {output_dir}/final/)
P1_06B_GSM="${P1_ROOT}/exp1_06b_gsm/final"
P1_06B_SCI="${P1_ROOT}/exp2_06b_sci/final"
P1_17B_GSM="${P1_ROOT}/exp3_17b_gsm/final"
P1_17B_SCI="${P1_ROOT}/exp4_17b_sci/final"

# ── Phase 2: REINFORCE search (tied vs untied) ────────────────────────────────
echo "Phase 2: REINFORCE search — {tied, untied} × 4 checkpoints — 8 runs"
echo "  (1) tied   : embed_tokens abstract rows = lm_head abstract rows (Qwen3 default)"
echo "  (2) untied : embed_tokens and lm_head abstract rows train independently"
echo ""

# Batch 2a: 0.6B checkpoints
echo "  Batch 2a: 0.6B × {gsm8k, scienceqa} × {tied, untied}"
run_p2 "06b_gsm_tied"   $P1_06B_GSM $M06 gsm8k
run_p2 "06b_gsm_untied" $P1_06B_GSM $M06 gsm8k    --untie_embeddings
run_p2 "06b_sci_tied"   $P1_06B_SCI $M06 scienceqa
run_p2 "06b_sci_untied" $P1_06B_SCI $M06 scienceqa --untie_embeddings
wait

# Batch 2b: 1.7B checkpoints
echo "  Batch 2b: 1.7B × {gsm8k, scienceqa} × {tied, untied}"
run_p2 "17b_gsm_tied"   $P1_17B_GSM $M17 gsm8k
run_p2 "17b_gsm_untied" $P1_17B_GSM $M17 gsm8k    --untie_embeddings
run_p2 "17b_sci_tied"   $P1_17B_SCI $M17 scienceqa
run_p2 "17b_sci_untied" $P1_17B_SCI $M17 scienceqa --untie_embeddings
wait

echo ""
echo "=== All phases complete. $(date) ==="
echo ""
echo "  Phase 1 checkpoints:"
echo "    0.6B gsm8k : ${P1_06B_GSM}"
echo "    0.6B sciQA : ${P1_06B_SCI}"
echo "    1.7B gsm8k : ${P1_17B_GSM}"
echo "    1.7B sciQA : ${P1_17B_SCI}"
echo ""
echo "  Phase 2 results: ${P2_ROOT}/"