#!/bin/bash
# Continuously train from SoRL-trained ckpt via REINFORCE abstract routing search.
#
# Setup (run once before this script):
#   huggingface-cli upload fangyuan-ksgk/sorl-06b-128v ckpt/06b-128v --repo-type model
#
# Then launch this script.
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
N_GPUS=2

MODEL_NAME="Qwen/Qwen3-0.6B"
ABS_VOCAB=128

# HF checkpoint (downloaded automatically by train_sorl_search.py)
HF_CKPT_06B="Ksgk-fy/sorl-06b-128v"

TIMESTAMP=$(date +%Y%m%d_%H%M)
EXP_IDX=0

# REINFORCE hyperparams
K=4
N=4
MAX_ITER=4
EVAL_ABS_PREFIX_MAX=8    # abs_prefix_max used when the SoRL ckpt was trained
LR=1e-5
BATCH_SIZE=2
MAX_STEPS=1000
EVAL_EVERY=200
SAVE_EVERY=500
EVAL_SAMPLES=500
BASELINE_EVAL_SAMPLES=100
EVAL_BATCH_SIZE=8   # generate is no-KV-cache; batch>8 OOMs with 256 new tokens
MAX_NEW_TOKENS=256

# ── run_bg: launch one train_sorl_search.py job on a single GPU ───────────────
run_bg() {
  EXP_IDX=$((EXP_IDX + 1))
  local idx=$EXP_IDX
  local gpu=$(( (idx - 1) % N_GPUS ))
  local port=$((BASE_PORT + idx))
  local tag=$1; shift
  local ckpt=$1; shift
  local dataset=$1; shift
  local output_dir="./ckpt/search_${TIMESTAMP}/exp${idx}_${tag}"

  echo "  Exp ${idx}: ${tag}  ckpt=${ckpt}  dataset=${dataset}  [GPU=${gpu}]"

  CUDA_VISIBLE_DEVICES=$gpu python train_sorl_search.py \
    --model_name    $MODEL_NAME \
    --abstract_vocab_size $ABS_VOCAB \
    --ckpt_dir      $ckpt \
    --dataset       $dataset \
    --K $K --N $N --max_iterations $MAX_ITER --eval_abs_prefix_max $EVAL_ABS_PREFIX_MAX \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --max_steps $MAX_STEPS \
    --eval_every $EVAL_EVERY \
    --save_every $SAVE_EVERY \
    --eval_samples $EVAL_SAMPLES \
    --baseline_eval_samples $BASELINE_EVAL_SAMPLES \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --max_new_tokens $MAX_NEW_TOKENS \
    --output_dir $output_dir \
    "$@" &
}

echo "=== REINFORCE Search Sweep === $(date)"
echo ""

# ── Phase 1: tied vs untied  ×  GSM8K + SciQA  (0.6B)  ──────────────────────
# Tests whether untying embed_tokens / lm_head for abstract rows helps REINFORCE.
#
#   tied   : embed_tokens.weight IS lm_head.weight (Qwen3 default)
#   untied : lm_head abstract rows train independently from embed_tokens rows

echo "Phase 1: tied vs untied — GSM8K"
run_bg "tied_gsm"   $HF_CKPT_06B gsm8k
run_bg "untied_gsm" $HF_CKPT_06B gsm8k --untie_embeddings
wait

# - the loaded ckpt is trained on GSM8K with SoRL, not trained on SciQA
# echo "Phase 1: tied vs untied — ScienceQA"
# run_bg "tied_sci"   $HF_CKPT_06B scienceqa
# run_bg "untied_sci" $HF_CKPT_06B scienceqa --untie_embeddings
# wait

echo "Phase 2: N rollouts ablation (GSM8K, untied)"
run_bg "untied_gsm_N2" $HF_CKPT_06B gsm8k --untie_embeddings --N 2
run_bg "untied_gsm_N8" $HF_CKPT_06B gsm8k --untie_embeddings --N 8
wait

echo ""
echo "All runs launched and finished."
echo "  Phase 1: 4 runs  (tied/untied × gsm8k/scienceqa)"
echo "  Phase 2: 2 runs  (N=2,8 on gsm8k, untied)"
echo "  Total  : 6 experiments"