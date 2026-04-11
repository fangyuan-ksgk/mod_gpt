#!/bin/bash
set -e

# Ablation on v7 "deep supervision"

# --- nvidia pod  specifics ------
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
NUM_EPOCHS=1

LOG_EVERY=10
EVAL_EVERY=99999
SAVE_EVERY=99999
EVAL_SAMPLES=1300
EVAL_BATCH_SIZE=256

# ============================================================================================
# similar_magnitude routing + v7 deep supervision sweep
#
# Structure:
#   Phase A — 0.6B baseline across ALL datasets (default config: pfx=8, iter=4, V=32)
#   Phase B — Config ablations (0.6B, GSM8K only)
#             B1: max_iterations
#             B2: abs_prefix_max
#             B3: abstract_vocab_size
#   Phase C — Other models across ALL datasets (default config)
#   Phase D — Outer-loop ablation
#             D1: 0.6B across ALL datasets
#             D2: Other models on GSM8K
# ============================================================================================

TIMESTAMP=$(date +%Y%m%d_%H%M)
EXP_IDX=0

# Model shorthands
M06="Qwen/Qwen3-0.6B"
M17="Qwen/Qwen3-1.7B"
ML1="meta-llama/Llama-3.2-1B"
ML3="meta-llama/Llama-3.2-3B"
M4B="Qwen/Qwen3-4B"

# Dataset shorthands
DS_GSM="gsm8k"
DS_SCI="scienceqa"
DS_ARC="arc"
DS_MMLU="mmlu"
DS_CSQA="commonsenseqa"
DS_BOOLQ="boolq"
DS_OBQA="openbookqa"
DS_AQUA="aqua"
DS_HPQA="hotpotqa"

ALL_DS=("$DS_GSM" "$DS_ARC" "$DS_SCI" "$DS_MMLU" "$DS_CSQA")
OTHER_MODELS=("$M17" "$ML1" "$ML3" "$M4B")
OTHER_MODEL_TAGS=("17b" "l1b" "l3b" "q4b")

# Common SoRL flags (default config)
ABS="--abstract_vocab_size 128 --prefix_abs --alpha_traj 1.0 --abs_routing_mode similar_magnitude"
DEFAULT="--use_v7 $ABS --abs_prefix_max 4 --max_iterations 2"

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
    --output_dir $output_dir \
    --untie_embedding \
    "$@" &
}


# =============================================================================
# Ablation: {Qwen3-4B, Qwen3-1.7B} × {GSM8K, SciQA}
# Base:  v7, similar_magnitude, LoRA, prefix_K=4, max_iter=2, emb_lr_mult=1.0
# Sweep (1): V ∈ {128 (ref), 256, 1024}          (emb_lr fixed at 1.0)
# Sweep (2): emb_lr_mult ∈ {1.0 (ref), 5.0, 10.0} (V fixed at 128)
# 5 run per (model × dataset)  ×  4 combos  =  20 total runs
# 4 GPUs, 4 experiments per batch, 5 batches
# =============================================================================

LORA="--use_lora --lora_rank 16 --lora_alpha 32"
BASE="--use_v7 --abs_routing_mode similar_magnitude \
  --prefix_abs --abs_prefix_max 4 --K 4 \
  --max_iterations 2 --eval_K 4 $LORA"

echo "=== v7 LoRA Ablation === $(date)"

# # ── Batch 1: 4B × GSM8K ───────────────────────────────────────────────────────
# echo "Batch 1: Qwen3-4B × GSM8K  [ref | v256 | v1024 | emb5]"
# run_bg "q4b_gsm_ref"   $M4B $DS_GSM $BASE --abstract_vocab_size 128  --emb_lr_mult 1.0  --eval_batch_size 8
# run_bg "q4b_gsm_v256"  $M4B $DS_GSM $BASE --abstract_vocab_size 128  --emb_lr_mult 20.0  --eval_batch_size 8
# run_bg "q4b_gsm_v1024" $M4B $DS_GSM $BASE --abstract_vocab_size 128 --emb_lr_mult 100.0  --eval_batch_size 8
# run_bg "q4b_gsm_emb5"  $M4B $DS_GSM $BASE --abstract_vocab_size 128  --emb_lr_mult 5.0  --eval_batch_size 8
# wait

# # ── Batch 2: 4B × GSM8K (emb10) + 4B × SciQA (ref, v256, v1024) ─────────────
# echo "Batch 2: 4B GSM emb10 + 4B SciQA [ref | v256 | v1024]"
# run_bg "q4b_gsm_emb10" $M4B $DS_GSM $BASE --abstract_vocab_size 128  --emb_lr_mult 10.0 --eval_batch_size 8
# run_bg "q4b_sci_ref"   $M4B $DS_SCI $BASE --abstract_vocab_size 128  --emb_lr_mult 1.0  --eval_batch_size 8
# run_bg "q4b_sci_v256"  $M4B $DS_SCI $BASE --abstract_vocab_size 256  --emb_lr_mult 1.0  --eval_batch_size 8
# run_bg "q4b_sci_v1024" $M4B $DS_SCI $BASE --abstract_vocab_size 1024 --emb_lr_mult 1.0  --eval_batch_size 8
# wait

# # ── Batch 3: 4B × SciQA (emb5, emb10) + 1.7B × GSM8K (ref, v256) ────────────
# echo "Batch 3: 4B SciQA [emb5 | emb10] + 1.7B GSM8K [ref | v256]"
# run_bg "q4b_sci_emb5"   $M4B $DS_SCI $BASE --abstract_vocab_size 128  --emb_lr_mult 5.0  --eval_batch_size 8
# run_bg "q4b_sci_emb10"  $M4B $DS_SCI $BASE --abstract_vocab_size 128  --emb_lr_mult 10.0 --eval_batch_size 8
# run_bg "q17b_gsm_ref"   $M17 $DS_GSM $BASE --abstract_vocab_size 128  --emb_lr_mult 1.0  --eval_batch_size 8
# run_bg "q17b_gsm_v256"  $M17 $DS_GSM $BASE --abstract_vocab_size 128  --emb_lr_mult 50.0  --eval_batch_size 8
# wait

# # ── Batch 4: 1.7B × GSM8K (v1024, emb5, emb10) + 1.7B × SciQA (ref) ─────────
# echo "Batch 4: 1.7B GSM8K [v1024 | emb5 | emb10] + 1.7B SciQA [ref]"
# run_bg "q17b_gsm_v1024" $M17 $DS_GSM $BASE --abstract_vocab_size 1024 --emb_lr_mult 1.0  --eval_batch_size 8
# run_bg "q17b_gsm_emb5"  $M17 $DS_GSM $BASE --abstract_vocab_size 128  --emb_lr_mult 5.0  --eval_batch_size 8
# run_bg "q17b_gsm_emb10" $M17 $DS_GSM $BASE --abstract_vocab_size 128  --emb_lr_mult 10.0 --eval_batch_size 8
# run_bg "q17b_sci_ref"   $M17 $DS_SCI $BASE --abstract_vocab_size 128  --emb_lr_mult 1.0  --eval_batch_size 8
# wait

# # ── Batch 5: 1.7B × SciQA (v256, v1024, emb5, emb10) ────────────────────────
# echo "Batch 5: 1.7B SciQA [v256 | v1024 | emb5 | emb10]"
# run_bg "q17b_sci_v256"  $M17 $DS_SCI $BASE --abstract_vocab_size 128  --emb_lr_mult 50.0  --eval_batch_size 8
# run_bg "q17b_sci_v1024" $M17 $DS_SCI $BASE --abstract_vocab_size 1024 --emb_lr_mult 1.0  --eval_batch_size 8
# run_bg "q17b_sci_emb5"  $M17 $DS_SCI $BASE --abstract_vocab_size 128  --emb_lr_mult 5.0  --eval_batch_size 8
# run_bg "q17b_sci_emb10" $M17 $DS_SCI $BASE --abstract_vocab_size 128  --emb_lr_mult 10.0 --eval_batch_size 8
# wait

# # ── Batch A: 1.7B × {ARC, MMLU, HotpotQA} + 1B × ARC ────────────────────────
# echo "Batch A: Qwen3-1.7B × {ARC, MMLU, HotpotQA}  +  Llama-1B × ARC"
# run_bg "q17b_arc"  $M17 $DS_ARC  $BASE --abstract_vocab_size 128 --emb_lr_mult 1.0 --eval_batch_size 8
# run_bg "q17b_mmlu" $M17 $DS_MMLU $BASE --abstract_vocab_size 128 --emb_lr_mult 1.0 --eval_batch_size 8
# run_bg "q17b_hpqa" $M17 $DS_HPQA $BASE --abstract_vocab_size 128 --emb_lr_mult 1.0 --eval_batch_size 8
# run_bg "l1b_arc"   $ML1 $DS_ARC  $BASE --abstract_vocab_size 128 --emb_lr_mult 1.0 --eval_batch_size 8
# wait

# echo "Batch A: Qwen3-1.7B × {ARC, MMLU, HotpotQA}  +  Llama-1B × ARC"
# run_bg "q17b_arc"  $M17 $DS_ARC  $BASE --abstract_vocab_size 256 --emb_lr_mult 1.0 --eval_batch_size 8
# run_bg "q17b_mmlu" $M17 $DS_MMLU $BASE --abstract_vocab_size 256 --emb_lr_mult 1.0 --eval_batch_size 8
# run_bg "q17b_hpqa" $M17 $DS_HPQA $BASE --abstract_vocab_size 256 --emb_lr_mult 1.0 --eval_batch_size 8
# run_bg "l1b_arc"   $ML1 $DS_ARC  $BASE --abstract_vocab_size 256 --emb_lr_mult 1.0 --eval_batch_size 8
# wait

# echo "Batch A: Qwen3-1.7B × {ARC, MMLU, HotpotQA}  +  Llama-1B × ARC"
# run_bg "q17b_arc"  $M17 $DS_ARC  $BASE --abstract_vocab_size 256 --emb_lr_mult 10.0 --eval_batch_size 8
# run_bg "q17b_mmlu" $M17 $DS_MMLU $BASE --abstract_vocab_size 256 --emb_lr_mult 10.0 --eval_batch_size 8
# run_bg "q17b_hpqa" $M17 $DS_HPQA $BASE --abstract_vocab_size 256 --emb_lr_mult 10.0 --eval_batch_size 8
# run_bg "l1b_arc"   $ML1 $DS_ARC  $BASE --abstract_vocab_size 256 --emb_lr_mult 10.0 --eval_batch_size 8
# wait


# # ── Batch B: 1B × {MMLU, HotpotQA} ──────────────────────────────────────────
# echo "Batch B: Llama-1B × {MMLU, HotpotQA}"
# run_bg "l1b_mmlu" $ML1 $DS_MMLU $BASE --abstract_vocab_size 128 --emb_lr_mult 1.0 --eval_batch_size 8
# run_bg "l1b_hpqa" $ML1 $DS_HPQA $BASE --abstract_vocab_size 256 --emb_lr_mult 1.0 --eval_batch_size 8
# run_bg "l1b_mmlu" $ML1 $DS_MMLU $BASE --abstract_vocab_size 256 --emb_lr_mult 1.0 --eval_batch_size 8
# run_bg "l1b_hpqa" $ML1 $DS_HPQA $BASE --abstract_vocab_size 256 --emb_lr_mult 1.0 --eval_batch_size 8
# wait

# # ── Batch B: 1B × {MMLU, HotpotQA} ──────────────────────────────────────────
# echo "Batch B: Llama-1B × {MMLU, HotpotQA}"
# run_bg "l1b_mmlu" $ML1 $DS_MMLU $BASE --abstract_vocab_size 128 --emb_lr_mult 1.0 --eval_batch_size 8
# run_bg "l1b_hpqa" $ML1 $DS_HPQA $BASE --abstract_vocab_size 256 --emb_lr_mult 1.0 --eval_batch_size 8
# run_bg "l1b_mmlu" $ML1 $DS_MMLU $BASE --abstract_vocab_size 256 --emb_lr_mult 10.0 --eval_batch_size 8
# run_bg "l1b_hpqa" $ML1 $DS_HPQA $BASE --abstract_vocab_size 256 --emb_lr_mult 10.0 --eval_batch_size 8
# wait


# =============================================================================
# v7 SoRL sweep | LoRA, ep=1, baseline config (V=128, pfx=4, iter=2, emb_lr=1.0)
# scienceQA : {q4b, q17b, l1b, l3b}
# mmlu/hotpotqa/aqua/obookqa/boolq/csqa : all 5 models  {q06, q17b, l1b, l3b, q4b}
# =============================================================================

SORL="$BASE --abstract_vocab_size 128 --emb_lr_mult 1.0"

echo "=== v7 SoRL Sweep === $(date)"

# ── ScienceQA ─────────────────────────────────────────────────────────────────
echo "SciQA SoRL × {q4b, q17b, l1b, l3b}"
run_bg "sci_sorl_q4b"  $M4B $DS_SCI $SORL --eval_batch_size 8
run_bg "sci_sorl_q17b" $M17 $DS_SCI $SORL --eval_batch_size 8
run_bg "sci_sorl_l1b"  $ML1 $DS_SCI $SORL --eval_batch_size 8
run_bg "sci_sorl_l3b"  $ML3 $DS_SCI $SORL --eval_batch_size 8
wait

# ── MMLU ──────────────────────────────────────────────────────────────────────
echo "MMLU SoRL × {q06, q17b, l1b, l3b}"
run_bg "mmlu_sorl_q06"  $M06 $DS_MMLU $SORL --eval_batch_size 8
run_bg "mmlu_sorl_q17b" $M17 $DS_MMLU $SORL --eval_batch_size 8
run_bg "mmlu_sorl_l1b"  $ML1 $DS_MMLU $SORL --eval_batch_size 8
run_bg "mmlu_sorl_l3b"  $ML3 $DS_MMLU $SORL --eval_batch_size 8
wait
run_bg "mmlu_sorl_q4b"  $M4B $DS_MMLU $SORL --eval_batch_size 8
wait

# ── HotpotQA ──────────────────────────────────────────────────────────────────
echo "HotpotQA SoRL × {q06, q17b, l1b, l3b}"
run_bg "hpqa_sorl_q06"  $M06 $DS_HPQA $SORL --eval_batch_size 8
run_bg "hpqa_sorl_q17b" $M17 $DS_HPQA $SORL --eval_batch_size 8
run_bg "hpqa_sorl_l1b"  $ML1 $DS_HPQA $SORL --eval_batch_size 8
run_bg "hpqa_sorl_l3b"  $ML3 $DS_HPQA $SORL --eval_batch_size 8
wait
run_bg "hpqa_sorl_q4b"  $M4B $DS_HPQA $SORL --eval_batch_size 8
wait

# ── AQuA ──────────────────────────────────────────────────────────────────────
echo "AQuA SoRL × {q06, q17b, l1b, l3b}"
run_bg "aqua_sorl_q06"  $M06 $DS_AQUA $SORL --eval_batch_size 8
run_bg "aqua_sorl_q17b" $M17 $DS_AQUA $SORL --eval_batch_size 8
run_bg "aqua_sorl_l1b"  $ML1 $DS_AQUA $SORL --eval_batch_size 8
run_bg "aqua_sorl_l3b"  $ML3 $DS_AQUA $SORL --eval_batch_size 8
wait
run_bg "aqua_sorl_q4b"  $M4B $DS_AQUA $SORL --eval_batch_size 8
wait

# ── OpenBookQA ────────────────────────────────────────────────────────────────
echo "ObookQA SoRL × {q06, q17b, l1b, l3b}"
run_bg "obqa_sorl_q06"  $M06 $DS_OBQA $SORL --eval_batch_size 8
run_bg "obqa_sorl_q17b" $M17 $DS_OBQA $SORL --eval_batch_size 8
run_bg "obqa_sorl_l1b"  $ML1 $DS_OBQA $SORL --eval_batch_size 8
run_bg "obqa_sorl_l3b"  $ML3 $DS_OBQA $SORL --eval_batch_size 8
wait
run_bg "obqa_sorl_q4b"  $M4B $DS_OBQA $SORL --eval_batch_size 8
wait

# ── BoolQ ─────────────────────────────────────────────────────────────────────
echo "BoolQ SoRL × {q06, q17b, l1b, l3b}"
run_bg "boolq_sorl_q06"  $M06 $DS_BOOLQ $SORL --eval_batch_size 8
run_bg "boolq_sorl_q17b" $M17 $DS_BOOLQ $SORL --eval_batch_size 8
run_bg "boolq_sorl_l1b"  $ML1 $DS_BOOLQ $SORL --eval_batch_size 8
run_bg "boolq_sorl_l3b"  $ML3 $DS_BOOLQ $SORL --eval_batch_size 8
wait
run_bg "boolq_sorl_q4b"  $M4B $DS_BOOLQ $SORL --eval_batch_size 8
wait

# ── CommonsenseQA ─────────────────────────────────────────────────────────────
echo "CSQA SoRL × {q06, q17b, l1b, l3b}"
run_bg "csqa_sorl_q06"  $M06 $DS_CSQA $SORL --eval_batch_size 8
run_bg "csqa_sorl_q17b" $M17 $DS_CSQA $SORL --eval_batch_size 8
run_bg "csqa_sorl_l1b"  $ML1 $DS_CSQA $SORL --eval_batch_size 8
run_bg "csqa_sorl_l3b"  $ML3 $DS_CSQA $SORL --eval_batch_size 8
wait
run_bg "csqa_sorl_q4b"  $M4B $DS_CSQA $SORL --eval_batch_size 8
wait


echo ""
echo "=== All 20 experiments complete. $(date) ==="


