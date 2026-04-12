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
SORL_EPOCHS=2         # SoRL: 2 ep × max_iter=2 ≈ 4 effective forward passes/sample
SFT_EPOCHS=4          # SFT:  4 ep × 1 = 4 effective forward passes/sample (compute-matched)
EVAL_SAMPLES=1000
EVAL_BATCH_SIZE=128
NUM_LOG_SAMPLES=3

TIMESTAMP=$(date +%Y%m%d_%H%M)

# Models
M06="Qwen/Qwen3-0.6B"
M16="Qwen/Qwen3-1.7B"

# Mixed dataset string
MIX_DS="gsm8k,scienceqa,arc,mmlu,commonsenseqa"

# echo "============================================================"
# echo "Mixed-CPT experiment | ${TIMESTAMP}"
# echo "  Train: ${MIX_DS} | SoRL=${SORL_EPOCHS}ep×2iter, SFT=${SFT_EPOCHS}ep | EBS=${BATCH_SIZE}x${GRAD_ACCUM}=8"
# echo "  Eval:  all 5 datasets (auto)"
# echo "============================================================"

# # ============================================================================
# # Run 1: SoRL v7 on mixed datasets (GPU 0)
# # ============================================================================
# SORL_DIR="./ckpt/cpt_${TIMESTAMP}/sorl_v7_q06_mix"

# echo ""
# echo ">>> [GPU 0] SoRL v7 — q06 on mixed datasets"

# CUDA_VISIBLE_DEVICES=0 torchrun \
#   --nproc_per_node=1 \
#   --master_addr=$MASTER_ADDR \
#   --master_port=$((BASE_PORT + 1)) \
#   train_sorl_post.py \
#   --model_name $M06 \
#   --dataset $MIX_DS \
#   --num_epochs $SORL_EPOCHS \
#   --lr $LR \
#   --warmup_steps $WARMUP_STEPS \
#   --batch_size $BATCH_SIZE \
#   --gradient_accumulation_steps $GRAD_ACCUM \
#   --use_v7 \
#   --abs_routing_mode similar_magnitude \
#   --prefix_abs --abs_prefix_max 8 \
#   --K 8 --eval_K 8 \
#   --max_iterations 2 \
#   --emb_lr_mult 1.0 \
#   --abstract_vocab_size 128 \
#   --eval_every 99999 \
#   --save_every 99999 \
#   --eval_samples $EVAL_SAMPLES \
#   --eval_batch_size $EVAL_BATCH_SIZE \
#   --num_log_samples $NUM_LOG_SAMPLES \
#   --log_every 10 \
#   --output_dir $SORL_DIR &

# # ============================================================================
# # Run 2: SFT on mixed datasets (GPU 1)
# # ============================================================================
# SFT_DIR="./ckpt/cpt_${TIMESTAMP}/sft_q06_mix"

# echo ">>> [GPU 1] SFT — q06 on mixed datasets"

# CUDA_VISIBLE_DEVICES=1 torchrun \
#   --nproc_per_node=1 \
#   --master_addr=$MASTER_ADDR \
#   --master_port=$((BASE_PORT + 2)) \
#   train_sft_pt.py \
#   --model_name $M06 \
#   --dataset $MIX_DS \
#   --num_epochs $SFT_EPOCHS \
#   --lr $LR \
#   --warmup_steps $WARMUP_STEPS \
#   --batch_size $BATCH_SIZE \
#   --gradient_accumulation_steps $GRAD_ACCUM \
#   --eval_every 99999 \
#   --save_every 99999 \
#   --eval_samples $EVAL_SAMPLES \
#   --eval_batch_size $EVAL_BATCH_SIZE \
#   --num_log_samples $NUM_LOG_SAMPLES \
#   --log_every 10 \
#   --log_samples_every 99999 \
#   --output_dir $SFT_DIR &

# wait

# echo ""
# echo "============================================================"
# echo "Stage 1 (CPT) complete."
# echo "  SoRL: ${SORL_DIR}/train.log"
# echo "  SFT:  ${SFT_DIR}/train.log"
# echo "============================================================"

# # ============================================================================
# # Stage 2: Fine-tune SoRL on each single dataset from the CPT checkpoint
# #
# # The CPT SoRL checkpoint is at ${SORL_DIR}/final/
# # We fine-tune for 1 epoch on each dataset, evaluating on that dataset.
# # 4 GPUs => 4 datasets in parallel, then the 5th.
# # ============================================================================

# SORL_CPT_CKPT="${SORL_DIR}/final"
# FT_EPOCHS=1
# DATASETS=("gsm8k" "scienceqa" "arc" "mmlu" "commonsenseqa")

# echo ""
# echo "============================================================"
# echo "Stage 2: Fine-tune SoRL from CPT checkpoint on each dataset"
# echo "  Checkpoint: ${SORL_CPT_CKPT}"
# echo "  Datasets: ${DATASETS[*]}"
# echo "  Epochs: ${FT_EPOCHS}"
# echo "============================================================"

# # Batch 1: 4 datasets in parallel (GPU 0-3)
# for i in 0 1 2 3; do
#   DS=${DATASETS[$i]}
#   FT_DIR="./ckpt/cpt_${TIMESTAMP}/sorl_v7_q06_ft_${DS}"
#   echo ">>> [GPU ${i}] SoRL v7 fine-tune: ${DS}"

#   CUDA_VISIBLE_DEVICES=$i torchrun \
#     --nproc_per_node=1 \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$((BASE_PORT + 10 + i)) \
#     train_sorl_post.py \
#     --model_name $M06 \
#     --resume_ckpt $SORL_CPT_CKPT \
#     --dataset $DS \
#     --num_epochs $FT_EPOCHS \
#     --lr $LR \
#     --warmup_steps 20 \
#     --batch_size $BATCH_SIZE \
#     --gradient_accumulation_steps $GRAD_ACCUM \
#     --use_v7 \
#     --abs_routing_mode similar_magnitude \
#     --prefix_abs --abs_prefix_max 8 \
#     --K 8 --eval_K 8 \
#     --max_iterations 2 \
#     --emb_lr_mult 10.0 \
#     --abstract_vocab_size 128 \
#     --eval_every 99999 \
#     --save_every 99999 \
#     --eval_samples $EVAL_SAMPLES \
#     --eval_batch_size $EVAL_BATCH_SIZE \
#     --num_log_samples $NUM_LOG_SAMPLES \
#     --log_every 10 \
#     --output_dir $FT_DIR &
# done

# wait

# # Batch 2: 5th dataset (GPU 0)
# DS=${DATASETS[4]}
# FT_DIR="./ckpt/cpt_${TIMESTAMP}/sorl_v7_q06_ft_${DS}"
# echo ">>> [GPU 0] SoRL v7 fine-tune: ${DS}"

# CUDA_VISIBLE_DEVICES=0 torchrun \
#   --nproc_per_node=1 \
#   --master_addr=$MASTER_ADDR \
#   --master_port=$((BASE_PORT + 20)) \
#   train_sorl_post.py \
#   --model_name $M06 \
#   --resume_ckpt $SORL_CPT_CKPT \
#   --dataset $DS \
#   --num_epochs $FT_EPOCHS \
#   --lr $LR \
#   --warmup_steps 20 \
#   --batch_size $BATCH_SIZE \
#   --gradient_accumulation_steps $GRAD_ACCUM \
#   --use_v7 \
#   --abs_routing_mode similar_magnitude \
#   --prefix_abs --abs_prefix_max 8 \
#   --K 8 --eval_K 8 \
#   --max_iterations 2 \
#   --emb_lr_mult 10.0 \
#   --abstract_vocab_size 128 \
#   --eval_every 99999 \
#   --save_every 99999 \
#   --eval_samples $EVAL_SAMPLES \
#   --eval_batch_size $EVAL_BATCH_SIZE \
#   --num_log_samples $NUM_LOG_SAMPLES \
#   --log_every 10 \
#   --output_dir $FT_DIR

# echo ""
# echo "============================================================"
# echo "All stages complete. Results in ./ckpt/cpt_${TIMESTAMP}/"
# echo ""
# echo "  Stage 1 (CPT):"
# echo "    SoRL: ${SORL_DIR}/train.log"
# echo "    SFT:  ${SFT_DIR}/train.log"
# echo ""
# echo "  Stage 2 (fine-tune from SoRL CPT):"
# for DS in "${DATASETS[@]}"; do
#   echo "    ${DS}: ./ckpt/cpt_${TIMESTAMP}/sorl_v7_q06_ft_${DS}/train.log"
# done
# echo "============================================================"



# ============================================================================
# Stage 3: V1 vs V2 (separate_abs_params) — single-dataset fine-tuning
#
# Qwen3-0.6B, 7 datasets, V=128, mi=2, emb_lr=1.0, 1 ep
# V1: abstract rows in expanded embed_tokens/lm_head (tied with NL)
# V2: standalone abs_embed + abs_proj (decoupled from NL tying)
# 7 datasets × 2 variants = 14 experiments, batches of 4
# ============================================================================

V2_DS=("gsm8k" "scienceqa" "arc" "mmlu" "commonsenseqa" "openbookqa" "boolq")
V2_DT=("gsm" "sci" "arc" "mmlu" "csqa" "obqa" "boolq")

echo ""
echo "============================================================"
echo "Stage 3: V1 vs V2 ablation — 7 ds × 2 = 14 exps"
echo "============================================================"

V2_IDX=0
for di in "${!V2_DS[@]}"; do
  ds=${V2_DS[$di]}
  dt=${V2_DT[$di]}

  for variant in v1 v2; do
    V2_IDX=$((V2_IDX + 1))
    gpu=$(( (V2_IDX - 1) % 2 + 2 ))   # cycle through GPU 2,3 only
    port=$((BASE_PORT + 30 + V2_IDX))
    out="./ckpt/cpt_${TIMESTAMP}/v1v2_q06_${dt}_${variant}"

    EXTRA=""
    if [ "$variant" = "v2" ]; then
      EXTRA="--separate_abs_params"
    fi

    echo ">>> [GPU ${gpu}] q06/${dt}/${variant}"

    CUDA_VISIBLE_DEVICES=$gpu torchrun \
      --nproc_per_node=1 \
      --master_addr=$MASTER_ADDR \
      --master_port=$port \
      train_sorl_post.py \
      --model_name $M06 \
      --dataset $ds \
      --num_epochs 1 \
      --lr $LR \
      --warmup_steps $WARMUP_STEPS \
      --batch_size $BATCH_SIZE \
      --gradient_accumulation_steps $GRAD_ACCUM \
      --use_v7 \
      --abs_routing_mode similar_magnitude \
      --prefix_abs --abs_prefix_max 8 \
      --K 8 --eval_K 8 \
      --max_iterations 2 \
      --emb_lr_mult 1.0 \
      --abstract_vocab_size 128 \
      --eval_every 99999 \
      --save_every 99999 \
      --eval_samples $EVAL_SAMPLES \
      --eval_batch_size $EVAL_BATCH_SIZE \
      --num_log_samples $NUM_LOG_SAMPLES \
      --log_every 10 \
      $EXTRA \
      --output_dir $out &

    if (( V2_IDX % 2 == 0 )); then wait; fi
  done
done
wait

echo ""
echo "============================================================"
echo "V1 vs V2 ablation complete."
echo "============================================================"
