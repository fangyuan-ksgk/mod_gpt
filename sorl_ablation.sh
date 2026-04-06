#!/bin/bash
set -e

# Ablation method for SoRL
# 1. Token Assorted
# 2. Pause Token
# 3. Gist Token

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

# ============================================================================
# Token Assorted — GSM8K + Qwen3-1.7B + LoRA
# ============================================================================
MODEL="Qwen/Qwen3-1.7B"
DATASET="gsm8k"
TIMESTAMP=$(date +%Y%m%d_%H%M)
OUTPUT_DIR="./ckpt/ablation_${TIMESTAMP}/ta_gsm_1.7B"
VQVAE_CKPT="./ckpt/vqvae/ta_gsm_1.7B.pt"

torchrun \
  --nproc_per_node=1 \
  --master_addr=127.0.0.1 \
  --master_port=29600 \
  train_ta_pt.py \
  --model_name $MODEL \
  --dataset $DATASET \
  --max_length 512 \
  --lr 1e-5 \
  --warmup_steps 50 \
  --batch_size 2 \
  --gradient_accumulation_steps 4 \
  --num_epochs 3 \
  --log_every 10 \
  --log_samples_every 999999 \
  --num_log_samples 3 \
  --eval_every 99999 \
  --save_every 99999 \
  --eval_samples 1319 \
  --eval_batch_size 64 \
  --max_new_tokens 256 \
  --output_dir $OUTPUT_DIR \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --vqvae_steps 20000 \
  --vqvae_l 16 \
  --vqvae_c_size 1024 \
  --vqvae_d_bot 256 \
  --vqvae_lr 1e-5 \
  --vqvae_batch_size 32 \
  --vqvae_save_ckpt $VQVAE_CKPT

# ============================================================================
# Pause Token — GSM8K + Qwen3-1.7B + LoRA
# ============================================================================
OUTPUT_DIR="./ckpt/ablation_${TIMESTAMP}/pause_gsm_1.7B"

torchrun \
  --nproc_per_node=1 \
  --master_addr=127.0.0.1 \
  --master_port=29601 \
  train_pause_pt.py \
  --model_name $MODEL \
  --dataset $DATASET \
  --max_length 512 \
  --lr 1e-5 \
  --warmup_steps 50 \
  --batch_size 2 \
  --gradient_accumulation_steps 4 \
  --num_epochs 3 \
  --log_every 10 \
  --log_samples_every 999999 \
  --num_log_samples 3 \
  --eval_every 99999 \
  --save_every 99999 \
  --eval_samples 1319 \
  --eval_batch_size 64 \
  --max_new_tokens 256 \
  --output_dir $OUTPUT_DIR \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --k_pause 8