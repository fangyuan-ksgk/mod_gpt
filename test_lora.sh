#!/bin/bash
set -e

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

MASTER_ADDR=127.0.0.1
BASE_PORT=29501
TIMESTAMP=$(date +%Y%m%d_%H%M)
EXP_IDX=0

run_bg() {
  EXP_IDX=$((EXP_IDX + 1))
  local idx=$EXP_IDX
  local gpu=$(( (idx - 1) % 4 ))
  local port=$((BASE_PORT + idx))
  local tag=$1; shift
  local model=$1; shift
  local dataset=$1; shift
  local output_dir="./ckpt/test_lora_${TIMESTAMP}/exp${idx}_${tag}"

  echo "  Exp ${idx}: ${tag}  model=$(basename $model)  dataset=${dataset}  [GPU=${gpu}]"

  CUDA_VISIBLE_DEVICES=$gpu torchrun \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$port \
    train_sorl_post.py \
    --model_name $model \
    --dataset $dataset \
    --max_length 512 \
    --lr 1e-5 \
    --warmup_steps 10 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_epochs 1 \
    --log_every 5 \
    --eval_every 50 \
    --save_every 50 \
    --eval_samples 100 \
    --eval_batch_size 16 \
    --max_new_tokens 64 \
    --output_dir $output_dir \
    --use_lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    "$@" &
}

# SoRL flags for quick test
R_V1="--alpha_info_gain 1.0 --alpha_abs 0.5"
R_V6="--use_v6 --abstract_vocab_size 32 --K 16 --eval_K 16"

echo "Quick LoRA save-logic test: SFT + SoRL (v1, v6) on 4 GPUs"

# GPU 0: plain SFT + LoRA (Qwen3-1.7B)
run_bg "sft_csqa_1.7B"  Qwen/Qwen3-1.7B         commonsenseqa
# GPU 1: plain SFT + LoRA (Llama-3.2-1B)
run_bg "sft_csqa_L1"    meta-llama/Llama-3.2-1B  commonsenseqa
# GPU 2: SoRL v1 + LoRA (Qwen3-1.7B)
run_bg "v1_csqa_1.7B"   Qwen/Qwen3-1.7B         commonsenseqa $R_V1
# GPU 3: SoRL v6 + LoRA (Qwen3-1.7B)
run_bg "v6_csqa_1.7B"   Qwen/Qwen3-1.7B         commonsenseqa $R_V6

wait
echo "Done. Check ./ckpt/test_lora_${TIMESTAMP}/ for saved checkpoints."
