#!/bin/bash

# Fast SORL Evaluation Script
# Usage:
#   Single GPU:  bash eval_fast_sorl.sh
#   Multi-GPU:   bash eval_fast_sorl.sh 2

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

NUM_GPUS=${1:-2}
MASTER_ADDR=127.0.0.2
MASTER_PORT=$((29500 + RANDOM % 1000))

COMMON_ARGS=(
    --hf_repo_id "Ksgk-fy/sorl"
    --hf_filename "ts-k4-v128.pt"
    --model_size "small"
    --abstract_vocab_size 128
    --val_files "data/tinystories/tinystory_val_*.bin"
    --val_tokens 10485760
    --val_seq_len 16384
    --num_rollouts 2
    --K 4
    --max_iterations 2
    --min_temperature 0.0
    --max_temperature 5.0
    --avoid_prefix_truncation
    --save_path "eval_fast_sorl.csv"
)

if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Running single-GPU fast evaluation..."
    python eval_sorl.py "${COMMON_ARGS[@]}"
else
    echo "Running distributed fast evaluation on $NUM_GPUS GPUs..."
    torchrun \
      --nproc_per_node=$NUM_GPUS \
      --master_addr=$MASTER_ADDR \
      --master_port=$MASTER_PORT \
      eval_sorl.py "${COMMON_ARGS[@]}"
fi
