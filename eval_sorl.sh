#!/bin/bash

# SORL Evaluation Script
# Usage:
#   Single GPU:  bash eval_sorl.sh
#   Multi-GPU:   bash eval_sorl.sh 4   (for 4 GPUs)

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
MASTER_PORT=29503

# -------------------- "slow" script (fair evaluation) --------------------
COMMON_ARGS=(
    --hf_repo_id "Ksgk-fy/sorl"
    --hf_filename "ts-k4-v128.pt"
    --hf_filename_base "gpt2-small-ts.pt"
    --model_size "small"
    --abstract_vocab_size 128
    --num_rollouts 5
    --K 4
    --max_iterations 2
    --min_temperature 0.0
    --max_temperature 5.0
    --save_path "eval_sorl.csv"
    --split "validation"
    --num_stories 1000
    --max_len 1024
    --batch_size 4
    --use_compile
)

if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Running single-GPU evaluation..."
    python eval_sorl_slow.py "${COMMON_ARGS[@]}"
else
    echo "Running distributed evaluation on $NUM_GPUS GPUs..."
    torchrun \
      --nproc_per_node=$NUM_GPUS \
      --master_addr=$MASTER_ADDR \
      --master_port=$MASTER_PORT \
      eval_sorl_slow.py "${COMMON_ARGS[@]}"
fi

# -------------------- "fast" script (less fair, but closer to training logs) --------------------
