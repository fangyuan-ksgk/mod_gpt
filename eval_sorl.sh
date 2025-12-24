#!/bin/bash

# SORL Evaluation Script
# Usage:
#   Single GPU:  bash eval_sorl.sh
#   Multi-GPU:   bash eval_sorl.sh 4   (for 4 GPUs)

NUM_GPUS=${1:-2}

COMMON_ARGS=(
    --hf_repo_id "Ksgk-fy/sorl"
    --hf_filename "ts-k4-v128.pt"
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
    torchrun --nproc_per_node=$NUM_GPUS eval_sorl_slow.py "${COMMON_ARGS[@]}"
fi
