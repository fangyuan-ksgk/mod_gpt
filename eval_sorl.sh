#!/bin/bash

# SORL Evaluation Script || Alien token suffix probes
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
# Sweep across different pad_shift values (alien token indices)
PAD_SHIFTS=(1 2 5 10 20 50 100 200 500 1000 2000 5000 10000)

OUTPUT_DIR="eval_results"
mkdir -p "$OUTPUT_DIR"

for PAD_SHIFT in "${PAD_SHIFTS[@]}"; do
    echo ""
    echo "========================================"
    echo "Running evaluation with pad_shift=$PAD_SHIFT"
    echo "========================================"
    
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
        --save_path "${OUTPUT_DIR}/eval_sorl_pad${PAD_SHIFT}.csv"
        --split "validation"
        --num_stories 1000
        --max_len 1024
        --batch_size 4
        --use_compile
        --compare_against_base
        --pad_shift $PAD_SHIFT
    )

    if [ "$NUM_GPUS" -eq 1 ]; then
        python eval_sorl_slow.py "${COMMON_ARGS[@]}"
    else
        torchrun \
          --nproc_per_node=$NUM_GPUS \
          --master_addr=$MASTER_ADDR \
          --master_port=$MASTER_PORT \
          eval_sorl_slow.py "${COMMON_ARGS[@]}"
    fi
done

echo ""
echo "========================================"
echo "All evaluations complete!"
echo "Results saved to: $OUTPUT_DIR/"
echo "========================================"

# -------------------- "fast" script (less fair, but closer to training logs) --------------------
