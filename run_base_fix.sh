#!/bin/bash
# run_dist_test.sh

# =================================================================
# Environment Fixes for NCCL on NVIDIA Pods
# =================================================================

# Dummy tuner config to avoid NCCL crashing
DUMMY_CONFIG_PATH="/workspace/mod_gpt/dummy_tuner_config.txt"
rm -f "$DUMMY_CONFIG_PATH"
touch "$DUMMY_CONFIG_PATH"

export NCCL_TUNER_CONFIG_PATH="$DUMMY_CONFIG_PATH"
export NCCL_TUNER_PLUGIN=""
export NCCL_NET_PLUGIN=""
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

# =================================================================
# Run the debug script
# =================================================================
echo "--- Running distributed test with NCCL backend and environment fixes ---"

torchrun --standalone \
  --nproc_per_node=2 \
  --master_addr=127.0.0.1 \
  --master_port=29500 \
  train_base.py \
    --batch_size 32 \
    --num_iterations 1750 \
    --train_seq_len 8192 \
    --val_seq_len 8192

echo "--- Test finished ---"