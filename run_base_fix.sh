#!/bin/bash
# run_dist_test.sh

# =================================================================
# Environment Fixes for NCCL on NVIDIA Pods
# =================================================================

# Dummy tuner config to avoid NCCL crashing
export NCCL_TUNER_CONFIG_PATH="/workspace/mod_gpt/dummy_tuner_config.txt"

# Disable the crashing NCCL plugins that are auto-loaded by the pod's environment
export NCCL_TUNER_PLUGIN=""
export NCCL_NET_PLUGIN=""

# Force use of the reliable loopback network interface for single-node communication
export NCCL_SOCKET_IFNAME=lo

# Disable InfiniBand to prevent auto-detection conflicts
export NCCL_IB_DISABLE=1

# Use WARN for cleaner logs. Change to INFO to see verbose NCCL output.
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
    --batch_size 16 \
    --num_iterations 1750 \
    --train_seq_len 8192 \
    --val_seq_len 8192

echo "--- Test finished ---"