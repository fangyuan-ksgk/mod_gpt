#!/bin/bash

# =================================================================
# FINAL NCCL Environment Fixes for NVIDIA Pods
# =================================================================

# 1. Unset the custom network plugin that is causing the crash.
#    This forces NCCL to use its default, reliable networking.
unset NCCL_NET_PLUGIN

# 2. Force NCCL to use a specific, reliable network interface (lo is safest for single-node).
export NCCL_SOCKET_IFNAME=lo

# 3. Disable InfiniBand to prevent conflicts.
export NCCL_IB_DISABLE=1

# 4. Give a long timeout just in case.
export NCCL_TIMEOUT=600

# 5. Keep NCCL logging enabled so we can see what it's doing.
export NCCL_DEBUG=INFO

# =================================================================
# Run the training script
# =================================================================
echo "--- Starting training with FINAL NCCL fixes ---"

torchrun --standalone \
  --nproc_per_node=2 \
  --master_addr=127.0.0.1 \
  --master_port=29500 \
  train_base.py \
  --batch_size 16 \
  --num_iterations 100 # Use a small number of iterations for the test

echo "--- Training script finished ---"