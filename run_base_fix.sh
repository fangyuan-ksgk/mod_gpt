#!/bin/bash

# =================================================================
# NCCL Environment Fixes for Containerized Environments
# =================================================================
# 1. Force NCCL to use a specific network interface (lo is safest for single-node)
export NCCL_SOCKET_IFNAME=lo

# 2. Disable InfiniBand if it's causing conflicts
export NCCL_IB_DISABLE=1

# 3. Give a very long timeout for initialization
export NCCL_TIMEOUT=600

# 4. Enable verbose NCCL logging to see what it's doing
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# =================================================================
# Run the training script
# =================================================================
echo "--- Starting training with NCCL fixes ---"

torchrun --standalone \
  --nproc_per_node=2 \
  --master_addr=127.0.0.1 \
  --master_port=29500 \
  train_base.py \
  --batch_size 16 \
  --num_iterations 100 # Use a small number of iterations for the test

echo "--- Training script finished ---"