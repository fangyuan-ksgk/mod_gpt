#!/bin/bash
# =============================================================================
# Gradient Alignment Experiment: CE vs MBE & CE vs Frobenius
# =============================================================================
# Tracks cosine similarity between CE gradients and both MBE/Frobenius
# gradients during pre-training. Results saved to logs/<run_id>/.
#
# Usage:
#   bash run_grad_alignment.sh
# =============================================================================

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

# ============================================================================
# Configuration
# ============================================================================
BATCH_SIZE=32
TRAIN_SEQ_LEN=$((32 * 1024))
VAL_SEQ_LEN=$((32 * 1024))
NUM_ITERATIONS=5000
N_GPUS=4
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500

# ============================================================================
# Gradient Alignment Tracking: CE vs MBE & CE vs Frobenius
# ============================================================================
# Runs train_iblm_log.py with --log_grad_info to record gradient cosine
# similarity between CE and both MBE/Frobenius losses at every training step.
#
# After training, load results with:
#   from src.gradtracker import MultiGradStatsRecorder
#   rec = MultiGradStatsRecorder.load("logs/<run_id>/multi_grad_stats_step005000.pkl")
#   mbe_df = rec.global_df("mbe")
#   frob_df = rec.global_df("frob")
# ============================================================================

for MODEL_SIZE in "small" "medium" "large"; do
    torchrun \
        --nproc_per_node=$N_GPUS \
        --master_addr=$MASTER_ADDR \
        --master_port=$((MASTER_PORT++)) \
        train_iblm_log.py \
        --batch_size $BATCH_SIZE \
        --train_seq_len $TRAIN_SEQ_LEN \
        --val_seq_len $VAL_SEQ_LEN \
        --num_iterations $NUM_ITERATIONS \
        --log_grad_info \
        --no_reg \
        --model_size $MODEL_SIZE \
        --run_info "GradAlign: ModelSize=$MODEL_SIZE | CE vs MBE & Frobenius"
done
