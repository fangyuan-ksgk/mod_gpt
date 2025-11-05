# ============================================================================
# Configuration
# ============================================================================
BATCH_SIZE=30  # Closer to benchmark batch_size=32
TRAIN_SEQ_LEN=$((16 * 1024))
VAL_SEQ_LEN=$((16 * 1024))
NUM_ITERATIONS=1750
N_GPUS=3


# ============================================================================
# BASELINE EXPERIMENTS
# ============================================================================

echo "========================================="
echo "GAPT: Gated Phase Transition Training"
echo "========================================="

torchrun --standalone --nproc_per_node=$N_GPUS train_iblm.py \
  --batch_size $BATCH_SIZE \
  --train_seq_len $TRAIN_SEQ_LEN \
  --val_seq_len $VAL_SEQ_LEN \
  --num_iterations $NUM_ITERATIONS \
  --use_gapt \
  --entropy_patience 125 \
  --entropy_min_delta 0.01 \
  --mbe_patience 125 \
  --mbe_min_delta 0.01 \
  --patch_size 8

echo "Training complete!"