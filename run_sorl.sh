# Experiment Script for SoRL

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
BATCH_SIZE=30  # Closer to benchmark batch_size=32
TRAIN_SEQ_LEN=$((16 * 1024))
VAL_SEQ_LEN=$((16 * 1024))
NUM_ITERATIONS=1750
NUM_ROLLOUTS=2
MAX_ITERATIONS=2
N_GPUS=6
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500
ALPHA_LOSS=0.1
EXPLORATION_MODE=0 # SGPO - exploration


# echo "========================================="
# echo "BASELINE: No Abstraction (Standard GPT)"
# echo "========================================="

# torchrun --standalone --nproc_per_node=$N_GPUS train_base.py \
#   --batch_size $BATCH_SIZE \
#   --train_seq_len $TRAIN_SEQ_LEN \
#   --val_seq_len $VAL_SEQ_LEN \
#   --num_iterations $NUM_ITERATIONS


# ===============================================
# Exp 12.1: 
# - hypothesis: free-bit entropy regularization + select-best per abs SoRL gives non-collapsed vocabulary with positive search advantage
# -             free-bit entropy regularization + select-best per doc SoRL gives non-collapsed vocabulary with positive search advantage
# - obs #1. 'select-best per-abs' gives negative search advantage. 
# - obs #2. alpha_entropy=0.1 + target entropy=1.2 gives 2 word vocabulary (not exactly collapsed)
# - hypo #1. I suspect the key is larger 'target_entropy', the weak alpha_entropy essentially losen such constraint.
# Get to extreme, just to see if we can avoid collapse, we'd worry about optimal setting later
# ===============================================
for ALPHA_ENTROPY in 0.1 1.0; do
  for TARGET_ENTROPY in 12.0 11.0 10.0 9.0 8.0 7.0 6.0 5.0; do
    torchrun \
      --nproc_per_node=$N_GPUS \
      --master_addr=$MASTER_ADDR \
      --master_port=$MASTER_PORT \
      train_sorl_v5.py \
      --batch_size $BATCH_SIZE \
      --train_seq_len $TRAIN_SEQ_LEN \
      --val_seq_len $VAL_SEQ_LEN \
      --num_iterations $NUM_ITERATIONS \
      --num_rollouts $NUM_ROLLOUTS \
      --K 8 \
      --max_iterations $MAX_ITERATIONS \
      --min_temperature 0.0 \
      --temperature 5.0 \
      --alpha_loss $ALPHA_LOSS \
      --alpha_entropy $ALPHA_ENTROPY \
      --target_entropy $TARGET_ENTROPY \
      --run_info "Exp12.1: Select-best SoRL with entropy regularization (alpha_entropy=$ALPHA_ENTROPY, target_entropy=$TARGET_ENTROPY, use_per_doc_selection)"
  done
done