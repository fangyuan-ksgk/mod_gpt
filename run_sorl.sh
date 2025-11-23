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
N_GPUS=3
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500
ALPHA_LOSS=0.1
EXPLORATION_MODE=0 # SGPO - exploration


# Try the 'all-rollout SoRL' again
# I'd like to ablate on some more variants of its adv formulation
# - (1). adv \propto 1 / p(a | s)
# - (2). adv \propto 
# - p(a | s) / p(s | a) --> favor more familiar abstraction at same utility, favor less utility abstractino at same familiarity
# - can we somehow add the topological smimilarity term into the adv formulation to 'regulate' the preference for horrble abstraction? 


# echo "========================================="
# echo "BASELINE: No Abstraction (Standard GPT)"
# echo "========================================="

# torchrun --standalone --nproc_per_node=$N_GPUS train_base.py \
#   --batch_size $BATCH_SIZE \
#   --train_seq_len $TRAIN_SEQ_LEN \
#   --val_seq_len $VAL_SEQ_LEN \
#   --num_iterations $NUM_ITERATIONS


# --------------------------------------------------------------
# Exp 10: curiosity advantage SoRL (with all rollouts)
# 


TOPO_MODE_DESC=("dot product" "correlation" "covariance")

echo "========================================="
echo "Exp 10.1: Off-policy + curiosity advantage SoRL (with all rollouts)"
echo "========================================="
torchrun \
  --nproc_per_node=$N_GPUS \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train_sorl_v2.py \
  --batch_size $BATCH_SIZE \
  --train_seq_len $TRAIN_SEQ_LEN \
  --val_seq_len $VAL_SEQ_LEN \
  --num_iterations $NUM_ITERATIONS \
  --num_rollouts $NUM_ROLLOUTS \
  --K 8 \
  --max_iterations $MAX_ITERATIONS \
  --use_static_memory_span \
  --min_temperature 0.0 \
  --temperature 5.0 \
  --alpha_loss $ALPHA_LOSS \
  --mode 4 \
  --steps_per_cycle $MAX_ITERATIONS \
  --exploration_fraction 1.0 \
  --exploration_till_vocab_util 1.0 \
  --run_info "Exp10.1: Off-policy + curiosity advantage SoRL (with all rollouts)"


echo "========================================="
echo "Exp 10.2: Off-policy + curiosity advantage SoRL (with all rollouts)"
echo "========================================="
torchrun \
  --nproc_per_node=$N_GPUS \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train_sorl_v2.py \
  --batch_size $BATCH_SIZE \
  --train_seq_len $TRAIN_SEQ_LEN \
  --val_seq_len $VAL_SEQ_LEN \
  --num_iterations $NUM_ITERATIONS \
  --num_rollouts $NUM_ROLLOUTS \
  --K 8 \
  --max_iterations $MAX_ITERATIONS \
  --use_static_memory_span \
  --min_temperature 0.0 \
  --temperature 5.0 \
  --alpha_loss $ALPHA_LOSS \
  --mode 5 \
  --steps_per_cycle $MAX_ITERATIONS \
  --exploration_fraction 1.0 \
  --exploration_till_vocab_util 1.0 \
  --run_info "Exp10.2: Off-policy + curiosity + utility preference SoRL (with all rollouts)"