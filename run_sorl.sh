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
NUM_ITERATIONS=1000
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
# Exp 12.6: 
# - hypothesis: utility scaling helps avoid vocabulary collapse, whilst retaining search adv for select-best SoRL
# - we'll sweep on different choices on alpha_marg_ent, alpha_cond_ent, decay and target_vocab_util, starting with the first 2
# ===============================================
# ALPHA_MARG_ENT=1.0
# DECAY=0.8
# TARGET_VOCAB_UTIL=0.8
# K=8
# torchrun \
#   --nproc_per_node=$N_GPUS \
#   --master_addr=$MASTER_ADDR \
#   --master_port=$MASTER_PORT \
#   train_sorl_v5.py \
#   --batch_size $BATCH_SIZE \
#   --train_seq_len $TRAIN_SEQ_LEN \
#   --val_seq_len $VAL_SEQ_LEN \
#   --num_iterations $NUM_ITERATIONS \
#   --num_rollouts $NUM_ROLLOUTS \
#   --K $K \
#   --max_iterations $MAX_ITERATIONS \
#   --min_temperature 0.0 \
#   --temperature 5.0 \
#   --alpha_loss $ALPHA_LOSS \
#   --alpha_marg_ent $ALPHA_MARG_ENT \
#   --decay $DECAY \
#   --target_vocab_util $TARGET_VOCAB_UTIL \
#   --use_orthogonal_init \
#   --utility_scaling \
#   --run_info "Exp12.6: SoRL with Utility Scaling (alpha_marg_ent=$ALPHA_MARG_ENT, decay=$DECAY, target_vocab_util=$TARGET_VOCAB_UTIL)"


# ======================
# TinyStories Dataset 
# - well-trained ckpt
# -> full tinystories dataset got about 1GB tokens
# -> I don't think 500 step is sufficient here
# ======================
ALPHA_MARG_ENT=1.0
DECAY=0.8
TARGET_VOCAB_UTIL=0.8
K=4
ABSTRACT_VOCAB_SIZE=16
torchrun \
  --nproc_per_node=$N_GPUS \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train_sorl_v3.py \
  --batch_size $BATCH_SIZE \
  --train_seq_len $TRAIN_SEQ_LEN \
  --val_seq_len $VAL_SEQ_LEN \
  --num_iterations $NUM_ITERATIONS \
  --num_rollouts $NUM_ROLLOUTS \
  --K $K \
  --abstract_vocab_size $ABSTRACT_VOCAB_SIZE \
  --save_checkpoint \
  --max_iterations $MAX_ITERATIONS \
  --min_temperature 0.0 \
  --temperature 5.0 \
  --alpha_loss $ALPHA_LOSS \
  --alpha_marg_ent $ALPHA_MARG_ENT \
  --decay $DECAY \
  --target_vocab_util $TARGET_VOCAB_UTIL \
  --use_orthogonal_init \
  --utility_scaling \
  --run_info "Exp2.1 TinyStories Dataset (K=$K, abstract_vocab_size=$ABSTRACT_VOCAB_SIZE)"