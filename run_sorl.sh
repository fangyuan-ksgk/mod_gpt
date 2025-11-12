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
N_GPUS=2
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500
ALPHA_LOSS=0.1


# ============================================================================
# BASELINE EXPERIMENTS
# ============================================================================

# echo "========================================="
# echo "BASELINE: No Abstraction (Standard GPT)"
# echo "========================================="

# torchrun --standalone --nproc_per_node=$N_GPUS train_base.py \
#   --batch_size $BATCH_SIZE \
#   --train_seq_len $TRAIN_SEQ_LEN \
#   --val_seq_len $VAL_SEQ_LEN \
#   --num_iterations $NUM_ITERATIONS


# ============================================================================
# SGPO + EMA: Sweep EMA Decay × GAPT
# ============================================================================

echo "========================================="
echo "SGPO + EMA EXPERIMENTS"
echo "Testing: ema_decay × gapt"
echo "========================================="

EXP_NUM=1
for EMA_DECAY in 0.90 0.95 0.99; do    
  echo "[Exp $EXP_NUM/6] ema_decay=$EMA_DECAY, use gapt"
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
    --ema_decay $EMA_DECAY \
    --alpha_loss $ALPHA_LOSS \
    --use_gapt \
    --traj_perplexity_patience 100 \
    --abs_perplexity_patience 100 \
    --run_info "SGPO+EMA: decay=$EMA_DECAY, use gapt"

  EXP_NUM=$((EXP_NUM + 1))
 done


for EMA_DECAY in 0.90 0.95 0.99; do    
  echo "[Exp $EXP_NUM/6] ema_decay=$EMA_DECAY, no gapt"
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
    --ema_decay $EMA_DECAY \
    --alpha_loss $ALPHA_LOSS \
    --run_info "SGPO+EMA: decay=$EMA_DECAY, no gapt"

  EXP_NUM=$((EXP_NUM + 1))
 done

echo "========================================="
echo "All SGPO+EMA experiments completed!"
echo "========================================="