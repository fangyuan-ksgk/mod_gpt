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
BATCH_SIZE=32  # Closer to benchmark batch_size=32
TRAIN_SEQ_LEN=$((32 * 1024))
VAL_SEQ_LEN=$((32 * 1024))
NUM_ITERATIONS=500
N_GPUS=4
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500

# ====================================
# Catastrohpic Forgetting Probe
# ====================================
MBE_COMP_MODE="spike"
MODEL_SIZE="small"
TINY_STORIES_FILES="data/tiny_stories/tiny_stories_train_*.bin"
FINEWEB_VAL_FILES="data/fineweb10B/fineweb_val_*.bin"

torchrun \
  --nproc_per_node=$N_GPUS \
  --master_addr=$MASTER_ADDR \
  --master_port=$((MASTER_PORT++)) \
  train_iblm.py \
  --continue_from_ckpt "ckpt/fineweb10B-gpt2-small-20k.pt" \
  --train_files $TINY_STORIES_FILES \
  --val_files $FINEWEB_VAL_FILES \
  --batch_size $BATCH_SIZE \
  --train_seq_len $TRAIN_SEQ_LEN \
  --val_seq_len $VAL_SEQ_LEN \
  --num_iterations $NUM_ITERATIONS \
  --no_reg \
  --model_size $MODEL_SIZE \
  --save_checkpoint \
  --run_info "Continuous Pretrain w. Baseline on TinyStories: ModelSize=$MODEL_SIZE" 


for MBE_WEIGHT in 20.0; do
  torchrun \
      --nproc_per_node=$N_GPUS \
      --master_addr=$MASTER_ADDR \
      --master_port=$((MASTER_PORT++)) \
      train_iblm.py \
      --continue_from_ckpt "ckpt/fineweb10B-gpt2-small-20k.pt" \
      --train_files $TINY_STORIES_FILES \
      --val_files $FINEWEB_VAL_FILES \
      --batch_size $BATCH_SIZE \
      --train_seq_len $TRAIN_SEQ_LEN \
      --val_seq_len $VAL_SEQ_LEN \
      --num_iterations $NUM_ITERATIONS \
      --use_gapt \
      --entropy_patience 125 \
      --entropy_min_delta 0.01 \
      --mbe_patience 75 \
      --mbe_min_delta 0.01 \
      --mbe_weight $MBE_WEIGHT \
      --mbe_comp_mode $MBE_COMP_MODE \
      --mbe_schedule "all_middle" \
      --model_size $MODEL_SIZE \
      --save_checkpoint \
      --run_info "continuous pretrain w. GAPT on TinyStories: ModelSize=$MODEL_SIZE | w=$MBE_WEIGHT | MBE comp mode: $MBE_COMP_MODE"
done