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
NUM_ITERATIONS=1750
N_GPUS=4
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500


# ============================================================================
# BASELINE EXPERIMENTS
# ============================================================================

# torchrun \
#   --nproc_per_node=$N_GPUS \
#   --master_addr=$MASTER_ADDR \
#   --master_port=$((MASTER_PORT++)) \
#   train_base.py
#   --batch_size $BATCH_SIZE \
#   --train_seq_len $TRAIN_SEQ_LEN \
#   --val_seq_len $VAL_SEQ_LEN \
#   --num_iterations $NUM_ITERATIONS \
#   --run_info "Sanity-checking baseline"


# ========================================="
# Exp 1. FineWeb0.8B (Base, GAPT, MBE, L2)
#        | MBE composition mode sweep: naive, max, softmax
# Exp 2. FineWeb10B (GPT2 small, medium, large, xl)
#        | 'num_iterations' change the 'attention blocksize' at each step (smaller for bigger iterations number)
#        | therefore it's unfair to compare results at same step across run with different 'num_itrations'
# ========================================="
MBE_COMP_MODE="spike"
for MODEL_SIZE in "small"; do
  torchrun \
      --nproc_per_node=$N_GPUS \
      --master_addr=$MASTER_ADDR \
      --master_port=$((MASTER_PORT++)) \
      train_iblm.py \
      --batch_size 32 \
      --train_seq_len $TRAIN_SEQ_LEN \
      --val_seq_len $VAL_SEQ_LEN \
      --num_iterations 1750 \
      --use_gapt \
      --entropy_patience 125 \
      --entropy_min_delta 0.01 \
      --mbe_patience 75 \
      --mbe_min_delta 0.01 \
      --mbe_weight 20.0 \
      --mbe_comp_mode $MBE_COMP_MODE \
      --mbe_schedule "all_middle" \
      --model_size $MODEL_SIZE \
      --save_checkpoint \
      --run_info "GAPT: ModelSize=$MODEL_SIZE | GAPT | w=20.0 | MBE comp mode: $MBE_COMP_MODE | regularize on all middle layers" 
done

for MODEL_SIZE in "small"; do
  torchrun \
      --nproc_per_node=$N_GPUS \
      --master_addr=$MASTER_ADDR \
      --master_port=$((MASTER_PORT++)) \
      train_iblm.py \
      --batch_size 32 \
      --train_seq_len $TRAIN_SEQ_LEN \
      --val_seq_len $VAL_SEQ_LEN \
      --num_iterations 1750 \
      --no_reg \
      --model_size $MODEL_SIZE \
      --save_checkpoint \
      --run_info "GAPT: ModelSize=$MODEL_SIZE | 1750 steps | no regularization" 
done

# 贼心不死
# MODEL_SIZE="medium"
# for MBE_COMP_MODE in "valley"; do
#   torchrun \
#       --nproc_per_node=$N_GPUS \
#       --master_addr=$MASTER_ADDR \
#       --master_port=$((MASTER_PORT++)) \
#       train_iblm.py \
#       --batch_size 32 \
#       --train_seq_len $TRAIN_SEQ_LEN \
#       --val_seq_len $VAL_SEQ_LEN \
#       --num_iterations 1750 \
#       --use_gapt \
#       --entropy_patience 125 \
#       --entropy_min_delta 0.01 \
#       --mbe_patience 75 \
#       --mbe_min_delta 0.01 \
#       --mbe_weight 20.0 \
#       --mbe_comp_mode $MBE_COMP_MODE \
#       --mbe_schedule "all_middle" \
#       --model_size $MODEL_SIZE \
#       --save_checkpoint \
#       --run_info "GAPT: ModelSize=$MODEL_SIZE | GAPT | w=20.0 | MBE comp mode: $MBE_COMP_MODE | regularize on all middle layers" 
# done


# NUM_ITERATIONS=20000
# # Baseline experiment (10B fineweb dataset)
# # # -----------------------------------------
# for MODEL_SIZE in "xl"; do
#   torchrun \
#     --nproc_per_node=$N_GPUS \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$((MASTER_PORT++)) \
#     train_iblm.py \
#     --batch_size $BATCH_SIZE \
#     --train_seq_len $TRAIN_SEQ_LEN \
#     --val_seq_len $VAL_SEQ_LEN \
#     --num_iterations $NUM_ITERATIONS \
#     --no_reg \
#     --model_size $MODEL_SIZE \
#     --save_checkpoint \
#     --save_checkpoint_every 2000 \
#     --run_info "Baseline: ModelSize=$MODEL_SIZE" 
# done 




# Experiment (I). The best balance in 'patience' for entropy & mbe
# entropy_patience 250 
# mbe_patience 50 
# is a good balance, we get to 3.28 (0.02 better than baseline)

# Experiment (II). Patch size sweep & entropy min_delta, mbe_min_delta sweep
