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

# Baseline: No GAPT (for comparison)
echo "========================================="
echo "Baseline: Training WITHOUT GAPT"
echo "========================================="
# torchrun \
#   --nproc_per_node=$N_GPUS \
#   --master_addr=$MASTER_ADDR \
#   --master_port=$((MASTER_PORT++)) \
#   train_iblm.py \
#   --batch_size $BATCH_SIZE \
#   --train_seq_len $TRAIN_SEQ_LEN \
#   --val_seq_len $VAL_SEQ_LEN \
#   --num_iterations $NUM_ITERATIONS \
#   --patch_size 8


# echo "========================================="
# echo "GAPT: Gated Phase Transition Training"
# echo "========================================="

#   torchrun \
#     --nproc_per_node=$N_GPUS \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$((MASTER_PORT++)) \
#     train_iblm.py \
#   --batch_size $BATCH_SIZE \
#   --train_seq_len $TRAIN_SEQ_LEN \
#   --val_seq_len $VAL_SEQ_LEN \
#   --num_iterations $NUM_ITERATIONS \
#   --use_gapt \
#   --entropy_patience 250 \
#   --entropy_min_delta 0.01 \
#   --mbe_patience 50 \
#   --mbe_min_delta 0.01 \
#   --patch_size 8



# Large scale model sweep (10B fineweb)

NUM_ITERATIONS=8000
for MODEL_SIZE in "medium" "large" "xl"; do
  torchrun \
      --nproc_per_node=$N_GPUS \
      --master_addr=$MASTER_ADDR \
      --master_port=$((MASTER_PORT++)) \
      train_iblm.py \
      --batch_size $BATCH_SIZE \
      --train_seq_len $TRAIN_SEQ_LEN \
      --val_seq_len $VAL_SEQ_LEN \
      --num_iterations $NUM_ITERATIONS \
      --use_gapt \
      --entropy_patience 250 \
      --entropy_min_delta 0.01 \
      --mbe_patience 50 \
      --mbe_min_delta 0.01 \
      --patch_size 8 \
      --model_size $MODEL_SIZE \
      --run_info "GAPT Sweep: ModelSize=$MODEL_SIZE | CEPat=250 | MBEPat=50" 
done

# Sweep on MBE schedule across layers (which one gets regularized when)
# ---------------------------------------------------------------------
for MBE_SCHEDULE in "rotate" "rotate_accum" "progressive" "weighted_valley" "weighted_mountain"; do
  torchrun \
      --nproc_per_node=$N_GPUS \
      --master_addr=$MASTER_ADDR \
      --master_port=$((MASTER_PORT++)) \
      train_iblm.py \
      --batch_size $BATCH_SIZE \
      --train_seq_len $TRAIN_SEQ_LEN \
      --val_seq_len $VAL_SEQ_LEN \
      --num_iterations 1750 \
      --use_gapt \
      --entropy_patience 250 \
      --entropy_min_delta 0.01 \
      --mbe_patience 50 \
      --mbe_min_delta 0.01 \
      --patch_size 8 \
      --model_size "small" \
      --run_info "GAPT Sweep: MBEschedule=$MBE_SCHEDULE | CEPat=250 | MBEPat=50 | Model=GPT2-small" 
done


# # Sweep on entropy min delta (allow more oscillation)
# for ENTROPY_MIN_DELTA in 0.005 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 0.20 0.30; do
#   torchrun \
#       --nproc_per_node=$N_GPUS \
#       --master_addr=$MASTER_ADDR \
#       --master_port=$((MASTER_PORT++)) \
#       train_iblm.py \
#       --batch_size $BATCH_SIZE \
#       --train_seq_len $TRAIN_SEQ_LEN \
#       --val_seq_len $VAL_SEQ_LEN \
#       --num_iterations $NUM_ITERATIONS \
#       --use_gapt \
#       --entropy_patience 250 \
#       --entropy_min_delta $ENTROPY_MIN_DELTA \
#       --mbe_patience 50 \
#       --mbe_min_delta 0.01 \
#       --patch_size 8 \
#       --run_info "GAPT Sweep: EntMinDelta=$ENTROPY_MIN_DELTA | CEPat=250 | MBEPat=50" 
# done 

# # Sweep on mbe min delta (allow more oscillation)
# for MBE_MIN_DELTA in 0.005 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 0.20 0.30; do
#   torchrun \
#       --nproc_per_node=$N_GPUS \
#       --master_addr=$MASTER_ADDR \
#       --master_port=$((MASTER_PORT++)) \
#       train_iblm.py \
#       --batch_size $BATCH_SIZE \
#       --train_seq_len $TRAIN_SEQ_LEN \
#       --val_seq_len $VAL_SEQ_LEN \
#       --num_iterations $NUM_ITERATIONS \
#       --use_gapt \
#       --entropy_patience 250 \
#       --entropy_min_delta 0.01 \
#       --mbe_patience 50 \
#       --mbe_min_delta $MBE_MIN_DELTA \
#       --patch_size 8 \
#       --run_info "GAPT Sweep: MBEMinDelta=$MBE_MIN_DELTA | CEPat=250 | MBEPat=50" 
# done




# Experiment (I). The best balance in 'patience' for entropy & mbe
# entropy_patience 250 
# mbe_patience 50 
# is a good balance, we get to 3.28 (0.02 better than baseline)

# Experiment (II). Patch size sweep & entropy min_delta, mbe_min_delta sweep
