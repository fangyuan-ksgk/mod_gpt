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
BATCH_SIZE=16  # Closer to benchmark batch_size=32
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
#   --patch_size 8 \
#   --no_reg \
#   --run_info "Baseline (no reg)"

# (a.1) Sweep on MBE weight (for soft-add MBE reg)
# for MBE_WEIGHT in 0.1 0.3 0.5; do
#   torchrun \
#     --nproc_per_node=$N_GPUS \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$((MASTER_PORT++)) \
#     train_iblm.py \
#     --batch_size $BATCH_SIZE \
#     --train_seq_len $TRAIN_SEQ_LEN \
#     --val_seq_len $VAL_SEQ_LEN \
#     --num_iterations $NUM_ITERATIONS \
#     --mbe_weight $MBE_WEIGHT \
#     --patch_size 8 \
#     --run_info "Baseline (with soft-add MBE reg, MBE_WEIGHT=$MBE_WEIGHT)"
# done 

# echo "========================================="
# echo "GAPT: Gated Phase Transition Training"
# echo "========================================="
# (a.2) Sweep on Patch size (for soft-add MBE reg) | no difference
# (a.2.1) Patch Size currculum (8 -> 1024)
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
  --entropy_patience 125 \
  --entropy_min_delta 0.01 \
  --mbe_patience 75 \
  --mbe_min_delta 0.01 \
  --run_info "IBLM (soft-add GAPT, entropy_patience=125, mbe_patience=75, patch size 8 -> 1024)"


# The real problem, is GAPT doesn't shrink MBE 

# (a.3) Sweep on mbe patience and entropy_min_delta
for MBE_PATIENCE in 10 20 30 40 50; do
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
    --entropy_patience $MBE_PATIENCE \
    --entropy_min_delta 0.01 \
    --mbe_patience $MBE_PATIENCE \
    --mbe_min_delta 0.01 \
    --min_a 1e-5 \
    --patch_size 8 \
    --run_info "IBLM (soft-add GAPT, MBE_PATIENCE=$MBE_PATIENCE, ENTROPY_PATIENCE=$MBE_PATIENCE)"
done

# # (a.4) Sweep on entropy min_delta (for soft-add GAPT)
# for ENTROPY_DELTA in 0.02 0.05 0.1; do
#   torchrun \
#     --nproc_per_node=$N_GPUS \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$((MASTER_PORT++)) \
#     train_iblm.py \
#     --batch_size $BATCH_SIZE \
#     --train_seq_len $TRAIN_SEQ_LEN \
#     --val_seq_len $VAL_SEQ_LEN \
#     --num_iterations $NUM_ITERATIONS \
#     --use_gapt \
#     --entropy_patience 250 \
#     --entropy_min_delta $ENTROPY_DELTA \
#     --mbe_patience 50 \
#     --mbe_min_delta 0.01 \
#     --min_a 1e-5 \
#     --patch_size 8 \
#     --run_info "IBLM (soft-add GAPT, ENTROPY_DELTA=$ENTROPY_DELTA)"
# done



# Large scale model sweep (10B fineweb)
# Issue #1. MBE collapse to nan value for 'large' GPT model training
#           better set a lower-bound on mbe values during optimization
# Fix #1.   Added MBE clamping gadget to avoid 'over-optimization' of mbe loss
#           it's worth sweeping through different 'minMBE' effect on things

# Scaling experiment (10B fineweb dataset)
# -----------------------------------------
# NUM_ITERATIONS=8000
# for MODEL_SIZE in "small" "medium" "large" "xl"; do
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
#       --mbe_min_delta 0.01 \
#       --patch_size 8 \
#       --model_size $MODEL_SIZE \
#       --min_a 1e-5 \
#       --save_checkpoint \
#       --run_info "GAPT Sweep: ModelSize=$MODEL_SIZE | CEPat=250 | MBEPat=50 | ClampMinMBE=1e-5" 
# done

# NUM_ITERATIONS=8000
# # Baseline experiment (10B fineweb dataset)
# # -----------------------------------------
# for MODEL_SIZE in "small" "medium" "large"; do
#   torchrun \
#     --nproc_per_node=$N_GPUS \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$((MASTER_PORT++)) \
#     train_iblm.py \
#     --batch_size $BATCH_SIZE \
#     --train_seq_len $TRAIN_SEQ_LEN \
#     --val_seq_len $VAL_SEQ_LEN \
#     --num_iterations $NUM_ITERATIONS \
#     --patch_size 8 \
#     --no_reg \
#     --model_size $MODEL_SIZE \
#     --save_checkpoint \
#     --run_info "Baseline: ModelSize=$MODEL_SIZE" 
# done 




# Experiment (I). The best balance in 'patience' for entropy & mbe
# entropy_patience 250 
# mbe_patience 50 
# is a good balance, we get to 3.28 (0.02 better than baseline)

# Experiment (II). Patch size sweep & entropy min_delta, mbe_min_delta sweep
