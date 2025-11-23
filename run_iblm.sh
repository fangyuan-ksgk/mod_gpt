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
N_GPUS=3
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500

# ============================================================================
# BASELINE EXPERIMENTS
# ============================================================================

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
#   --entropy_patience 125 \
#   --entropy_min_delta 0.01 \
#   --mbe_patience 125 \
#   --mbe_min_delta 0.01 \
#   --patch_size 8


# ============================================================================
# GAPT Patience Sweep Experiments
# ============================================================================

# Baseline: No GAPT (for comparison)
echo "========================================="
echo "Baseline: Training WITHOUT GAPT"
echo "========================================="
torchrun \
  --nproc_per_node=$N_GPUS \
  --master_addr=$MASTER_ADDR \
  --master_port=$((MASTER_PORT++)) \
  train_iblm.py \
  --batch_size $BATCH_SIZE \
  --train_seq_len $TRAIN_SEQ_LEN \
  --val_seq_len $VAL_SEQ_LEN \
  --num_iterations $NUM_ITERATIONS \
  --patch_size 8

# =================================================================================
# Part 1: Symmetric Patience (entropy_patience = mbe_patience)
# =================================================================================

# Exp 1.1: Very Impatient (fast phase switching)
echo "========================================="
echo "Exp 1.1: GAPT - Very Impatient (patience=25)"
echo "========================================="
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
  --entropy_patience 25 \
  --entropy_min_delta 0.01 \
  --mbe_patience 25 \
  --mbe_min_delta 0.01 \
  --patch_size 8

# Exp 1.2: Impatient
echo "========================================="
echo "Exp 1.2: GAPT - Impatient (patience=50)"
echo "========================================="
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
  --entropy_patience 50 \
  --entropy_min_delta 0.01 \
  --mbe_patience 50 \
  --mbe_min_delta 0.01 \
  --patch_size 8

# Exp 1.3: Moderate (original setting)
echo "========================================="
echo "Exp 1.3: GAPT - Moderate (patience=125) ⭐ BASELINE"
echo "========================================="
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
  --mbe_patience 125 \
  --mbe_min_delta 0.01 \
  --patch_size 8

# Exp 1.4: Patient
echo "========================================="
echo "Exp 1.4: GAPT - Patient (patience=250)"
echo "========================================="
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
  --mbe_patience 250 \
  --mbe_min_delta 0.01 \
  --patch_size 8

# Exp 1.5: Very Patient (slow phase switching)
echo "========================================="
echo "Exp 1.5: GAPT - Very Patient (patience=500)"
echo "========================================="
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
  --entropy_patience 500 \
  --entropy_min_delta 0.01 \
  --mbe_patience 500 \
  --mbe_min_delta 0.01 \
  --patch_size 8

# =================================================================================
# Part 2: Asymmetric Patience - Entropy-Focused
# (More patient with entropy, less patient with MBE)
# =================================================================================

# Exp 2.1: Entropy-focused (high entropy patience)
echo "========================================="
echo "Exp 2.1: GAPT - Entropy-focused (E=250, M=50)"
echo "========================================="
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
  --patch_size 8

# Exp 2.2: Strong entropy-focused
echo "========================================="
echo "Exp 2.2: GAPT - Strong Entropy-focused (E=500, M=50)"
echo "========================================="
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
  --entropy_patience 500 \
  --entropy_min_delta 0.01 \
  --mbe_patience 50 \
  --mbe_min_delta 0.01 \
  --patch_size 8

# Exp 2.3: Moderate entropy-focused
echo "========================================="
echo "Exp 2.3: GAPT - Moderate Entropy-focused (E=250, M=125)"
echo "========================================="
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
  --mbe_patience 125 \
  --mbe_min_delta 0.01 \
  --patch_size 8

# =================================================================================
# Part 3: Asymmetric Patience - MBE-Focused
# (Less patient with entropy, more patient with MBE)
# =================================================================================

# Exp 3.1: MBE-focused
echo "========================================="
echo "Exp 3.1: GAPT - MBE-focused (E=50, M=250)"
echo "========================================="
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
  --entropy_patience 50 \
  --entropy_min_delta 0.01 \
  --mbe_patience 250 \
  --mbe_min_delta 0.01 \
  --patch_size 8

# Exp 3.2: Strong MBE-focused
echo "========================================="
echo "Exp 3.2: GAPT - Strong MBE-focused (E=50, M=500)"
echo "========================================="
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
  --entropy_patience 50 \
  --entropy_min_delta 0.01 \
  --mbe_patience 500 \
  --mbe_min_delta 0.01 \
  --patch_size 8

# Exp 3.3: Moderate MBE-focused
echo "========================================="
echo "Exp 3.3: GAPT - Moderate MBE-focused (E=125, M=250)"
echo "========================================="
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
  --mbe_patience 250 \
  --mbe_min_delta 0.01 \
  --patch_size 8

# =================================================================================
# Part 4: Extreme Asymmetric (test edge cases)
# =================================================================================

# Exp 4.1: Extreme entropy priority
echo "========================================="
echo "Exp 4.1: GAPT - EXTREME Entropy priority (E=750, M=25)"
echo "========================================="
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
  --entropy_patience 750 \
  --entropy_min_delta 0.01 \
  --mbe_patience 25 \
  --mbe_min_delta 0.01 \
  --patch_size 8

# Exp 4.2: Extreme MBE priority
echo "========================================="
echo "Exp 4.2: GAPT - EXTREME MBE priority (E=25, M=750)"
echo "========================================="
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
  --entropy_patience 25 \
  --entropy_min_delta 0.01 \
  --mbe_patience 750 \
  --mbe_min_delta 0.01 \
  --patch_size 8

echo "========================================="
echo "All GAPT experiments complete!"
echo "========================================="
