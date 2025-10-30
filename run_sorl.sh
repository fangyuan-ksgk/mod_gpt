# Experiment Script for SoRL

# Common settings
BATCH_SIZE=15
TRAIN_SEQ_LEN=$((16 * 1024))  # 16K tokens (fits in 48GB)
VAL_SEQ_LEN=$((16 * 1024))
NUM_ITERATIONS=1750
N_GPUS=3

# ============================================================================
# BASELINE EXPERIMENTS
# ============================================================================

echo "========================================="
echo "BASELINE: No Abstraction (Standard GPT)"
echo "========================================="

torchrun --standalone --nproc_per_node=$N_GPUS train_base.py \
  --batch_size $BATCH_SIZE \
  --train_seq_len $TRAIN_SEQ_LEN \
  --val_seq_len $VAL_SEQ_LEN \
  --num_iterations $NUM_ITERATIONS


# ============================================================================
# HYPOTHESIS 1: Placeholder tokens help learning
# ============================================================================

echo "========================================="
echo "EXP 1a: No-Search SoRL (n=1, max_iterations=0)"
echo "Keeps placeholder tokens at abstract positions"
echo "========================================="
torchrun --standalone --nproc_per_node=$N_GPUS train_sorl.py \
  --batch_size $BATCH_SIZE \
  --train_seq_len $TRAIN_SEQ_LEN \
  --val_seq_len $VAL_SEQ_LEN \
  --num_iterations $NUM_ITERATIONS \
  --num_rollouts 1 \
  --K 8 \
  --max_iterations 0 \
  --temperature 0.0

echo "========================================="
echo "EXP 1b: Full SoRL with Search (baseline)"
echo "========================================="
torchrun --standalone --nproc_per_node=$N_GPUS train_sorl.py \
  --batch_size $BATCH_SIZE \
  --train_seq_len $TRAIN_SEQ_LEN \
  --val_seq_len $VAL_SEQ_LEN \
  --num_iterations $NUM_ITERATIONS \
  --num_rollouts 2 \
  --K 8 \
  --max_iterations 1 \
  --temperature 1.0

# ============================================================================
# HYPOTHESIS 2: Memory compression effect
# ============================================================================

echo "========================================="
echo "EXP 2a: No Memory Compression (static span=full sequence)"
echo "Tests if memory compression is necessary"
echo "========================================="
torchrun --standalone --nproc_per_node=$N_GPUS train_sorl.py \
  --batch_size $BATCH_SIZE \
  --train_seq_len $TRAIN_SEQ_LEN \
  --val_seq_len $VAL_SEQ_LEN \
  --num_iterations $NUM_ITERATIONS \
  --num_rollouts 2 \
  --K 8 \
  --max_iterations 1 \
  --temperature 1.0 \
  --use_static_memory_span  # No curriculum, fixed at 1792

echo "========================================="
echo "EXP 2b: With Memory Compression Curriculum"
echo "Default behavior: 1792 -> 64 over training"
echo "========================================="
torchrun --standalone --nproc_per_node=$N_GPUS train_sorl.py \
  --batch_size $BATCH_SIZE \
  --train_seq_len $TRAIN_SEQ_LEN \
  --val_seq_len $VAL_SEQ_LEN \
  --num_iterations $NUM_ITERATIONS \
  --num_rollouts 2 \
  --K 8 \
  --max_iterations 1 \
  --temperature 1.0
  # No --use_static_memory_span flag = uses curriculum

# ============================================================================
# HYPOTHESIS 3: Abstraction vocabulary size
# ============================================================================

echo "========================================="
echo "EXP 3: Ablate Abstraction Vocabulary Size"
echo "========================================="
for VOCAB in 64 256 512; do
  echo "Testing abstract_vocab_size=$VOCAB"
  torchrun --standalone --nproc_per_node=$N_GPUS train_sorl.py \
    --batch_size $BATCH_SIZE \
    --train_seq_len $TRAIN_SEQ_LEN \
    --val_seq_len $VAL_SEQ_LEN \
    --num_iterations $NUM_ITERATIONS \
    --num_rollouts 2 \
    --K 8 \
    --max_iterations 1 \
    --temperature 1.0 \
    --abstract_vocab_size $VOCAB
done

# ============================================================================
# HYPOTHESIS 4: Abstraction ratio K
# ============================================================================

echo "========================================="
echo "EXP 4: Ablate Abstraction Interval K"
echo "========================================="
for K in 2 8 32; do
  echo "Testing K=$K (insert abstract token every $K trajectory tokens)"
  torchrun --standalone --nproc_per_node=$N_GPUS train_sorl.py \
    --batch_size $BATCH_SIZE \
    --train_seq_len $TRAIN_SEQ_LEN \
    --val_seq_len $VAL_SEQ_LEN \
    --num_iterations $NUM_ITERATIONS \
    --num_rollouts 2 \
    --K $K \
    --max_iterations 1 \
    --temperature 1.0
done

# ============================================================================
# HYPOTHESIS 5: Temperature matters
# ============================================================================

echo "========================================="
echo "EXP 5: Ablate Temperature"
echo "========================================="
for TEMP in 0.1 0.5 1.0 2.0 5.0 10.0; do
  echo "Testing temperature=$TEMP"
  torchrun --standalone --nproc_per_node=$N_GPUS train_sorl.py \
    --batch_size $BATCH_SIZE \
    --train_seq_len $TRAIN_SEQ_LEN \
    --val_seq_len $VAL_SEQ_LEN \
    --num_iterations $NUM_ITERATIONS \
    --num_rollouts 2 \
    --K 8 \
    --max_iterations 1 \
    --temperature $TEMP
done

# ============================================================================
# HYPOTHESIS 6: More iterations help
# ============================================================================

echo "========================================="
echo "EXP 6: Ablate Number of Search Iterations"
echo "========================================="
for ITERS in 2 3; do
  echo "Testing max_iterations=$ITERS"
  torchrun --standalone --nproc_per_node=$N_GPUS train_sorl.py \
    --batch_size $BATCH_SIZE \
    --train_seq_len $TRAIN_SEQ_LEN \
    --val_seq_len $VAL_SEQ_LEN \
    --num_iterations $NUM_ITERATIONS \
    --num_rollouts 2 \
    --K 8 \
    --max_iterations $ITERS \
    --temperature 1.0
done

# ============================================================================
# HYPOTHESIS 7: Number of rollouts (n)
# ============================================================================

echo "========================================="
echo "EXP 7: Ablate Number of Rollouts"
echo "========================================="
for N in 2 4; do
  echo "Testing n=$N (1 greedy + $(($N-1)) stochastic)"
  torchrun --standalone --nproc_per_node=$N_GPUS train_sorl.py \
    --batch_size $BATCH_SIZE \
    --train_seq_len $TRAIN_SEQ_LEN \
    --val_seq_len $VAL_SEQ_LEN \
    --num_iterations $NUM_ITERATIONS \
    --num_rollouts $N \
    --K 8 \
    --max_iterations 1 \
    --temperature 1.0
done