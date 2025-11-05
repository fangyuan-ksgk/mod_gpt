# Experiment Script for SoRL


# ============================================================================
# Configuration
# ============================================================================
BATCH_SIZE=30  # Closer to benchmark batch_size=32
TRAIN_SEQ_LEN=$((16 * 1024))
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

# Second round of experiments || we shall use batch_size=30 since we got all night, this allows closer to benchmark batch size (32) too. 

# - previous discoveries: max_iterations = 2 > max iterations = 1 | traj perplexity of SoRL lags behind baseline
# - hypothesis: 
#   (1). GAPT + SoRL > SoRL in terms of traj perplexity 
#   (2). SoRL v2 (use_greedy_retention=False) > SoRL (use_greedy_retention=True) in terms of traj perplexity
#   (3). max_iterations = 2 > max iterations = 1 in terms of traj perplexity, but max_iterations > 2 has moderate benefits
#   (4). use_curiosity_reward=True > use_curiosity_reward=False in terms of abstract vocab utilization rate
#   (5). high temperature (10.0) > low temperature (5.0, 1.0) in terms of abstract vocab utilization rate
#   (6). no memory compression (use_static_memory_span=True) > memory_compression in terms of traj perplexity, not clear its effect on abstract vocab util rate
#   (7). larger minimal memory span (say 128, 256) > smaller minimal memory span (say 64) in terms of traj perplexity
#   (8). num_rollouts = 3 > num_rollouts = 2 in terms of traj perplexity (last time we tried 4 which exceeds memory budget)
#   (9). a bigger K (say 16, 32) might be better than K=8, a smaller K (say 2, 4) is also better than K=8, this is ambiguous for now. 


# ============================================================================
# HYPOTHESIS 1: GAPT improves traj perplexity
# ============================================================================
echo "========================================="
echo "H1: GAPT + SoRL vs Baseline SoRL"
echo "========================================="


# H1a: Baseline SoRL (no GAPT)
echo "Running: SoRL baseline (no GAPT)..."
torchrun --standalone --nproc_per_node=$N_GPUS train_sorl.py \
  --batch_size $BATCH_SIZE \
  --train_seq_len $TRAIN_SEQ_LEN \
  --val_seq_len $VAL_SEQ_LEN \
  --num_iterations $NUM_ITERATIONS \
  --num_rollouts 2 \
  --K 8 \
  --max_iterations 2 \
  --temperature 10.0


# H1b: SoRL + GAPT : This one produces error, halting the entire script.

torchrun --standalone --nproc_per_node=$N_GPUS train_base.py \
  --batch_size $BATCH_SIZE \
  --train_seq_len $TRAIN_SEQ_LEN \
  --val_seq_len $VAL_SEQ_LEN \
  --num_iterations $NUM_ITERATIONS
  
echo "Running: SoRL + GAPT..."
torchrun --standalone --nproc_per_node=$N_GPUS train_sorl.py \
  --batch_size $BATCH_SIZE \
  --train_seq_len $TRAIN_SEQ_LEN \
  --val_seq_len $VAL_SEQ_LEN \
  --num_iterations $NUM_ITERATIONS \
  --num_rollouts 2 \
  --K 8 \
  --max_iterations 2 \
  --temperature 10.0 \
  --use_gapt

# ============================================================================
# HYPOTHESIS 2: SoRL v2 (no greedy) > SoRL (greedy retention)
# ============================================================================
echo "========================================="
echo "H2: Greedy Retention Impact"
echo "========================================="

# H2a: SoRL v2 (no greedy retention) - already ran in H1a
echo "SoRL v2 result from H1a"

# H2b: SoRL with greedy retention
echo "Running: SoRL with greedy retention..."
torchrun --standalone --nproc_per_node=$N_GPUS train_sorl.py \
  --batch_size $BATCH_SIZE \
  --train_seq_len $TRAIN_SEQ_LEN \
  --val_seq_len $VAL_SEQ_LEN \
  --num_iterations $NUM_ITERATIONS \
  --num_rollouts 2 \
  --K 8 \
  --max_iterations 2 \
  --temperature 10.0 \
  --use_greedy_retention

# ============================================================================
# HYPOTHESIS 3: max_iterations sweep (1 vs 2 vs 3)
# ============================================================================
echo "========================================="
echo "H3: Max Iterations Sweep"
echo "========================================="

# H3a: max_iterations=1
echo "Running: max_iterations=1..."
torchrun --standalone --nproc_per_node=$N_GPUS train_sorl.py \
  --batch_size $BATCH_SIZE \
  --train_seq_len $TRAIN_SEQ_LEN \
  --val_seq_len $VAL_SEQ_LEN \
  --num_iterations $NUM_ITERATIONS \
  --num_rollouts 2 \
  --K 8 \
  --max_iterations 1 \
  --temperature 10.0

# H3b: max_iterations=2 (already ran in H1a)
echo "max_iterations=2 result from H1a"

# H3c: max_iterations=3
echo "Running: max_iterations=3..."
torchrun --standalone --nproc_per_node=$N_GPUS train_sorl.py \
  --batch_size $BATCH_SIZE \
  --train_seq_len $TRAIN_SEQ_LEN \
  --val_seq_len $VAL_SEQ_LEN \
  --num_iterations $NUM_ITERATIONS \
  --num_rollouts 2 \
  --K 8 \
  --max_iterations 3 \
  --temperature 10.0

# ============================================================================
# HYPOTHESIS 4 & 5: Curiosity reward + Temperature (vocab utilization)
# ============================================================================
echo "========================================="
echo "H4 & H5: Curiosity Reward + Temperature"
echo "========================================="

# H4a: Curiosity reward + temp=10.0
echo "Running: Curiosity reward + temp=10.0..."
torchrun --standalone --nproc_per_node=$N_GPUS train_sorl.py \
  --batch_size $BATCH_SIZE \
  --train_seq_len $TRAIN_SEQ_LEN \
  --val_seq_len $VAL_SEQ_LEN \
  --num_iterations $NUM_ITERATIONS \
  --num_rollouts 2 \
  --K 8 \
  --max_iterations 2 \
  --temperature 10.0 \
  --use_curiosity_reward

# H5a: No curiosity + temp=5.0
echo "Running: No curiosity + temp=5.0..."
torchrun --standalone --nproc_per_node=$N_GPUS train_sorl.py \
  --batch_size $BATCH_SIZE \
  --train_seq_len $TRAIN_SEQ_LEN \
  --val_seq_len $VAL_SEQ_LEN \
  --num_iterations $NUM_ITERATIONS \
  --num_rollouts 2 \
  --K 8 \
  --max_iterations 2 \
  --temperature 5.0

# H5b: Curiosity + temp=5.0
echo "Running: Curiosity + temp=5.0..."
torchrun --standalone --nproc_per_node=$N_GPUS train_sorl.py \
  --batch_size $BATCH_SIZE \
  --train_seq_len $TRAIN_SEQ_LEN \
  --val_seq_len $VAL_SEQ_LEN \
  --num_iterations $NUM_ITERATIONS \
  --num_rollouts 2 \
  --K 8 \
  --max_iterations 2 \
  --temperature 5.0 \
  --use_curiosity_reward

# ============================================================================
# HYPOTHESIS 6 & 7: Memory span configurations
# ============================================================================
echo "========================================="
echo "H6 & H7: Memory Span Impact"
echo "========================================="

# H6: Static memory span (no compression)
echo "Running: Static memory span..."
torchrun --standalone --nproc_per_node=$N_GPUS train_sorl.py \
  --batch_size $BATCH_SIZE \
  --train_seq_len $TRAIN_SEQ_LEN \
  --val_seq_len $VAL_SEQ_LEN \
  --num_iterations $NUM_ITERATIONS \
  --num_rollouts 2 \
  --K 8 \
  --max_iterations 2 \
  --temperature 10.0 \
  --use_static_memory_span

# H7a: Larger min_memory_span=128
echo "Running: min_memory_span=128..."
torchrun --standalone --nproc_per_node=$N_GPUS train_sorl.py \
  --batch_size $BATCH_SIZE \
  --train_seq_len $TRAIN_SEQ_LEN \
  --val_seq_len $VAL_SEQ_LEN \
  --num_iterations $NUM_ITERATIONS \
  --num_rollouts 2 \
  --K 8 \
  --max_iterations 2 \
  --temperature 10.0 \
  --min_memory_span 128

# H7b: Larger min_memory_span=256
echo "Running: min_memory_span=256..."
torchrun --standalone --nproc_per_node=$N_GPUS train_sorl.py \
  --batch_size $BATCH_SIZE \
  --train_seq_len $TRAIN_SEQ_LEN \
  --val_seq_len $VAL_SEQ_LEN \
  --num_iterations $NUM_ITERATIONS \
  --num_rollouts 2 \
  --K 8 \
  --max_iterations 2 \
  --temperature 10.0 \
  --min_memory_span 256

# ============================================================================
# HYPOTHESIS 8: num_rollouts impact
# ============================================================================
echo "========================================="
echo "H8: Number of Rollouts"
echo "========================================="

# H8a: num_rollouts=2 (already ran in H1a)
echo "num_rollouts=2 result from H1a"

# H8b: num_rollouts=3
echo "Running: num_rollouts=3..."
torchrun --standalone --nproc_per_node=$N_GPUS train_sorl.py \
  --batch_size $BATCH_SIZE \
  --train_seq_len $TRAIN_SEQ_LEN \
  --val_seq_len $VAL_SEQ_LEN \
  --num_iterations $NUM_ITERATIONS \
  --num_rollouts 3 \
  --K 8 \
  --max_iterations 2 \
  --temperature 10.0

# ============================================================================
# HYPOTHESIS 9: K (abstraction ratio) sweep
# ============================================================================
echo "========================================="
echo "H9: K (Abstraction Ratio) Sweep"
echo "========================================="

# H9a: K=2 (high compression)
echo "Running: K=2..."
torchrun --standalone --nproc_per_node=$N_GPUS train_sorl.py \
  --batch_size $BATCH_SIZE \
  --train_seq_len $TRAIN_SEQ_LEN \
  --val_seq_len $VAL_SEQ_LEN \
  --num_iterations $NUM_ITERATIONS \
  --num_rollouts 2 \
  --K 2 \
  --max_iterations 2 \
  --temperature 10.0

# H9b: K=4
echo "Running: K=4..."
torchrun --standalone --nproc_per_node=$N_GPUS train_sorl.py \
  --batch_size $BATCH_SIZE \
  --train_seq_len $TRAIN_SEQ_LEN \
  --val_seq_len $VAL_SEQ_LEN \
  --num_iterations $NUM_ITERATIONS \
  --num_rollouts 2 \
  --K 4 \
  --max_iterations 2 \
  --temperature 10.0

# H9c: K=8 (already ran in H1a)
echo "K=8 result from H1a"

# H9d: K=16 (low compression)
echo "Running: K=16..."
torchrun --standalone --nproc_per_node=$N_GPUS train_sorl.py \
  --batch_size $BATCH_SIZE \
  --train_seq_len $TRAIN_SEQ_LEN \
  --val_seq_len $VAL_SEQ_LEN \
  --num_iterations $NUM_ITERATIONS \
  --num_rollouts 2 \
  --K 16 \
  --max_iterations 2 \
  --temperature 10.0

# ============================================================================
# BEST CONFIGURATION (based on toy experiments)
# ============================================================================
echo "========================================="
echo "BEST: Combined optimal settings"
echo "========================================="

echo "Running: Best config (SoRL v2 + curiosity + GAPT + temp=10)..."
torchrun --standalone --nproc_per_node=$N_GPUS train_sorl.py \
  --batch_size $BATCH_SIZE \
  --train_seq_len $TRAIN_SEQ_LEN \
  --val_seq_len $VAL_SEQ_LEN \
  --num_iterations $NUM_ITERATIONS \
  --num_rollouts 2 \
  --K 8 \
  --max_iterations 2 \
  --temperature 10.0 \
  --use_curiosity_reward \
  --use_gapt \
  --min_memory_span 128

echo "========================================="
echo "All experiments complete!"
echo "========================================="