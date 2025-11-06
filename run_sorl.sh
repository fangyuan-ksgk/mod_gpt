# Experiment Script for SoRL

# --- nvidia pod specifics ------
# export NCCL_NET_PLUGIN=""
# export NCCL_SOCKET_IFNAME=lo
# export NCCL_IB_DISABLE=1
# export NCCL_DEBUG=WARN

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

# echo "========================================="
# echo "BASELINE: No Abstraction (Standard GPT)"
# echo "========================================="

# torchrun --standalone --nproc_per_node=$N_GPUS train_base.py \
#   --batch_size $BATCH_SIZE \
#   --train_seq_len $TRAIN_SEQ_LEN \
#   --val_seq_len $VAL_SEQ_LEN \
#   --num_iterations $NUM_ITERATIONS

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


# (TBD). included 'run_info' argument so that it's easy to see what's going on with the runs 

# ============================================================================
# HYPOTHESIS 1: GAPT improves traj perplexity
# ============================================================================
echo "========================================="
echo "H1: GAPT + SoRL vs Baseline SoRL"
echo "========================================="

# H1a: Baseline SoRL (no GAPT, no memory compression)
echo "Running: SoRL baseline (no GAPT | no Memory compression | no greedy retention)..."
torchrun --standalone --nproc_per_node=$N_GPUS train_sorl.py \
  --run_info "SoRL baseline (no GAPT | no Memory compression | no greedy retention)" \
  --batch_size $BATCH_SIZE \
  --train_seq_len $TRAIN_SEQ_LEN \
  --val_seq_len $VAL_SEQ_LEN \
  --num_iterations $NUM_ITERATIONS \
  --num_rollouts 2 \
  --K 8 \
  --max_iterations 2 \
  --temperature 10.0 \
  --use_static_memory_span

echo "Running: SoRL baseline (no GAPT | no Memory compression | greedy retention)..."
torchrun --standalone --nproc_per_node=$N_GPUS train_sorl.py \
  --run_info "SoRL baseline (no GAPT | no Memory compression | greedy retention)" \
  --batch_size $BATCH_SIZE \
  --train_seq_len $TRAIN_SEQ_LEN \
  --val_seq_len $VAL_SEQ_LEN \
  --num_iterations $NUM_ITERATIONS \
  --num_rollouts 2 \
  --K 8 \
  --max_iterations 2 \
  --temperature 10.0 \
  --use_static_memory_span \
  --use_greedy_retention


# H1b: SoRL + GAPT
# -------------------------------------------------------------

echo "Running: SoRL + GAPT (no Memory compression | no greedy retention)..."
torchrun --standalone --nproc_per_node=$N_GPUS train_sorl.py \
  --run_info "SoRL + GAPT (no Memory compression | no greedy retention)" \
  --batch_size $BATCH_SIZE \
  --train_seq_len $TRAIN_SEQ_LEN \
  --val_seq_len $VAL_SEQ_LEN \
  --num_iterations $NUM_ITERATIONS \
  --num_rollouts 2 \
  --K 8 \
  --max_iterations 2 \
  --temperature 10.0 \
  --use_static_memory_span \
  --use_gapt

echo "Running: SoRL + GAPT (no Memory compression | greedy retention)..."
torchrun --standalone --nproc_per_node=$N_GPUS train_sorl.py \
  --run_info "SoRL + GAPT (no Memory compression | greedy retention)" \
  --batch_size $BATCH_SIZE \
  --train_seq_len $TRAIN_SEQ_LEN \
  --val_seq_len $VAL_SEQ_LEN \
  --num_iterations $NUM_ITERATIONS \
  --num_rollouts 2 \
  --K 8 \
  --max_iterations 2 \
  --temperature 10.0 \
  --use_static_memory_span \
  --use_greedy_retention \
  --use_gapt


# Patience in GAPT sweep
# -------------------------------------------------------------
for patience in 50 100 150 200; do
  echo "Running: GAPT patience sweep | patience=${patience} | no greedy retention"
  torchrun --standalone --nproc_per_node=$N_GPUS train_sorl.py \
    --run_info "GAPT patience sweep | patience=${patience} | no greedy retention" \
    --batch_size $BATCH_SIZE \
    --train_seq_len $TRAIN_SEQ_LEN \
    --val_seq_len $VAL_SEQ_LEN \
    --num_iterations $NUM_ITERATIONS \
    --num_rollouts 2 \
    --K 8 \
    --max_iterations 2 \
    --temperature 10.0 \
    --use_static_memory_span \
    --use_gapt \
    --traj_perplexity_patience $patience

  echo "Running: GAPT patience sweep | patience=${patience} | greedy retention"
  torchrun --standalone --nproc_per_node=$N_GPUS train_sorl.py \
    --run_info "GAPT patience sweep | patience=${patience} | greedy retention" \
    --batch_size $BATCH_SIZE \
    --train_seq_len $TRAIN_SEQ_LEN \
    --val_seq_len $VAL_SEQ_LEN \
    --num_iterations $NUM_ITERATIONS \
    --num_rollouts 2 \
    --K 8 \
    --max_iterations 2 \
    --temperature 10.0 \
    --use_static_memory_span \
    --use_greedy_retention \
    --use_gapt \
    --traj_perplexity_patience $patience
done

# Curiosity reward ablation
# -------------------------------------------------------------
echo "Running: Curiosity reward | GAPT | no greedy retention"
torchrun --standalone --nproc_per_node=$N_GPUS train_sorl.py \
  --run_info "Curiosity reward | GAPT | no greedy retention" \
  --batch_size $BATCH_SIZE \
  --train_seq_len $TRAIN_SEQ_LEN \
  --val_seq_len $VAL_SEQ_LEN \
  --num_iterations $NUM_ITERATIONS \
  --num_rollouts 2 \
  --K 8 \
  --max_iterations 2 \
  --temperature 10.0 \
  --use_static_memory_span \
  --use_gapt \
  --use_curiosity_reward

echo "Running: Curiosity reward | GAPT | greedy retention"
torchrun --standalone --nproc_per_node=$N_GPUS train_sorl.py \
  --run_info "Curiosity reward | GAPT | greedy retention" \
  --batch_size $BATCH_SIZE \
  --train_seq_len $TRAIN_SEQ_LEN \
  --val_seq_len $VAL_SEQ_LEN \
  --num_iterations $NUM_ITERATIONS \
  --num_rollouts 2 \
  --K 8 \
  --max_iterations 2 \
  --temperature 10.0 \
  --use_static_memory_span \
  --use_greedy_retention \
  --use_gapt \
  --use_curiosity_reward


# Num rollouts sweep 
# -------------------------------------------------------------
for num_rollouts in 3 4 5; do
  echo "Running: Num rollouts sweep | num_rollouts=${num_rollouts} | no greedy retention"
  torchrun --standalone --nproc_per_node=$N_GPUS train_sorl.py \
    --run_info "Num rollouts sweep | num_rollouts=${num_rollouts} | no greedy retention" \
    --batch_size $BATCH_SIZE \
    --train_seq_len $TRAIN_SEQ_LEN \
    --val_seq_len $VAL_SEQ_LEN \
    --num_iterations $NUM_ITERATIONS \
    --num_rollouts $num_rollouts \
    --K 8 \
    --max_iterations 2 \
    --temperature 10.0 \
    --use_static_memory_span \
    --use_gapt \
    --use_curiosity_reward

  echo "Running: Num rollouts sweep | num_rollouts=${num_rollouts} | greedy retention"
  torchrun --standalone --nproc_per_node=$N_GPUS train_sorl.py \
    --run_info "Num rollouts sweep | num_rollouts=${num_rollouts} | greedy retention" \
    --batch_size $BATCH_SIZE \
    --train_seq_len $TRAIN_SEQ_LEN \
    --val_seq_len $VAL_SEQ_LEN \
    --num_iterations $NUM_ITERATIONS \
    --num_rollouts $num_rollouts \
    --K 8 \
    --max_iterations 2 \
    --temperature 10.0 \
    --use_static_memory_span \
    --use_gapt \
    --use_greedy_retention \
    --use_curiosity_reward
done 


# Max iterations sweep
# -------------------------------------------------------------
for max_iterations in 1 3 4 5; do
  echo "Running: Max iterations sweep | max_iterations=${max_iterations} | no greedy retention"
  torchrun --standalone --nproc_per_node=$N_GPUS train_sorl.py \
    --run_info "Max iterations sweep | max_iterations=${max_iterations} | no greedy retention" \
    --batch_size $BATCH_SIZE \
    --train_seq_len $TRAIN_SEQ_LEN \
    --val_seq_len $VAL_SEQ_LEN \
    --num_iterations $NUM_ITERATIONS \
    --num_rollouts 2 \
    --K 8 \
    --max_iterations $max_iterations \
    --temperature 10.0 \
    --use_static_memory_span \
    --use_gapt \
    --use_curiosity_reward

  echo "Running: Max iterations sweep | max_iterations=${max_iterations} | greedy retention"
  torchrun --standalone --nproc_per_node=$N_GPUS train_sorl.py \
    --run_info "Max iterations sweep | max_iterations=${max_iterations} | greedy retention" \
    --batch_size $BATCH_SIZE \
    --train_seq_len $TRAIN_SEQ_LEN \
    --val_seq_len $VAL_SEQ_LEN \
    --num_iterations $NUM_ITERATIONS \
    --num_rollouts 2 \
    --K 8 \
    --max_iterations $max_iterations \
    --temperature 10.0 \
    --use_static_memory_span \
    --use_gapt \
    --use_greedy_retention \
    --use_curiosity_reward
done 

# Here onwards has yet to be run. 
# ==============================================

# Temperature sweep
# -------------------------------------------------------------
for temperature in 1.0 5.0 15.0 20.0; do
  echo "Running: Temperature sweep | temperature=${temperature} | no greedy retention"
  torchrun --standalone --nproc_per_node=$N_GPUS train_sorl.py \
    --run_info "Temperature sweep | temperature=${temperature} | no greedy retention" \
    --batch_size $BATCH_SIZE \
    --train_seq_len $TRAIN_SEQ_LEN \
    --val_seq_len $VAL_SEQ_LEN \
    --num_iterations $NUM_ITERATIONS \
    --num_rollouts 2 \
    --K 8 \
    --max_iterations 2 \
    --temperature $temperature \
    --use_static_memory_span \
    --use_gapt \
    --use_curiosity_reward

  echo "Running: Temperature sweep | temperature=${temperature} | greedy retention"
  torchrun --standalone --nproc_per_node=$N_GPUS train_sorl.py \
    --run_info "Temperature sweep | temperature=${temperature} | greedy retention" \
    --batch_size $BATCH_SIZE \
    --train_seq_len $TRAIN_SEQ_LEN \
    --val_seq_len $VAL_SEQ_LEN \
    --num_iterations $NUM_ITERATIONS \
    --num_rollouts 2 \
    --K 8 \
    --max_iterations 2 \
    --temperature $temperature \
    --use_static_memory_span \
    --use_gapt \
    --use_greedy_retention \
    --use_curiosity_reward
done 

# Memory compression sweep (enables memory compression by NOT using --use_static_memory_span)
# -------------------------------------------------------------
for min_memory_span in 1280 1024 512 256; do
  echo "Running: Memory compression sweep | min_memory_span=${min_memory_span} | no greedy retention"
  torchrun --standalone --nproc_per_node=$N_GPUS train_sorl.py \
    --run_info "Memory compression sweep | min_memory_span=${min_memory_span} | no greedy retention" \
    --batch_size $BATCH_SIZE \
    --train_seq_len $TRAIN_SEQ_LEN \
    --val_seq_len $VAL_SEQ_LEN \
    --num_iterations $NUM_ITERATIONS \
    --num_rollouts 2 \
    --K 8 \
    --max_iterations 2 \
    --temperature 10.0 \
    --use_gapt \
    --use_curiosity_reward \
    --min_memory_span $min_memory_span

  echo "Running: Memory compression sweep | min_memory_span=${min_memory_span} | greedy retention"
  torchrun --standalone --nproc_per_node=$N_GPUS train_sorl.py \
    --run_info "Memory compression sweep | min_memory_span=${min_memory_span} | greedy retention" \
    --batch_size $BATCH_SIZE \
    --train_seq_len $TRAIN_SEQ_LEN \
    --val_seq_len $VAL_SEQ_LEN \
    --num_iterations $NUM_ITERATIONS \
    --num_rollouts 2 \
    --K 8 \
    --max_iterations 2 \
    --temperature 10.0 \
    --use_gapt \
    --use_greedy_retention \
    --use_curiosity_reward \
    --min_memory_span $min_memory_span
done 

echo "========================================="
echo "All experiments complete!"
echo "========================================="