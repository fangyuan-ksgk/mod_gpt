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
MAX_ITERATIONS=2
N_GPUS=3
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500
ALPHA_LOSS=0.1
EXPLORATION_MODE=0 # SGPO - exploration


# Try the 'all-rollout SoRL' again
# I'd like to ablate on some more variants of its adv formulation
# - (1). adv \propto 1 / p(a | s)
# - (2). adv \propto 
# - p(a | s) / p(s | a) --> favor more familiar abstraction at same utility, favor less utility abstractino at same familiarity
# - can we somehow add the topological smimilarity term into the adv formulation to 'regulate' the preference for horrble abstraction? 


# echo "========================================="
# echo "BASELINE: No Abstraction (Standard GPT)"
# echo "========================================="

# torchrun --standalone --nproc_per_node=$N_GPUS train_base.py \
#   --batch_size $BATCH_SIZE \
#   --train_seq_len $TRAIN_SEQ_LEN \
#   --val_seq_len $VAL_SEQ_LEN \
#   --num_iterations $NUM_ITERATIONS


# --------------------------------------------------------------
# Exp 10: curiosity advantage SoRL (with all rollouts)
# --------------------------------------------------------------

# echo "========================================="
# Exp 11.1: SGPO --> offline distillation (gapt -- only in exploitation phase)
# echo "========================================="

# for TRAJ_PATIENCE in {100..500..100}; do
#   torchrun \
#     --nproc_per_node=$N_GPUS \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$MASTER_PORT \
#     train_sorl_v2.py \
#     --batch_size $BATCH_SIZE \
#     --train_seq_len $TRAIN_SEQ_LEN \
#     --val_seq_len $VAL_SEQ_LEN \
#     --num_iterations $NUM_ITERATIONS \
#     --num_rollouts $NUM_ROLLOUTS \
#     --K 8 \
#     --max_iterations $MAX_ITERATIONS \
#     --use_static_memory_span \
#     --min_temperature 0.0 \
#     --temperature 5.0 \
#     --alpha_loss $ALPHA_LOSS \
#     --mode 0 \
#     --steps_per_cycle $NUM_ITERATIONS \
#     --exploration_fraction 0.5 \
#     --exploration_till_vocab_util 0.5 \
#     --use_off_policy_distillation \
#     --use_gapt \
#     --traj_perplexity_patience $TRAJ_PATIENCE \
#     --do_reinit \
#     --reinit_mode 0 \
#     --run_info "Exp11.1: SGPO -> offline distillation (gapt -- traj perplexity patience=$TRAJ_PATIENCE)"
# done

# # Exp 11.2: Distillation with "active supression" on non-greedy rollouts
# # echo "========================================="
# # echo "Exp 11.2: Distillation with 'active supression' on non-greedy rollouts"
# # echo "========================================="






# # Exp 10.6. SGPO (with topo reg, correlation) --> off-policy distillation
# # echo "========================================="
# # echo "Exp 10.6. SGPO (with topo reg, correlation) --> off-policy distillation"
# # echo "========================================="
# ALPHA_TOPOS=(0.5 1.0 1.5 2.0)

# for ALPHA_TOPO in "${ALPHA_TOPOS[@]}"; do
#   torchrun \
#     --nproc_per_node=$N_GPUS \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$MASTER_PORT \
#     train_sorl_v3.py \
#     --batch_size $BATCH_SIZE \
#     --train_seq_len $TRAIN_SEQ_LEN \
#     --val_seq_len $VAL_SEQ_LEN \
#     --num_iterations $NUM_ITERATIONS \
#     --num_rollouts $NUM_ROLLOUTS \
#     --K 8 \
#     --max_iterations $MAX_ITERATIONS \
#     --use_static_memory_span \
#     --min_temperature 0.0 \
#     --temperature 5.0 \
#     --alpha_loss $ALPHA_LOSS \
#     --mode 0 \
#   --steps_per_cycle $NUM_ITERATIONS \
#   --exploration_fraction 0.5 \
#   --exploration_till_vocab_util 0.5\
#   --use_off_policy_distillation \
#   --do_reinit \
#   --reinit_mode 0 \
#   --topo_mode 1 \
#     --alpha_topo $ALPHA_TOPO \
#     --run_info "Exp10.6: SGPO -> off-policy distillation (topo reg alpha=$ALPHA_TOPO)"
# done

# Observation #1. 
# Combining GAPT with SGPO -> Exploitation pipeline yields no benefit, we are better off training for longer. (3.26 is possible with longer exploit phase)

# Exp 12.1 || Longer exploitation phase (SGPO -> Offline distillation)
# Fix MAX_ITERATIONS as a shell variable, not Python assignment
MAX_ITERATIONS=3500

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
  --K 8 \
  --max_iterations $MAX_ITERATIONS \
  --use_static_memory_span \
  --min_temperature 0.0 \
  --temperature 5.0 \
  --alpha_loss $ALPHA_LOSS \
  --mode 0 \
  --steps_per_cycle $NUM_ITERATIONS \
  --exploration_fraction 0.5 \
  --exploration_till_vocab_util 0.5 \
  --use_off_policy_distillation \
  --do_reinit \
  --reinit_mode 0 \
  --run_info "Exp11.1: SGPO -> offline distillation (2x compute)"

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
  --K 8 \
  --max_iterations $MAX_ITERATIONS \
  --use_static_memory_span \
  --min_temperature 0.0 \
  --temperature 5.0 \
  --alpha_loss $ALPHA_LOSS \
  --mode 0 \
  --steps_per_cycle $NUM_ITERATIONS \
  --exploration_fraction 0.5 \
  --exploration_till_vocab_util 0.5 \
  --use_off_policy_distillation \
  --do_reinit \
  --reinit_mode 4 \
  --run_info "Exp11.1: SGPO -> offline distillation (all parameter re-initialized, 2x compute, justified as we start from scratch again)"


# Question #1. 
# - It's not clear whether on language modeling task, select-one SoRL with [0.5, 5.0] temperature leads to collapsed vocabulary, when 
#   we adopt GAPT to optimize the traj_perplexity. The explore->exploit pipeline reaches a 3.34 traj loss at the end, this means we underperform
#   baseline by 0.04 here. Plus, the explore -> exploit pipeline produces only 1 - 2% search adv at the moment. If we repeat the explore->exploit
#   loop for longer, we can reach a similar traj loss (so exploit phase needs to be longer), let's run the prior experiment to verify that 'greedy'
#   vocabulary collapse with select-one SoRL. 

MAX_ITERATIONS=1750
torchrun \
  --nproc_per_node=$N_GPUS \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train_sorl.py \
  --batch_size $BATCH_SIZE \
  --train_seq_len $TRAIN_SEQ_LEN \
  --val_seq_len $VAL_SEQ_LEN \
  --num_iterations $NUM_ITERATIONS \
  --num_rollouts $NUM_ROLLOUTS \
  --K 8 \
  --max_iterations $MAX_ITERATIONS \
  --use_static_memory_span \
  --min_temperature 0.5 \
  --temperature 5.0 \
  --alpha_loss $ALPHA_LOSS \
  --use_gapt \
  --traj_perplexity_patience 50 \
  --run_info "Exp9.1: select-one SoRL (t=[0.5, 5.0]) + GAPT produces non-collapsed vocabulary with minimal search adv?"

# Question #2. 
# - Does EMA based distillation produce better 'search advantage'? 
