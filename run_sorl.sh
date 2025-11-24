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
# 


ADV_MODE_DESC=(
    "utility disadvantage (SGPO)"
    "all equal (baseline MLE)"
    "familiarity advantage (distillation)"
    "utility preference (distillation)"
    "curiosity (favor unfamiliar)"
    "curiosity + 0.3 * utility preference"
    "curiosity + 0.1 * utility preference"
    "SGPO + 0.3 * familiarity preference"
    "0.1 * curiosity + 0.9 * utility preference"
    "0.25 * curiosity + 0.75 * utility preference"
    "variance scaling (normalized curiosity, coef=0.5)"
    "conditional curiosity (dampened by alpha=0.3)"
    "clipped curiosity (element-wise, 1x utility)"
    "clipped curiosity + weighting (coef=0.5)"
    "gated curiosity (utility-gated exploration, coef=0.3)"
)

# echo "========================================="
# echo "Exp 10.1: SGPO Advantage Mode Sweep"
# echo "========================================="
# # run 3 to ensure positive search adv is achieved

# fpr ADV_MODE in 3 
# for ADV_MODE in {10..14}; do
#   echo "Running ADV_MODE=${ADV_MODE}: ${ADV_MODE_DESC[ADV_MODE]}"
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
#     --mode $ADV_MODE \
#     --steps_per_cycle $NUM_ITERATIONS \
#     --exploration_fraction 1.0 \
#     --exploration_till_vocab_util 1.0 \
#     --run_info "Exp10.1.${ADV_MODE}: ${ADV_MODE_DESC[ADV_MODE]}"
# done

# echo "========================================="
# echo "Exp 10.2: SGPO Advantage Mode Sweep with Explicit Explore Phase"
# echo "========================================="
# --> These doesn't work well, vocabulary collapsed with 5:1 ratio

# for EXPLORE_EVERY in 2 5; do
#   echo "Running explicit explore phase every ${EXPLORE_EVERY} steps"
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
#     --mode 3 \
#     --steps_per_cycle $NUM_ITERATIONS \
#     --exploration_fraction 1.0 \
#     --exploration_till_vocab_util 1.0 \
#     --use_explicit_explore_phase \
#     --explore_every $EXPLORE_EVERY \
#     --run_info "Exp10.2: explicit explore phase every $EXPLORE_EVERY steps"
# done

# # echo "========================================="
# echo "Exp 10.3: Curiosity SoRL --> offline distillation"
# echo "========================================="

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
  --mode 4 \
  --steps_per_cycle $NUM_ITERATIONS \
  --exploration_fraction 0.5 \
  --exploration_till_vocab_util 0.5 \
  --use_off_policy_distillation \
  --do_reinit \
  --reinit_mode 0 \
  --run_info "Exp10.3: curiosity -> offline distillation"


# Exp 10.4, curiosity SoRL with topo reg (correlation)
# echo "========================================="
# echo "Exp 10.4: Curiosity SoRL with Topo Reg (correlation)"
# echo "========================================="

torchrun \
  --nproc_per_node=$N_GPUS \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train_sorl_v3.py \
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
  --mode 4 \
  --steps_per_cycle $NUM_ITERATIONS \
  --exploration_fraction 1.0 \
  --exploration_till_vocab_util 1.0 \
  --topo_mode 1 \
  --alpha_topo 2.0 \


# Exp 10.5: curiosity SoRL with topo reg (correlation) -> off-policy distillation
# echo "========================================="
# echo "Exp 10.5: curiosity SoRL with topo reg (correlation) -> off-policy distillation"
# echo "========================================="

torchrun \
  --nproc_per_node=$N_GPUS \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train_sorl_v3.py \
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
  --mode 4 \
  --steps_per_cycle $NUM_ITERATIONS \
  --exploration_fraction 0.5 \
  --exploration_till_vocab_util 0.5\
  --use_off_policy_distillation \
  --do_reinit \
  --reinit_mode 0 \
  --topo_mode 1 \
  --alpha_topo 2.0 \
  --run_info "Exp10.5: curiosity -> off-policy distillation"


# Exp 10.6. SGPO (with topo reg, correlation) --> off-policy distillation
# echo "========================================="
# echo "Exp 10.6. SGPO (with topo reg, correlation) --> off-policy distillation"
# echo "========================================="

torchrun \
  --nproc_per_node=$N_GPUS \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train_sorl_v3.py \
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
  --exploration_till_vocab_util 0.5\
  --use_off_policy_distillation \
  --do_reinit \
  --reinit_mode 0 \
  --topo_mode 1 \
  --alpha_topo 2.0 \
  --run_info "Exp10.6: SGPO -> off-policy distillation"