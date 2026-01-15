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

# ========================================
# Exp 4. Log run: probability, entropy, mbe, gradient magnitude, per-token loss 
# Exp 5. Log run: 
#        | MBE range regularization (all layer mean MBE) -> Range constrained MBE regularization (0.45 - 0.75)
#        | MBE variance regularization (all layer mean MBE)
#        | GAPT + MBE range regularization (bottlneck layer MBE) | 0.45 - 0.75
#        | GAPT + MBE variance regularization (bottleneck layer MBE)
# Exp 6. Generalization on OOD is maximized when adopting direct MBE regularization. 
#        What if validation fineweb still probes 'overfitting' instead of 'generalization'?
#        Why don't we just over compress and see what happens?
# ========================================

MODEL_SIZE="medium" 
torchrun \
    --nproc_per_node=$N_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$((MASTER_PORT++)) \
    train_iblm.py \
    --batch_size 32 \
    --train_seq_len $TRAIN_SEQ_LEN \
    --val_seq_len $VAL_SEQ_LEN \
    --num_iterations 20000 \
    --reg_mbe \
    --mbe_weight 1.0 \
    --mbe_comp_mode "naive" \
    --mbe_schedule "all_middle" \
    --model_size $MODEL_SIZE \
    --save_checkpoint \
    --run_info "ModelSize=$MODEL_SIZE | reg_mbe | w=1.0 | MBE comp mode: naive" 


# MODEL_SIZE="xl" 
# torchrun \
#     --nproc_per_node=$N_GPUS \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$((MASTER_PORT++)) \
#     train_iblm.py \
#     --batch_size 32 \
#     --train_seq_len $TRAIN_SEQ_LEN \
#     --val_seq_len $VAL_SEQ_LEN \
#     --reg_mbe \
#     --mbe_weight 1.0 \
#     --mbe_comp_mode "naive" \
#     --mbe_schedule "all_middle" \
#     --model_size $MODEL_SIZE \
#     --run_info "ModelSize=$MODEL_SIZE | reg_mbe | w=1.0 | MBE comp mode: naive" 

# ============ Direct MBE Regularization (reg_mbe) ============
# Comparing: mbe_range vs mbe_variance with same settings

# torchrun \
#     --nproc_per_node=$N_GPUS \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$((MASTER_PORT++)) \
#     train_iblm_log.py \
#     --batch_size 32 \
#     --train_seq_len $TRAIN_SEQ_LEN \
#     --val_seq_len $VAL_SEQ_LEN \
#     --num_iterations 10000 \
#     --no_reg \
#     --model_size $MODEL_SIZE \
#     --run_info "Log: ModelSize=$MODEL_SIZE | no reg"

# torchrun \
#     --nproc_per_node=$N_GPUS \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$((MASTER_PORT++)) \
#     train_iblm_log.py \
#     --batch_size 32 \
#     --train_seq_len $TRAIN_SEQ_LEN \
#     --val_seq_len $VAL_SEQ_LEN \
#     --num_iterations 10000 \
#     --use_gapt \
#     --entropy_patience 125 \
#     --entropy_min_delta 0.01 \
#     --mbe_patience 75 \
#     --mbe_min_delta 0.01 \
#     --mbe_weight 20.0 \
#     --mbe_schedule "all_middle" \
#     --mbe_comp_mode "spike" \
#     --model_size $MODEL_SIZE \
#     --use_softplus_gapt \
#     --run_info "Log: ModelSize=$MODEL_SIZE | GAPT | w=20.0 | softplus"

# ========================================="
# Exp 1. FineWeb0.8B (Base, GAPT, MBE, L2)
#        | MBE composition mode sweep: naive, max, softmax
# Exp 2. FineWeb10B (GPT2 small, medium, large, xl)
#        | 'num_iterations' change the 'attention blocksize' at each step (smaller for bigger iterations number)
#        | therefore it's unfair to compare results at same step across run with different 'num_itrations'
# Exp 3. What if skip connection are removed, so that bottleneck is "real" bottleneck that can't be bypassed?

# Exp 4. How does the distribution of token confidence & entropy look like?
# Exp 5. Does EAFT loss help pre-training? 
# ========================================="
# MBE_COMP_MODE="spike"
# for MODEL_SIZE in "small"; do
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
#       --use_softplus_gapt \
#       --reg_mode "mbe_var" \
#       --run_info "GAPT: ModelSize=$MODEL_SIZE | GAPT | w=20.0 | MBE comp mode: $MBE_COMP_MODE" 

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
#       --mbe_schedule "all_middle" \s
#       --model_size $MODEL_SIZE \
#       --use_softplus_gapt \
#       --reg_mode "mbe_var" \
#       --run_info "GAPT: ModelSize=$MODEL_SIZE | GAPT | w=20.0 | MBE comp mode: $MBE_COMP_MODE" 

#   torchrun \
#       --nproc_per_node=$N_GPUS \
#       --master_addr=$MASTER_ADDR \
#       --master_port=$((MASTER_PORT++)) \
#       train_iblm.py \
#       --batch_size 32 \
#       --train_seq_len $TRAIN_SEQ_LEN \
#       --val_seq_len $VAL_SEQ_LEN \
#       --num_iterations 1750 \
#       --no_reg \
#       --model_size $MODEL_SIZE \
#       --run_info "Baseline: ModelSize=$MODEL_SIZE" 
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


# # ========================================="
# # Layer Pruning Experiments 
# # ========================================="
# # (Make sure ckpt is downloaded from huggingface repo: Ksgk-fy/iblm-gpt2-ckpt)
# declare -A IBLM_CKPT=(
#     ["small"]="ckpt/fineweb10B-iblm-gpt2-small-spike.pt"
#     ["medium"]="ckpt/fineweb10B-iblm-gpt2-medium-spike.pt"
#     ["large"]="ckpt/fineweb10B-iblm-gpt2-large-softplus-spike.pt"
#     ["xl"]="ckpt/fineweb10B-iblm-gpt2-xl-spike.pt"
# )
# declare -A IBLM_BLAYER=(
#     ["small"]="5,10"
#     ["medium"]="11,15"
#     ["large"]="3,16,2"
#     ["xl"]="22,23"
# )

# # --- layer pruned continuous pre-training (with GAPT) ---
# MBE_COMP_MODE="spike"
# for MODEL_SIZE in "small" "medium" "large"; do
#   torchrun \
#       --nproc_per_node=$N_GPUS \
#       --master_addr=$MASTER_ADDR \
#       --master_port=$((MASTER_PORT++)) \
#       train_iblm.py \
#       --batch_size 32 \
#       --train_seq_len $TRAIN_SEQ_LEN \
#       --val_seq_len $VAL_SEQ_LEN \
#       --num_iterations 875 \
#       --prune_layers "${IBLM_BLAYER[$MODEL_SIZE]}" \
#       --continue_from_ckpt "${IBLM_CKPT[$MODEL_SIZE]}" \
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
#       --run_info "LayerPruned: ModelSize=$MODEL_SIZE | GAPT | w=20.0 | MBE comp mode: $MBE_COMP_MODE" 
# done

# # --- layer pruned continuous pre-training (without GAPT) ---
# for MODEL_SIZE in "small" "medium" "large"; do
#   torchrun \
#       --nproc_per_node=$N_GPUS \
#       --master_addr=$MASTER_ADDR \
#       --master_port=$((MASTER_PORT++)) \
#       train_iblm.py \
#       --batch_size 32 \
#       --train_seq_len $TRAIN_SEQ_LEN \
#       --val_seq_len $VAL_SEQ_LEN \
#       --num_iterations 875 \
#       --prune_layers "${IBLM_BLAYER[$MODEL_SIZE]}" \
#       --continue_from_ckpt "${IBLM_CKPT[$MODEL_SIZE]}" \
#       --no_reg \
#       --model_size $MODEL_SIZE \
#       --save_checkpoint \
#       --run_info "LayerPruned: ModelSize=$MODEL_SIZE | no GAPT | no reg" 
# done