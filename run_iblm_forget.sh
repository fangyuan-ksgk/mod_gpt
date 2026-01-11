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
NUM_ITERATIONS=500 # has to be bigger than 125 for MBE regularization phase to kick in
N_GPUS=2
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500

# ====================================
# Catastrohpic Forgetting Probe
# 1. CPT on OOD dataset:
#    (without prior memories on CPT data)
#    what's the forgetting level like for prior knowledge? 
# 2. CPT on ID dataset: 
#    (with prior memories on CPT data)
#    what's the forgetting level like for prior knowledge? 
# ====================================
NUM_SHARDS=10
# - prepare data
# python data/forget_fineweb.py $NUM_SHARDS

BASE_CKPT="ckpt/fineweb10B-gpt2-small-20k.pt"
GAPT_CKPT="ckpt/fineweb10B-iblm-gpt2-small-spike.pt"

MBE_COMP_MODE="spike"
MODEL_SIZE="small"
FORGET_MODE="fineweb"
MBE_WEIGHT=20.0

# -------------------------------
# (I). Pure CPT on random shards 
# -------------------------------

ALL_SHARDS_FINEWEB="data/fineweb/fineweb_train_*.bin"
torchrun \
  --nproc_per_node=$N_GPUS \
  --master_addr=$MASTER_ADDR \
  --master_port=$((MASTER_PORT++)) \
  train_iblm_forget.py \
  --continue_from_ckpt $BASE_CKPT \
  --train_files "$ALL_SHARDS_FINEWEB" \
  --batch_size $BATCH_SIZE \
  --train_seq_len $TRAIN_SEQ_LEN \
  --val_seq_len $VAL_SEQ_LEN \
  --num_iterations $NUM_ITERATIONS \
  --no_reg \
  --model_size $MODEL_SIZE \
  --save_checkpoint \
  --forget_mode $FORGET_MODE \
  --method_name "base_cpt_from_base" \
  --run_info "CPT on random shards: modelSize=$MODEL_SIZE | base CPT from base Ckpt"

torchrun \
  --nproc_per_node=$N_GPUS \
  --master_addr=$MASTER_ADDR \
  --master_port=$((MASTER_PORT++)) \
  train_iblm_forget.py \
  --continue_from_ckpt $BASE_CKPT \
  --train_files "$ALL_SHARDS_FINEWEB" \
  --batch_size $BATCH_SIZE \
  --train_seq_len $TRAIN_SEQ_LEN \
  --val_seq_len $VAL_SEQ_LEN \
  --num_iterations $NUM_ITERATIONS \
  --use_gapt \
  --entropy_patience 125 \
  --entropy_min_delta 0.01 \
  --mbe_patience 75 \
  --mbe_min_delta 0.01 \
  --mbe_weight $MBE_WEIGHT \
  --mbe_comp_mode $MBE_COMP_MODE \
  --mbe_schedule "all_middle" \
  --model_size $MODEL_SIZE \
  --save_checkpoint \
  --method_name "gapt_from_base" \
  --run_info "CPT w. GAPT on random shards: ModelSize=$MODEL_SIZE | w=$MBE_WEIGHT | $MBE_COMP_MODE | GAPT CPT from base Ckpt"


# ------------------------------
# (II). CPT on specific shards
# ------------------------------

for n in $(seq 0 $((NUM_SHARDS - 1))); do
  FINEWEB_TRAIN_FILES="data/forget_fineweb/bin${n}/fineweb_train_*.bin"
  
  # Method 1: Baseline (no regularization)
  torchrun \
    --nproc_per_node=$N_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$((MASTER_PORT++)) \
    train_iblm_forget.py \
    --continue_from_ckpt $BASE_CKPT \
    --train_files "$FINEWEB_TRAIN_FILES" \
    --batch_size $BATCH_SIZE \
    --train_seq_len $TRAIN_SEQ_LEN \
    --val_seq_len $VAL_SEQ_LEN \
    --num_iterations $NUM_ITERATIONS \
    --no_reg \
    --model_size $MODEL_SIZE \
    --save_checkpoint \
    --forget_mode $FORGET_MODE \
    --shard_idx $n \
    --method_name "baseline_from_base" \
    --run_info "CPT w. Baseline on FineWeb shard${n}: ModelSize=$MODEL_SIZE" 

  # Method 2: GAPT
  torchrun \
    --nproc_per_node=$N_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$((MASTER_PORT++)) \
    train_iblm_forget.py \
    --continue_from_ckpt $BASE_CKPT \
    --train_files "$FINEWEB_TRAIN_FILES" \
    --batch_size $BATCH_SIZE \
    --train_seq_len $TRAIN_SEQ_LEN \
    --val_seq_len $VAL_SEQ_LEN \
    --num_iterations $NUM_ITERATIONS \
    --use_gapt \
    --entropy_patience 125 \
    --entropy_min_delta 0.01 \
    --mbe_patience 75 \
    --mbe_min_delta 0.01 \
    --mbe_weight $MBE_WEIGHT \
    --mbe_comp_mode $MBE_COMP_MODE \
    --mbe_schedule "all_middle" \
    --model_size $MODEL_SIZE \
    --save_checkpoint \
    --shard_idx $n \
    --method_name "gapt_from_base" \
    --run_info "CPT w. GAPT on FineWeb shard${n}: ModelSize=$MODEL_SIZE | w=$MBE_WEIGHT | $MBE_COMP_MODE"

  # Method 3: GAPT (softplus)
  torchrun \
    --nproc_per_node=$N_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$((MASTER_PORT++)) \
    train_iblm_forget.py \
    --continue_from_ckpt $BASE_CKPT \
    --train_files "$FINEWEB_TRAIN_FILES" \
    --batch_size $BATCH_SIZE \
    --train_seq_len $TRAIN_SEQ_LEN \
    --val_seq_len $VAL_SEQ_LEN \
    --num_iterations $NUM_ITERATIONS \
    --use_gapt \
    --entropy_patience 125 \
    --entropy_min_delta 0.01 \
    --mbe_patience 75 \
    --mbe_min_delta 0.01 \
    --mbe_weight $MBE_WEIGHT \
    --mbe_comp_mode $MBE_COMP_MODE \
    --mbe_schedule "all_middle" \
    --model_size $MODEL_SIZE \
    --use_softplus_gapt \
    --save_checkpoint \
    --shard_idx $n \
    --method_name "gapt_softplus_from_base" \
    --run_info "CPT w. GAPT(softplus) on FineWeb shard${n}: ModelSize=$MODEL_SIZE | w=$MBE_WEIGHT | $MBE_COMP_MODE"
done


for n in $(seq 0 $((NUM_SHARDS - 1))); do
  FINEWEB_TRAIN_FILES="data/forget_fineweb/bin${n}/fineweb_train_*.bin"
  
  # Method 1: Baseline (no regularization) from GAPT ckpt
  torchrun \
    --nproc_per_node=$N_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$((MASTER_PORT++)) \
    train_iblm_forget.py \
    --continue_from_ckpt $GAPT_CKPT \
    --train_files "$FINEWEB_TRAIN_FILES" \
    --batch_size $BATCH_SIZE \
    --train_seq_len $TRAIN_SEQ_LEN \
    --val_seq_len $VAL_SEQ_LEN \
    --num_iterations $NUM_ITERATIONS \
    --no_reg \
    --model_size $MODEL_SIZE \
    --save_checkpoint \
    --forget_mode $FORGET_MODE \
    --shard_idx $n \
    --method_name "baseline_from_gapt" \
    --run_info "CPT w. Baseline on FineWeb shard${n} (from GAPT): ModelSize=$MODEL_SIZE" 

  # Method 2: GAPT from GAPT ckpt
  torchrun \
    --nproc_per_node=$N_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$((MASTER_PORT++)) \
    train_iblm_forget.py \
    --continue_from_ckpt $GAPT_CKPT \
    --train_files "$FINEWEB_TRAIN_FILES" \
    --batch_size $BATCH_SIZE \
    --train_seq_len $TRAIN_SEQ_LEN \
    --val_seq_len $VAL_SEQ_LEN \
    --num_iterations $NUM_ITERATIONS \
    --use_gapt \
    --entropy_patience 125 \
    --entropy_min_delta 0.01 \
    --mbe_patience 75 \
    --mbe_min_delta 0.01 \
    --mbe_weight $MBE_WEIGHT \
    --mbe_comp_mode $MBE_COMP_MODE \
    --mbe_schedule "all_middle" \
    --model_size $MODEL_SIZE \
    --save_checkpoint \
    --shard_idx $n \
    --method_name "gapt_from_gapt" \
    --run_info "CPT w. GAPT on FineWeb shard${n} (from GAPT): ModelSize=$MODEL_SIZE | w=$MBE_WEIGHT | $MBE_COMP_MODE"

  # Method 3: GAPT (softplus) from GAPT ckpt
  torchrun \
    --nproc_per_node=$N_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$((MASTER_PORT++)) \
    train_iblm_forget.py \
    --continue_from_ckpt $GAPT_CKPT \
    --train_files "$FINEWEB_TRAIN_FILES" \
    --batch_size $BATCH_SIZE \
    --train_seq_len $TRAIN_SEQ_LEN \
    --val_seq_len $VAL_SEQ_LEN \
    --num_iterations $NUM_ITERATIONS \
    --use_gapt \
    --entropy_patience 125 \
    --entropy_min_delta 0.01 \
    --mbe_patience 75 \
    --mbe_min_delta 0.01 \
    --mbe_weight $MBE_WEIGHT \
    --mbe_comp_mode $MBE_COMP_MODE \
    --mbe_schedule "all_middle" \
    --model_size $MODEL_SIZE \
    --use_softplus_gapt \
    --save_checkpoint \
    --shard_idx $n \
    --method_name "gapt_softplus_from_gapt" \
    --run_info "CPT w. GAPT(softplus) on FineWeb shard${n} (from GAPT): ModelSize=$MODEL_SIZE | w=$MBE_WEIGHT | $MBE_COMP_MODE"
done