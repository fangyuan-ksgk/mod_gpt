
# ===========================================================================
# Batch 1 — Prefix-first ABS: [Q][ABS×N][CoT][#### ans]
#   Search & training operate on the SAME prefix layout → closed loop.
#   --prefix_abs + --abs_prefix_max=N → insert_prefix_abs (contiguous block)
#
# V1: trainable abstract projection (alpha_abs > 0)
# V6: frozen diagonal projection  (--use_v6, alpha_abs=0 by default)
#
# (a) v1 prefix-8  — trainable ABS prefix, constrained eval
# (b) v6 prefix-8  — frozen diagonal ABS prefix, constrained eval
# (c) v1 prefix-8 + NL drop — same as (a) + TA-style NL dropping
# (d) v6 prefix-8 + NL drop — same as (b) + TA-style NL dropping
# ===========================================================================


# # (a) v1 prefix-8: trainable ABS, closed-loop
# run_bg "v1_pfx8" $M06 $DS_GSM \
#   --K 8 --abstract_vocab_size 32 \
#   --prefix_abs --abs_prefix_max 8 \
#   --alpha_abs 1.0

# # (b) v6 prefix-8: frozen diagonal, closed-loop
# run_bg "v6_pfx8" $M06 $DS_GSM \
#   --use_v6 --K 8 --abstract_vocab_size 32 \
#   --prefix_abs --abs_prefix_max 8

# # (c) v1 prefix-8 + NL drop
# run_bg "v1_pfx8_drop" $M06 $DS_GSM \
#   --K 8 --abstract_vocab_size 32 \
#   --prefix_abs --abs_prefix_max 8 \
#   --alpha_abs 1.0 \
#   --compress_m_set 0,8,16,24,32,64

# wait

# # (d) v6 prefix-8 + NL drop
# run_bg "v6_pfx8_drop" $M06 $DS_GSM \
#   --use_v6 --K 8 --abstract_vocab_size 32 \
#   --prefix_abs --abs_prefix_max 8 \
#   --compress_m_set 0,8,16,24,32,64

# wait

# ===========================================================================
# Batch 2 — v6 prefix-8 scale-up: Qwen3-1.7B, sweep max_iterations
#   v6 search refines abstract tokens via Jacobi iterations.
#   max_iterations = abs_prefix_max → perfect train-inference alignment.
#   Axes: max_iterations ∈ {1, 4, 8}, dataset ∈ {gsm8k, scienceqa}
# ===========================================================================

# # --- Batch 2a: GSM8K × max_iterations ---
# echo ""
# echo "Batch 2a: v6 prefix-8, Qwen1.7B, GSM8K, sweep max_iterations"

# run_bg "v6_pfx8_gsm_i1" $M17 $DS_GSM \
#   --use_v6 --K 8 --abstract_vocab_size 32 \
#   --prefix_abs --abs_prefix_max 8 \
#   --max_iterations 1

# run_bg "v6_pfx8_gsm_i4" $M17 $DS_GSM \
#   --use_v6 --K 8 --abstract_vocab_size 32 \
#   --prefix_abs --abs_prefix_max 8 \
#   --max_iterations 4

# run_bg "v6_pfx8_gsm_i8" $M17 $DS_GSM \
#   --use_v6 --K 8 --abstract_vocab_size 32 \
#   --prefix_abs --abs_prefix_max 8 \
#   --max_iterations 8

# wait

# # --- Batch 2b: ScienceQA × max_iterations ---
# echo ""
# echo "Batch 2b: v6 prefix-8, Qwen1.7B, ScienceQA, sweep max_iterations"

# run_bg "v6_pfx8_sci_i1" $M17 $DS_SCI \
#   --use_v6 --K 8 --abstract_vocab_size 32 \
#   --prefix_abs --abs_prefix_max 8 \
#   --max_iterations 1

# run_bg "v6_pfx8_sci_i4" $M17 $DS_SCI \
#   --use_v6 --K 8 --abstract_vocab_size 32 \
#   --prefix_abs --abs_prefix_max 8 \
#   --max_iterations 4

# run_bg "v6_pfx8_sci_i8" $M17 $DS_SCI \
#   --use_v6 --K 8 --abstract_vocab_size 32 \
#   --prefix_abs --abs_prefix_max 8 \
#   --max_iterations 8

# wait

# # --- Batch 2c: best iter (8) + NL compression ---
# echo ""
# echo "Batch 2c: v6 prefix-8 iter=8, Qwen1.7B, +compression"

# run_bg "v6_pfx8_gsm_i8_drop" $M17 $DS_GSM \
#   --use_v6 --K 8 --abstract_vocab_size 32 \
#   --prefix_abs --abs_prefix_max 8 \
#   --max_iterations 8 \
#   --compress_m_set 0,8,16,24,32,64

# run_bg "v6_pfx8_sci_i8_drop" $M17 $DS_SCI \
#   --use_v6 --K 8 --abstract_vocab_size 32 \
#   --prefix_abs --abs_prefix_max 8 \
#   --max_iterations 8 \
#   --compress_m_set 0,8,16,24,32,64

# run_bg "v6_pfx16_gsm_i16" $M17 $DS_GSM \
#   --use_v6 --K 16 --abstract_vocab_size 32 \
#   --prefix_abs --abs_prefix_max 16 \
#   --max_iterations 16

# wait

# ===========================================================================
# Batches 3a–3e — v6 prefix deeper sweep (2 per batch, 10 total)
#   All use --use_v6, iter=abs_prefix_max (perfect alignment) unless noted.
#   Base: Qwen3-1.7B, GSM8K, abs_vocab=32, pfx8, iter=8, emb_lr=1x, 1ep
# ===========================================================================

# --- Batch 3a: prefix length (GSM8K, 1.7B) ---
echo ""
echo "Batch 3a: prefix length sweep — pfx4 vs pfx16"

run_bg "v6_pfx4_gsm_i4" $M17 $DS_GSM \
  --use_v6 --K 4 --abstract_vocab_size 32 \
  --prefix_abs --abs_prefix_max 4 \
  --max_iterations 4

run_bg "v6_pfx16_sci_i16" $M17 $DS_SCI \
  --use_v6 --K 16 --abstract_vocab_size 32 \
  --prefix_abs --abs_prefix_max 16 \
  --max_iterations 16

wait

# --- Batch 3b: emb_lr_mult fine-grained (GSM8K, 1.7B, pfx8, iter=8) ---
echo ""
echo "Batch 3b: emb_lr_mult — 3x vs 5x"

run_bg "v6_pfx8_gsm_emb3" $M17 $DS_GSM \
  --use_v6 --K 8 --abstract_vocab_size 32 \
  --prefix_abs --abs_prefix_max 8 \
  --max_iterations 8 \
  --emb_lr_mult 3.0

run_bg "v6_pfx8_gsm_emb5" $M17 $DS_GSM \
  --use_v6 --K 8 --abstract_vocab_size 32 \
  --prefix_abs --abs_prefix_max 8 \
  --max_iterations 8 \
  --emb_lr_mult 5.0

wait

# --- Batch 3c: emb_lr_mult continued (GSM8K, 1.7B, pfx8, iter=8) ---
echo ""
echo "Batch 3c: emb_lr_mult — 10x vs 20x"

run_bg "v6_pfx8_gsm_emb10" $M17 $DS_GSM \
  --use_v6 --K 8 --abstract_vocab_size 32 \
  --prefix_abs --abs_prefix_max 8 \
  --max_iterations 8 \
  --emb_lr_mult 10.0

run_bg "v6_pfx8_gsm_emb20" $M17 $DS_GSM \
  --use_v6 --K 8 --abstract_vocab_size 32 \
  --prefix_abs --abs_prefix_max 8 \
  --max_iterations 8 \
  --emb_lr_mult 20.0

wait

# --- Batch 3d: prefix length × dataset (1.7B, iter=match) ---
echo ""
echo "Batch 3d: prefix length — pfx4 GSM vs pfx16 GSM"

run_bg "v6_pfx4_gsm" $M17 $DS_GSM \
  --use_v6 --K 4 --abstract_vocab_size 32 \
  --prefix_abs --abs_prefix_max 4 \
  --max_iterations 4

run_bg "v6_pfx16_gsm" $M17 $DS_GSM \
  --use_v6 --K 16 --abstract_vocab_size 32 \
  --prefix_abs --abs_prefix_max 16 \
  --max_iterations 16

wait

# --- Batch 3e: best emb_lr on SciQA + prefix length on SciQA ---
echo ""
echo "Batch 3e: SciQA — pfx4 vs pfx16"

run_bg "v6_pfx4_sci" $M17 $DS_SCI \
  --use_v6 --K 4 --abstract_vocab_size 32 \
  --prefix_abs --abs_prefix_max 4 \
  --max_iterations 4

run_bg "v6_pfx16_sci" $M17 $DS_SCI \
  --use_v6 --K 16 --abstract_vocab_size 32 \
  --prefix_abs --abs_prefix_max 16 \
  --max_iterations 16

wait


# ---- Baseline runners (separate scripts) ----
run_pause_bg() {
  EXP_IDX=$((EXP_IDX + 1))
  local idx=$EXP_IDX
  local gpu=$(( (idx - 1) % N_GPUS ))
  local port=$((BASE_PORT + idx))
  local tag=$1; shift
  local model=$1; shift
  local dataset=$1; shift
  local grad_accum=$((8 / BATCH_SIZE))
  local output_dir="./ckpt/sweep_${TIMESTAMP}/exp${idx}_${tag}"

  echo "  Exp ${idx}: ${tag}  model=$(basename $model)  dataset=${dataset}  [GPU=${gpu}] [pause]"

  CUDA_VISIBLE_DEVICES=$gpu torchrun \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$port \
    train_pause_pt.py \
    --model_name $model \
    --dataset $dataset \
    --max_length $MAX_LENGTH \
    --lr $LR \
    --warmup_steps $WARMUP_STEPS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $grad_accum \
    --num_epochs 1 \
    --log_every $LOG_EVERY \
    --eval_every $EVAL_EVERY \
    --save_every $SAVE_EVERY \
    --eval_samples $EVAL_SAMPLES \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --max_new_tokens $MAX_NEW_TOKENS \
    --output_dir $output_dir \
    "$@" &
}

run_ta_bg() {
  EXP_IDX=$((EXP_IDX + 1))
  local idx=$EXP_IDX
  local gpu=$(( (idx - 1) % N_GPUS ))
  local port=$((BASE_PORT + idx))
  local tag=$1; shift
  local model=$1; shift
  local dataset=$1; shift
  local grad_accum=$((8 / BATCH_SIZE))
  local output_dir="./ckpt/sweep_${TIMESTAMP}/exp${idx}_${tag}"

  echo "  Exp ${idx}: ${tag}  model=$(basename $model)  dataset=${dataset}  [GPU=${gpu}] [ta]"

  CUDA_VISIBLE_DEVICES=$gpu torchrun \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$port \
    train_ta_pt.py \
    --model_name $model \
    --dataset $dataset \
    --max_length $MAX_LENGTH \
    --lr $LR \
    --warmup_steps $WARMUP_STEPS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $grad_accum \
    --num_epochs 1 \
    --log_every $LOG_EVERY \
    --eval_every $EVAL_EVERY \
    --save_every $SAVE_EVERY \
    --eval_samples $EVAL_SAMPLES \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --max_new_tokens $MAX_NEW_TOKENS \
    --output_dir $output_dir \
    "$@" &
}

# # ============================================================================
# # Batch 4 — Baselines: pause token + token assorted
# #           4 models × 5 datasets × 2 baselines = 40 experiments, 1 epoch each
# #           Each (model, dataset) pair launches pause + ta in parallel (1 GPU each),
# #           then waits before the next pair.
# # ============================================================================
# echo ""
# echo "============================================================"
# echo "Batch 4: Baselines (pause + TA) — 4 models × 5 datasets (${TIMESTAMP})"
# echo "============================================================"

# for mp in "17:$M17" "4b:$M4" "l1:$ML1" "l3:$ML3"; do
#     mtag="${mp%%:*}"; model="${mp#*:}"
#     for dp in "gsm:$DS_GSM" "sci:$DS_SCI" "arc:$DS_ARC" "mml:$DS_MMLU" "csqa:$DS_CSQA"; do
#         dtag="${dp%%:*}"; ds="${dp#*:}"
#         run_pause_bg "pause_${dtag}_${mtag}" "$model" "$ds"
#         run_ta_bg    "ta_${dtag}_${mtag}"    "$model" "$ds"
#         echo "  Waiting for pause_${dtag}_${mtag} + ta_${dtag}_${mtag}..."
#         wait
#     done
# done

# echo "  Batch 4 complete."

# # echo ""
# # echo "============================================================"
# # echo "All 16 experiments complete. Results in ./ckpt/sweep_${TIMESTAMP}/"
# # echo "============================================================"