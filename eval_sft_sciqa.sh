#!/bin/bash
# ===========================================================================
# Evaluate all 5 SFT checkpoints on ScienceQA (test split, full 2224 samples).
# Runs sequentially on a single GPU (CUDA_VISIBLE_DEVICES=0).
# Produces one .pt per run under ./analysis_out/codes/  plus an index.json.
#
# Usage:
#   ./eval_sft_sciqa.sh           # all 5 ckpts
#   ./eval_sft_sciqa.sh l1 q06    # subset by tag
# ===========================================================================
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

NUM_SAMPLES=${NUM_SAMPLES:-2224}
EVAL_BATCH=${EVAL_BATCH:-8}
MAX_NEW=${MAX_NEW:-256}
OUT_DIR=${OUT_DIR:-./analysis_out/codes}
DATASET=scienceqa

CKPT_0344=./ckpt/sft_sciqa_20260421_0344
CKPT_0546=./ckpt/sft_sciqa_20260421_0546

# tag -> (dir, tokenizer_fallback)
declare -A C_DIR C_FALLBACK
C_DIR[l1]="${CKPT_0344}/l1_sciqa_ep1";  C_FALLBACK[l1]="meta-llama/Llama-3.2-1B"
C_DIR[q17]="${CKPT_0344}/q17_sciqa_ep1"; C_FALLBACK[q17]="Qwen/Qwen3-1.7B"
C_DIR[q06]="${CKPT_0344}/q06_sciqa_ep1"; C_FALLBACK[q06]="Qwen/Qwen3-0.6B"
C_DIR[q4b]="${CKPT_0546}/q4b_sciqa_ep1"; C_FALLBACK[q4b]="Qwen/Qwen3-4B"
C_DIR[l3]="${CKPT_0546}/l3_sciqa_ep1";   C_FALLBACK[l3]="meta-llama/Llama-3.2-3B"

ORDER=(l1 q06 q17 l3 q4b)   # small -> large; adjust if OOM

if [ "$#" -gt 0 ]; then
  TAGS=("$@")
else
  TAGS=("${ORDER[@]}")
fi

mkdir -p "$OUT_DIR"
echo "============================================================"
echo "SFT ScienceQA eval | GPU=${CUDA_VISIBLE_DEVICES} | N=${NUM_SAMPLES}"
echo "  tags: ${TAGS[*]}"
echo "  out : ${OUT_DIR}"
echo "============================================================"

for tag in "${TAGS[@]}"; do
  dir=${C_DIR[$tag]:-}
  fb=${C_FALLBACK[$tag]:-}
  if [ -z "$dir" ]; then
    echo "!! unknown tag: $tag  (known: ${!C_DIR[*]})"; continue
  fi
  if [ ! -d "$dir" ]; then
    echo "!! missing dir: $dir  (skipping $tag)"; continue
  fi

  echo ""
  echo "--- [${tag}] ${dir} ---"
  python analyze_latent_codes.py \
    --sft_local "$dir" \
    --sft_dataset "$DATASET" \
    --sft_tokenizer_fallback "$fb" \
    --num_samples "$NUM_SAMPLES" \
    --eval_batch "$EVAL_BATCH" \
    --max_new_tokens "$MAX_NEW" \
    --out_dir "$OUT_DIR"
done

echo ""
echo "============================================================"
echo "Done. Results in ${OUT_DIR}/  (see index.json)"
echo "============================================================"
