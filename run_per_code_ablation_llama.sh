#!/usr/bin/env bash
# Llama-only per-code ablation sweep (l1 + l3), intended to run in
# parallel with `run_per_code_ablation.sh` (which is grinding through
# the Qwen runs).
#
# Override settings via env, e.g.:
#   CUDA_VISIBLE_DEVICES=1  NUM_SAMPLES=2000  BATCH_SIZE=16  \
#     bash run_per_code_ablation_llama.sh
#
# Recommended: pin to a different GPU than the Qwen sweep, e.g.
#   tmux new -s pca_llama
#   CUDA_VISIBLE_DEVICES=1 bash run_per_code_ablation_llama.sh \
#       2>&1 | tee log/per_code_ablation_llama.log

set -euo pipefail
cd "$(dirname "$0")"

NUM_SAMPLES=${NUM_SAMPLES:-2000}
BATCH_SIZE=${BATCH_SIZE:-16}
MAX_NEW=${MAX_NEW:-512}
SEED=${SEED:-0}
OUT_DIR=${OUT_DIR:-analysis_out/per_code_ablation}

REPO_L="Ksgk-fy/sciqa_ckpt_20260416_1452"

RUNS=(
  "${REPO_L}|l1_sciqa_v9_C32_detach_az0.5_aa0.5"
  "${REPO_L}|l3_sciqa_v6_C32_base"
)

FORCE=${FORCE:-0}

for row in "${RUNS[@]}"; do
  REPO="${row%%|*}"
  RUN="${row##*|}"
  SUMMARY="${OUT_DIR}/${RUN}/summary_by_code.json"

  echo "=============================================================="
  echo "[run] ${RUN}  (repo=${REPO})  GPU=${CUDA_VISIBLE_DEVICES:-default}"
  echo "=============================================================="

  if [[ -f "${SUMMARY}" && "${FORCE}" != "1" ]]; then
    echo "[skip] ${SUMMARY} exists (FORCE=1 to recompute)"
    continue
  fi

  python eval_per_code_ablation.py \
    --repo "${REPO}" \
    --run  "${RUN}" \
    --num-samples "${NUM_SAMPLES}" \
    --batch-size  "${BATCH_SIZE}" \
    --max-new-tokens "${MAX_NEW}" \
    --seed "${SEED}" \
    --out-dir "${OUT_DIR}"
done

echo "done (llama) -> ${OUT_DIR}"
