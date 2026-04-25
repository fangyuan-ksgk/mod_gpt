#!/usr/bin/env bash
# Per-code unigram ablation sweep on the FULL ScienceQA test set
# for all 5 main SciQA runs.
#
# For each run, sweeps all 32 codes with `eval_per_code_ablation.py`,
# producing per-(sample, code) outcomes (help / hurt / same_ok / same_bad).
#
# Outputs land under  ${OUT_DIR}/<run>/ :
#   baseline.jsonl
#   ablations.jsonl
#   summary_by_code.json
#   summary_by_topic_code.json
#
# Override any setting via env, e.g.
#   NUM_SAMPLES=500  BATCH_SIZE=8  bash run_per_code_ablation.sh

set -euo pipefail
cd "$(dirname "$0")"

NUM_SAMPLES=${NUM_SAMPLES:-2000}
BATCH_SIZE=${BATCH_SIZE:-16}
MAX_NEW=${MAX_NEW:-512}
SEED=${SEED:-0}
OUT_DIR=${OUT_DIR:-analysis_out/per_code_ablation}

REPO_Q="Ksgk-fy/sciqa_ckpt_20260416_0942"
REPO_L="Ksgk-fy/sciqa_ckpt_20260416_1452"

# rows: "<repo>|<run>"
RUNS=(
  "${REPO_Q}|q06_sciqa_v9_C32_detach_az0.1_aa0.5"
  "${REPO_Q}|q17_sciqa_v9_C32_detach_az0.1_aa0.5"
  "${REPO_Q}|q4b_sciqa_v9_C32_detach_az0.1_aa0.1"
  "${REPO_L}|l1_sciqa_v9_C32_detach_az0.5_aa0.5"
  "${REPO_L}|l3_sciqa_v6_C32_base"
)

# Optional: skip a run whose summary already exists. Set FORCE=1 to recompute.
FORCE=${FORCE:-0}

for row in "${RUNS[@]}"; do
  REPO="${row%%|*}"
  RUN="${row##*|}"
  SUMMARY="${OUT_DIR}/${RUN}/summary_by_code.json"

  echo "=============================================================="
  echo "[run] ${RUN}  (repo=${REPO})"
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

echo "done -> ${OUT_DIR}"
