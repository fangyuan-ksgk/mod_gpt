# 2026-04-25 — top-rescuer code ablation eval (Qwen3 0.6B / 1.7B / 4B, SciQA test, N=2000)

## Run config

- Repo: `Ksgk-fy/sciqa_ckpt_20260416_0942`
- Runs:
  - `q06_sciqa_v9_C32_detach_az0.1_aa0.5` (Qwen3-0.6B, C=32, scale=0.5)
  - `q17_sciqa_v9_C32_detach_az0.1_aa0.5` (Qwen3-1.7B, C=32, scale=0.5)
  - `q4b_sciqa_v9_C32_detach_az0.1_aa0.1` (Qwen3-4B, C=32, scale=0.1)
- `num_samples=2000`, `seed=0`
- Single-code ablation: random-swap one code at a time over the test set.
- Output: `log/analysis_out/top_ablate/` (one JSONL per (run, mode) + `summary.json`)

## Results — aggregate accuracy under single-code ablation

### Qwen3-0.6B (steered = 69.75%)

| mode      | correct/total | acc    | Δ vs steered |
|-----------|---------------|--------|--------------|
| steered   | 1395/2000     | 69.75% | —            |
| ablate_0  | 1323/2000     | 66.15% | −3.60        |
| ablate_1  | 1300/2000     | 65.00% | −4.75        |
| ablate_5  | 1300/2000     | 65.00% | −4.75        |

### Qwen3-1.7B (steered = 77.30%)

| mode      | correct/total | acc    | Δ vs steered |
|-----------|---------------|--------|--------------|
| steered   | 1546/2000     | 77.30% | —            |
| ablate_0  | 1503/2000     | 75.15% | −2.15        |
| ablate_1  | 1518/2000     | 75.90% | −1.40        |
| ablate_3  | 1532/2000     | 76.60% | −0.70        |

### Qwen3-4B (steered = 76.35%)

| mode      | correct/total | acc    | Δ vs steered |
|-----------|---------------|--------|--------------|
| steered   | 1527/2000     | 76.35% | —            |
| ablate_0  | 1504/2000     | 75.20% | −1.15        |
| ablate_1  | 1499/2000     | 74.95% | −1.40        |
| ablate_9  | 1504/2000     | 75.20% | −1.15        |

## Reading the numbers

- **Single-code ablation is reliably negative.** Every (model, top-code) cell drops vs steered (Δ ∈ [−4.75, −0.70]).
- **Drops shrink with model scale.** Worst drop: 0.6B (−4.75pp). Mildest: 4B (≤ −1.40pp).
  - Consistent with the codebook-diversity story: larger models distribute work over more codes, so removing any one code hurts less in aggregate.
  - Qwen3-4B utilizes 56% of its codebook (vs 41% for 0.6B and 31% for 1.7B); the load is more spread.
- **No code is uniquely catastrophic at 1.7B / 4B.** The three top codes cluster within ~1pp of each other on these models — graceful degradation rather than single-point-of-failure.
- **0.6B is the outlier**: codes 1 and 5 are tied at the worst drop and ~1pp worse than code 0 — at the smallest scale a couple of codes are doing disproportionate work.

## Relation to other tables

- **Table 2 (global ablation, analysis.tex)** uses different eval settings (different sample subset / scale config) and is not directly comparable; numbers there are 55.3 / 64.1 / 72.3% steered vs 49.1 / 57.7 / 60.1% under `scale → 0`. Those drops (4–12pp) are larger than the single-code drops here, as expected — global zeroing removes *all* steering, not one code.
- **Table 4 (rescue, analysis.tex)** is the *complementary* per-code view on the failure subset: fix-rate-by-letter-bucket. Top codes that look mildly negative in aggregate here (e.g., q4b code 9 at −1.15pp) are simultaneously the strongest "no-answer" rescuers (70.6% no-answer fix rate). Net aggregate impact is the sum of rescues (failure → success) minus de-rails (success → failure); the small negative net is consistent with code 9 being a commitment-inducer that occasionally over-commits on cases the baseline already had right.

## Interpretation

- The aggregate-accuracy view confirms structural redundancy at scale: at 1.7B and 4B, no single code in the top-rescuer set is irreplaceable. Targeted, *topic-conditional* ablation (vs blanket) remains the right knob for behavior shaping (cf. `0423-ablate5.md`).
- The 0.6B numbers (−4.75pp for codes 1 and 5) deserve a follow-up: candidate that the small model concentrates more of the "commitment-induction" function on a few codes. A per-topic pivot of these JSONLs would confirm.

## Files

- `log/analysis_out/top_ablate/summary.json`
- `log/analysis_out/top_ablate/q{06,17,4b}_*__{steered,ablate_*}.jsonl`
