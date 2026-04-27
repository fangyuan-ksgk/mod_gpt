# 2026-04-25 — failure-set per-code ablation, all 32 codes (includes L1, L3 partial)

## Setup

- For each model, sweep ablation of every code $c \in \{0, \dots, 31\}$ over the steered-baseline failure set.
- `n_failures = 200` per model, `n_runs = 2` (different random swap seeds), so `n = 400` per (model, code).
- "Fix rate" = fraction of failures that flip to correct under single-code ablation.
- Output: `log/analysis_out/failure_per_code/{run}/summary.json`

## Aggregate fix-rate distribution per model

| Model       | baseline | mean fix | min   | max   | top-3 codes (fix-rate)              |
|-------------|----------|----------|-------|-------|-------------------------------------|
| Qwen3-0.6B  | 0.690    | 0.245    | 0.230 | 0.310 | code 1 (0.310), code 5 (0.300), code 0 (0.287) |
| Qwen3-1.7B  | 0.770    | 0.232    | 0.200 | 0.355 | code 0 (0.355), code 3 (0.328), code 1 (0.297) |
| Qwen3-4B    | 0.776    | 0.368    | 0.320 | 0.522 | code 9 (0.522), code 0 (0.477), code 1 (0.448) |
| Llama3.2-1B | 0.690    | 0.234    | 0.223 | 0.253 | code 5 (0.253), code 9 (0.253), code 2 (0.245) |
| Llama3.2-3B | —        | —        | —     | —     | (no `failure_ablations.jsonl` yet)  |

## Reading the numbers

- **Qwen models show concentrated rescuers.** The top code beats the bottom by 5–20 pp:
  - q06: top − bottom = 8 pp (31.0% vs 23.0%).
  - q17: top − bottom = 15.5 pp (35.5% vs 20.0%).
  - q4b: top − bottom = 20.2 pp (52.2% vs 32.0%); also the highest *mean* fix rate at 36.8%.
- **Llama3.2-1B is qualitatively different: flat distribution.** Range is 22.3%–25.3%, only 3 pp from worst to best. There is no "top rescuer" code on L1; every code rescues at roughly the mean rate.
- **The flat L1 distribution lines up with codebook geometry.** L1 has near-orthogonal `svec`s (|cos| ≈ 0.012) and 78% utilization (vs 31–56% for Qwen). Rescue capacity is spread evenly across the codebook rather than concentrated on a few hubs.
- **Q4B is the high-fix outlier.** Mean fix rate of 36.8% is +12 pp above the other models. Combined with its broader utilization (56%) and higher eRank, it has both more codes "doing real work" *and* more rescue capacity per code.

## Implication for the analysis storyline

This adds a missing piece to the codebook-geometry narrative:

> Geometry → behavior. **Concentrated codebooks (Qwen) ⇒ concentrated rescuers; near-orthogonal, broadly-utilized codebooks (Llama) ⇒ flat rescue distribution.**

The single-code-ablation story (existing `tab:rescue` in `analysis.tex`) is *consistent* with the diversity finding (`tab:codebook-diversity`) but only sampled top-4 codes per model. The full 32-code sweep here makes the contrast quantitative: the *shape* of the per-code rescue distribution is itself a fingerprint of codebook geometry.

## Files

- `log/analysis_out/failure_per_code/{q06,q17,q4b,l1}_sciqa_v9_C32_detach_*/summary.json`
- `l3_sciqa_v6_C32_base/failure_ablations.jsonl` is empty — pending re-run.
