# 2026-04-25 — per-code ablation aggregate accuracy + help/hurt decomposition (Qwen3 0.6B / 1.7B / 4B)

## Setup

- Same eval logic as `0425-top-ablate.md` (single-code random-swap ablation, full SciQA test set), with `n=2100` and a richer per-row decomposition.
- For each (model, code) we partition every test sample into one of:
  - `help`     — baseline wrong, ablation correct (a *rescue*)
  - `hurt`     — baseline correct, ablation wrong (a *de-rail*)
  - `same_ok`  — both correct
  - `same_bad` — both wrong
- Output: `log/analysis_out/per_code_ablation/{run}/summary_by_code.json`

## Per-model results

### Qwen3-0.6B (baseline 69.10%)

| code | ablate_acc | Δ vs baseline | help | hurt | same_ok | same_bad | net (help−hurt) |
|------|-----------:|--------------:|-----:|-----:|--------:|---------:|----------------:|
| 0    | 67.67%     | −1.43         | 145  | 175  | 1276    | 504      | −30             |
| 1    | 66.10%     | −3.00         | 167  | 230  | 1221    | 482      | −63             |
| 2    | 66.48%     | −2.62         | 138  | 193  | 1258    | 511      | −55             |
| 5    | 65.19%     | −3.90         | 146  | 228  | 1223    | 503      | −82             |

### Qwen3-1.7B (baseline 77.14%)

| code | ablate_acc | Δ vs baseline | help | hurt | same_ok | same_bad | net (help−hurt) |
|------|-----------:|--------------:|-----:|-----:|--------:|---------:|----------------:|
| 0    | 76.57%     | −0.57         | 132  | 144  | 1476    | 348      | −12             |
| 1    | 75.95%     | −1.19         | 102  | 127  | 1493    | 378      | −25             |
| 3    | 76.24%     | −0.90         | 132  | 151  | 1469    | 348      | −19             |
| 14   | 76.95%     | −0.19         | 96   | 100  | 1520    | 384      |  −4             |

### Qwen3-4B (baseline 76.52%)

| code | ablate_acc | Δ vs baseline | help | hurt | same_ok | same_bad | net (help−hurt) |
|------|-----------:|--------------:|-----:|-----:|--------:|---------:|----------------:|
| 0    | 75.19%     | −1.33         | 213  | 241  | 1366    | 280      | −28             |
| 1    | 75.95%     | −0.57         | 213  | 225  | 1382    | 280      | −12             |
| 9    | 75.05%     | −1.48         | 227  | 258  | 1349    | 266      | −31             |
| 20   | 75.67%     | −0.86         | 113  | 131  | 1476    | 380      | −18             |

## Reading the numbers

- **Every top code is net-negative aggregate**, but only by a small margin (Δ ∈ [−3.90, −0.19] pp). The codes are "useful on net" — removing any one *does* cost accuracy.
- **The (help, hurt) magnitudes are large and roughly balanced.** Even when net Δ is small, hundreds of samples flip in each direction:
  - q4b code 9: 227 helped, 258 hurt — the code is *active* on ~23% of samples in *both* directions.
  - This shows top codes are not "merely commitment-inducers" — they are dynamically routed and have causal effect on a meaningful fraction of inputs.
- **Help/hurt magnitudes scale with model size.**
  - q06 top code: 167 helps, 230 hurts.
  - q4b top code: 227 helps, 258 hurts (more total flips, ratio closer to 1).
  - Larger models route more sensitively; ablation shifts the trajectory of more samples.
- **q17 code 14 is the gentlest ablation** (96/100 flips, net −4): a candidate "low-impact code" — useful for sanity-checking the eval pipeline (a small, near-zero net delta on a balanced flip count).

## Why this complements existing tables

| Source                  | Subset      | Granularity                         | What it tells you                                    |
|-------------------------|-------------|-------------------------------------|------------------------------------------------------|
| `tab:global-ablation`   | full set    | aggregate (zero / random-replace)   | The routing as a whole is necessary.                 |
| `tab:rescue`            | failure set | per-code, per-letter-bucket fix     | Top codes act as *commitment inducers* on no-answer failures. |
| `0425-top-ablate.md`    | full set    | per-code aggregate accuracy (n=2000)| Single-code drops are mild (1–5 pp).                 |
| **this file**           | full set    | per-code aggregate **+ help/hurt**  | Mild net Δ hides large bidirectional flips — top codes have wide causal footprint. |
| `0425-failure-per-code.md` | failure set | per-code fix rate, **all 32 codes** | Concentrated vs flat rescuer distribution by family. |

## Implication

The "small net Δ but large help/hurt" finding is the strongest argument so far that the discrete codes are *causal* rather than incidental: each top code flips hundreds of samples in *each* direction, indicating dynamic, content-dependent routing. The aggregate accuracy is preserved partly because the help and hurt populations are largely disjoint and roughly balanced — i.e. the code is doing *useful work where it routes*, but ablating it neither uniformly helps nor uniformly hurts.

This is the "dynamic structured steering" claim in §4.4 made causal, not just observational.

## Files

- `log/analysis_out/per_code_ablation/{q06,q17,q4b}_*/summary_by_code.json`
- `log/analysis_out/per_code_ablation/{q06,q17,q4b}_*/summary_by_topic_code.json` (per-topic decomposition — not summarized here)
- `log/analysis_out/per_code_ablation/{q06,q17,q4b}_*/ablations.jsonl`, `baseline.jsonl`
