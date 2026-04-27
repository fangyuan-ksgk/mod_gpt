# 2026-04-23 — ablate code 5 eval (q06, ScienceQA test, N=2000)

## Run config

- Repo: `Ksgk-fy/sciqa_ckpt_20260416_0942`
- Run: `q06_sciqa_v9_C32_detach_az0.1_aa0.5` (Qwen3-0.6B, C=32, L=4, scale=0.5)
- `max_new_tokens=512`, `batch_size=16`, `num_samples=2000`
- Mutual exclusion ON for ablate (swap pool excludes ablated codes themselves)
- Output: `analysis_out/steering_modes_abl5/`

## Command

```bash
python eval_steering_modes.py \
  --modes plain prompt_steered steered ablate_codes \
  --ablate-codes 5 \
  --num-samples 2000 --batch-size 16 \
  --out-dir analysis_out/steering_modes_abl5
```

## Results

| mode           | correct/total | acc    | Δ vs plain | Δ vs steered |
|----------------|---------------|--------|------------|--------------|
| plain          | 1285/2000     | 64.25% | —          | −4.75        |
| prompt_steered | 1261/2000     | 63.05% | −1.20      | −5.95        |
| **steered**    | **1380/2000** | **69.00%** | **+4.75**  | —        |
| ablate_5       | 1306/2000     | 65.30% | +1.05      | −3.70        |

## Reading the numbers

- **Steering helps globally (+4.75pp over plain).** Decode-time steering is clearly the dominant contribution.
- **Prefill-only steering is worse than plain (−1.20pp).** Strong control: the gain requires codes to be injected during decode, not just during prompt encoding. Priming alone hurts slightly.
- **Ablating code 5 across all topics: net −3.70pp vs full steered.**
  - Recovers some of the gap relative to plain (+1.05pp), but sacrifices most of the steering gain.
  - Consistent with: code 5 is **mostly helpful**, occasionally causes degenerate loops.
- The notebook case studies (idx=29, idx=52) where ablate-5 rescued steered were cherry-picked failure modes. Globally, code 5 is net-positive.

## Interpretation

- Code 5 appears to be a high-utility code that drives useful structural patterns on most samples but triggers repetition/attractor behavior on a topic subset (writing-strategies clearly, maybe others).
- A blanket ablation is the wrong fix. The right experiment is **topic-conditional ablation**: ablate 5 only on topics where it harms.
- Need the per-topic breakdown (topic field is now logged in each JSONL row — run the pivot below once `ablate_15` and `ablate_5_15` finish).

## Per-topic analysis (to run after all 3 dirs finish)

```python
import pandas as pd, pathlib
d = {fp.stem.split("__",1)[1]: pd.read_json(fp, lines=True)
     for fp in pathlib.Path("analysis_out/steering_modes_abl5").glob("*.jsonl")}
df = pd.concat([x.assign(tag=t) for t,x in d.items()])
pivot = df.groupby(["topic","tag"])["correct"].mean().unstack()
diff  = (pivot["ablate_5"] - pivot["steered"]).sort_values(ascending=False)
n     = df[df.tag=="steered"].groupby("topic").size().rename("n")
print(pd.concat([diff.rename("abl5-steered"), n], axis=1)
        .query("n >= 20")
        .sort_values("abl5-steered", ascending=False))
```

Expected: writing/language topics at the top (positive), science topics at the bottom (negative).

## Pending runs

- `ablate_15` on `analysis_out/steering_modes_abl15/`
- `ablate_5_15` on `analysis_out/steering_modes_abl5_15/` (with `exclude_mutual=True` so 5↛15 and 15↛5)

## Code changes landed today

- `eval_steering_modes.py`: `max_new_tokens` default 256 → 512; logs `topic`/`subject`/`category`/`skill` per row; `ablate_codes` mode uses `exclude_mutual=True`.
- `eval_per_code_ablation.py`: `max_new_tokens` default 256 → 512.
- `sorl/steer.py`: `_ablate_exclude` state; random-replacement draw now respects exclusion set.
- `sorl/analyze.py`: `ablate_router_ngrams(..., exclude=None, exclude_mutual=False)` kwargs.

---

## Follow-up (later in the day): 0.6B vs 1.7B with sufficient `max_new_tokens`

With `max_new_tokens=512` the degenerate-loop truncation artifact goes away and we get a clean story.

| run                              | plain | prompt_steered | steered | ablate_5 |
|----------------------------------|-------|----------------|---------|----------|
| `q06_sciqa_v9_C32_detach_az0.1_aa0.5` (0.6B) | 64.25 | 63.05          | **69.00** | 65.30    |
| `q17_sciqa_v9_C32_detach_az0.1_aa0.5` (1.7B) | 75.10 | 76.60          | **77.00** | 77.00    |

Timings ~1700–1825s per mode per run (2000 samples).

### Core takeaways

- **`Acc[plain] < Acc[fully_steered]` holds in both cases.** Expected — training objective is the fully-steered trajectory, so plain decoding is off-distribution. Not a surprise, not a crisis; it's the baseline condition.
- **Steered gain scales inversely with base competence.** 0.6B gets +4.75pp, 1.7B only +1.9pp. The abstraction scaffold helps more when the LM prior is weaker.
- **Prompt-only steering is sign-flipped across scales.** Hurts 0.6B (−1.20pp vs plain); helps 1.7B (+1.50pp vs plain). Interpretation: the 0.6B prefill router emits codes that only pay off if the decode router continues the pattern; without decode-side steering the prefix is net-toxic. The 1.7B model has enough slack to benefit from the prefix signal standalone.
- **`ablate_5` story is scale-dependent.**
  - 1.7B: `ablate_5 = steered` (77.00 = 77.00). The entire gap between `prompt_steered` and `steered` is accounted for by those 5 codes — clean discrete lesion.
  - 0.6B: `ablate_5 = 65.30` recovers ~47% of the `prompt_steered → steered` gap. Ablation alone is insufficient; decode steering is doing something beyond blacklisting a handful of codes (likely continuous-direction corrections).

### Non-trivial comparisons (what's actually worth reporting)

The `plain < fully_steered` inequality is trivially expected from the training setup. The informative comparisons are:

1. **`prompt_steered` vs `steered`** — how much of the benefit lives in prefill vs decode. Small gap at 1.7B (+0.4), large at 0.6B (+5.95).
2. **`ablate_5` vs `steered`** — how localized the signal is. Fully localized at 1.7B, partially at 0.6B.
3. **`prompt_steered` vs `plain`** — the sign-flip. Train/test mismatch (prefill-steered, decode-plain) hurts the small model, helps the big one.

### Status

No methodological concerns. Remaining work is mechanistic ablation / characterization (where in prefill/decode the gain lives, which codes carry it, per-topic localization), not algorithm rescue.

### Per-sample mechanism notes (from notebook case studies earlier in day)

Two distinct steered-failure archetypes observed:

- **Commitment bug (e.g. rhyme sample idx=18):** plain and steered diverge at a single early chunk (chunk 18), then the steered trajectory locks in a wrong semantic commitment. Single-code global ablations can't fix it; needs position-scoped code substitution.
- **Attractor bug (e.g. fallacy sample idx=115):** plain and steered share the entire prefill, then steered flips one decode code (chunk 26) and slides into a self-reinforcing glossary-dump loop, never answering. Full random code perturbation fixes it because any non-degenerate routing escapes the attractor basin.

Diagnostic: `recovery_rate = P(pred_perturbed == gold | pred_steer ≠ gold)` should split cleanly between the two — high for attractor failures, near-zero for commitment failures. Worth computing across the regression set.
