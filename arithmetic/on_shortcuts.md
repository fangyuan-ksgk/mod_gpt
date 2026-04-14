# On Shortcuts: S6 vs C6 and the Information Bottleneck

## The S6 anomaly

On undersized models (2L1H128d), SoRL K=1 abs30 LOSES to baseline on add_S6:
- Baseline: 55%
- SoRL: 21%

But SoRL WINS massively on add_C6 (same cascade depth, varied answers):
- Baseline: 39%
- SoRL: 96%

## Why?

S6 examples have **degenerate answers**: always `1000000`-`1000009`. All digit pairs sum to exactly 9 or 10, creating a full cascade of Quirke's U (uncertain) states. The answer is always `1` followed by `0`s.

C6 examples have **varied answers** like `1500014`, `1060004`, `1546000`. Some positions have definite carries (STn=1) that break the cascade chain.

## The shortcut

The baseline can learn "when all digit sums are 9-10, output `1` then `0`s" — a **holistic pattern match** without computing the cascade. The SoRL information bottleneck prevents this:

- The `block_mask` in SorlModelWrapper forces information to flow through abstraction tokens
- The `skip_abs` rule prevents trajectory tokens from attending past accumulated abstractions
- The model MUST compute carry propagation step-by-step through the abstraction layer
- No shortcut that bypasses the bottleneck is possible

## Implications

This is actually a **feature, not a bug** for interpretability:
- SoRL's failures are mechanistically predictable (cascade propagation fails at the final carry)
- The error pattern (outputs `099` instead of `100`) reveals exactly where the computation breaks down
- Per-digit accuracy shows SoRL gets MORE digits right even when full-sequence accuracy is lower

## Recommendation

Use C-splits (C3-C6) for the paper's hard-case comparisons, not S-splits. C-splits have:
- Same cascade depth as S-splits
- Diverse answer digits (no shortcut possible)
- Fair comparison between baseline and SoRL

S-splits are still valid for easy cases (S0-S3) where the answer diversity is adequate.

## Data

```
2L1H128d @ 100K:
            Baseline  SoRL    Gap
C3 (hot):    44%      93%    +49pp
C4 (hot):    38%      94%    +56pp
C5 (hot):    33%      85%    +52pp
C6 (hot):    39%      96%    +57pp
S6 (degen):  55%      21%    -34pp  ← shortcut advantage for baseline
```
