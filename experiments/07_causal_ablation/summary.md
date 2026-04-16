# Causal Ablation

**Model:** `add_sub_sorl_v1_abs30_K1_100K_2L1H128d`
**Eval set:** canonical N=100 from HuggingFace

## Overall

| Intervention | Accuracy | Delta |
|--------------|----------|-------|
| baseline | 0.0% | +0.0% |
| knockout | 0.0% | +0.0% |
| shuffle | 0.0% | +0.0% |
| random | 0.0% | +0.0% |

![Causal Ablation](fig_causal_ablation.png)

## Interpretation

If abstraction tokens are causally encoding carry information:
- Knockout should drop accuracy (model loses carry signal)
- Shuffle should drop accuracy (wrong carry info at each position)
- Random should drop accuracy (random noise instead of signal)
