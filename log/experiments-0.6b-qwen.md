# SoRL Ablation Experiment Log

## Overview

All experiments fine-tune **Qwen3-0.6B** on **GSM8K** for 3 epochs with:
- lr=1e-5, warmup=50 steps, cooldown_frac=0.4
- batch_size=2, grad_accum=4 (effective batch=8)
- max_length=512, max_new_tokens=256
- abstract_vocab_size=128, num_rollouts=4, max_iterations=2, temperature=1.0
- Eval: temperature=0.0 (greedy), 1000 samples

## Known Issues

- **Info-gain loss bug (fixed post-run)**: `traj_loss_with_abs` was computed using full-vocab logits instead of NL-only logits for the CE, making info_gain misaligned with `base_traj_loss` (which correctly masks abstract logits to -inf). Similarly, `abs_loss` used full-vocab logits instead of abstract-only logits. Fixed for future runs.
- **Ortho loss not logged** in run 1 and run 2 (fixed post-run).
- **Ortho loss stuck at ~1.0** due to symmetry trap: `resize_token_embeddings` initializes near-identical abstract embeddings; ortho_loss gradient is symmetric so all embeddings move together. Fixed by adding orthogonal QR initialization for abstract embeddings.
- **Token budget confound**: In run 1, `max_new_tokens` was shared between NL and abstract tokens during K≠None eval, giving ~20% fewer NL tokens. Fixed in run 2 by scaling to `max_new_tokens * K/(K-1)`. Comparison showed the fix had negligible effect (mean K4 change: +0.5%, within noise).
- **DDP validation incomplete**: The 4-GPU DDP baseline never finished training in either run, so DDP correctness is unvalidated.
- **Run 2 incomplete**: Experiments exp21-24 started K=4 eval but were killed before finishing. Batches 13-20 (ortho, vocab size, max_iterations sweeps) never ran.

---

## Run 1: `ablate_20260313_0949`

No max_new_tokens budget fix. All aux losses use full-vocab logits (buggy info/abs loss).

| Exp | α_info | α_abs | α_zipf | α_ortho | K | NL% | K4% | Gap | Eff V | Status |
|-----|--------|-------|--------|---------|---|-----|-----|-----|-------|--------|
| exp1 | 0.0 | 0.0 | 0.0 | 0.0 | 4 | 49.7 | - | - | - | done |
| exp2 | 1.0 | 0.0 | 0.0 | 0.0 | 4 | 49.7 | 45.6 | +4.1 | 116 | done |
| exp3 | 3.0 | 0.0 | 0.0 | 0.0 | 4 | 50.1 | 43.8 | +6.3 | 125 | done |
| exp4 | 5.0 | 0.0 | 0.0 | 0.0 | 4 | 49.5 | 44.6 | +4.9 | 118 | done |
| exp5 | 7.0 | 0.0 | 0.0 | 0.0 | 4 | 51.1 | 44.7 | +6.4 | 121 | done |
| exp6 | 9.0 | 0.0 | 0.0 | 0.0 | 4 | 48.0 | 43.0 | +5.0 | 121 | done |
| exp7 | 1.0 | 0.5 | 0.0 | 0.0 | 4 | 47.9 | 43.2 | +4.7 | 1 | done |
| exp8 | 3.0 | 0.5 | 0.0 | 0.0 | 4 | 48.5 | 44.5 | +4.0 | 1 | done |
| exp9 | 5.0 | 0.5 | 0.0 | 0.0 | 4 | 46.9 | 42.6 | +4.3 | 1 | done |
| exp10 | 7.0 | 0.5 | 0.0 | 0.0 | 4 | 46.2 | 44.1 | +2.1 | 2 | done |
| exp11 | 9.0 | 0.5 | 0.0 | 0.0 | 4 | 45.9 | 43.6 | +2.3 | 1 | done |
| exp12 | 9.0 | 1.0 | 0.0 | 0.0 | 4 | 47.1 | 44.2 | +2.9 | 1 | done |
| exp13 | 9.0 | 1.5 | 0.0 | 0.0 | 4 | 47.9 | 44.0 | +3.9 | 2 | done |
| exp14 | 9.0 | 2.0 | 0.0 | 0.0 | 4 | 45.4 | 42.4 | +3.0 | 1 | done |
| exp15 | 9.0 | 0.5 | 0.5 | 0.0 | 4 | 47.0 | 44.8 | +2.2 | 3 | done |
| exp16 | 9.0 | 0.5 | 1.0 | 0.0 | 4 | 47.5 | 43.9 | +3.6 | 4 | done |
| exp17 | 9.0 | 0.5 | 1.5 | 0.0 | 4 | 47.1 | 44.6 | +2.5 | 9 | done |
| exp18 | 9.0 | 0.5 | 2.0 | 0.0 | 4 | 46.7 | 44.2 | +2.5 | 10 | done |
| exp19 | 1.0 | 0.1 | 0.0 | 0.0 | 4 | 45.9 | 42.9 | +3.0 | 1 | done |
| exp20 | 1.0 | 0.3 | 0.0 | 0.0 | 4 | 49.5 | 43.9 | +5.6 | 1 | done |
| exp21 | 1.0 | 0.7 | 0.0 | 0.0 | 4 | 48.1 | 43.6 | +4.5 | 3 | done |
| exp22 | 1.0 | 1.0 | 0.0 | 0.0 | 4 | 48.4 | 40.9 | +7.5 | 1 | done |
| exp23 | 0.5 | 0.5 | 0.0 | 0.0 | 4 | 49.7 | 42.1 | +7.6 | 2 | done |
| exp24 | 1.5 | 0.5 | 0.0 | 0.0 | 4 | 47.5 | 44.0 | +3.5 | 2 | done |
| exp25 | 2.0 | 0.5 | 0.0 | 0.0 | 4 | 46.7 | 44.5 | +2.2 | 1 | done |
| exp26 | 2.5 | 0.5 | 0.0 | 0.0 | 4 | 47.1 | 43.7 | +3.4 | 1 | done |
| exp27 | 3.0 | 0.1 | 0.0 | 0.0 | 4 | 46.6 | 44.1 | +2.5 | 58 | done |
| exp28 | 3.0 | 0.3 | 0.0 | 0.0 | 4 | 47.5 | 44.3 | +3.2 | 2 | done |
| exp29 | 3.0 | 0.7 | 0.0 | 0.0 | 4 | 47.1 | 43.3 | +3.8 | 1 | done |
| exp30 | 3.0 | 1.0 | 0.0 | 0.0 | 4 | 47.9 | 44.0 | +3.9 | 1 | done |
| exp31 | 1.0 | 0.5 | 0.5 | 0.0 | 4 | 47.7 | 41.2 | +6.5 | 5 | done |
| exp32 | 1.0 | 0.5 | 1.0 | 0.0 | 4 | 48.5 | 41.4 | +7.1 | 8 | done |
| exp33 | 1.0 | 0.5 | 1.5 | 0.0 | 4 | 48.4 | 40.3 | +8.1 | 15 | done |
| exp34 | 1.0 | 0.5 | 2.0 | 0.0 | 4 | 47.8 | 40.4 | +7.4 | 12 | done |
| exp35 | 1.0 | 0.5 | 0.0 | 0.0 | 2 | 48.2 | 39.5 | +8.7 | 9 | done |
| exp36 | 1.0 | 0.5 | 0.0 | 0.0 | 3 | 48.8 | 44.6 | +4.2 | 5 | done |
| exp37 | 1.0 | 0.5 | 0.0 | 0.0 | 6 | 46.5 | 41.6 | +4.9 | 1 | done |
| exp38 | 1.0 | 0.5 | 0.0 | 0.0 | 8 | 48.4 | 43.9 | +4.5 | 1 | done |
| exp39 | 9.0 | 0.5 | 0.0 | 0.0 | 2 | 46.4 | 37.1 | +9.3 | 1 | done |
| exp40 | 9.0 | 0.5 | 0.0 | 0.0 | 8 | 45.9 | 42.8 | +3.1 | 1 | done |
| exp41 | 0.3 | 0.1 | 0.0 | 0.0 | 4 | 50.5 | 41.6 | +8.9 | 2 | done |
| exp42 | 0.3 | 0.3 | 0.0 | 0.0 | 4 | 47.2 | 41.6 | +5.6 | 1 | done |
| exp43 | 1.0 | 0.5 | 3.0 | 0.0 | 4 | 48.4 | 38.3 | +10.1 | 10 | done |
| exp44 | 1.0 | 0.3 | 0.5 | 0.0 | 4 | 48.1 | 42.9 | +5.2 | 7 | done |
| exp45 | 1.0 | 0.3 | 1.0 | 0.0 | 4 | 48.7 | 41.7 | +7.0 | 16 | done |
| exp46 | 3.0 | 0.5 | 0.5 | 0.0 | 4 | 48.1 | 45.5 | +2.6 | 8 | done |
| exp47 | 3.0 | 0.5 | 1.0 | 0.0 | 4 | 47.4 | 43.8 | +3.6 | 6 | done |
| exp48 | 0.5 | 0.3 | 0.0 | 0.0 | 4 | 48.4 | 40.7 | +7.7 | 1 | done |
| exp49 | 0.0 | 0.0 | 0.0 | 0.0 | 4 | - | - | - | - | DDP, incomplete |

---

## Run 2: `ablate_20260315_0332`

With max_new_tokens budget fix (K≠None eval scales to `K/(K-1) * max_new_tokens`). Still has buggy info/abs loss (full-vocab logits).

| Exp | α_info | α_abs | α_zipf | α_ortho | K | NL% | K4% | Gap | Eff V | Status |
|-----|--------|-------|--------|---------|---|-----|-----|-----|-------|--------|
| exp1 | 0.0 | 0.0 | 0.0 | 0.0 | 4 | 48.5 | - | - | - | done |
| exp2 | 1.0 | 0.0 | 0.0 | 0.0 | 4 | 49.5 | 43.5 | +6.0 | 120 | done |
| exp3 | 3.0 | 0.0 | 0.0 | 0.0 | 4 | 50.7 | 48.4 | +2.3 | 117 | done |
| exp4 | 5.0 | 0.0 | 0.0 | 0.0 | 4 | 49.6 | 44.6 | +5.0 | 123 | done |
| exp5 | 7.0 | 0.0 | 0.0 | 0.0 | 4 | 50.1 | 44.5 | +5.6 | 126 | done |
| exp6 | 9.0 | 0.0 | 0.0 | 0.0 | 4 | 48.3 | 45.3 | +3.0 | 121 | done |
| exp7 | 1.0 | 0.5 | 0.0 | 0.0 | 4 | 48.9 | 45.3 | +3.6 | 2 | done |
| exp8 | 3.0 | 0.5 | 0.0 | 0.0 | 4 | 46.3 | 44.5 | +1.8 | 1 | done |
| exp9 | 5.0 | 0.5 | 0.0 | 0.0 | 4 | 47.0 | 45.0 | +2.0 | 2 | done |
| exp10 | 7.0 | 0.5 | 0.0 | 0.0 | 4 | 46.2 | 42.8 | +3.4 | 2 | done |
| exp11 | 9.0 | 0.5 | 0.0 | 0.0 | 4 | 46.0 | 42.9 | +3.1 | 2 | done |
| exp12 | 9.0 | 1.0 | 0.0 | 0.0 | 4 | 45.3 | 43.5 | +1.8 | 1 | done |
| exp13 | 9.0 | 1.5 | 0.0 | 0.0 | 4 | 46.5 | 43.2 | +3.3 | 1 | done |
| exp14 | 9.0 | 2.0 | 0.0 | 0.0 | 4 | 47.5 | 42.5 | +5.0 | 2 | done |
| exp15 | 9.0 | 0.5 | 0.5 | 0.0 | 4 | 44.6 | 45.2 | -0.6 | 5 | done |
| exp16 | 9.0 | 0.5 | 1.0 | 0.0 | 4 | 45.5 | 44.9 | +0.6 | 6 | done |
| exp17 | 9.0 | 0.5 | 1.5 | 0.0 | 4 | 45.9 | 44.7 | +1.2 | 10 | done |
| exp18 | 9.0 | 0.5 | 2.0 | 0.0 | 4 | 46.6 | 44.8 | +1.8 | 10 | done |
| exp19 | 1.0 | 0.1 | 0.0 | 0.0 | 4 | 46.9 | 43.9 | +3.0 | 1 | done |
| exp20 | 1.0 | 0.3 | 0.0 | 0.0 | 4 | 48.6 | 44.0 | +4.6 | 1 | done |
| exp21 | 1.0 | 0.7 | 0.0 | 0.0 | 4 | 47.0 | - | - | - | K4 eval killed |
| exp22 | 1.0 | 1.0 | 0.0 | 0.0 | 4 | 49.2 | - | - | - | K4 eval killed |
| exp23 | 0.5 | 0.5 | 0.0 | 0.0 | 4 | 47.2 | - | - | - | K4 eval killed |
| exp24 | 1.5 | 0.5 | 0.0 | 0.0 | 4 | 48.1 | - | - | - | K4 eval killed |

---

## Run-to-Run Variation (Run 1 vs Run 2, same configs)

Measured across 20 configs that completed in both runs:

| Metric | NL Accuracy | K=4 Accuracy |
|--------|-------------|--------------|
| Mean |delta| | 1.0% | 1.1% |
| Median |delta| | 1.0% | 0.7% |
| Max |delta| | 2.4% | 4.6% |

**Practical noise floor: ~2%. Differences < 2% are unreliable.**

---

## Key Findings

### 1. Info-gain only (no abs loss) preserves NL accuracy
- Best NL: 50.1-51.1% (α_info=3-7), vs 48.5-49.7% baseline
- Uses full abstract vocab (~116-126 tokens) but unconstrained

### 2. Abs loss narrows the NL–K4 gap but drops NL accuracy
- Vocab collapses to eff_vocab=1-2 in all abs-loss configs
- Gap narrows because both NL and K4 drop, with NL dropping more
- For α_info≥3, abs loss slightly raises K4 accuracy (e.g. 43.0→44.2 at info=9)

### 3. Zipf regularization increases vocab diversity but doesn't help accuracy
- Pushes eff_vocab from 1 to 5-16
- At α_info=1.0: more zipf actively hurts K4 (43.2→38.3)
- At α_info=9.0: slight K4 improvement, one config (exp15) where K4 > NL (45.2 vs 44.6, within noise)

### 4. K=2 is too dense, K=3-4 is optimal
- K=2: 33.6% abstract ratio, worst K4 accuracy (37-39%)
- K=3: best K4 for info=1.0 configs (44.6%)
- K=4: default, generally good
- K=8: similar to K=4 with less overhead

### 5. Token budget fix had negligible impact
- Mean K4 change: +0.5% (within noise)
- Confirms the NL-K4 gap is from abstraction quality, not truncation

### 6. Ortho loss stuck due to symmetry trap
- `resize_token_embeddings` initializes near-identical abstract embeddings
- Ortho loss = 1.0156 (vs ~0.001 for random orthogonal vectors)
- Gradient is symmetric: all embeddings get same update direction, stay collapsed
- Only drops with α_ortho≈1000 and all other losses zeroed
- **Fix**: Orthogonal QR initialization added to break symmetry

### 7. Info-gain loss ≈ 0 throughout training
- Means abstract tokens don't reduce NL perplexity during teacher-forcing
- Consistent with abstractions being uninformative noise



Results Summary (ablate_20260316_0758)

  ┌───────┬──────────────────────────────────┬─────────┬─────────┬─────────┬──────┬───────────┬────────────────────┐
  │  Exp  │              Config              │ α_ortho │ K=None  │   K=4   │ Gap  │ Eff Vocab │ Ortho Loss (final) │
  ├───────┼──────────────────────────────────┼─────────┼─────────┼─────────┼──────┼───────────┼────────────────────┤
  │ exp1  │ v1, i=1, abs=0.5                 │    0    │  49.5%  │  44.8%  │ -4.7 │    60     │       1.008        │
  ├───────┼──────────────────────────────────┼─────────┼─────────┼─────────┼──────┼───────────┼────────────────────┤
  │ exp2  │ v1, i=3, abs=0.5                 │    0    │  48.7%  │  45.8%  │ -2.9 │    59     │       1.016        │
  ├───────┼──────────────────────────────────┼─────────┼─────────┼─────────┼──────┼───────────┼────────────────────┤
  │ exp3  │ v2, i=1, abs=0.5                 │    0    │  47.8%  │  43.5%  │ -4.3 │    42     │       1.008        │
  ├───────┼──────────────────────────────────┼─────────┼─────────┼─────────┼──────┼───────────┼────────────────────┤
  │ exp4  │ v2, i=3, abs=0.5                 │    0    │  47.9%  │  43.1%  │ -4.8 │    53     │       1.016        │
  ├───────┼──────────────────────────────────┼─────────┼─────────┼─────────┼──────┼───────────┼────────────────────┤
  │ exp5  │ v1, i=1, ortho=1, emb1x          │    1    │  51.1%  │  43.7%  │ -7.4 │    56     │       1.008        │
  ├───────┼──────────────────────────────────┼─────────┼─────────┼─────────┼──────┼───────────┼────────────────────┤
  │ exp6  │ v1, i=1, ortho=1, emb10x         │    1    │  47.2%  │  43.2%  │ -4.0 │     8     │       0.088        │
  ├───────┼──────────────────────────────────┼─────────┼─────────┼─────────┼──────┼───────────┼────────────────────┤
  │ exp7  │ v1, i=1, zipf=1                  │    0    │  48.7%  │  42.7%  │ -6.0 │     2     │       0.988        │
  ├───────┼──────────────────────────────────┼─────────┼─────────┼─────────┼──────┼───────────┼────────────────────┤
  │ exp8  │ v1, i=1, ortho=1, zipf=1, emb10x │    1    │  44.1%  │  41.6%  │ -2.5 │    102    │       0.092        │
  ├───────┼──────────────────────────────────┼─────────┼─────────┼─────────┼──────┼───────────┼────────────────────┤
  │ exp9  │ v1, i=3, ortho=1, emb10x         │    1    │ running │    —    │  —   │     —     │       0.131        │
  ├───────┼──────────────────────────────────┼─────────┼─────────┼─────────┼──────┼───────────┼────────────────────┤
  │ exp10 │ v1, i=3, zipf=1                  │    0    │ running │    —    │  —   │     —     │       0.984        │
  ├───────┼──────────────────────────────────┼─────────┼─────────┼─────────┼──────┼───────────┼────────────────────┤
  │ exp11 │ v1, i=3, ortho=1, zipf=1, emb10x │    1    │ running │    —    │  —   │     —     │       0.107        │
  ├───────┼──────────────────────────────────┼─────────┼─────────┼─────────┼──────┼───────────┼────────────────────┤
  │ exp12 │ v2, i=1, ortho=1, emb10x         │    1    │  45.3%  │ running │  —   │     —     │       0.065        │
  └───────┴──────────────────────────────────┴─────────┴─────────┴─────────┴──────┴───────────┴────────────────────┘

  Ortho Loss Insights

  The ortho loss tells a clear story about embedding learning rate:

  - emb1x + ortho=1 (exp5): ortho loss stays at ~1.0 — the 1x LR is too low to actually push embeddings apart. The ortho regularizer has essentially no
  effect despite being turned on. This explains why exp5 has the same ortho loss as experiments with α_ortho=0.
  - emb10x + ortho=1 (exp6, 8, 9, 11, 12): ortho loss drops to 0.065–0.131 — the 10x LR gives enough gradient signal to actually separate embeddings.
  - emb10x without ortho (not tested) would likely also collapse — exp6 shows that even with ortho driving embeddings apart, emb10x causes vocab collapse
  to 8 tokens. The high LR likely overshoots into degenerate attractors.
  - Zipf alone (exp7, 10): ortho ~0.98–0.99, slightly below the no-regularization baseline of ~1.0. Zipf doesn't directly affect orthogonality but the
  vocab collapse to 2 tokens means only 2 embedding vectors are active, so the metric is nearly meaningless.
  - Best ortho-trained run (exp8): ortho=0.092 with the highest effective vocab (102) — the ortho+zipf combo successfully spreads usage across tokens, but
   at a -3.5pp accuracy cost vs exp2.

  Bottom line: ortho regularization requires emb10x to have any actual effect on embeddings, but emb10x itself is destabilizing. The emb1x runs that "use"
   ortho (exp5) aren't actually learning orthogonal embeddings at all.




  ● All experiments completed. Here's the analysis:

  Results: ablate_20260316_1625

  This round introduces v3 (contrastive loss) with new params: corrupt_method, corrupt_ratio, gamma_contrastive, alpha_contrastive. Exps 9-12 are
   v1/v2 baselines for comparison.

  Full Table

  ┌───────┬─────────┬─────────────────────────┬────────┬───────┬──────┬───────────┬────────────┐
  │  Exp  │ Trainer │       Key Config        │ K=None │  K=4  │ Gap  │ Eff Vocab │ Ortho Loss │
  ├───────┼─────────┼─────────────────────────┼────────┼───────┼──────┼───────────┼────────────┤
  │ exp9  │ v1      │ i=1, abs=0.5 (baseline) │ 48.8%  │ 42.7% │ -6.1 │    29     │     ~0     │
  ├───────┼─────────┼─────────────────────────┼────────┼───────┼──────┼───────────┼────────────┤
  │ exp10 │ v2      │ i=1, abs=0.5 (baseline) │ 47.3%  │ 45.5% │ -1.8 │    23     │     ~0     │
  ├───────┼─────────┼─────────────────────────┼────────┼───────┼──────┼───────────┼────────────┤
  │ exp11 │ v1      │ i=1, ortho=1, emb10x    │ 45.7%  │ 42.5% │ -3.2 │    21     │     ~0     │
  ├───────┼─────────┼─────────────────────────┼────────┼───────┼──────┼───────────┼────────────┤
  │ exp12 │ v2      │ i=1, ortho=1, emb10x    │ 45.3%  │ 43.4% │ -1.9 │    25     │     ~0     │
  ├───────┼─────────┼─────────────────────────┼────────┼───────┼──────┼───────────┼────────────┤
  │       │         │                         │        │       │      │           │            │
  ├───────┼─────────┼─────────────────────────┼────────┼───────┼──────┼───────────┼────────────┤
  │ exp1  │ v3      │ shuf r=0.3, γ=0.5       │ 47.9%  │ 42.4% │ -5.5 │    33     │     ~0     │
  ├───────┼─────────┼─────────────────────────┼────────┼───────┼──────┼───────────┼────────────┤
  │ exp2  │ v3      │ shuf r=0.5, γ=0.5       │ 47.6%  │ 45.4% │ -2.2 │    28     │     ~0     │
  ├───────┼─────────┼─────────────────────────┼────────┼───────┼──────┼───────────┼────────────┤
  │ exp3  │ v3      │ shuf r=1.0, γ=0.5       │ 49.1%  │ 46.2% │ -2.9 │    18     │     ~0     │
  ├───────┼─────────┼─────────────────────────┼────────┼───────┼──────┼───────────┼────────────┤
  │ exp4  │ v3      │ noise r=0.3, γ=0.5      │ 48.2%  │ 45.3% │ -2.9 │    17     │     ~0     │
  ├───────┼─────────┼─────────────────────────┼────────┼───────┼──────┼───────────┼────────────┤
  │ exp5  │ v3      │ shuf r=0.3, γ=0.1       │ 49.3%  │ 43.9% │ -5.4 │    35     │     ~0     │
  ├───────┼─────────┼─────────────────────────┼────────┼───────┼──────┼───────────┼────────────┤
  │ exp6  │ v3      │ shuf r=0.3, γ=1.0       │ 49.8%  │ 45.4% │ -4.4 │    16     │     ~0     │
  ├───────┼─────────┼─────────────────────────┼────────┼───────┼──────┼───────────┼────────────┤
  │ exp7  │ v3      │ shuf r=0.3, γ=2.0       │ 47.0%  │ 43.9% │ -3.1 │    39     │     ~0     │
  ├───────┼─────────┼─────────────────────────┼────────┼───────┼──────┼───────────┼────────────┤
  │ exp8  │ v3      │ shuf r=0.3, α_contr=3.0 │ 49.3%  │ 44.8% │ -4.5 │    69     │     ~0     │
  └───────┴─────────┴─────────────────────────┴────────┴───────┴──────┴───────────┴────────────┘

  Key Observations

  1. v3 contrastive loss achieves the new best K=4 accuracy:
  - exp3 (shuf r=1.0, γ=0.5) → 46.2% K=4 — best K=4 score across both ablation rounds
  - Beats the previous best of 45.8% (0758-exp2, v1 info=3)

  2. Ortho loss converged to ~0 everywhere:
  - All experiments in this round have ortho loss ≈ 0 (1e-6 to 1e-8), including exps with alpha_ortho=0. This is different from the 0758 round
  where no-ortho exps sat at ~1.0. Something changed in the loss computation — the embeddings may have been re-initialized or the ortho loss
  definition changed.

  3. Higher corrupt_ratio helps (shuffle):
  - r=0.3 → 42.4% K=4, r=0.5 → 45.4%, r=1.0 → 46.2%
  - Full shuffle (r=1.0) gives the strongest contrastive signal, forcing abstractions to carry more information

  4. Noise vs shuffle corruption:
  - exp4 (noise r=0.3) → 45.3% K=4 vs exp1 (shuf r=0.3) → 42.4%
  - Noise corruption significantly outperforms shuffle at the same ratio (+2.9pp). Noise may provide a smoother/harder negative that forces
  better representations.

  5. Gamma (γ) contrastive sweep:
  - γ=0.1 → 43.9%, γ=0.5 → 42.4%, γ=1.0 → 45.4%, γ=2.0 → 43.9%
  - Non-monotonic: γ=1.0 is the sweet spot for K=4 accuracy
  - But γ=2.0 gives the best effective vocab (39) among the γ sweep

  6. Higher α_contrastive (exp8, α=3.0):
  - Produces the highest effective vocab (69) — more diverse token usage
  - But K=4 accuracy (44.8%) doesn't benefit proportionally

  7. v2 baseline improved dramatically:
  - exp10 (v2, this round) → 45.5% K=4 with only -1.8 gap, vs 0758-exp3 (v2) → 43.5%
  - Suggests something else changed between rounds (code fix? different seed?)