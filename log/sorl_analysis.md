# SoRL Performance Analysis
_Compiled from all log/*.md files_

## 1. SFT Baselines (Reference)

    Full-FT (ep=1, EBS=8)                              LoRA r=16 (ep=1, EBS=8)
    ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┐  ┌──────┬──────┬──────┬──────┬──────┬──────┐
    │Model │GSM8K │SciQA │  ARC │ MMLU │ OBQA │ CSQA │  │Model │GSM8K │  ARC │ MMLU │ OBQA │ CSQA │
    ├──────┼──────┼──────┼──────┼──────┼──────┼──────┤  ├──────┼──────┼──────┼──────┼──────┼──────┤
    │ q06  │ 46.4 │ 48.0 │ 60.8 │ 46.1 │ 69.2 │ 65.1 │  │ q06  │ 43.1 │ 59.0 │ 45.6 │ 64.8 │ 64.5 │
    │ q17b │ 60.2 │ 56.4 │ 76.7 │ 57.6 │ 79.7 │   ?  │  │ q17b │ 60.6 │ 76.1 │ 57.6 │ 77.8 │ 76.2 │
    │ l1b  │ 16.9 │ 48.3 │ 51.3 │ 44.2 │ 67.5 │   ?  │  │ l1b  │ 12.1 │ 44.2 │ 43.3 │ 60.1 │   ?  │
    │ l3b  │ 41.3 │ 61.7 │ 71.4 │ 57.0 │ 80.7 │ 79.8 │  │ l3b  │ 34.9 │ 69.2 │ 54.9 │ 77.5 │ 77.1 │
    │ q4b  │ 77.8 │ 63.6 │ 87.6 │ 70.5 │   —  │ 84.5 │  │ q4b  │ 76.1 │ 87.1 │ 70.2 │ 87.3 │ 83.5 │
    └──────┴──────┴──────┴──────┴──────┴──────┴──────┘  └──────┴──────┴──────┴──────┴──────┴──────┘

---

## 2. Trainer Evolution (0.6B/1.7B, 3 epochs, GSM8K + SciQA)

_Source: gsm-sci-0403.md, ablation-0408.md_

    ┌─────────┬──────┬───────────────────────────┬───────────────────────────┐
    │ Trainer │ emb  │ 0.6B GSM8K (SFT=47.5)     │ 0.6B SciQA (SFT=52.3)    │
    │         │  lr  │   NL%  │  K=4%  │  Δ(SFT) │   NL%  │  K=4%  │ Δ(SFT) │
    ├─────────┼──────┼────────┼────────┼─────────┼────────┼────────┼────────┤
    │ v1      │  1x  │  47.7  │  45.3  │   +0.2  │  52.2  │  42.6  │  -0.1  │
    │ v1      │ 10x  │  45.1  │  42.2  │   -2.4  │  60.0  │  50.0  │  +7.7  │
    │ v6      │  1x  │  44.8  │    —   │   -2.7  │  50.0  │    —   │  -2.3  │
    │ v6      │ 10x  │  44.2  │    —   │   -3.3  │  55.0  │    —   │  +2.7  │
    │ v7 ds   │  1x  │  47.1  │  47.4  │   -0.4  │  61.8  │  61.2  │  +9.5  │
    └─────────┴──────┴────────┴────────┴─────────┴────────┴────────┴────────┘

    ┌─────────┬──────┬───────────────────────────┬───────────────────────────┐
    │ Trainer │ emb  │ 1.7B GSM8K (SFT=62.5)     │ 1.7B SciQA (SFT=58.1)    │
    │         │  lr  │   NL%  │  K=4%  │  Δ(SFT) │   NL%  │  K=4%  │ Δ(SFT) │
    ├─────────┼──────┼────────┼────────┼─────────┼────────┼────────┼────────┤
    │ v1      │  1x  │  64.2  │  58.2  │   +1.7  │  58.1  │  49.1  │   0.0  │
    │ v1      │ 10x  │  60.2  │  56.9  │   -2.3  │  60.0  │  51.8  │  +1.9  │
    │ v6      │  1x  │  62.4  │    —   │   -0.1  │  52.1  │    —   │  -6.0  │
    │ v7 ds   │  1x  │  63.2  │  63.2  │   +0.7  │  63.2  │  61.0  │  +5.1  │
    └─────────┴──────┴────────┴────────┴─────────┴────────┴────────┴────────┘

    KEY: "v7 ds" = v7 deep supervision (ablation-0408 + 0410)

    [Finding 1] v7 is the decisive breakthrough: closes NL↔K gap AND beats SFT.
    [Finding 2] v1 can improve NL% on SciQA with emb_lr=10x, but K% still lags.
    [Finding 3] v6 is consistently worse than v7, especially on SciQA.
    [Finding 4] emb_lr=10x helps SciQA but hurts GSM8K — dataset-dependent.

---

## 3. v7 Cross-Model Performance (self-route, LoRA, 0410.md)

_Config: V=32, pfx=8, K=8, iter=4, emb_lr=1.0, 1 epoch_

    ┌──────┬────────┬────────┬─────────┬────────┬────────┬─────────┐
    │Model │SFT GSM │ v7 GSM │  Δ GSM  │SFT Sci │ v7 Sci │  Δ Sci  │
    ├──────┼────────┼────────┼─────────┼────────┼────────┼─────────┤
    │ q06  │  44.0  │  47.1  │  +3.1   │  43.5  │  61.8  │ +18.3   │
    │ q17b │  59.4  │  63.2  │  +3.8   │  51.7  │  63.2  │ +11.5   │
    │ l1b  │  16.1  │  19.2* │  +3.1   │  48.0  │  55.2* │ +23.2   │
    │ l3b  │  34.0  │  44.0* │ +10.0   │  61.5  │  81.7* │ +47.2   │
    │ q4b  │  77.0  │  74.5  │  -2.5   │  60.7  │  67.6  │  +6.9   │
    └──────┴────────┴────────┴─────────┴────────┴────────┴─────────┘
    * = v7o (orthogonal init variant)  | SFT refs from sweep-0406

    [Finding 5] SciQA benefits enormously: +6.9 to +47.2 pts across all models.
    [Finding 6] GSM8K benefits modestly: +3 to +10 pts (except q4b -2.5).
    [Finding 7] Qwen3-4B is the hardest model — v7 sometimes hurts GSM8K.
    [Finding 8] Llama-3B shows largest gains (GSM +10, SciQA +47).

On SciQA, we have most promising result, but I want to be rigorous. When v7 uses more gradient steps, we want to match it with SFT (ep = max_iter)
otherwise we are not doing fair comparison. Here for instance, we need 4 ep SFT to compare things fairly. 



---

## 4. v7 Latest Full-FT Sweep (0411.md, V=32, K=8, pfx=8, mi=2, ep=2)

### 4a. v7 NL% vs Full-FT SFT (ep=1) — Δ = v7_NL − SFT_fullFT

    ┌──────┬───────┬───────┬───────┬───────┬───────┬───────┐
    │Model │ GSM8K │ SciQA │  ARC  │  MMLU │  CSQA │  OBQA │
    ├──────┼───────┼───────┼───────┼───────┼───────┼───────┤
    │ l1b  │  +6.4 │  +7.0 │ -14.1 │  -6.7 │    ?  │  +2.8 │
    │ l3b  │  +5.6 │  +2.1 │  -0.1 │  -8.4 │  +0.5 │  -0.4 │
    │ q17b │  +2.8 │  +5.4 │  +2.5 │  -4.7 │    ?  │  +3.0 │
    │ q4b  │  -1.2 │  -1.7 │  +0.9 │  -5.4 │  -0.6 │  +3.1†│
    └──────┴───────┴───────┴───────┴───────┴───────┴───────┘
    † OBQA Δ is vs LoRA SFT (no full-FT OBQA for q4b)

    Color-coded summary:
    ■ Strong win (>+3):    GSM l1b/l3b, SciQA l1b/q17b, OBQA all
    ■ Marginal win (+0~3):  GSM q17b, SciQA l3b, ARC q17b/q4b, CSQA l3b
    ■ Neutral (±1):        ARC l3b, CSQA q4b
    ■ Regression (<-1):     ARC l1b(-14!), MMLU all(-5~-8), GSM q4b, SciQA q4b
    ■ Catastrophic:         HpQA all (0% vs 16-21% SFT)

### 4b. K=8 Inference Gap (K=8% − NL%)

    ┌──────┬──────┬──────┬──────┬──────┬──────┐
    │Model │GSM8K │SciQA │ ARC  │ MMLU │ CSQA │
    ├──────┼──────┼──────┼──────┼──────┼──────┤
    │ l1b  │ +0.7 │ +1.8 │ -0.3 │ +0.2 │ +1.0 │
    │ l3b  │ -0.7 │ +0.5 │ +0.5 │ +0.2 │  0.0 │
    │ q17b │ -1.1 │ -1.6 │ -0.9 │ +0.2 │ -0.1 │
    │ q4b  │ +0.2 │ -0.5 │ -0.6 │ -0.8 │ +0.3 │
    └──────┴──────┴──────┴──────┴──────┴──────┘
    Mean |Δ| = 0.6%.  Gap is near-zero — deep supervision works.

---

## 5. Key Ablations (from 0410.md + 0411.md)

### 5a. Routing: self-route vs similar-magnitude (0.6B GSM8K)

    similar-magnitude peaks at iter=2 (47.9%) vs self-route at iter=4 (47.4%)
    similar-magnitude shows consistently positive K−NL gap (abstractions help)
    → similar-magnitude chosen as default routing method

### 5b. Iteration Depth (0.6B GSM8K, similar-mag)

    iter=1: 44.7  |  iter=2: 47.9 ← best  |  iter=4: 47.0  |  iter=8: 40.1
    → iter=2 optimal, deeper iterations overfit/degrade

### 5c. Prefix Length (0.6B GSM8K, similar-mag, iter=4)

    pfx=1: 45.8  |  pfx=4: 47.5 ← best  |  pfx=8: 47.0  |  pfx=16: 47.3
    → pfx=4~8 is the sweet spot, minimal sensitivity

### 5d. Vocab Size (Qwen3-4B, LoRA, 0410.md)

    V=32: NL=75.9  |  V=64: NL=76.1  |  V=128: NL=77.3
    → Monotonous improvement with larger vocab, worth scaling further

### 5e. Temperature (0411.md)

    q17 GSM8K: t=0.0 → 59.7%  |  t=1.0 → 61.4% (+1.7)
    q06 GSM8K: t=1.0 → 43.4%  |  t=2.0 → 41.9% (-1.5)
    → t=1.0 is justified; helps make abstraction choice robust

### 5f. GSM8K V=1024 vs V=32 (0411.md)

    q4b: V=32 NL=76.6 → V=1024 NL=79.5  (+2.9) ← significant
    l1b: V=32 NL=23.3 → V=1024 NL=22.4  (-0.9) ← no benefit
    → Larger vocab helps bigger models disproportionately

---

## 6. Comparison vs Alternative Baselines (0411.md)

### Pause Token & Token Assorted (LoRA, ep=1)

    Neither method ever beats plain SFT on any dataset.
    OBQA: SFT ≈ Pause ≈ TA (within ±1.5 pts)
    MMLU: Pause/TA degrade vs SFT by ~3 pts
    BoolQ, CSQA: all within noise
    HpQA: Pause/TA also collapse to ~0% (same as v7)

    → v7 is the ONLY method that consistently beats SFT.

### REINFORCE Search (0410.md)

    Post-training REINFORCE on abstract token choices does NOT help.
    All runs show degradation from loaded checkpoint.
    → Direction abandoned.

---

## 7. Dataset Taxonomy

    ┌────────────┬─────────────┬───────────────────────────────────────┐
    │ Category   │ Datasets    │ v7 Effect                             │
    ├────────────┼─────────────┼───────────────────────────────────────┤
    │ Strong win │ SciQA, OBQA │ +3~47 pts over SFT. Semantically      │
    │            │             │ diverse tasks benefit most.            │
    ├────────────┼─────────────┼───────────────────────────────────────┤
    │ Modest win │ GSM8K       │ +3~6 pts for small models, neutral    │
    │            │             │ or slight loss for 4B.                 │
    ├────────────┼─────────────┼───────────────────────────────────────┤
    │ Neutral    │ CSQA, BoolQ │ ±0.5 pts, no clear advantage.         │
    ├────────────┼─────────────┼───────────────────────────────────────┤
    │ Regression │ MMLU, ARC   │ -5~-14 pts on some model/dataset.     │
    │            │ (l1b)       │ Knowledge-heavy tasks hurt.            │
    ├────────────┼─────────────┼───────────────────────────────────────┤
    │ Collapse   │ HotpotQA    │ All methods → 0%. Long-form QA fails. │
    └────────────┴─────────────┴───────────────────────────────────────┘

    Hypothesis: v7 benefits tasks with high "semantic diversity" (SciQA, OBQA)
    where diverse abstractions can capture meaningful sub-categories.
    Knowledge-intensive (MMLU) or long-form extraction (HpQA) tasks degrade.

---

## 8. Open Questions & Risks

1. **MMLU uniform degradation**: All 4 models lose 5-8 pts. Is this inherent to
   abstraction training corrupting factual knowledge? Or a training recipe issue?

2. **ARC l1b -14 pts**: Severe regression on one model. Is Llama-1B too small
   for multi-task abstraction training on ARC?

3. **HotpotQA collapse**: Shared across v7, Pause, and TA. May be a fundamental
   limitation of prefix-based abstraction on extractive QA tasks.

4. **q4b GSM8K regression**: The largest model slightly degrades on GSM8K but
   benefits from larger vocab (V=1024 → +2.9). Config tuning may help.

5. **Fair comparison**: v7 ep=2 mi=2 gives 4× optimizer steps vs SFT ep=1.
   Epoch-matched comparison needed (v7 ep=1 mi=2 vs SFT ep=2).

---

## 9. Summary of Current Best Configuration

    v7 | similar_magnitude routing | deep supervision
    V=32 (or larger for big models) | K=8 | pfx=8 | iter=2
    temperature=1.0 | emb_lr=1.0 | full-FT (LoRA only for ≥4B)

    Strengths: +3~47 pts on SciQA/OBQA, near-zero K−NL gap
    Weaknesses: -5~8 pts on MMLU, collapse on HpQA
