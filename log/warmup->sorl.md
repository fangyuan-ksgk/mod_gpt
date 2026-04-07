# Warmup, Jacobi & Masked Trajectory Loss Experiments

## Methods

Three auxiliary mechanisms tested on top of SoRLv3 (shuffle corruption + hinge loss):

1. **SFT Warmup** — Cluster K-gram NL embeddings into |V| centers via K-means, initialize
   abstract embedding table, then SFT the model to predict with these pinned abstractions
   before SoRL search begins. Goal: teach "what can be said" before "searching what to say."

2. **Masked Trajectory Loss (m_traj)** — After searching for abstractions on clean data,
   replace a fraction of NL tokens with a fixed mask token, then train the model to predict
   the original NL targets. Forces reliance on abstract tokens (the only clean signal left).
   Fixed mask token >> random NL token for corruption.

3. **Jacobi Loss** — SoRL search uses Jacobi decoding (parallel), but CE loss only trains
   AR generation. Jacobi loss trains the model to take `[t t t <mask_abs> t t t <mask_abs>]`
   and predict `[t t t a t t t a]`, directly improving the search mechanism.

---

## Sweep 1: Qwen3-1.7B (GSM8K, 1.3K val, emb_lr_mult=10.0)

  ┌───────┬──────────────────────────────┬──────┬──────┬───────┐
  │  Exp  │ Config                       │  NL% │ K=4% │  Gap  │
  ├───────┼──────────────────────────────┼──────┼──────┼───────┤
  │       │ Warmup → v3 (wu=500)         │      │      │       │
  ├───────┼──────────────────────────────┼──────┼──────┼───────┤
  │ exp1  │ wu jacobi=0.5 → v3           │ 50.0 │ 40.5 │   9.5 │
  │ exp2  │ wu vanilla → v3              │ 53.0 │ 43.8 │   9.2 │
  │ exp3  │ wu jacobi=1.0 → v3           │ 57.1 │ 41.4 │  15.7 │
  │ exp4  │ wu jacobi=0.5 → v3+jacobi    │ 49.3 │ 42.5 │   6.8 │
  ├───────┼──────────────────────────────┼──────┼──────┼───────┤
  │       │ Warmup steps                 │      │      │       │
  ├───────┼──────────────────────────────┼──────┼──────┼───────┤
  │ exp5  │ wu=250, jacobi=0.5 → v3      │ 50.9 │ 43.4 │   7.5 │
  │ exp6  │ wu=1000, jacobi=0.5 → v3     │ 42.2 │ 34.6 │   7.6 │
  │ exp7  │ wu=250, vanilla → v3         │ 53.7 │ 47.4 │   6.3 │
  │ exp8  │ wu=1000, vanilla → v3        │ 30.3 │ 31.1 │  -0.8 │
  ├───────┼──────────────────────────────┼──────┼──────┼───────┤
  │       │ mask_traj                    │      │      │       │
  ├───────┼──────────────────────────────┼──────┼──────┼───────┤
  │ exp9  │ mtraj=1.0                    │ 64.4 │ 59.6 │   4.8 │
  │*exp10 │ jacobi=0.5 + mtraj=1.0      │ 64.0 │ 60.6 │   3.4*│
  │ exp11 │ mtraj=0.5                    │ 62.5 │ 59.4 │   3.1 │
  │ exp12 │ wu=500 → v3+mtraj=1.0        │ 58.6 │ 49.3 │   9.3 │
  └───────┴──────────────────────────────┴──────┴──────┴───────┘

### Inner-monologue diversity (1.7B)

  ┌────────────────────────┬───────┬───────┬──────┬───────────────────┐
  │ Exp                    │ Vocab │ Top3% │ K=4% │ Distribution      │
  ├────────────────────────┼───────┼───────┼──────┼───────────────────┤
  │                        │  No warmup                               │
  ├────────────────────────┼───────┼───────┼──────┼───────────────────┤
  │ exp9  (mtraj=1.0)      │    40 │  89.2 │ 59.6 │ 3-token dominant  │
  │ exp10 (jacobi+mtraj)   │    24 │  99.9 │ 60.6 │ 1-token collapse  │
  │ exp11 (mtraj=0.5)      │    16 │ 100.0 │ 59.4 │ 2-token collapse  │
  ├────────────────────────┼───────┼───────┼──────┼───────────────────┤
  │                        │  Warmup → v3                             │
  ├────────────────────────┼───────┼───────┼──────┼───────────────────┤
  │ exp1  (wu jacobi)      │    67 │  59.9 │ 40.5 │ 6 tokens spread   │
  │ exp2  (wu vanilla)     │    68 │  56.1 │ 43.8 │ 6 tokens spread   │
  │ exp3  (wu jacobi=1.0)  │    60 │  67.7 │ 41.4 │ 5 tokens spread   │
  │ exp4  (wu+jacobi)      │    43 │  83.4 │ 42.5 │ semi-spread       │
  │ exp12 (wu+mtraj)       │    76 │  61.1 │ 49.3 │ 7 tokens spread   │
  ├────────────────────────┼───────┼───────┼──────┼───────────────────┤
  │                        │  Warmup steps                            │
  ├────────────────────────┼───────┼───────┼──────┼───────────────────┤
  │ exp5  (wu=250)         │    52 │  64.7 │ 43.4 │ 5 tokens spread   │
  │ exp6  (wu=1000)        │    98 │  63.7 │ 34.6 │ 7+ tokens spread  │
  │ exp7  (wu=250 vanilla) │    51 │  63.6 │ 47.4 │ 5 tokens spread   │
  │ exp8  (wu=1000 vanilla)│   100 │  65.9 │ 31.1 │ 7+ tokens spread  │
  └────────────────────────┴───────┴───────┴──────┴───────────────────┘

### Takeaways (1.7B)

1. **Warmup is catastrophic.** All warmup→v3 land 30-53% NL vs 64% baseline. More steps = worse
   (wu=1000 at 30-42% vs wu=250 at 50-54%). Warmup damages base NL ability before SoRL starts.
2. **mask_traj is the strongest signal.** exp10 (jacobi+mtraj) achieves best K=4=60.6%,
   gap=3.4% — the only config that meaningfully improves K=4 over v3 baseline (58.8%).
3. **Warmup diversity is the bad kind.** Warmup spreads the vocab (43-100 effective tokens,
   top3%=56-84%) but this inversely correlates with accuracy. The SFT phase teaches spread-out
   but uncertain embeddings that SoRL can't recover from. More warmup = more spread = worse.
4. **Best configs have collapsed vocabs.** exp10 (K=4=60.6%) uses just 24 effective tokens,
   top3%=99.9%. Confident concentration >> uncertain diversity.

---

## Sweep 2: Qwen3-0.6B (GSM8K, 1.3K val, emb_lr_mult=1.0)

Base: v3, traj=1.0, abs=0.5, hinge(shuffle, r=1.0, γ=0.5)

### Pure SoRL + Jacobi (3 epochs)

  ┌─────┬──────────────────────┬──────┬──────┬─────┬───────┬───────┬─────────┐
  │ Exp │ Config               │  NL% │ K=4% │ Gap │ Vocab │ Top3% │ AbsLoss │
  ├─────┼──────────────────────┼──────┼──────┼─────┼───────┼───────┼─────────┤
  │   1 │ v3 baseline          │ 47.0 │ 43.8 │ 3.2 │    24 │  99.9 │   0.011 │
  ├─────┼──────────────────────┼──────┼──────┼─────┼───────┼───────┼─────────┤
  │   2 │ v3 + jacobi=0.5      │ 45.1 │ 42.6 │ 2.5 │    55 │  96.0 │   0.080 │
  ├─────┼──────────────────────┼──────┼──────┼─────┼───────┼───────┼─────────┤
  │   3 │ v3 + jacobi=1.0      │ 45.2 │ 43.4 │ 1.8 │    39 │  99.9 │   0.012 │
  ├─────┼──────────────────────┼──────┼──────┼─────┼───────┼───────┼─────────┤
  │   4 │ v3 + jacobi=0.25     │ 46.9 │ 41.1 │ 5.8 │    16 │ 100.0 │   0.005 │
  └─────┴──────────────────────┴──────┴──────┴─────┴───────┴───────┴─────────┘

### Warmup 500 steps → SoRL (2 epochs)

  ┌─────┬────────────────────────────┬──────┬──────┬─────┬───────┬───────┬─────────┐
  │ Exp │ Config                     │  NL% │ K=4% │ Gap │ Vocab │ Top3% │ AbsLoss │
  ├─────┼────────────────────────────┼──────┼──────┼─────┼───────┼───────┼─────────┤
  │   5 │ wu(jacobi=0.5) → v3        │ 37.6 │ 34.9 │ 2.7 │    74 │  58.5 │   0.790 │
  ├─────┼────────────────────────────┼──────┼──────┼─────┼───────┼───────┼─────────┤
  │   6 │ wu(vanilla) → v3           │ 41.3 │ 36.6 │ 4.7 │    77 │  68.7 │   0.834 │
  ├─────┼────────────────────────────┼──────┼──────┼─────┼───────┼───────┼─────────┤
  │   7 │ wu(jacobi=1.0) → v3        │ 38.7 │ 36.9 │ 1.8 │    70 │  70.2 │   0.750 │
  ├─────┼────────────────────────────┼──────┼──────┼─────┼───────┼───────┼─────────┤
  │   8 │ wu(jacobi=0.5) → v3+jacobi │ 33.9 │ 27.5 │ 6.4 │    68 │  67.2 │   0.748 │
  └─────┴────────────────────────────┴──────┴──────┴─────┴───────┴───────┴─────────┘

### Warmup length sweep

  ┌─────┬──────────────────────────────┬──────┬──────┬─────┬───────┬───────┬─────────┐
  │ Exp │ Config                       │  NL% │ K=4% │ Gap │ Vocab │ Top3% │ AbsLoss │
  ├─────┼──────────────────────────────┼──────┼──────┼─────┼───────┼───────┼─────────┤
  │   9 │ wu=250(jacobi) → v3 3ep      │ 41.3 │ 38.7 │ 2.6 │    66 │  73.3 │   0.618 │
  ├─────┼──────────────────────────────┼──────┼──────┼─────┼───────┼───────┼─────────┤
  │  10 │ wu=1000(jacobi) → v3 1ep     │ 35.9 │ 33.7 │ 2.2 │    98 │  52.6 │   1.624 │
  ├─────┼──────────────────────────────┼──────┼──────┼─────┼───────┼───────┼─────────┤
  │  11 │ wu=250(vanilla) → v3 3ep     │ 37.6 │ 37.6 │ 0.0 │    64 │  76.0 │   0.612 │
  ├─────┼──────────────────────────────┼──────┼──────┼─────┼───────┼───────┼─────────┤
  │  12 │ wu=1000(vanilla) → v3 1ep    │ 31.6 │ 31.6 │ 0.0 │    96 │  54.1 │   1.578 │
  └─────┴──────────────────────────────┴──────┴──────┴─────┴───────┴───────┴─────────┘

### Masked traj loss (fixed mode, ratio=0.3)

  ┌──────┬─────────────────────────────┬──────┬──────┬──────┬───────┬───────┬─────────┐
  │  Exp │ Config                      │  NL% │ K=4% │  Gap │ Vocab │ Top3% │ AbsLoss │
  ├──────┼─────────────────────────────┼──────┼──────┼──────┼───────┼───────┼─────────┤
  │   13 │ v3 + mtraj=1.0              │ 46.4 │ 42.3 │  4.1 │    19 │ 100.0 │   0.019 │
  ├──────┼─────────────────────────────┼──────┼──────┼──────┼───────┼───────┼─────────┤
  │ **14 │ v3 + jacobi=0.5 + mtraj=1.0 │ 47.6 │ 45.5 │  2.1 │    44 │  99.5 │   0.061 │**
  ├──────┼─────────────────────────────┼──────┼──────┼──────┼───────┼───────┼─────────┤
  │   15 │ v3 + mtraj=0.5              │ 46.3 │ 44.1 │  2.2 │    46 │  99.8 │   0.009 │
  ├──────┼─────────────────────────────┼──────┼──────┼──────┼───────┼───────┼─────────┤
  │   16 │ wu(jacobi) → v3+mtraj       │ 37.9 │ 38.0 │ -0.1 │    40 │  97.7 │   0.111 │
  └──────┴─────────────────────────────┴──────┴──────┴──────┴───────┴───────┴─────────┘

### Takeaways (0.6B)

1. **Same pattern as 1.7B** — warmup hurts (5-15pp NL drop), jacobi+mtraj is best config.
2. **Best: exp14** (jacobi=0.5 + mtraj=1.0) — NL=47.6%, K=4=45.5%, gap=2.1pp.
   Beats baseline (exp1) on both NL (+0.6pp) and K=4 (+1.7pp).
3. **Jacobi narrows gap** but costs ~2pp NL (exp3 gap=1.8 vs exp1 gap=3.2).
   Combined with mtraj (exp14), the NL cost disappears — synergistic.
4. **Warmup + mtraj doesn't rescue** (exp16: 37.9/38.0) — warmup damage is too deep.
5. **Vocab collapse persists** in all good runs (top3% ≥ 96%). Warmup runs with
   spread vocabs (52-76% top3%) always have high abs_loss (0.6-1.6) — uncertainty, not diversity.

---

## Conclusion

- **Warmup is harmful** — it damages base NL ability and creates uncertain (not diverse) abstractions.
- **Jacobi loss alone is marginal** — narrows the gap slightly but costs NL accuracy.
- **mask_traj is the key mechanism** — forces dependency on abstractions without hurting NL.
- **Best config: jacobi=0.5 + mtraj=1.0** — on both model sizes, this is the only config that
  meaningfully improves K=4 accuracy over v3 baseline while maintaining or improving NL accuracy.
  The combination is synergistic: jacobi alone hurts NL, mtraj alone is neutral, together they
  push both NL and K=4 above baseline.
