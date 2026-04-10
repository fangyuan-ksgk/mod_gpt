# Arithmetic Interpretability Study

Goal: show that SoRL externalizes arithmetic reasoning mechanisms (carry, borrow circuits)
as explicit abstraction tokens — observable and intervenable without activation-level tooling.

Detailed interpretability plan: [`docs/interpretability_study.md`](../docs/interpretability_study.md)

### References

- Quirke et al., ["Understanding Addition and Subtraction in Transformers"](https://arxiv.org/abs/2402.02619) (2024)
  - Code: [quanta_maths](https://github.com/PhilipQuirke/quanta_maths) | [quanta_mech_interp](https://github.com/PhilipQuirke/quanta_mech_interp)
  - Models: [PhilipQuirke/VerifiedArithmetic](https://huggingface.co/PhilipQuirke) (49 models on HF)
  - Local PDF: [`docs/understanding_addition_subtraction_transformers.pdf`](../docs/understanding_addition_subtraction_transformers.pdf)
- Yu, Su & Abdullah, ["Intention-Level Alignment with Weak Supervision"](../docs/SoRL_alignment_weak_supervision.pdf) (SoRL paper)

### HuggingFace

- Models: [thoughtworks/arithmetic-sorl](https://huggingface.co/thoughtworks/arithmetic-sorl)
- Datasets: [thoughtworks/arithmetic-sorl-data](https://huggingface.co/datasets/thoughtworks/arithmetic-sorl-data)

### Code

```
arithmetic/
├── datasets/
│   └── addition.py              # data gen, Quirke labels, enrichment, eval sets
├── interp_utils/
│   ├── interventions.py         # token-level interventions (SoRL analog of mech interp)
│   ├── test_interventions.py    # 20 tests, all passing
│   └── sae_trainer.py           # SAE via EleutherAI sparsify
├── evaluate.py                  # ArithmeticEvaluator class
├── catalog.py                   # ModelCatalog — index HF models
├── hub.py                       # HuggingFace save/load
├── train.py                     # unified training: baseline SFT + SoRL v1
├── eval_sets/                   # cached deterministic eval sets (seed=42)
└── scripts/
    ├── gpu_queue.py             # GPU job scheduler
    ├── sweep_enriched.txt       # main 30-job sweep
    ├── sweep_baselines_10ep.txt # 10-epoch baseline re-runs
    ├── sweep_low_data_sorl.txt  # low-data SoRL K=1 experiments
    ├── sweep_undersize.txt      # undersized model sweep
    ├── auto_zipf_sweep.py       # autonomous zipf diversity pipeline
    └── reeval_hf_models.py      # re-evaluate uploaded models
```

Key links:
[`train.py`](../arithmetic/train.py) |
[`addition.py`](../arithmetic/datasets/addition.py) |
[`interventions.py`](../arithmetic/interp_utils/interventions.py) |
[`hub.py`](../arithmetic/hub.py) |
[`evaluate.py`](../arithmetic/evaluate.py) |
[`catalog.py`](../arithmetic/catalog.py)

---

## Setup

### Model

Tiny Qwen3 from scratch (random init), wrapped in `SorlModelWrapper` for clean SoRL trainer access.

```
  ┌────────────────────┬────────────┐
  │ Parameter          │ Value      │
  ├────────────────────┼────────────┤
  │ Architecture       │ Qwen3      │
  │ Layers             │ 2          │
  │ Attention heads    │ 3          │
  │ Hidden dim         │ 510        │
  │ Total params       │ ~162M      │
  │ Transformer params │ ~7.8M      │
  │ Pretrained weights │ None       │
  │ Tokenizer          │ Qwen3-0.6B │
  └────────────────────┴────────────┘
```

Qwen3 tokenizer maps each digit/operator to exactly 1 token (uniform 21-token sequences).

### SoRL: Trainer v1 (info-gain)

`SoRLTrainer` (v1): info_gain loss + abs loss. Forces abstractions to actually reduce
prediction uncertainty via `alpha_info_gain`. v6 (self-routing, traj-only loss) was tried
but produces 0% accuracy from scratch — switched to v1.

Baseline = same Qwen3 model, standard SFT, no abstraction tokens.

---

## Data

6-digit addition and subtraction, 21 tokens per sequence.

```
  Addition:    D5 D4 D3 D2 D1 D0  +  D'5 D'4 D'3 D'2 D'1 D'0  =  A6 A5 A4 A3 A2 A1 A0
  Subtraction: D5 D4 D3 D2 D1 D0  -  D'5 D'4 D'3 D'2 D'1 D'0  =  A6 A5 A4 A3 A2 A1 A0
               ──── prompt (14 tokens) ────                         ── answer (7 tokens) ──
```

**Enrichment:** 40% of positions forced to equal digits (creates borrow cascades),
40% forced to sum-to-9 (creates carry cascades). Without enrichment, M3+ borrows
are extremely rare (<1%).

### Quirke's algorithmic sub-tasks (paper sections 3.2-3.4)

The model must implement these sub-tasks internally. Quirke discovers them via ablation.

**Addition (section 3.2):**

```
  ┌──────┬────────────────┬──────────────────────────────────────────────────────┐
  │ Task │ Name           │ Definition                                           │
  ├──────┼────────────────┼──────────────────────────────────────────────────────┤
  │ SA   │ Base Add       │ (Dn + D'n) mod 10                                   │
  │ ST   │ TriCase        │ 1 if sum>=10, 0 if sum<=8, U if sum=9               │
  │ SV   │ Cascade carry  │ TriAdd(ST_n..ST_0) — resolves U's left-to-right     │
  └──────┴────────────────┴──────────────────────────────────────────────────────┘
  Answer digit: An = (SAn + SV_{n-1}) mod 10
```

**Subtraction (section 3.3):**

```
  ┌──────┬────────────────┬──────────────────────────────────────────────────────┐
  │ Task │ Name           │ Definition                                           │
  ├──────┼────────────────┼──────────────────────────────────────────────────────┤
  │ MD   │ Base Diff      │ (Dn - D'n) mod 10                                   │
  │ MB   │ TriCase borrow │ 1 if Dn<D'n, 0 if Dn>D'n, U if Dn=D'n             │
  │ MV   │ Cascade borrow │ TriAdd(MB_n..MB_0) — resolves U's left-to-right     │
  │ ND   │ Neg Diff       │ (D'n - Dn) mod 10 — for negative answers            │
  │ SGN  │ Sign           │ attends to answer sign (+ or -)                      │
  └──────┴────────────────┴──────────────────────────────────────────────────────┘
```

**Mixed (section 3.4):** adds OPR (operator detection — attends to + or - token).

### Per-digit outcome labels (our dataset)

We label each answer digit by its outcome. SA, SC, SS match Quirke's `MathsTask` enum
exactly. UC, US are our additions for "incoming carry" cases (Quirke captures this via
ST+SV mechanism rather than per-digit labels).

```
  ┌───────────────────────────────────────────────────────────────────────────────┐
  │ Addition                                │ Subtraction (x >= y)               │
  ├─────┬──────────────┬────────────────────┼─────┬──────────────┬───────────────┤
  │ SA  │ Base Add     │ no carry  (=Quirke)│ MD  │ Base Diff    │ no borrow (=) │
  │ SC  │ Make Carry   │ ST=1      (=Quirke)│ MB  │ Make Borrow  │ MB=1      (=) │
  │ SS  │ Sum is 9     │ ST=U      (=Quirke)│ ME  │ Equal digits │ MB=U (≈MT)    │
  │ UC  │ Use Carry    │ carry in, ST!=U    │ UB  │ Use Borrow   │ borrow in     │
  │ US  │ Use Sum-9    │ carry in, ST=U     │ UD  │ Use Equal    │ borrow + equal│
  └─────┴──────────────┴────────────────────┴─────┴──────────────┴───────────────┘
  (=) = exact match with Quirke's MathsTask enum
  UC, US, UB, UD = our labels for "incoming carry/borrow" cases
  Quirke captures these via the ST/SV (or MT/MV) cascade resolution mechanism
```

### Complexity (Quirke Table 8)

Complexity = max carry/borrow cascade length. Example: `555555+444448=1000003` is S6 —
carry from D0 cascades through 5 consecutive sum-9 positions.

```
  ┌─────┬─────────────────────────────────┬──────────┐
  │     │ Description                     │ Freq     │
  ├─────┼─────────────────────────────────┼──────────┤
  │ S0  │ no carries                      │  ~10%    │
  │ S1  │ carries, no cascade             │  ~50%    │
  │ S2  │ cascade of 2                    │  ~26%    │
  │ S3  │ cascade of 3                    │   ~9%    │
  │ S4  │ cascade of 4                    │   ~3%    │
  │ S5  │ cascade of 5                    │   ~1%    │
  │ S6  │ cascade of 6 (full chain)       │  <0.5%   │
  ├─────┼─────────────────────────────────┼──────────┤
  │ M0-M6 same structure for subtraction borrows     │
  └─────┴─────────────────────────────────┴──────────┘
  Note: M6 is impossible for 6-digit subtraction (x >= y constraint). Max is M5.
```

### Eval sets

Fixed eval set (seed=42, cached in `eval_sets/`). Includes:
- S0-S6: carry cascade complexity (addition)
- M0-M5: borrow cascade complexity (subtraction)
- C3-C6: hot carry chains (consecutive carry positions)
- B3-B5: hot borrow chains (consecutive borrow positions)

### Datasets on HuggingFace

```
  ┌────────────────────┬────────────────────────────────────────────────────────┐
  │ Config             │ Contents                                              │
  ├────────────────────┼────────────────────────────────────────────────────────┤
  │ add_6digit         │ 500K train, 10K val, stratified eval (S0-S6)          │
  │ add_sub_6digit     │ 500K train, 10K val, stratified eval (S0-S6 + M0-M6) │
  │ add_handcrafted    │ 108 Quirke test questions (S0-S5)                     │
  │ sub_handcrafted    │ 68 Quirke test questions (M0-M3)                      │
  └────────────────────┴────────────────────────────────────────────────────────┘
```

Columns: `tokens, labels, op, complexity, cascade_depth, x_digits, y_digits, z_digits`

---

## Results

All models: 2L/3H/510d Qwen3 from scratch on 6-digit add+sub with enrichment.
SoRL: v1 trainer (alpha_info_gain=10, alpha_abs=0.1, alpha_zipf=1.0).
Eval: recursion + teacher-forced (matches training procedure).

### 1. Baselines (SFT, 5 epochs, add_sub)

```
  ┌─────────┬─────────┬────────┬────────┬────────┬────────┬────────┐
  │ Data    │ Overall │  S5    │  S6    │  C6    │  M5    │  B5    │
  ├─────────┼─────────┼────────┼────────┼────────┼────────┼────────┤
  │  10K    │   10%   │  18%   │  60%   │   2%   │   2%   │   0%   │
  │  25K    │   55%   │  20%   │  34%   │  26%   │  10%   │  34%   │
  │  50K    │   84%   │  42%   │  72%   │  70%   │   8%   │  74%   │
  │ 250K    │  100%   │ 100%   │ 100%   │ 100%   │ 100%   │ 100%   │
  │ 500K    │  100%   │ 100%   │ 100%   │ 100%   │ 100%   │ 100%   │
  └─────────┴─────────┴────────┴────────┴────────┴────────┴────────┘
  Hard cases fail at 10K-50K. 250K+ saturates.
```

<!-- PLACEHOLDER: 10-epoch baseline results (12 jobs queued) — apples-to-apples with SoRL -->

### 2. SoRL v1 K=1 (10 epochs, 500K add_sub)

All vocab sizes tested (2, 16, 20, 30, 50, 70, 100): 100% on ALL splits including
hard ones (S5, S6, C6, M5, B5). K=1 is universally robust.

### 3. SoRL v1 K=4 (10 epochs, 500K add_sub)

Most vocab sizes reach 100%. Known fragile spot: abs30 K=4 is broken (S5=54%, M5=34%).
abs16 K=4: M5=96%. K=4 is less robust than K=1 at certain vocab sizes.

<!-- PLACEHOLDER: full K=4 results table across all vocab sizes and hard splits -->

### 4. Data efficiency: SoRL vs baseline at low data

At 500K, both baseline and SoRL saturate at 100%. The differentiation story is at
low data (10K-50K) where baselines fail on cascades.

<!-- PLACEHOLDER: SoRL K=1 vocab=10 at 10K, 25K, 50K, 100K, 250K (5 jobs queued) -->

### 5. Undersized models

Three architectures (1L/3H/510d, 1L/2H/256d, 2L/1H/128d) tested across
5 data sizes, baseline + SoRL = 30 jobs total.

<!-- PLACEHOLDER: undersized model results (30 jobs queued) -->

### 6. Zipf diversity sweep

3 zipf values x 2 K x 4 best vocabs = 24 jobs. Testing whether higher
alpha_soft_zipf produces genuine vocabulary diversity vs uncertainty noise.

<!-- PLACEHOLDER: zipf sweep results (24 jobs queued) -->

### 7. Vocabulary utilization

How many abstraction tokens does the model actually use?

**Vocab utilization at K=4 (add_sub, 500K, 100 examples):**

```
  ┌───────────┬────────┬────────┬─────────────────────────────────────────┐
  │ abs_vocab  │ used   │ top-3  │ distribution (top 5 tokens)            │
  ├───────────┼────────┼────────┼─────────────────────────────────────────┤
  │  1         │  1/1   │ 100%   │ t1:100%                                │
  │  2         │  1/2   │ 100%   │ t1:100%                                │
  │  5         │  3/5   │ 100%   │ t3:72% t1:19% t2:10%                  │
  │ 10         │  7/10  │  69%   │ t4:27% t2:22% t1:20% t3:12% t5:10%   │
  │ 16         │ 12/16  │  70%   │ t1:33% t2:28% t3:8% t5:6% t9:4%      │
  │ 20         │ 15/20  │  51%   │ t1:21% t6:19% t5:11% t2:11% t7:6%    │
  └───────────┴────────┴────────┴─────────────────────────────────────────┘
```

Pattern: ~70% of vocab is used. Top-3 concentration decreases with larger vocab
(100% at vocab=5 to 51% at vocab=20). Distribution is Zipf-like, not uniform.

**Vocab utilization across K (abs_vocab=10, add_sub, 500K):**

```
  ┌─────┬────────┬────────┬──────────────────────────────────┐
  │ K   │ used   │ top-3  │ top 3 tokens                     │
  ├─────┼────────┼────────┼──────────────────────────────────┤
  │  1  │  7/10  │  68%   │ t7:34% t3:20% t1:14%            │
  │  2  │  7/10  │  87%   │ t1:43% t3:25% t4:19%            │
  │  3  │  7/10  │  84%   │ t2:50% t1:22% t5:11%            │
  │  4  │  7/10  │  69%   │ t4:27% t2:22% t1:20%            │
  └─────┴────────┴────────┴──────────────────────────────────┘
```

K does NOT affect utilization count (always 7/10). But K=2,3 concentrate more
on fewer tokens (87% top-3) vs K=1,4 (~68% top-3).

<!-- PLACEHOLDER: vocab 25-100 utilization results -->

---

## Interpretability

Plan and methodology: [`docs/interpretability_study.md`](../docs/interpretability_study.md)
Token-level intervention utils: [`interventions.py`](../arithmetic/interp_utils/interventions.py)
(tested: [`test_interventions.py`](../arithmetic/interp_utils/test_interventions.py), 20/20 passing)
