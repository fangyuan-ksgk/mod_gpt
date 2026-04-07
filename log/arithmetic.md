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
│   └── addition.py              # data gen, Quirke labels, complexity, eval sets
├── interp_utils/
│   ├── interventions.py         # token-level interventions (SoRL analog of mech interp)
│   ├── test_interventions.py    # 20 tests, all passing
│   └── sae_trainer.py           # SAE via EleutherAI sparsify
├── hub.py                       # HuggingFace save/load
├── train.py                     # Qwen3 + SorlModelWrapper + v6
└── scripts/
    └── sweep.sh                 # full ablation sweep across 3 GPUs
```

Key links:
[`train.py`](../arithmetic/train.py) |
[`addition.py`](../arithmetic/datasets/addition.py) |
[`interventions.py`](../arithmetic/interp_utils/interventions.py) |
[`hub.py`](../arithmetic/hub.py)

---

## Setup

### Model

Tiny Qwen3 from scratch (random init), wrapped in `SorlModelWrapper` for clean SoRL trainer access.

```
  ┌────────────────────┬────────────┐
  │ Parameter          │ Value      │
  ├────────────────────┼────────────┤
  │ Architecture       │ Qwen3      │
  │ Layers             │ 3          │
  │ Attention heads    │ 4          │
  │ Hidden dim         │ 512        │
  │ MLP dim            │ 2048       │
  │ Total params       │ ~168M      │
  │ Active params      │ ~3M        │
  │ Pretrained weights │ None       │
  │ Tokenizer          │ Qwen3-0.6B │
  └────────────────────┴────────────┘
```

Qwen3 tokenizer maps each digit/operator to exactly 1 token (uniform 21-token sequences).

### SoRL: Trainer v6 (self-routing)

`SoRLTrainerv6` (`sorl/selfroute.py`): fixed diagonal lm_head for abstract tokens + grad
hook freeze + traj-only loss. Cleanest SoRL variant. See [`docs/interpretability_study.md`](../docs/interpretability_study.md)
for the full comparison of classical mech interp vs SoRL interp.

Baseline = same Qwen3 model, standard SFT, no abstraction tokens.

---

## Data

6-digit addition and subtraction, 21 tokens per sequence.

```
  Addition:    D5 D4 D3 D2 D1 D0  +  D'5 D'4 D'3 D'2 D'1 D'0  =  A6 A5 A4 A3 A2 A1 A0
  Subtraction: D5 D4 D3 D2 D1 D0  -  D'5 D'4 D'3 D'2 D'1 D'0  =  A6 A5 A4 A3 A2 A1 A0
               ──── prompt (14 tokens) ────                         ── answer (7 tokens) ──
```

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
```

Data enrichment (Quirke): 60% of batches, 40% of positions forced to sum-to-9.

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

## Ablation Sweep (36 unique runs)

All models: tiny Qwen3 3L/4H/512d, trained from scratch, 3 epochs.
All eval: random (256) + stratified (S0-S6, M0-M6) + Quirke handcrafted.

### Ablation 1+2: Baseline vs SoRL x Data efficiency

```
  ┌──────────┬────────────┬──────────────────────────────────────────┐
  │ Task     │ Mode       │ Dataset sizes                            │
  ├──────────┼────────────┼──────────────────────────────────────────┤
  │ add      │ baseline   │ 10K, 50K, 100K, 250K, 500K              │  5 runs
  │ add      │ SoRL v6/16 │ 10K, 50K, 100K, 250K, 500K              │  5 runs
  │ add_sub  │ baseline   │ 10K, 50K, 100K, 250K, 500K              │  5 runs
  │ add_sub  │ SoRL v6/16 │ 10K, 50K, 100K, 250K, 500K              │  5 runs
  └──────────┴────────────┴──────────────────────────────────────────┘  = 20 runs
```

Questions: does SoRL match baseline accuracy? Does it need less data for S4-S6?

### Ablation 3: Vocab size sweep (all at 500K)

```
  ┌──────────┬────────────────────────────────────────────────────────┐
  │ Task     │ abs_vocab (0=baseline, rest=SoRL v6)                   │
  ├──────────┼────────────────────────────────────────────────────────┤
  │ add      │ 1, 2, 4, 5*, 8, 10, 20, 24                           │  8 runs
  │ add_sub  │ 1, 2, 4, 5, 8, 10*, 20, 24                           │  8 runs
  └──────────┴────────────────────────────────────────────────────────┘  = 16 runs
  (vocab=0 and vocab=16 already covered by ablation 1+2 at 500K)
  * = matches sub-task count (5 add / 10 total)
```

Questions: does vocab=5 give 1-to-1 token-mechanism mapping? What happens at vocab=1?

### Ablation 4: K sweep (abstraction insertion frequency, at 500K)

K = every K trajectory tokens, insert one abstraction token.

```
  ┌──────────┬────────────────────────────────────────────────────────┐
  │ Task     │ K values (at vocab=5, 10, 16)                         │
  ├──────────┼────────────────────────────────────────────────────────┤
  │ add      │ K=2, 3, 4* (each x vocab 5, 10, 16)                  │  9 runs
  │ add_sub  │ K=2, 3, 4* (each x vocab 5, 10, 16)                  │  9 runs
  └──────────┴────────────────────────────────────────────────────────┘  = 18 runs
  * K=4 already in vocab sweep
```

K=2: ~7 abs tokens per sequence (dense reasoning). K=4: ~3-4 abs tokens (sparse).

### Ablation 5: Undersized model (2L/3H/510d, add_sub at 500K)

```
  abs_vocab: 0 (baseline), 1, 2, 5, 8, 10, 16                      = 7 runs
```

Does SoRL help when the model is undersized for mixed add+sub?

### Total: ~69 runs + 10 SAE training runs

```
  Ablation 1+2 (data eff.):    20 runs
  Ablation 3 (vocab sweep):    16 runs
  Ablation 4 (K sweep):        12 runs (K=2,3 only; K=4 in ablation 3)
  Ablation 5 (undersized):      7 runs
  Extra vocab gap-fills:         4 runs
  SAEs (key models):            10 runs (3 layers, k={32} or {8,16,32,64})
  ──────────────────────────────────────────
  Total:                        69 jobs
```

---

## Interpretability

Plan and methodology: [`docs/interpretability_study.md`](../docs/interpretability_study.md)
Token-level intervention utils: [`interventions.py`](../arithmetic/interp_utils/interventions.py)
(tested: [`test_interventions.py`](../arithmetic/interp_utils/test_interventions.py), 20/20 passing)

---

## Results

(pending — tables will be added here as runs complete)
