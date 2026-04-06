# Arithmetic Interpretability Study

Goal: show that SoRL externalizes arithmetic reasoning mechanisms (carry, borrow circuits)
as explicit abstraction tokens — observable and intervenable without activation-level tooling.

References:
- Quirke et al., "Understanding Addition and Subtraction in Transformers" (2024)
- SoRL paper: `docs/SoRL_alignment_weak_supervision.pdf`
- Experiment plan: `docs/interpretability_study.md`

HuggingFace:
- Models: [thoughtworks/arithmetic-sorl](https://huggingface.co/thoughtworks/arithmetic-sorl)
- Datasets: [thoughtworks/arithmetic-sorl-data](https://huggingface.co/datasets/thoughtworks/arithmetic-sorl-data)

---

## Setup

### Model: Tiny Qwen3 (from scratch, random init)

We use `Qwen3ForCausalLM` with a custom small config so that `SorlModelWrapper` and all
existing trainers (v1-v6) work with zero adaptation. No pretrained weights — trained from
scratch on arithmetic only.

```python
Qwen3Config(
    hidden_size=512,
    num_hidden_layers=3,       # Quirke uses 3
    num_attention_heads=4,     # Quirke uses 4
    num_key_value_heads=4,
    intermediate_size=2048,    # 4 * hidden_size
    vocab_size=151936,         # standard Qwen3 vocab
    max_position_embeddings=128,
)
```

Params: ~168M (mostly embedding table; active params ~3M for 3 layers).
Qwen3 tokenizer maps each digit/operator to exactly 1 token -> uniform sequence length.

### SoRL: Trainer v6 (self-routing)

`SoRLTrainerv6` (`sorl/selfroute.py`) inherits v3 and simplifies:
1. **Fixed diagonal lm_head** for abstract tokens — no learned projection
2. **Gradient hook** freezes abstract rows in lm_head
3. **Loss = traj_loss only** — no info_gain, no abs_loss, no zipf

This is the cleanest SoRL variant. Abstract tokens map 1-to-1 from hidden state dimensions,
making them directly interpretable.

For baseline (no SoRL): standard SFT on the same Qwen3 model, loss on answer tokens only.

---

## Data: 6-digit arithmetic

### Sequence format

```
  Position:   0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20
  Addition:   D5 D4 D3 D2 D1 D0  +  D'5D'4D'3D'2D'1D'0  =  A6 A5 A4 A3 A2 A1 A0
  Subtraction:D5 D4 D3 D2 D1 D0  -  D'5D'4D'3D'2D'1D'0  =  A6 A5 A4 A3 A2 A1 A0

  prompt_len = 14 (everything up to and including '=')
  answer_len = 7  (n_digits + 1 for overflow/sign)
  total      = 21 tokens

  Token IDs: 0-9 = digits, 10 = '+', 11 = '=', 12 = '-'
  Qwen3 IDs: 15-24 = digits, 10 = '+', 28 = '=', 12 = '-'
```

### Per-digit sub-task labels (Quirke et al.)

Each answer digit is labeled by the arithmetic operation it requires.
These are the building blocks of the carry/borrow circuits.

**Addition** — the model must learn to cascade carries left-to-right:

```
  SA — Base Add        Dn + D'n, no carry involved        (simplest)
  SC — Make Carry      Dn + D'n >= 10, generates carry
  SS — Sum is 9        Dn + D'n == 9, carry uncertain     (propagator)
  UC — Use Carry       carry arrives, Dn + D'n != 9       (consumes carry)
  US — Use Sum-9       carry arrives, Dn + D'n == 9       (cascade, hardest)
```

**Subtraction** (x >= y) — mirrors addition with borrows:

```
  MD — Base Diff       Dn > D'n, no borrow                (simplest)
  MB — Make Borrow     Dn < D'n, generates borrow
  ME — Equal digits    Dn == D'n, borrow uncertain         (propagator)
  UB — Use Borrow      borrow arrives, Dn != D'n           (consumes borrow)
  UD — Use Equal       borrow arrives, Dn == D'n           (cascade, hardest)
```

### Complexity classification (Quirke Table 8)

The complexity of a problem = the length of the longest carry/borrow cascade.
A cascade happens when a carry/borrow propagates through consecutive "uncertain"
positions (SS or ME digits).

**Example: 555555 + 444448 = 1000003**

```
  Position (LSB first):  D0   D1   D2   D3   D4   D5
  Digits:                5+8  5+4  5+4  5+4  5+4  5+4
  Digit sum:              13    9    9    9    9    9
  ST (tri-state):          1    U    U    U    U    U
  Cascade:              [SC]  [US   US   US   US   US]  <- depth 6 = S6
```

The carry from D0 (5+8=13) cascades through D1-D5 (all sum to 9).
Each "U" position is uncertain until the carry arrives. This is the hardest case.

**Addition complexity S0-S6 (6-digit):**

```
  S0: no carries at all                 555555+111111=0666666      ~10%
  S1: carries, no cascade               111111+000009=0111120      ~50%
  S2: cascade of 2                      111111+000089=0111200      ~26%
  S3: cascade of 3                      111111+000889=0112000      ~9%
  S4: cascade of 4                      111111+008889=0120000      ~3%
  S5: cascade of 5                      111111+088889=0200000      ~1%
  S6: cascade of 6                      111111+888889=1000000      <0.5%
```

**Subtraction complexity M0-M6:** same structure with borrow cascades.

### Data enrichment (Quirke)

Without enrichment, S5/S6 cases are extremely rare (~1% and <0.5%).
Following Quirke: **60% of batches** have **40% of digit positions** forced to sum-to-9.
This increases cascade frequency so the model sees enough hard cases to learn.

### Dataset structure on HuggingFace

```
  thoughtworks/arithmetic-sorl-data/
  ├── add_6digit/                    # addition only
  │   ├── train.parquet              500K examples
  │   ├── val.parquet                10K examples
  │   ├── eval_stratified.parquet    50 per complexity level (S0-S6) + 200 random
  │   └── config.json
  ├── add_sub_6digit/                # mixed addition + subtraction
  │   ├── train.parquet              500K examples (50/50 add/sub)
  │   ├── val.parquet                10K examples
  │   ├── eval_stratified.parquet    50 per level (S0-S6, M0-M6) + 200 random each
  │   └── config.json
  ├── add_handcrafted/               # Quirke's manual test questions (addition)
  │   ├── test.parquet               108 questions: S0(24), S1(23), S2(23), S3(17), S4(9), S5(12)
  │   └── config.json
  └── sub_handcrafted/               # Quirke's manual test questions (subtraction)
      ├── test.parquet               68 questions: M0(27), M1(18), M2(15), M3(8)
      └── config.json
```

Each row has columns: `tokens, labels, op, complexity, cascade_depth, x_digits, y_digits, z_digits`

The `eval_stratified` splits have an additional `eval_category` column (e.g., `add_S0`, `sub_M3`).
The `handcrafted` splits have `expected_complexity`, `x_val`, `y_val` — values taken directly
from Quirke's [quanta_maths](https://github.com/PhilipQuirke/quanta_maths) repo.
All complexity labels verified to match Quirke's expected values exactly.

**Complexity distribution (add_6digit train, 500K):**

```
  S0:  50,860  (10.2%)
  S1: 247,567  (49.5%)
  S2: 130,699  (26.1%)
  S3:  46,995   (9.4%)
  S4:  16,540   (3.3%)
  S5:   5,611   (1.1%)
  S6:   1,728   (0.3%)
```

---

## Code structure

```
arithmetic/
├── datasets/
│   └── addition.py          # data gen, Quirke labels, complexity, eval sets
├── interp_utils/
│   └── sae_trainer.py       # SAE via EleutherAI sparsify (SparseCoder)
├── hub.py                   # HuggingFace save/load utilities
├── train.py                 # unified training: Qwen3 + SorlModelWrapper + v6
└── scripts/
    └── sweep.sh             # sweep over tasks x vocab sizes across 3 GPUs
```

Key files:
- **[`arithmetic/train.py`](../arithmetic/train.py)** — entry point, model factory, dataset adapter, accuracy callback
- **[`arithmetic/datasets/addition.py`](../arithmetic/datasets/addition.py)** — data generation with Quirke sub-task labels + complexity
- **[`arithmetic/hub.py`](../arithmetic/hub.py)** — save/load models and datasets to HuggingFace

SoRL core (Fangyuan's code, unchanged):
- [`sorl/sorl_wrapper.py`](../sorl/sorl_wrapper.py) — SorlModelWrapper (extends HF model with abstract vocab)
- [`sorl/selfroute.py`](../sorl/selfroute.py) — SoRLTrainerv6 (self-routing, traj-only loss)
- [`sorl/trainer_ablate.py`](../sorl/trainer_ablate.py) — SoRLTrainer (v1), v2, v3, v4
- [`sorl/sorl_trainer.py`](../sorl/sorl_trainer.py) — sorl_search, loss functions

### Compute

3x NVIDIA RTX PRO 6000 Blackwell (96GB each).

---

## Experiment Plan

Three ablation dimensions, all evaluated on random + stratified + handcrafted eval sets.
All models: tiny Qwen3 (3L/4H/512d, ~168M params), trained from scratch.

### Ablation 1: Baseline vs SoRL

Does SoRL maintain arithmetic accuracy? Does it hurt or help on hard cases (S4-S6)?

```
  ┌──────────┬────────────┬─────────────────────────────────────────────────┐
  │ Task     │ Mode       │ Config                                          │
  ├──────────┼────────────┼─────────────────────────────────────────────────┤
  │ add      │ baseline   │ no abstraction tokens                           │
  │ add      │ SoRL v6    │ abs_vocab=16, K=4                               │
  │ add_sub  │ baseline   │ no abstraction tokens                           │
  │ add_sub  │ SoRL v6    │ abs_vocab=16, K=4                               │
  └──────────┴────────────┴─────────────────────────────────────────────────┘
```

Eval: accuracy on random (256), stratified (S0-S6, M0-M6), handcrafted (Quirke).
Report per-complexity and per-subtask (SA, SC, SS, UC, US / MD, MB, ME, UB, UD).

### Ablation 2: Data efficiency

How much training data does each mode need to reach target accuracy?
Train baseline and SoRL v6 (abs_vocab=16) with varying dataset sizes.

```
  ┌──────────┬────────────┬─────────────────────────────────────────────────┐
  │ Task     │ Mode       │ Dataset sizes                                   │
  ├──────────┼────────────┼─────────────────────────────────────────────────┤
  │ add      │ baseline   │ 10K, 50K, 100K, 250K, 500K                     │
  │ add      │ SoRL v6    │ 10K, 50K, 100K, 250K, 500K                     │
  │ add_sub  │ baseline   │ 10K, 50K, 100K, 250K, 500K                     │
  │ add_sub  │ SoRL v6    │ 10K, 50K, 100K, 250K, 500K                     │
  └──────────┴────────────┴─────────────────────────────────────────────────┘
```

Question: does SoRL reach high accuracy with less data? Especially on S4-S6 (rare cases)?

### Ablation 3: SoRL vocab size

How does abstract vocabulary size affect accuracy and token interpretability?

```
  ┌──────────┬────────────┬─────────────────────────────────────────────────┐
  │ Task     │ abs_vocab  │ Notes                                           │
  ├──────────┼────────────┼─────────────────────────────────────────────────┤
  │ add      │ 0          │ baseline — no SoRL                              │
  │ add      │ 1          │ single token — can it encode anything?          │
  │ add      │ 2          │ binary — carry/no-carry?                        │
  │ add      │ 4          │ undercomplete                                   │
  │ add      │ 5          │ = number of addition sub-tasks (SA,SC,SS,UC,US) │
  │ add      │ 8          │                                                 │
  │ add      │ 10         │ = total sub-tasks (5 add + 5 sub)               │
  │ add      │ 16         │ default                                         │
  │ add      │ 20         │ overcomplete                                    │
  │ add      │ 24         │ overcomplete                                    │
  ├──────────┼────────────┼─────────────────────────────────────────────────┤
  │ add_sub  │ 0          │ baseline — no SoRL                              │
  │ add_sub  │ 1          │ single token                                    │
  │ add_sub  │ 2          │ binary — add/sub or carry/borrow?               │
  │ add_sub  │ 4          │                                                 │
  │ add_sub  │ 5          │ = number of addition sub-tasks                  │
  │ add_sub  │ 8          │                                                 │
  │ add_sub  │ 10         │ = total sub-tasks (5 add + 5 sub)               │
  │ add_sub  │ 16         │ default                                         │
  │ add_sub  │ 20         │ overcomplete                                    │
  │ add_sub  │ 24         │ overcomplete                                    │
  └──────────┴────────────┴─────────────────────────────────────────────────┘
```

Track: accuracy, vocab utilization, top-3 concentration, abs_loss, token-subtask correlation.
Key questions:
- Does vocab=5 give 1-to-1 token-mechanism mapping for addition?
- Does vocab=10 give 1-to-1 for mixed add+sub?
- What does the model do with 1 or 2 tokens? Collapse to carry/no-carry binary?

### Total runs

```
  Ablation 1:  4 runs   (2 tasks x 2 modes)
  Ablation 2: 20 runs   (2 tasks x 2 modes x 5 sizes)
  Ablation 3: 20 runs   (2 tasks x 10 vocab sizes, includes baseline)
  ─────────────────────
  Total:      44 runs   (minus overlap: baselines counted once → ~38 unique)
```

3 GPUs, ~25 min per baseline run, SoRL ~2-4x slower.
Estimated wall time: ~10-12 hours with parallelism.

### Phase 4: Interpretability (after training)

Full plan in [`docs/interpretability_study.md`](../docs/interpretability_study.md).

**Token-level interventions** ([`arithmetic/interp_utils/interventions.py`](../arithmetic/interp_utils/interventions.py)):

Every activation-level intervention from Quirke has a token-level SoRL analog:

```
  ┌─────────────────────────────┬──────────────────────────────────────────────────┐
  │ Quirke (activation-level)   │ SoRL (token-level)                               │
  ├─────────────────────────────┼──────────────────────────────────────────────────┤
  │ Mean ablation of a node     │ token_knockout: replace abs token with placeholder│
  │ Activation patching (pairs) │ token_swap: swap abs tokens between paired Qs     │
  │ Zero ablation               │ token_replace: set to fixed value                 │
  │ Random perturbation         │ token_shuffle: permute abs tokens in sequence      │
  │ Per-digit node knockout     │ knockout_at_digit: mask abs tokens before digit N  │
  │ Paired causal intervention  │ swap_at_digit: swap abs tokens near digit N        │
  └─────────────────────────────┴──────────────────────────────────────────────────┘
```

Key advantage: Quirke needs TransformerLens hooks + cached activations + paired forward
passes to patch a single node. SoRL needs one line: `tokens[abs_pos] = new_value`.

Analysis pipeline:
1. Token-subtask correlation: P(token | SA), P(token | SC), etc.
2. PCA of hidden states at abs positions (compare to Quirke's 3-cluster finding)
3. Token knockout per digit → accuracy drop by complexity (quanta maps)
4. Token swap between paired questions → causal verification
5. SAE on baseline → feature-token mapping via Hungarian matching
6. Polysemanticity check: do tokens map 1-to-1 or many-to-many?
7. Auto-interpretability: LLM describes each token's role across examples

---

## Results

(pending)
