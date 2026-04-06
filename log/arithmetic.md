# Arithmetic Interpretability Study

Goal: show that SoRL externalizes arithmetic reasoning mechanisms (carry, borrow circuits)
as explicit abstraction tokens — observable and intervenable without activation-level tooling.

References:
- Quirke et al., "Understanding Addition and Subtraction in Transformers" (2024)
- SoRL paper: `docs/SoRL_alignment_weak_supervision.pdf`
- Experiment plan: `docs/interpretability_study.md`

---

## Setup

### Model: Tiny Qwen3 (from scratch, random init)

We use `Qwen3ForCausalLM` with a custom small config so that `SorlModelWrapper` and all
existing trainers (v1–v6) work with zero adaptation. No pretrained weights — trained from
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
Qwen3 tokenizer maps each digit/operator to exactly 1 token → uniform sequence length.

### SoRL: Trainer v6 (self-routing)

`SoRLTrainerv6` (`sorl/selfroute.py`) inherits v3 and simplifies:
1. **Fixed diagonal lm_head** for abstract tokens — no learned projection
2. **Gradient hook** freezes abstract rows in lm_head
3. **Loss = traj_loss only** — no info_gain, no abs_loss, no zipf

This is the cleanest SoRL variant. Abstract tokens map 1-to-1 from hidden state dimensions,
making them directly interpretable.

For baseline (no SoRL): standard SFT on the same Qwen3 model, loss on answer tokens only.

### Data: 6-digit addition + subtraction (online generation)

Format: `XXXXXX+YYYYYY=ZZZZZZZ` or `XXXXXX-YYYYYY=ZZZZZZZ` (21 tokens, answer = 7 tokens).
Tokenized uniformly: prompt_len=14, answer_len=7.

Data enrichment (matching Quirke): 60% of batches have 40% of digit positions forced to
sum-to-9 (increases carry cascade frequency).

Subtraction: x >= y guaranteed. Answer zero-padded to 7 digits.

### Sub-tasks tracked

Addition:
- BA  — Base Add (no carry in/out)
- MC1 — Make Carry (digit_sum >= 10)
- MS9 — Make Sum 9 (digit_sum == 9, propagates carry)
- UC1 — Use Carry (carry_in=1, digit_sum != 9)
- US9 — Use Sum 9 (carry_in=1, digit_sum == 9 — cascade, hardest)

Subtraction:
- BS  — Base Sub (no borrow)
- MB1 — Make Borrow (x_i < y_i)
- MD9 — Make Diff 9 (x_i == y_i, propagates borrow)
- UB1 — Use Borrow (borrow_in=1, x_i != y_i)
- UD9 — Use Diff 9 (borrow_in=1, x_i == y_i — cascade)

### Code structure

```
arithmetic/
├── datasets/
│   └── addition.py          # data gen, sub-task labels, eval sets
├── interp_utils/
│   └── sae_trainer.py       # SAE via EleutherAI sparsify (SparseCoder)
├── train.py                 # unified training: Qwen3 + SorlModelWrapper + v6
└── scripts/
    └── sweep.sh             # sweep over tasks × vocab sizes across 3 GPUs
```

SoRL core (Fangyuan's code, unchanged):
- `sorl/sorl_wrapper.py` — SorlModelWrapper (extends HF model with abstract vocab)
- `sorl/selfroute.py` — SoRLTrainerv6 (self-routing, traj-only loss)
- `sorl/trainer_ablate.py` — SoRLTrainer (v1), v2, v3, v4
- `sorl/sorl_trainer.py` — sorl_search, loss functions

### Compute

3x NVIDIA RTX PRO 6000 Blackwell (96GB each).
With bf16 + compile + batch=512: ~14 it/s baseline, 20K steps ~25 min.

---

## Experiment Plan

### Phase 1: Baselines (no SoRL)

Sweep: {add, add_sub} × {2L, 3L, 4L} × {3H/510d, 4H/512d}
Metric: full-sequence accuracy + per-subtask accuracy (BA, MC1, MS9, UC1, US9, BS, MB1, MD9, UB1, UD9)
Goal: reproduce Quirke's >99% accuracy, establish baseline for each sub-task.

### Phase 2: SoRL vocab size sweep

Sweep: {add, add_sub} × abs_vocab {4, 8, 16, 32, 64} × trainer {v6}
Metric: same as Phase 1 + vocab utilization, abs token distribution
Question: does SoRL maintain accuracy? Which vocab sizes lead to interpretable tokens?

### Phase 3: Interpretability

- Correlate abstraction tokens with sub-task labels (BA, MC1, US9, etc.)
- Paired interventions: token swap vs activation patching
- SAE on hidden states (baseline) vs direct token analysis (SoRL)
- Auto-interpretability on top-k abstraction token usages

---

## Results

(pending)
