# Do DLR's interpretability results hold on a real pretrained LLM? — Arithmetic

**Question addressed.** Whether abstraction codes on a *real* pretrained language model
(a) specialise on known algorithmic sub-tasks, and (b) permit single-code edits that
repair wrong predictions.

**Setup.** Qwen3-0.6B (596M parameters), post-trained with DLR in the residual-steering
formulation — the same mechanism used for the main results, not a from-scratch variant.
One steering code per answer digit, codebook size 30, steering injected at layer 14 of
28. The injection layer was **fixed a priori at the network midpoint and never swept**,
so no test information enters the layer choice.

**Task.** Six-digit addition and subtraction, `abcdef±ghijkl=mnopqrs`. Qwen3's
digit-splitting tokenizer emits exactly one token per digit (21 tokens per problem,
verified on every example), so answer digit *i* is generated at decode step *i* and is
steered by exactly one code. Training: 100K problems, one epoch. Evaluation: 2,600
held-out problems spanning 24 difficulty splits, generated autoregressively from the
model's own predictions with no teacher forcing.

**Ground truth.** Each answer digit carries a sub-task label from the standard
addition/subtraction taxonomy — for addition: simple add, carry generation, sum-9
boundary, carry use, carry cascade; for subtraction: the borrow analogues. Cascade
labels are the hardest, requiring state to be tracked across digit positions.

---

## Measurements

**M1 — sub-task purity.** For each code, the distribution over sub-task labels of the
digits it steers. Reported as **purity** = P(label | code), the fraction of a code's
occurrences falling in its dominant label; **recall** = P(code | label), the share of
that label the code accounts for; and **lift** = purity divided by that label's base
rate in the data.

Lift is the metric that matters and purity alone is misleading. Labels are not
uniformly frequent — a code that is 40% pure on a label occurring 38% of the time has
learned nothing, while a code 36% pure on a label occurring 6% of the time is a strong
detector. Any purity figure quoted without its base rate is uninterpretable.

**M2 — single-code repair.** Two forms, because they answer different questions.

*M2-predictive.* For each wrong prediction and each answer position, read the
ground-truth label, look up the code M1 found purest for that label, force that code at
that position, regenerate. Matched one-for-one against forcing a uniformly random code
at the same position on the same examples. This is falsifiable: if codes carry sub-task
structure, the label-matched code should beat random.

*M2-existence.* For each wrong prediction, try all 30 codes at all 7 positions and count
how many examples *some* swap repairs. Reported for completeness, but it is an existence
measure over 210 interventions per example — a nonzero rate is expected from chance
alone, and it does not by itself evidence structure.

---

## Result

**Both measurements are negative on this model.** The codes carry a real but weak
statistical association with sub-tasks, and single-code edits do not repair errors.

### Accuracy, and the reason it is the headline

| Generation condition | Accuracy (2,600 held-out) |
|---|---|
| Steering codes **active** | **83.6%** |
| Steering codes **zeroed** | 83.4% |

Disabling every code changes accuracy by **0.2 percentage points**. The model solved
arithmetic in its weights and routed around the code channel. This is the finding that
governs everything below: a code that carries no causal load cannot be repaired by
editing, and its correlation with sub-task labels is a description of when it happens to
fire, not of what it does. The training signal agrees — information gain sat at
approximately zero throughout (−0.02 at the final step), meaning codes never improved
next-token prediction over the unsteered model.

### M1 — sub-task purity: not replicated

Only **5 of 30 codes** were used at all (≥30 occurrences). With 5 codes spanning 10
sub-task labels, a clean one-code-per-sub-task partition is arithmetically impossible.

```
┌────────┬────────┬──────────┬──────────┬──────────┬────────┬──────────┬────────┐
│  Code  │      n │ Top task │   Purity │ Marginal │   Lift │  Top pos │  #Pos  │
├────────┼────────┼──────────┼──────────┼──────────┼────────┼──────────┼────────┤
│    t20 │   4155 │       MD │   37.2%  │   15.4%  │   2.42 │       d0 │      5 │
│     t6 │    494 │       MB │   36.0%  │    6.2%  │   5.81 │       d6 │      4 │
│     t1 │   2219 │       SA │   33.3%  │   17.8%  │   1.87 │       d6 │      2 │
│     t7 │   6492 │       US │   22.5%  │   12.6%  │   1.78 │       d2 │      4 │
│    t19 │   4840 │       US │   17.2%  │   12.6%  │   1.36 │       d5 │      3 │
└────────┴────────┴──────────┴──────────┴──────────┴────────┴──────────┴────────┘
```

Peak purity is 37.2%, far below the ≥70% that would justify calling a code a
specialist. But the signal is not zero: `t6` fires on borrow-generation at **5.81× its
base rate**, and the median lift across active codes is 1.87×. The structure exists and
is diluted across too few codes rather than being absent.

Codes are also *not* position-locked — each active code appears at 2–5 of the 7 answer
positions.

### M2 — single-code repair: not replicated

426 wrong predictions available; 150 sampled for the sweep.

*Predictive form:*

| Intervention | Repaired |
|---|---|
| Label-matched code | 3 / 584 (0.5%) |
| Random code (matched control) | 2 / 584 (0.3%) |

Three versus two. No effect.

*Existence form (best-of-30 across all positions):*

| Position | d0 | d1 | d2 | d3 | d4 | d5 | d6 |
|---|---|---|---|---|---|---|---|
| Best-of-30 repairs | 3.3% | 2.0% | 1.3% | 0.7% | 0% | 0% | 0% |
| Single random code | 1.3% | 0.7% | 0.7% | 0% | 0% | 0% | 0% |

Even granted 210 attempts per example, the best position repairs 3.3% against a 1.3%
chance baseline. The gradient across positions — highest at d0, zero by d4 — is the one
suggestive pattern, consistent with early digits being more perturbable, but the
magnitudes are too small to carry weight.

---

## Interpretation

The honest reading is that **on this model the codes are close to inert**, and the
interpretability properties therefore have nothing to attach to. This is a coherent
story rather than a set of unrelated nulls: near-zero information gain during training →
no accuracy cost when codes are removed → weak sub-task association → no repair effect.
Each follows from the one before.

The causal direction matters for what this does and does not license. It does **not**
show that a real LLM cannot learn interpretable routing. It shows that *this* model did
not learn to use routing at all, and that when routing is unused, interpretability
questions about it are unanswerable rather than answered in the negative.

The most likely reason is capacity. The published interpretability results come from
models small enough that routing was load-bearing — models that could not solve the task
without it. A 596M-parameter pretrained model can do six-digit arithmetic unaided, so
gradient descent has no pressure to offload computation into a 30-vector codebook.
Interpretable routing may require the model to *need* the routing.

---

## Threats to validity

**A configuration mismatch, and what was done about it.** The run above used the
training script's default loss weights — information-gain 1.0, abstraction 0.5,
diversity 0.01 — whereas the published arithmetic configuration uses 10.0, 0.1, and
1.0. The information-gain weight is the term that pressures codes to be useful and the
diversity weight is what prevents codebook collapse; both were an order of magnitude or
more too low. Inert codes and a 5-of-30 collapse are exactly what those settings
predict. A second model was therefore trained at the published values, and **both runs
are reported** — see §Corrected-configuration run. No other hyperparameter was changed,
and no setting was selected on the basis of producing a better number.

**Measurement bug found and fixed, invalidating earlier figures.** The steering wrapper
defaults its decode-time scale to zero, so unless the trained scale is passed explicitly
at generation time, codes are routed and logged but multiplied by zero before reaching
the residual stream. An initial analysis run this way returned exactly 0/584 repairs in
both the treatment and control arms — which reads as a clean negative but was a no-op.
All figures in this document come from runs with steering verified active during
generation. The tell was that the two arms were *exactly* equal; a genuine null would
show sampling noise between them.

**Evaluation-set ordering.** The eval set is ordered by difficulty split, so any
truncated evaluation samples only the easiest problems. Training-time accuracy of
93–95% was measured this way and reflects no-carry addition alone. Every number here is
over the full 2,600 examples.

**Single seed.** One run per configuration. The accuracy figures are stable enough for
the comparison drawn, but the purity table should not be read at per-code resolution.

---

## Corrected-configuration run

Second model, identical in every respect except the three loss weights, set to the
published values: information-gain 10.0, abstraction 0.1, diversity 1.0.

### Summary against the first run

| | Default weights | Published weights |
|---|---|---|
| Accuracy (2,600 held-out) | 83.6% | **86.3%** |
| Active codes (≥30 uses) | 5 / 30 | **7 / 30** |
| Best purity | 37.2% | **78.3%** |
| — its base rate | 15.4% | 12.6% |
| — lift | 2.42× | **6.21×** |
| Median lift | 1.87× | 1.62× |
| **M1 sub-task purity** | not replicated | **replicated** |
| M2 label-matched repair | 3 / 584 (0.5%) | 1 / 843 (0.1%) |
| M2 random control | 2 / 584 (0.3%) | 2 / 843 (0.2%) |
| **M2 single-code repair** | not replicated | **not replicated** |

### M1 replicates

At the published weights a genuine specialist appears. **Code t6 steers sum-9 carry
cascades 78.3% of the time against a 12.6% base rate — 6.21× lift.** That clears the
≥70% threshold that would let it be called a sub-task specialist, on a 596M-parameter
pretrained model. Codebook usage also improves, from 5 active codes to 7.

The first run's failure was configuration, not scale: the diversity weight was 100×
too low and the information-gain weight 10× too low, which is precisely the recipe for
a collapsed, unused codebook. That run should not be read as evidence about real LLMs.

### M2 does not replicate, and the reason is not noise

The label-matched code repaired 1 of 843 attempts; a random code repaired 2. The
treatment loses to its own control. There is no reading of this in which single-code
repair works.

### Why both are true at once: the codes are inert

Same model, same weights, steering switched on and off at generation time:

| Generation condition | Accuracy (2,600 held-out) |
|---|---|
| Codes active | 86.3% (355 wrong) |
| Codes zeroed at decode | 86.2% (359 wrong) |
| **Difference** | **0.2 pp** |

Removing the codes entirely costs two tenths of a point. This reconciles the split
verdict into a single finding:

> **The router learned to predict sub-task structure without the codes acting on the
> computation.** Code t6 is a reliable sum-9-cascade detector — it reads the input and
> emits an informative symbol — but the steering vector it selects does not change what
> the model computes. The code is a *readout*, not a *control*.

This is why M2 fails without appeal to sampling noise: a computation cannot be repaired
by editing a channel it does not read. It also means M1's replication must be stated
carefully. The codes are interpretable in the sense of being *diagnostic of the input*.
They are not interpretable in the stronger sense of *externalising the mechanism*, which
is the claim that motivates the approach.

The 83.6% → 86.3% accuracy gain between the two runs is therefore a training-recipe
effect, not evidence that codes contribute to the answer — the within-model comparison
above settles that directly.

### Interpretation

On a real pretrained LLM at this scale, DLR produces codes that are **readable but not
load-bearing**. The most likely cause remains capacity: a 596M-parameter model already
solves six-digit arithmetic unaided, so nothing pressures it to route computation
through a 30-vector codebook, however much the information-gain term rewards doing so.
The router, being trained to be predictable from the input, converges on a sub-task
classifier instead — which is genuinely informative, just not mechanism.

A fair statement of what this replication establishes:

- **Yes**, sub-task-specialised codes do emerge outside toy-scale models — 78.3% purity
  at 6.21× lift on a real pretrained LLM.
- **No**, single-code edits do not repair predictions at this scale.
- **And** the codes carry no measurable causal load here, which is the honest reason for
  the second answer and a caveat on the first.

The measurement to run next, given more time, is the same protocol on a model small
enough that arithmetic is genuinely hard — which would test the capacity hypothesis
directly by predicting that load-bearingness, and with it repairability, returns.

---

## Cross-study summary

The same protocol was run on a second, unrelated domain — Python source from CodeNet,
with per-chunk syntactic constructs as ground truth in place of arithmetic sub-tasks
(see `codenet.md`). The two studies reach the same conclusion by different routes,
which is the strongest form the evidence takes.

| | Arithmetic | CodeNet |
|---|---|---|
| Structure type | algorithmic (carry/borrow) | syntactic (AST constructs) |
| Codebook behaviour | **collapsed** — 7 of 30 active | **healthy** — 12–15 of 30 active |
| Best specialist | sum-9 cascade, **78.3%**, 6.21× | function definition, 35.1%, 3.84×, **1 of 32 positions** |
| Median lift | 1.62× | 1.06× |
| Sub-task purity | **replicates** | does not replicate |
| Single-code repair | does not replicate | underpowered (0 vs 0) |
| **Causal load** | **+0.2 pp** | **−0.6 pp / −0.3 pp** |

The intermediate results differ in nearly every respect — one codebook collapses and the
other does not; purity replicates in one domain and not the other. The final measurement
does not differ: across three models and two domains, **removing the codes entirely
changes accuracy by at most two tenths of a percentage point**, and twice the change is
negative.

A single null invites the objection that the task, scale, or configuration was wrong.
Two nulls reached through different codebook behaviours and opposite purity outcomes
are much harder to attribute to any one of those. The consistent finding is that the
routing head learns to *predict* structure while the codes do no work on the
computation — a readout rather than a control.

---

## Method note: decode-time knockout

The knockout above zeroes the steering scale during *generation*, which is the condition
that matters for answer digits — each is produced at a decode step steered by exactly one
code. Prefill-time steering over the 14 prompt tokens is left intact, so this measures
the causal contribution of the codes at the positions being predicted, not of the whole
steering apparatus.

---

## Reproduction

```bash
bash amir_interp_rebuttal/run.sh arith          # default-configuration run
bash amir_interp_rebuttal/chain.sh              # corrected configuration + CodeNet
```

Analysis code: `amir_interp_rebuttal/{interp,runner,analyze}.py`.
Data and labels: `amir_interp_rebuttal/arith_dataset.py`, built on the existing
six-digit generator with its sub-task labelling intact. Raw results:
`amir_interp_rebuttal/results/arithmetic_r1r2.json`.
