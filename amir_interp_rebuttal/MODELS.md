# MODELS.md — reported checkpoints

HuggingFace repo: **[`thoughtworks/dlr-rebuttal-interp`](https://huggingface.co/thoughtworks/dlr-rebuttal-interp)** — **pushed, private**, one subfolder per
checkpoint tag. Naming follows the existing project convention
(`thoughtworks/arithmetic-sorl`, `thoughtworks/arithmetic-sorl-data`): one model
repo per study family, subfolders per run, config and metrics beside the weights.

Push tooling: `amir_interp_rebuttal/push_models.py`. **Dry run is the default** —
`--push` is required to upload anything.

```bash
python -m amir_interp_rebuttal.push_models              # dry run, recommended set
python -m amir_interp_rebuttal.push_models --all        # dry run, everything
python -m amir_interp_rebuttal.push_models --push       # actually upload
```

The registry inside `push_models.py` is the single source of truth for what gets
pushed and why; it is printed on every dry run, so this file and the tool cannot
disagree silently.

---

## 1. Registry

Two checkpoints are published — the two that the deliverables actually cite.
Everything else is a supporting row in a table or a negative result, and stays
local.

```
┌──────────────────────────────────┬────────────┬───────┬──────────────────────────────┐
│ Local checkpoint                 │ Study      │ Push? │ Role                         │
├──────────────────────────────────┼────────────┼───────┼──────────────────────────────┤
│ ckpt/codenet_s0.5_i10_z1_L8_n4000│ codenet    │ YES   │ the causal result (gate open)│
│ ckpt/arith_s0.5_i10_z1_u8        │ arithmetic │ YES   │ arithmetic causal (gate open)│
│ ckpt/arith_v9_paperhp            │ arithmetic │ YES   │ arithmetic R1 / #5 / #6 / #7 │
├──────────────────────────────────┼────────────┼───────┼──────────────────────────────┤
│ ckpt/arith_18d_500_MAX           │ arithmetic │ hold  │ escalation row 4             │
│ ckpt/arith_12d_10k               │ arithmetic │ hold  │ escalation row 2             │
│ ckpt/arith_12d_10k_s0.1_i10_z10u8│ arithmetic │ hold  │ escalation row 3             │
├──────────────────────────────────┼────────────┼───────┼──────────────────────────────┤
│ ckpt/codenet_FROZEN              │ codenet    │ hold  │ negative result, on record   │
│ ckpt/codenet_v9                  │ codenet    │ hold  │ superseded (scale=0.1)       │
│ ckpt/arith_v9                    │ arithmetic │ hold  │ no longer cited              │
│ ckpt/codenet_v9_20k              │ codenet    │ hold  │ budget control               │
└──────────────────────────────────┴────────────┴───────┴──────────────────────────────┘
```

**Rule: only push a checkpoint whose numbers appear in a deliverable.** A
checkpoint that supports one row of a sweep table is documented by its knockout
JSON, which is kilobytes; publishing 1.2 GB of weights for it is a bad trade.

## 2. Training config

Read from each checkpoint's own `args` dict. All share base model
`Qwen/Qwen3-0.6B` (596M) and `inject_layers=14` of 28 — **fixed a priori and
never swept**, so no test information enters the layer choice. `digits` and
`train size` are not in `args`; the datasets read `ARITH_DIGITS` / `ARITH_SIZE` /
`CODENET_SIZE` from the environment, so those columns are reconstructed from the
launch command and the run log.

```
┌──────────────────────────────┬────┬───┬───────┬────────┬───────┬────────┬──────────┬──────────┬────────┬───────┬───────────────┬────────────┐
│ Checkpoint                   │  C │ L │ scale │ a_info │ a_abs │ a_zipf │ tgt_util │ steer_lr │ epochs │ batch │ digits/chunk  │ train size │
├──────────────────────────────┼────┼───┼───────┼────────┼───────┼────────┼──────────┼──────────┼────────┼───────┼───────────────┼────────────┤
│ codenet_s0.5_i10_z1_L8_n4000 │ 30 │ 8 │   0.5 │   10.0 │   0.1 │    1.0 │      0.8 │     1e-3 │      2 │     8 │  8-tok chunk  │      4,000 │
│ arith_v9_paperhp             │ 30 │ 1 │   0.1 │   10.0 │   0.1 │    1.0 │    unset │     1e-3 │      1 │    32 │      6-digit  │    100,000 │
├──────────────────────────────┼────┼───┼───────┼────────┼───────┼────────┼──────────┼──────────┼────────┼───────┼───────────────┼────────────┤
│ arith_12d_10k                │ 30 │ 1 │   0.1 │   10.0 │   0.1 │    1.0 │    unset │     1e-3 │     10 │    32 │     12-digit  │     10,000 │
│ arith_12d_10k_s0.1_i10_z10u8 │ 30 │ 1 │   0.1 │   10.0 │   0.1 │   10.0 │      0.8 │     1e-3 │     10 │    32 │     12-digit  │     10,000 │
│ arith_18d_500_MAX            │ 30 │ 1 │   1.0 │   30.0 │   0.1 │   20.0 │      0.9 │     1e-3 │    200 │    32 │     18-digit  │        500 │
├──────────────────────────────┼────┼───┼───────┼────────┼───────┼────────┼──────────┼──────────┼────────┼───────┼───────────────┼────────────┤
│ codenet_FROZEN  (--freeze)   │ 30 │ 8 │   1.0 │   30.0 │   0.1 │   20.0 │      0.9 │     1e-2 │      8 │     8 │  8-tok chunk  │      4,000 │
│ codenet_v9                   │ 30 │ 8 │   0.1 │   10.0 │   0.1 │    1.0 │    unset │     1e-3 │      1 │     8 │  8-tok chunk  │      4,000 │
└──────────────────────────────┴────┴───┴───────┴────────┴───────┴────────┴──────────┴──────────┴────────┴───────┴───────────────┴────────────┘
```

`target_vocab_util` and `alpha_zipf` did not exist as flags when the `unset` rows
ran — the Zipf prior used its hardcoded 0.8 target. Both are now exposed on
`train_steer_pt.py`.

## 3. Headline metrics

```
┌──────────────────────────────┬──────────┬──────────────┬────────────────────────────────┬──────────────────────┐
│ Checkpoint                   │ Accuracy │ Active codes │ Best code                      │ Knockout (OFF_full)  │
├──────────────────────────────┼──────────┼──────────────┼────────────────────────────────┼──────────────────────┤
│ codenet_s0.5_i10_z1_L8_n4000 │   17.50% │  8 (23 used) │ t5 -> If 1.88x position-matched│ −6.87pp / −39.3% rel │
│ arith_v9_paperhp             │   86.35% │       7 / 30 │ t6 -> US 78.3% (6.21x lift)    │ +0.15pp / +0.2%  rel │
├──────────────────────────────┼──────────┼──────────────┼────────────────────────────────┼──────────────────────┤
│ arith_12d_10k                │   82.96% │      10 / 30 │ not measured                   │ +0.54pp / +0.6%  rel │
│ arith_12d_10k_s0.1_i10_z10u8 │      —   │ not measured │ not measured                   │ +0.23pp / +0.3%  rel │
│ arith_18d_500_MAX            │      —   │ not measured │ not measured                   │ +1.69pp / +5.9%  rel │
├──────────────────────────────┼──────────┼──────────────┼────────────────────────────────┼──────────────────────┤
│ codenet_FROZEN               │    9.25% │  — (23 used) │ not measured                   │ −1.12pp / −12.2% rel │
└──────────────────────────────┴──────────┴──────────────┴────────────────────────────────┴──────────────────────┘
```

Reading the columns:

- **Accuracy** is exact match, generated autoregressively from the model's own
  predictions, no teacher forcing.
- **Active codes** = codes above the 1%-of-firings threshold; "used" = codes that
  fire at all.
- **Knockout** = `acc(codes ON) − acc(steering_emb zeroed)`. Negative means
  removing the codes *costs* accuracy, i.e. the codes are load-bearing. The gate
  opens at ≥3pp **or** ≥15% relative.

**Only `codenet_s0.5_i10_z1_L8_n4000` opens the gate.** Every causal claim in the
rebuttal rests on it; the arithmetic checkpoints support claims about what the
codes *encode*, which do not depend on causal load.

### Accuracy is quoted per-harness, deliberately

The gated CodeNet checkpoint has three accuracies on disk — 17.50%, 17.0% and
19.5% — and they are not in conflict. All three are batch-1 measurements of the
same weights under different generation lengths (`--max_new_tokens` 32 for the
knockout, 8 for `codenet_confound.py` and `analyze.py`). Each table quotes the
figure from the run that produced the rest of its numbers; the four knockout arms
all come from a single file, so that comparison is internally consistent.

## 4. Which deliverable cites which checkpoint

```
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ Checkpoint                   │ Deliverable / table                                    │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ codenet_s0.5_i10_z1_L8_n4000 │ REBUTTAL_codenet.md — Finding #2 knockout, R1 purity   │
│                              │ repro/knockout.sh, repro/f3_codenet_purity.sh          │
│ arith_v9_paperhp             │ REBUTTAL_arithmetic.md — R1, Findings #5, #6, #7       │
│                              │ repro/r1_purity.sh, r5_sum9.sh, f6_polysemanticity.sh  │
│ arith_12d_10k                │ REBUTTAL_arithmetic.md — escalation table, row 2       │
│ arith_12d_10k_s0.1_i10_z10u8 │ REBUTTAL_arithmetic.md — escalation table, row 3       │
│ arith_18d_500_MAX            │ REBUTTAL_arithmetic.md — escalation table, row 4       │
│ codenet_FROZEN               │ MODELS.md — negative result (below)                    │
│ codenet_v9, arith_v9,        │ none — retained for provenance only                    │
│ codenet_v9_20k               │                                                        │
└──────────────────────────────┴────────────────────────────────────────────────────────┘
```

---

## Negative results, kept on the record

### The abstraction ratio was swept: 1:8 is where the codes are load-bearing

`L` is the number of trajectory tokens per abstraction code — the reasoning-to-
trajectory ratio. It is baked into the checkpoint, so each value is a separate
training run and a separate gate measurement, all on the identical recipe
(`scale=0.5, alpha_info=10, alpha_zipf=1.0, target_vocab_util=0.8`, 2 epochs,
4,000 files) with only `L` changed.

```
  ┌────────┬─────┬──────────┬──────────┬──────────┬──────────────────┬────────┐
  │ Ratio  │  L  │ codes ON │  RANDOM  │ OFF_full │ knockout         │  gate  │
  ├────────┼─────┼──────────┼──────────┼──────────┼──────────────────┼────────┤
  │  1:4   │   4 │   10.62% │   10.50% │   10.00% │ +0.62pp /  5.9%  │ closed │
  │  1:8   │   8 │   17.50% │   11.13% │   10.62% │ +6.87pp / 39.3%  │  OPEN  │
  │  1:16  │  16 │    9.88% │    6.25% │    7.50% │ +2.38pp / 24.1%  │  OPEN  │
  └────────┴─────┴──────────┴──────────┴──────────┴──────────────────┴────────┘
```

1:4 fails on both axes at once: the model trains to 10.62% against 1:8's 17.50%,
*and* its codes stop being load-bearing (5.9% relative, against a 15% bar).

1:16 does open the gate, at 24.1% relative, and its RANDOM arm is striking —
6.25% is *below* the 7.50% of removing steering entirely, so scrambling which
code is which is worse than having no codes at all. But it reaches 9.88%
accuracy, so that relative figure sits on a small denominator: 2.38pp absolute
against 1:8's 6.87pp. A large ratio on a model that barely performs the task is
weaker evidence than a large ratio on one that does.

This is not a negative result to be reported as a finding — it is the
justification for the reported configuration. `L=8` is not an unexamined default;
it is the setting at which the codes carry the computation, and the two
neighbouring ratios were trained and gated to establish that.

### `arith_HARD_cascade_s0.5` — enriching for cascades made the task EASIER

The intent was right and the implementation was wrong. To make the codes more
meaningful, the training data was reject-sampled for multi-column carry
cascades (`US`) and borrow propagation (`UD`), with `ARITH_AUG_PROB=0.8`
forcing digit pairs to sum to 9 — the cascade seed.

At that rate the distribution collapses. Forcing `y = 9 − x` on 80% of columns
makes `x + y = 999999`, so almost every answer is `1000000` plus a small
remainder:

```
  824475 + 175526 = 1000001      817069 + 182939 = 1000008
  422314 + 587688 = 1010002      035780 + 964819 = 1000599
```

The model learns to emit `100…` and reaches **99.35%** accuracy — the opposite
of harder. All three knockout arms returned an identical 0.9935, which is the
signature of a task so saturated that the steering channel is irrelevant.

Diagnosis: cascade frequency and operand diversity trade off, and the forcing
mechanism buys the first by destroying the second.

```
  ┌────────────────────┬───────┬───────┬────────────────┬──────────────────┐
  │ config             │   US  │   UD  │ distinct ans.  │ most common      │
  │                    │       │       │ prefixes/1200  │ prefix share     │
  ├────────────────────┼───────┼───────┼────────────────┼──────────────────┤
  │ baseline aug=0.4   │  4.3% │  2.9% │           180  │              8%  │
  │ hard=2 aug=0.2     │ 17.1% │ 15.6% │           152  │             15%  │
  │ hard=3 aug=0.35    │ 32.7% │ 13.7% │           116  │             34%  │
  │ hard=3 aug=0.8     │     — │     — │           125  │             26%  │
  └────────────────────┴───────┴───────┴────────────────┴──────────────────┘
```

`hard=2, aug=0.2` is the setting actually used: cascades enriched 4–5× over
baseline with the answer distribution intact. Rejection does the selecting;
forcing is kept low enough not to dictate the operands.

### Difficulty × scale: the grid, completed

The escalation sweep varied difficulty at `scale=0.1` and the scale sweep varied
scale at 6 digits, so each axis was tested at the other's dead value. Crossing
them:

```
  ┌───────────┬──────────────────────┬──────────────────────────────────┐
  │           │      scale 0.1       │            scale 0.5             │
  ├───────────┼──────────────────────┼──────────────────────────────────┤
  │  6-digit  │  +0.15pp   closed    │  −13 to −15pp  OPEN (2 retrains) │
  │ 12-digit  │  +0.54pp   closed    │  +0.23pp       closed            │
  │ 18-digit  │  +1.69pp   closed    │  not run                         │
  └───────────┴──────────────────────┴──────────────────────────────────┘
```

**Difficulty is not the axis.** At 12 digits the model reaches 70.08% — far from
saturated — and its codes remain inert at the scale that works for 6 digits
(ON 70.08 / RANDOM 69.96 / OFF 69.85). The arithmetic causal effect is specific
to 6-digit × `scale=0.5`, where it replicated across two independent retrains.

A harness bug was found here and is worth recording: `gate_then_repair.sh`
hardcoded `max_new_tokens=8`, which truncates any answer past 7 digits and
returns **0.0% in every arm** — indistinguishable from "the codes do nothing".
The script now derives generation length from `ds.answer_len`.

### Cascade enrichment does not create difficulty

`hard=2, aug=0.2` enriches carry cascades 4-5x over baseline while keeping the
answer distribution intact — but the resulting model reaches **97.31%**, well
above the 83% of unenriched data, and its gate is closed (+1.38pp). Selecting
for cascades selects for structure the model exploits. Two settings tried
(`aug=0.8` degenerate at 99.35%, `aug=0.2` merely easy at 97.31%); neither makes
the task harder.

### `codenet_FROZEN` — freezing the backbone does not force causal load

The idea: freeze all 596M model parameters and train only the 61K steering
parameters, so the model *cannot* adapt its weights to route around the
codebook and must carry information through the codes. It was worth trying and
it did not work.

What it got right: **codebook diversity**. 23 of 30 codes are used, against 8–11
active in the trainable-backbone runs. With no weight updates available the model
genuinely cannot route around the codebook.

What killed it: at `scale=1.0`, decode-time steering destroys generation. Three
arms from one settings-matched run —

```
  ┌────────────────────────────────┬──────────┬──────────────────────────────┐
  │ Arm                            │ accuracy │ what is steered              │
  ├────────────────────────────────┼──────────┼──────────────────────────────┤
  │ prefill steered, decode off    │   25.50% │ prompt only                  │
  │ no steering at all             │   10.37% │ nothing                      │
  │ fully steered (codes ON)       │    9.25% │ prompt + generation          │
  └────────────────────────────────┴──────────┴──────────────────────────────┘
```

Prefill steering is worth **+15pp** over no steering; adding decode steering at
this magnitude costs **16pp**. So the steering vectors do carry usable
information into the prompt representation — it is the decode-time injection at
`scale=1.0` that wrecks the model. The reported CodeNet checkpoint uses
`scale=0.5`, which is the regime where the codes help.

Do **not** compare 9.25% against the 15.75% untrained baseline quoted elsewhere:
that baseline was measured under different generation settings and is not
matched to these arms. The three rows above are matched, and are the comparison
to quote.

### The frozen arithmetic counterpart was deliberately not completed

A frozen Qwen3-0.6B cannot produce the `123456+654321=` answer format at all, so
its knockout would measure *"the codes taught the output format"* rather than
*"the codes carry carry-logic"*. The control needed to separate those does not
exist, so the result would not be reportable under this study's own rules. The
run was started, OOM'd against concurrent jobs, and was not relaunched.

### The arithmetic gate never opened

Four rungs, optimizer steps held roughly constant so training budget is not the
variable: +0.15pp → +0.54pp → +0.23pp → +1.69pp. Causal load rises ~11× with
difficulty and steering scale, in the predicted direction, without reaching the
3pp threshold. A 596M pretrained backbone solves six-digit arithmetic with enough
slack that the codes never have to carry it. Reproduce with
`repro/f2_escalation.sh`.

---

## What is actually in `final.pt`, and what gets uploaded

`final.pt` is **3.58 GB**, not the ~140 MB the parameter count suggests:

```
┌───────────────────────┬─────────┬──────────────────────────────────────┐
│ Component             │    Size │ Read by                              │
├───────────────────────┼─────────┼──────────────────────────────────────┤
│ model (bf16)          │ 1.50 GB │ load_local_steered, every analysis   │
│ optimizer (fp32 Adam) │ 2.38 GB │ train_steer_pt.py --resume_from only │
│ steering_emb          │ 0.12 MB │ load_local_steered                   │
│ abs_proj + args       │  < 1 MB │ load_local_steered                   │
└───────────────────────┴─────────┴──────────────────────────────────────┘
```

The optimizer state is 67% of the file and no interpretability code path reads
it. `push_models.py` **strips it by default**, taking each upload from 3.58 GB to
~1.19 GB; the result still loads through `load_local_steered` but will not resume
training. Pass `--keep-optimizer` to publish resumable weights.

Also uploaded per checkpoint: `history.json`, `steer_v9.pt` (the steering wrapper
alone, ~190 KB), and an auto-generated `README.md` model card carrying the exact
`args` config, the task, each metric with the results file it was read from, and
an explicit "what this model does NOT show" section.
