# MODELS.md - reported checkpoints

HuggingFace repo: **`thoughtworks/dlr-rebuttal-interp`**, one subfolder per checkpoint tag. Naming follows the
existing project convention (`thoughtworks/arithmetic-sorl`,
`thoughtworks/arithmetic-sorl-data`): one model repo per study family, subfolders
per run, config and metrics shipped beside the weights.

Push tooling: `amir_interp_rebuttal/push_models.py`. **Dry run is the default** -
`--push` is required to upload anything, and the tool refuses to push a HOLD
checkpoint without an interactive confirmation.

```bash
python -m amir_interp_rebuttal.push_models              # dry run, recommended set
python -m amir_interp_rebuttal.push_models --all        # dry run, everything
python -m amir_interp_rebuttal.push_models --push       # actually upload
```

---

## READ THIS FIRST - two headline checkpoints are PROVISIONAL

**`arith_v9_paperhp` (arithmetic) and `codenet_v9` (CodeNet) are the current
headline checkpoints and both may be superseded.** Two gate sweeps are running:

- `sweep_gate.py` - 7-rung arithmetic ladder. Rung 1 (`arith_12d_10k`) is done
  and its gate is closed (-0.23 pp). Rungs 2-7 pending.
- `codenet_sweep_gate.py` - CodeNet ladder, plus a stronger knockout arm
  (random-code substitution and full `steering_emb` zeroing, in prefill as well
  as decode) that may move the reported knockout number for `codenet_v9`.

If either sweep opens its gate, R1/R2/R3/R5 are regenerated on the gated
checkpoint and **the reported checkpoint changes**. Publishing a headline model
whose number is about to be superseded is worse than publishing nothing, so the
PROVISIONAL rows below should be pushed only once the sweeps have settled, or
pushed now with the understanding that their cards will need reissuing.

### Every knockout number in section 3 is a *decode-only* ablation

Checked against the logs that produced them
(`logs/knockout_paperhp.log`, `logs/codenet_knockout_125.log`,
`logs/arith_12d_10k_knockout.log`): all four report

```
  codes_ON   decode_scale=0.1
  codes_OFF  decode_scale=0.0
```

`StackedAbstractionWrapperV9.generate` documents `decode_scale` as leaving
**prefill untouched** — it silences steering during generation only. So in both
arms the prompt is still fully steered, and the reported delta bounds a weaker
claim than "the codes are not load-bearing".

`codenet_sweep_gate.py` already identifies this and now measures four arms
(`ON` / `OFF_decode` / `OFF_full` with `steering_emb` zeroed / `RANDOM`), with
`OFF_full` setting the gate. **`sweep_gate.py` (arithmetic) has not been updated**
— its `KNOCKOUT_SRC` still runs the two-arm version, so rungs 2–7 of the
arithmetic ladder are having their gate decided on the weaker ablation.

The effect is smaller on arithmetic than on CodeNet (the labelled structure *is*
the generated answer, whereas CodeNet's source sits almost entirely in the
prompt), but it is not zero: with `L=1` the 14-token arithmetic prompt carries 14
steered chunks in both arms. Until the arithmetic knockout is re-measured with
`OFF_full`, treat **+0.15 pp and −0.23 pp as lower bounds on the true knockout**,
and note that the whole R2-is-gated argument rests on them.

---

## 1. Registry

```
┌───────────────────────┬───────────────────────────────────────────────────┬────────────┬───────┬──────────────────┐
│ Local checkpoint      │ HF repo path                                      │ Study      │ Push? │ Status           │
├───────────────────────┼───────────────────────────────────────────────────┼────────────┼───────┼──────────────────┤
│ ckpt/arith_v9_paperhp │ thoughtworks/dlr-rebuttal-interp/arith_v9_paperhp │ arithmetic │ YES   │ PROVISIONAL      │
│ ckpt/arith_v9         │ thoughtworks/dlr-rebuttal-interp/arith_v9         │ arithmetic │ YES   │ stable           │
│ ckpt/codenet_v9       │ thoughtworks/dlr-rebuttal-interp/codenet_v9       │ codenet    │ YES   │ PROVISIONAL      │
├───────────────────────┼───────────────────────────────────────────────────┼────────────┼───────┼──────────────────┤
│ ckpt/codenet_v9_20k   │ (not pushed)                                      │ codenet    │ hold  │ see reason below │
│ ckpt/arith_12d_10k    │ (not pushed)                                      │ arithmetic │ hold  │ see reason below │
└───────────────────────┴───────────────────────────────────────────────────┴────────────┴───────┴──────────────────┘
```

## 2. Training config

Read from each checkpoint's own `args` dict. `tgt_util` = `target_vocab_util`:
the flag did not exist when any of these ran, so the Zipf prior used its
hardcoded 0.8 target. `digits` and `train size` are **not** in `args` - the
datasets read `ARITH_DIGITS` / `ARITH_SIZE` / `CODENET_SIZE` from the
environment, so those two columns are reconstructed from the launch command and
the run log.

All five share: base model `Qwen/Qwen3-0.6B` (596M), `inject_layers=14` of 28
(fixed a priori, never swept), `steer_lr=1e-3`, `num_rollouts=4`.

```
┌──────────────────┬──────┬────┬───┬───────┬────────┬───────┬────────┬──────────┬──────┬────────┬───────┬───────┬────────────────┬────────────┐
│ Checkpoint       │ mode │  C │ L │ scale │ a_info │ a_abs │ a_zipf │ tgt_util │   lr │ epochs │ batch │ steps │ digits / chunk │ train size │
├──────────────────┼──────┼────┼───┼───────┼────────┼───────┼────────┼──────────┼──────┼────────┼───────┼───────┼────────────────┼────────────┤
│ arith_v9_paperhp │   v9 │ 30 │ 1 │   0.1 │   10.0 │   0.1 │    1.0 │    unset │ 1e-5 │      1 │    32 │  3125 │        6-digit │    100,000 │
│ arith_v9         │   v9 │ 30 │ 1 │   0.1 │    1.0 │   0.5 │   0.01 │    unset │ 1e-5 │      1 │    32 │  3125 │        6-digit │    100,000 │
│ arith_12d_10k    │   v9 │ 30 │ 1 │   0.1 │   10.0 │   0.1 │    1.0 │    unset │ 1e-5 │     10 │    32 │  3130 │       12-digit │     10,000 │
├──────────────────┼──────┼────┼───┼───────┼────────┼───────┼────────┼──────────┼──────┼────────┼───────┼───────┼────────────────┼────────────┤
│ codenet_v9       │   v9 │ 30 │ 8 │   0.1 │   10.0 │   0.1 │    1.0 │    unset │ 1e-5 │      1 │     8 │   125 │    8-tok chunk │      4,000 │
│ codenet_v9_20k   │   v9 │ 30 │ 8 │   0.1 │   10.0 │   0.1 │    1.0 │    unset │ 1e-5 │      1 │     8 │   625 │    8-tok chunk │     20,000 │
└──────────────────┴──────┴────┴───┴───────┴────────┴───────┴────────┴──────────┴──────┴────────┴───────┴───────┴────────────────┴────────────┘
```

## 3. Headline metrics

```
┌──────────────────┬────────┬──────────┬──────────────┬──────────────────────────────────┬────────────────────────────┬────────────────┐
│ Checkpoint       │ n_eval │ Accuracy │ Active codes │ Best code (global lift)          │ Position-matched           │ Knockout delta │
├──────────────────┼────────┼──────────┼──────────────┼──────────────────────────────────┼────────────────────────────┼────────────────┤
│ arith_v9_paperhp │   2600 │   86.35% │       7 / 30 │ t6 -> US 78.3% (6.21x)           │ survives (pos 1 contested) │       +0.15 pp │
│ arith_v9         │   2600 │    83.6% │       5 / 30 │ t20 -> MD 37.2% (2.42x)          │ not audited                │   not measured │
│ arith_12d_10k    │   2600 │   82.96% │      10 / 30 │ not measured                     │ not audited                │       -0.23 pp │
├──────────────────┼────────┼──────────┼──────────────┼──────────────────────────────────┼────────────────────────────┼────────────────┤
│ codenet_v9       │    800 │   22.13% │      12 / 30 │ t13 -> BinOp 44.7% (3.31x), n=38 │ t5 -> If 1.70x, n=457      │       -0.63 pp │
│ codenet_v9_20k   │    800 │    4.88% │      15 / 30 │ t12 -> Call 38.2% (1.30x)        │ not audited                │       -0.25 pp │
└──────────────────┴────────┴──────────┴──────────────┴──────────────────────────────────┴────────────────────────────┴────────────────┘
```

Reading the columns:

- **Accuracy** is exact match, generated autoregressively from the model's own
  predictions with no teacher forcing.
- **Active codes** = codes used for at least 1% of firings, out of `C_SIZE=30`.
- **Best code (global lift)** = `P(label|code)` divided by that label's base rate
  over the whole eval set. This is the number that is easy to quote and easy to
  get wrong.
- **Position-matched** = the same purity divided by `P(label | the positions that
  code actually fires at)`. It is the column that decides whether a purity claim
  is learned structure or an inherited fact about the data.
- **Knockout delta** = `acc(codes ON) - acc(codes zeroed)`, in percentage points.
  A positive number means the codes help. **No checkpoint here exceeds
  +/-0.63 pp.**

## 4. Which deliverable cites which checkpoint

```
┌──────────────────┬──────────────────────────────────────────────────────────────────────┐
│ Checkpoint       │ Deliverable / table that cites it                                    │
├──────────────────┼──────────────────────────────────────────────────────────────────────┤
│ arith_v9_paperhp │ addition_followup.md - published-weights arm, the headline           │
│ arith_v9_paperhp │ r1_r3_r5_tables.md - R1 purity, R3 position, R5 sum-9, knockout      │
│ arith_v9_paperhp │ PLAN_arithmetic.md - Status, Honest current position                 │
│ arith_v9         │ addition_followup.md - default-weights arm of the weights comparison │
├──────────────────┼──────────────────────────────────────────────────────────────────────┤
│ codenet_v9       │ codenet.md - the 125-step run                                        │
│ codenet_v9       │ codenet_gate.md - position-confound audit (Results 1-4)              │
│ codenet_v9       │ r1_r3_r5_tables.md - knockout appendix                               │
│ codenet_v9       │ PLAN_codenet.md - Status, Honest current position                    │
│ codenet_v9_20k   │ codenet.md - matched-budget control (5x budget, worse)               │
│ codenet_v9_20k   │ r1_r3_r5_tables.md - knockout appendix                               │
├──────────────────┼──────────────────────────────────────────────────────────────────────┤
│ arith_12d_10k    │ none - results/sweep_gate_summary.json only (rung 1 of 7)            │
└──────────────────┴──────────────────────────────────────────────────────────────────────┘
```

---

## Per-checkpoint notes

### `arith_v9_paperhp` - PUSH, **PROVISIONAL**

The arithmetic headline. Published loss weights. One code (t6) is a real sub-task
specialist: 78.3% pure on the sum-9 carry-cascade label against a 12.6% base
rate, 6.21x lift, at answer position 1 - the one position where routing is
input-dependent rather than a position tag.

What it does not show: knockout is +0.15 pp, so the codes are a **read-out, not a
control**. R2 single-code repair is 1/843 label-matched against 2/843 random -
the treatment loses to its own control, and because R2 is gated on the knockout
that null is *uninformative*, not a negative result.

Provisional because `sweep_gate.py` may replace it.

### `arith_v9` - PUSH, stable

Default (misconfigured) loss weights: `alpha_zipf` 100x too low and `alpha_info`
10x too low relative to the published recipe. 5 of 30 codes active, peak purity
37.2% at a 15.4% base rate. Published as the **negative control** for the claim
that the first run's failure was configuration and not model scale. That claim is
only checkable if both arms are downloadable, and this arm is a fixed contrast
that no sweep result can supersede - hence not provisional.

### `codenet_v9` - PUSH, **PROVISIONAL**

The CodeNet checkpoint the position-confound audit ran on. **Its previously
headlined result is withdrawn:** `t20 -> FunctionDef 35.1% purity, 3.84x lift`
does not survive. t20 fires at exactly 1 of 32 chunk positions, and
`P(FunctionDef | chunk 0)` is 41.2% before any code is consulted, so t20's
`lift_pos` is 0.85x - below its own baseline. Worse, t20 fires at chunk 0 **iff**
the batch row's left-pad length is an exact multiple of `L=8` (228/228 aligned vs
0/572 misaligned, perfect separation over 800 files): it is a padding-alignment
detector, not a syntactic one.

What survives: **t5**, `If` at 25.4% purity vs a position-matched 14.9%,
`lift_pos = 1.70x`, n=457 over 31 positions, Bonferroni p=7.7e-7. Real,
position-independent, and weak - far below R1's 70% purity bar, and a generalist
code rather than a specialist. t13 (`BinOp`, `lift_pos = 2.22x`) has n=38 and does
not survive multiple-comparison correction.

Provisional because `codenet_sweep_gate.py` is running and a stronger knockout arm
is in flight.

**Accuracy bookkeeping.** Three files report three accuracies for this one
checkpoint: 0.22125 (`codenet_125step_knockout.json`), 0.2220
(`codenet_r1r2_125step.json`), 0.23625 (`codenet_position_confound.json`). They
come from three different harness invocations with different batching. Quote the
figure from the file the rest of the number came from, and reconcile these before
the CodeNet deliverable is finalised.

### `codenet_v9_20k` - **HOLD**

4.9% exact-match accuracy - below the CodeNet gate's own 10% analysis floor. Its
only role is the control showing that 5x the optimizer budget made things worse
on both axes (accuracy 22.1% -> 4.9%, peak lift 3.84x -> 1.30x), which rules out
undertraining as the explanation for the weak 125-step result. That claim is
fully documented by `history.json` and `train.log`, which are kilobytes.
Publishing 1.5 GB of degenerate weights to support one "it got worse" sentence is
a bad trade. Push only if a reviewer asks to verify the budget control directly.

### `arith_12d_10k` - **HOLD**

Rung 1 of a 7-rung sweep that is still running. Its gate is closed (-0.23 pp), so
it will never be the reported checkpoint, and no number from it appears in any
deliverable - it is one row in `results/sweep_gate_summary.json`. Publish the rung
that opens the gate, if one does. If none does, publish nothing from the sweep:
the reportable claim in that case is "unmeasurable at this scale", which needs the
summary JSON, not seven sets of weights.

---

## What is actually in `final.pt`, and what gets uploaded

`final.pt` is **3.58 GB**, not the ~140 MB one might assume from the parameter
count:

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
it. `push_models.py` therefore **strips it by default**, taking each upload from
3.58 GB to ~1.19 GB; the result still loads through `load_local_steered` but will
not resume training. Pass `--keep-optimizer` to publish resumable weights.

Also uploaded per checkpoint: `history.json` (when the run wrote one - `arith_v9`
did not), `steer_v9.pt` (the steering wrapper alone, ~190 KB - `arith_v9` predates
it), and an auto-generated `README.md` model card carrying the exact `args` config,
the task, the metrics with the results file each was read from, and an explicit
"what this model does NOT show" section.

## Rule

**Only push a checkpoint whose numbers appear in a deliverable, and whose numbers
are not about to be superseded.** Everything else stays local. The recommendation
for each checkpoint lives in the `CHECKPOINTS` registry in `push_models.py` and is
printed on every dry run, so this file and the tool cannot disagree silently.
