# CodeNet: position-confound audit and causal gate sweep

Qwen3-0.6B, DLR v9 residual steering, `inject_layers=14`, `C_SIZE=30`, `L=8`,
`scale=0.1`. Regenerate with `amir_interp_rebuttal/codenet_confound.py` and
`amir_interp_rebuttal/codenet_sweep_gate.py`.

> **Scope: this is the audit trail, not a deliverable.** It documents the
> investigation on `ckpt/codenet_v9` (`scale=0.1`) that (a) withdrew the
> `t20 -> FunctionDef 3.84x` result as a padding-alignment artefact and (b)
> motivated the scale sweep. That sweep produced
> `ckpt/codenet_s0.5_i10_z1_L8_n4000` (`scale=0.5`), which is the checkpoint
> every reported CodeNet number now comes from — see
> [REBUTTAL_codenet.md](REBUTTAL_codenet.md). Numbers below describe the
> superseded checkpoint and are retained as provenance for the withdrawal.

---

# Task 1 — is CodeNet's t20 the same position confound as arithmetic?

## Answer: yes, exactly the same confound, and totally so.

On arithmetic, position locking was degenerate: at 6 of 7 answer positions a
single code covered ~100% of examples, so `P(label|code)` was really
`P(label|position)` and the reported purity was a position artifact. The CodeNet
headline — **t20 -> FunctionDef, 35.1% purity, 3.84x lift** — has to survive the
same control.

It does not. Measured without the padding artifact described in §1.2:

* **t20 fires at exactly 1 of 32 chunk positions, and covers 100.0% of it**
  (800 of 800 files). It is a deterministic function of chunk position.
* Its purity is **41.2%**, and `P(FunctionDef | chunk 0) = 41.2%`. Identical to
  three decimal places. **`lift_pos = 1.00x`.**
* The reported 4.45x global lift is exactly the ratio of
  `P(FunctionDef | chunk 0)` to `P(FunctionDef | anywhere)`. It is the statement
  "chunk 0 of a Python file is usually a `def` or an `import`" — a fact about
  Python, contributed entirely by the dataset and not at all by the model.

**t20 carries zero information about the input. The result should be withdrawn.**

## 1.1 — The dataset alone already produces the number

`P(AST construct | chunk position)` computed straight from the CodeNet test split
with Python's `ast` and the Qwen tokenizer. No checkpoint is loaded at all.
1200 files, global `P(FunctionDef) = 7.0%`.

```
┌────────────┬──────────┬────────────────────┬────────┬─────────────────┬──────────┬─────────┐
│ Chunk pos  │  n files │  Top AST construct │  share │  P(FunctionDef) │  P(Call) │   P(If) │
├────────────┼──────────┼────────────────────┼────────┼─────────────────┼──────────┼─────────┤
│ 0          │     1200 │        FunctionDef │  30.7% │           30.7% │    24.6% │    0.0% │
│ 1          │     1200 │               Call │  42.6% │           16.2% │    42.6% │    7.7% │
│ 2          │     1200 │               Call │  33.5% │            7.9% │    33.5% │   16.4% │
│ 3          │     1199 │               Call │  24.7% │            7.4% │    24.7% │   18.6% │
│ 4          │     1180 │               Call │  25.6% │            6.4% │    25.6% │   19.8% │
│ 5          │     1099 │               Call │  29.7% │            5.0% │    29.7% │   19.4% │
│ 6          │      987 │               Call │  29.5% │            3.5% │    29.5% │   20.9% │
│ 7          │      868 │               Call │  26.4% │            4.4% │    26.4% │   24.3% │
│ 8          │      776 │               Call │  24.6% │            3.6% │    24.6% │   24.4% │
│ 9          │      699 │                 If │  26.0% │            3.6% │    26.0% │   26.0% │
└────────────┴──────────┴────────────────────┴────────┴─────────────────┴──────────┴─────────┘
```

## 1.2 — A padding artifact was corrupting every prior CodeNet number

Prefill chunks are cut from position 0 of the **left-padded batch row**
(`sorl/steer.py`: `src_idx = [c*L for c in range(n_chunks)]`, applied to the
padded sequence). With pad `P`, source token `t` sits at padded position `P+t`,
and the analysis maps padded chunk `P//L` onto "source chunk 0". That is exact
only when `P % L == 0`. At `eval_batch_size=32` only **28.5%** of rows satisfy
it (mean pad 66 tokens; the `P % 8` histogram is near-uniform).

The consequence was not subtle. At batch 32 the code at position 0 splits
perfectly on padding alignment:

```
┌───────────┬───────┬──────────────┬─────────────┬────────────┐
│ code@pos0 │     n │  pad-aligned │  misaligned │ P(aligned) │
├───────────┼───────┼──────────────┼─────────────┼────────────┤
│ t1        │   572 │            0 │         572 │       0.0% │
│ t20       │   228 │          228 │           0 │     100.0% │
└───────────┴───────┴──────────────┴─────────────┴────────────┘
    base rate P(pad aligned) = 228/800 = 28.5%
```

800/800 separation. The padded measurement made t20 look like a *selective*
detector firing on 28.5% of files, when in truth it fires on 100% of them and the
apparent selectivity was the batch padding. Re-running at `eval_batch_size=1`
(no padding, exact alignment) changes accuracy from 22.4% to **31.0%** and
reshuffles the routing entirely — different codes carry different labels.

**Every previously reported CodeNet number
(`codenet_r1r2_125step.json`, `codenet_r1r2_20k.json`) was computed at batch 32
and is contaminated.** Everything below uses the batch-1 measurement.

## 1.3 — Clean R1: per-code purity with position as an explicit control

`ckpt/codenet_v9`, 800 test files, batch 1, prefill codes vs source chunk labels.
`lift_pos = P(label|code) / P(label | the positions that code fires at)`,
weighted by the code's own firing distribution. `p_bonf` is a one-sided binomial
test against that position-matched baseline, Bonferroni-corrected over
13 codes x 16 labels.

```
┌───────┬───────┬──────────────┬─────────────┬─────────┬───────────┬────────────┬──────────┬──────┬─────────────┬──────────┬──────┐
│ Code  │     n │    Top label │ P(lab|code) │  P(lab) │ lift_glob │ P(lab|pos) │ lift_pos │ #pos │ top-pos cov │   p_bonf │ surv │
├───────┼───────┼──────────────┼─────────────┼─────────┼───────────┼────────────┼──────────┼──────┼─────────────┼──────────┼──────┤
│ t12   │    38 │         Call │       0.421 │   0.285 │     1.48x │      0.345 │    1.22x │   24 │          1% │        1 │   no │
│ t9    │    72 │           If │       0.417 │   0.137 │     3.05x │      0.147 │    2.84x │   27 │          1% │  5.5e-06 │  YES │
│ t20   │   800 │  FunctionDef │       0.412 │   0.093 │     4.45x │      0.412 │    1.00x │    1 │        100% │        1 │   no │
│ t13   │    31 │        BinOp │       0.323 │   0.137 │     2.36x │      0.192 │    1.68x │   16 │          1% │        1 │   no │
│ t0    │  5509 │         Call │       0.318 │   0.285 │     1.12x │      0.294 │    1.08x │   31 │         57% │   0.0067 │  YES │
│ t2    │  1232 │         Call │       0.317 │   0.285 │     1.11x │      0.296 │    1.07x │   31 │         16% │        1 │   no │
│ t7    │   109 │        BinOp │       0.312 │   0.137 │     2.28x │      0.143 │    2.18x │   24 │          2% │   0.0012 │  YES │
│ t6    │    89 │           If │       0.303 │   0.137 │     2.22x │      0.132 │    2.31x │   28 │          1% │   0.0039 │  YES │
│ t8    │    92 │       Return │       0.293 │   0.039 │     7.50x │      0.054 │    5.41x │   18 │          2% │  6.1e-11 │  YES │
│ t4    │   428 │           If │       0.285 │   0.137 │     2.09x │      0.159 │    1.80x │   29 │          5% │    6e-09 │  YES │
│ t1    │  1835 │         Call │       0.270 │   0.285 │     0.95x │      0.287 │    0.94x │   31 │         17% │        1 │   no │
│ t5    │   495 │           If │       0.265 │   0.137 │     1.94x │      0.153 │    1.73x │   31 │          6% │  2.8e-08 │  YES │
│ t3    │   807 │           If │       0.249 │   0.137 │     1.82x │      0.156 │    1.59x │   31 │         11% │  1.4e-09 │  YES │
└───────┴───────┴──────────────┴─────────────┴─────────┴───────────┴────────────┴──────────┴──────┴─────────────┴──────────┴──────┘
```

## 1.4 — Is position locking degenerate, as on arithmetic?

```
┌──────┬───────────┬─────────┬───────────┬────────┬──────────────────────────┬──────────────┬────────────┐
│ Pos  │  n chunks │  #codes │  top code │  share │              top-3 codes │    top label │ P(lab|pos) │
├──────┼───────────┼─────────┼───────────┼────────┼──────────────────────────┼──────────────┼────────────┤
│ 0    │       800 │       1 │       t20 │ 100.0% │                 t20:100% │  FunctionDef │      41.2% │
│ 1    │       800 │      12 │        t0 │  57.1% │     t0:57% t2:16% t1:10% │         Call │      33.6% │
│ 2    │       800 │      15 │        t0 │  53.1% │     t0:53% t2:12% t3:11% │         Call │      29.4% │
│ 3    │       796 │      13 │        t0 │  56.2% │     t0:56% t2:14% t1:11% │         Call │      22.5% │
│ 4    │       773 │      15 │        t0 │  49.2% │     t0:49% t1:14% t2:11% │         Call │      19.7% │
│ 5    │       724 │      14 │        t0 │  48.3% │     t0:48% t1:17% t2:11% │         Call │      22.8% │
│ 6    │       664 │      13 │        t0 │  50.7% │     t0:51% t1:17% t2:11% │         Call │      23.5% │
│ 7    │       597 │      16 │        t0 │  46.7% │      t0:47% t1:20% t2:9% │         Call │      21.1% │
│ 8    │       543 │      13 │        t0 │  48.8% │     t0:49% t1:21% t2:10% │         Call │      21.6% │
│ 9    │       491 │      12 │        t0 │  43.0% │     t0:43% t1:21% t2:13% │         Call │      23.0% │
│ 10   │       447 │      13 │        t0 │  43.9% │     t0:44% t1:20% t2:11% │         Call │      22.4% │
│ 11   │       401 │      12 │        t0 │  50.4% │      t0:50% t1:18% t2:9% │         Call │      22.9% │
│ 12   │       369 │      15 │        t0 │  49.9% │     t0:50% t1:21% t2:10% │        BinOp │      24.4% │
│ 13   │       338 │      15 │        t0 │  48.5% │      t0:49% t1:18% t2:9% │         Call │      22.2% │
└──────┴───────────┴─────────┴───────────┴────────┴──────────────────────────┴──────────────┴────────────┘
```

**Position 0 is the only degenerate position: 1 of 32.** It is owned 100% by t20.
Every other position spreads over 12-16 distinct codes, so CodeNet codes in
general are not position tags. The headline result just happened to come from the
one position that is.

## Task 1 verdict

1. **t20 is fully confounded — withdraw it.** It covers 100% of chunk position 0
   and nothing else, and `P(label|t20)` equals `P(label|pos 0)` exactly
   (`lift_pos = 1.00x`). This is the identical failure mode to arithmetic.
2. **A padding bug inflated the appearance of selectivity** and contaminated all
   prior CodeNet R1/R2 numbers. Analyses must run at batch 1 or be pad-corrected.
3. **Unlike arithmetic, the rest of the CodeNet codebook is not degenerate, and
   some of it is real.** 31 of 32 positions spread across 12-16 codes, and seven
   codes beat their position-matched baseline at Bonferroni-corrected
   significance. The strongest is **t8 -> `Return`, 29.3% vs a position-matched
   5.4%, `lift_pos = 5.41x`, n=92 over 18 positions, `p_bonf = 6e-11`** —
   a genuine, position-independent association. t9 (`If`, 2.84x), t6 (`If`,
   2.31x) and t7 (`BinOp`, 2.18x) follow.
4. **But nothing reaches R1's bar.** Peak absolute purity is 42% against a 70%
   threshold, and the strong-lift codes are all small-n. So: CodeNet has weak
   real structure, not the clean specialist codes the claim requires — and the
   specific number that was reported is an artifact.

---

# Task 2 — causal gate sweep

R1 purity is only worth measuring if the codes are causally load-bearing, so the
gate is the knockout delta. See `codenet_sweep_gate.py` for the ladder.

## A measurement bug in the prior CodeNet knockout

The earlier knockout compared `decode_scale=scale` against `decode_scale=0.0`.
But `StackedAbstractionWrapperV9.generate` documents `decode_scale` as
"leaves prefill untouched" — it silences steering only during *generation*. For
arithmetic that is the whole intervention. For CodeNet it is almost none of it:
the source lives in the prompt, so 32 prefill chunks stay fully steered in **both
arms** while at most `max_new_tokens/L` decode chunks differ. The published
-0.6pp bounds a far weaker claim than "the codes are not load-bearing".

The sweep therefore measures four arms, always passing `decode_scale` explicitly:

| arm | what it does |
|---|---|
| `ON` | `decode_scale = scale` — steering everywhere |
| `OFF_decode` | `decode_scale = 0.0` — prefill still steered (the old, partial measure) |
| `OFF_full` | `steering_emb` zeroed — no steering in prefill or decode. **Sets the gate.** |
| `RANDOM` | codes replaced uniformly in both phases — vectors present, identities destroyed |

`RANDOM` separates "the codes carry information" from "the model adapted to a
steering vector of roughly this magnitude".

**Gate:** `acc(ON) - acc(OFF_full) >= 3.0pp` and `0.10 <= acc(ON) <= 0.80`.
Optimizer steps held at ~250 on every rung by scaling epochs to data size, so
training budget is not a confound.
