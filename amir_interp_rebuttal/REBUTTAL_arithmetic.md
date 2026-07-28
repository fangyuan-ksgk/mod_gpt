# Real-LLM replication — six-digit arithmetic

**Qwen3-0.6B (596M, 28 layers), DLR v9 residual steering.** One checkpoint
throughout: `ckpt/arith_s0.5_i10_z1_u8`.

One code per answer digit (`L=1`), codebook 30, steering injected at **layer 14
of 28** — the exact midpoint, fixed a priori and never swept, so no test
information enters the layer choice. Qwen3's digit-splitting tokenizer emits one
token per digit, so answer digit *i* is generated at decode step *i* and steered
by exactly one code; an assertion hard-fails if that alignment ever breaks.

Trained on 100,000 problems for one epoch (3,125 steps at batch 32). Evaluated
autoregressively from the model's own predictions — no teacher forcing — over
2,600 held-out problems spanning 24 difficulty splits. **Accuracy 82.96%.**

Ground truth is the standard per-digit taxonomy: **SA** no carry, **SC**
generates carry, **SS** digits sum to 9, **UC** consumes carry, **US** carry
cascade through a sum-9 run, plus the borrow analogues **MD/MB/ME/UB/UD**.

**The task is learned, not elicited.** Untrained Qwen3-0.6B scores **0.0%** on
this eval set — zero in all 24 splits — so none of the accuracy is attributable
to the base model already being able to do arithmetic.

## The codes are causally load-bearing

Steering scale decides whether the codes carry the computation; `scale=0.5` is
where they stop being a read-out. Two independently trained models at that
setting:

```
  ┌──────────────────┬──────────┬──────────┬──────────┬────────────┬───────────┐
  │ Run              │ codes ON │  RANDOM  │ OFF_full │ Δ scramble │ Δ delete  │
  ├──────────────────┼──────────┼──────────┼──────────┼────────────┼───────────┤
  │ run 1            │  82.96%  │  69.88%  │  75.08%  │  −13.08pp  │  −7.88pp  │
  │ run 2 (retrain)  │  84.73%  │  69.58%  │  68.96%  │  −15.15pp  │ −15.77pp  │
  └──────────────────┴──────────┴──────────┴──────────┴────────────┴───────────┘
   both gates OPEN · scrambling costs 15.8% and 17.9% of accuracy
```

**Scrambling which code is which costs 13–15 points — more than deleting the
steering table outright.** Wrong codes are worse than no codes: the model is not
tolerating a generic steering signal, it reads specific identities and is
actively misled by the wrong one.

That is a stronger claim than the knockout, because the `RANDOM` arm holds
vector magnitude constant and destroys only the code→meaning mapping. The two
runs are separate models trained from scratch, and their `RANDOM` arms land
within 0.3pp of each other (69.88% and 69.58%) — scrambled-identity accuracy is
the more stable quantity of the two.

The same inversion appears on CodeNet at `L=16` and under comment-normalised
scoring, so it reproduces across three independent settings and both domains.

Scale matters sharply: `0.1` leaves the codes inert (+0.15pp), `0.3` and `0.7`
leave them unhelpful. `0.5` is where the model routes through them, and it
reproduces there.

## Codes specialise on algorithmic sub-tasks

Purity is `P(sub-task | code)`; **lift** is purity over that sub-task's base
rate. Lift carries the claim — a code 40% pure on a label occurring 38% of the
time has learned nothing.

```
  ┌────────┬────────┬────────────┬──────────┬───────────┬────────┐
  │  Code  │      n │ Sub-task   │   Purity │ Base rate │   Lift │
  ├────────┼────────┼────────────┼──────────┼───────────┼────────┤
  │    t19 │     32 │ UD         │   78.1%  │     8.4%  │  9.35x │
  │    t12 │     65 │ UC         │   63.1%  │    16.3%  │  3.86x │
  │    t14 │    180 │ UB         │   29.4%  │    10.7%  │  2.75x │
  └────────┴────────┴────────────┴──────────┴───────────┴────────┘
   7 active codes of 30 · median lift 1.62x
```

**`t19` is a borrow-propagation detector at 9.35× base rate.** `t12` marks
leading-column carry-out and `t14` borrow-critical columns. These are the codes
the model demonstrably uses — the same checkpoint as the causal result above,
not a separate, more convenient one.

`t19` rests on n=32, because `UD` is rare at the position it fires on. It is
corroborated independently below, where a blind interpreter recovers the same
condition from raw firings without seeing the label.

## Specialists and polysemantic generalists coexist

```
  ┌──────────────┬────────┬───────────┬──────────┬──────────┬────────────┐
  │ Role         │  codes │  firings  │ share of │ best     │ lift range │
  │              │        │           │ traffic  │ purity   │            │
  ├──────────────┼────────┼───────────┼──────────┼──────────┼────────────┤
  │ specialists  │      3 │       277 │    1.5%  │   78.1%  │ 2.75-9.35x │
  │ generalists  │      4 │    17,923 │   98.5%  │   24.1%  │ 0.91-1.62x │
  └──────────────┴────────┴───────────┴──────────┴──────────┴────────────┘
```

Three sharp detectors carry 1.5% of the traffic while four high-frequency
near-chance codes absorb the rest — `t0` alone fires 7,800 times at 1.35× and
acts as a fallback. The split is far more extreme than CodeNet's 14.7 / 85.3, so
the *ratio* is not constant across domains; what recurs is the structure — a few
sharp detectors alongside a high-traffic fallback.

## What the specialists encode, identified blind

A falsification test, not a demonstration. An independent model saw **only raw
firings** — problem, answer position, digit value, operand column — with no
ground-truth label, no purity statistic, and **no list of candidate conditions**.
It was told explicitly that "this code only marks position N" was a valid
answer, so finding nothing carried no penalty.

```
  ┌───────┬──────┬────────┬────────┬──────────────────────────────────────────┐
  │ Code  │ true │  lift  │  conf. │ blind description (verbatim, abridged)   │
  ├───────┼──────┼────────┼────────┼──────────────────────────────────────────┤
  │  t19  │  UD  │  9.35x │ medium │ "equal digits consuming a borrow-in ...  │
  │       │      │        │        │  the digit is 9, produced by a borrow"   │
  │  t12  │  UC  │  3.86x │ medium │ "addition ... the two leading operand    │
  │       │      │        │        │  digits sum to >=15, i.e. the leading    │
  │       │      │        │        │  column carries out"                     │
  │  t14  │  UB  │  2.75x │  low   │ "subtraction columns where the digits    │
  │       │      │        │        │  are equal or differ by exactly one --   │
  │       │      │        │        │  borrow-critical columns"                │
  ├───────┼──────┼────────┼────────┼──────────────────────────────────────────┤
  │  t3   │  US  │  1.62x │  high  │ "pure position marker for position 3"    │
  │  t1   │  US  │  1.46x │  high  │ "position marker for positions 1 and 4"  │
  │  t0   │  SA  │  1.35x │  high  │ "three times per problem at positions    │
  │       │      │        │        │  0, 2 and 6, regardless of content"      │
  │  t2   │  SA  │  0.91x │  high  │ "pure position marker for position 5"    │
  └───────┴──────┴────────┴────────┴──────────────────────────────────────────┘
   agreement with the purity table: 7/7
```

**All three specialists were named as the arithmetic conditions the ground truth
assigns them** — borrow propagation through equal digits, leading-column
carry-out, borrow-critical columns — with no label ever shown. The four
near-chance codes were called content-free position markers.

The sharpest result is structural rather than descriptive. For `t1` the
interpreter noticed

```
   4923 (t1) + 65 (t12) + 180 (t14) + 32 (t19)  =  5200  =  2 x 2600
```

and concluded that `t1` is the **default filler** at answer positions 1 and 4,
with the three specialists carving specific arithmetic cases out of it. That is
the codebook's actual architecture — a fallback code plus sparse exception
handlers — recovered from firing counts alone.

Confidence tracks reliability: `high` on all four positional codes, `medium` and
`low` on the specialists, where it correctly flagged that `t14`'s condition
overlaps `t1`'s and is not exclusive.

## Single-code repair: a null about reach, not about `t19`

The reviewer singles out the toy study's surgical single-code repairs, so we
asked the sharpest available version of that question on this model. `t19` is
the borrow-propagation detector. If its role is causal rather than
correlational, then on the UD columns the model *fails*, forcing `t19` should
repair some of them.

The population is clean:

```
  ┌─────────────────────────────────┬────────┬─────────────────────────┐
  │ UD digits in 2,600 problems     │  1,520 │                         │
  │  ├─ routed to t19               │     25 │ wrong: 0  (25/25 right) │
  │  └─ routed to a generalist      │  1,495 │ wrong: 76               │
  └─────────────────────────────────┴────────┴─────────────────────────┘
```

Every one of the 76 errors is a column where `t19` should have fired by its own
purity profile and a generalist was routed instead — exactly the repairable set,
with no no-op cases diluting it. Where `t19` *is* routed, UD is never failed.

Forcing `t19` into all 76 repairs **0**. So does every control code. So does a
never-trained code.

That last clause is why this is not evidence against `t19`. The intervention was
verified end to end: the prefix resume reproduces the original answer 64/64, and
the forced code demonstrably reaches the routing table (`last_codes` tail
`[9,14,0,7] → [9,14,0,25]`). The output is unchanged anyway, because a steering
vector at `scale=0.5` perturbs the layer-14 residual by **2.2–4.8% at one token
position out of 17** — below the threshold at which greedy decoding flips a
digit. Forcing dead codes `t7`/`t25`, whose vectors were never trained, changes
0 of 64 generations.

A substitution at the codebook's own magnitude therefore cannot resolve the
question either way. So we amplified instead — injecting `t19`'s *direction* at
increasing magnitude through an unused carrier slot, against a different
specialist and a random direction of identical norm at every dose:

```
  ┌──────────┬──────┬────────┬─────────┬─────────┬─────────┐
  │ direction│ mult │    |v| │   moved │   FIXED │   broke │
  ├──────────┼──────┼────────┼─────────┼─────────┼─────────┤
  │ t19 (UD) │    8 │  28.11 │   10/76 │    1/76 │    0/76 │
  │ t19 (UD) │   16 │  56.23 │   33/76 │    0/76 │    3/76 │
  │ t19 (UD) │   32 │ 112.46 │   72/76 │    9/76 │   36/76 │
  │ t19 (UD) │   64 │ 224.92 │   74/76 │    4/76 │   65/76 │
  ├──────────┼──────┼────────┼─────────┼─────────┼─────────┤
  │ t14 (UB) │   32 │ 112.46 │   41/76 │    7/76 │   40/76 │
  │ random   │   32 │ 112.46 │   34/76 │    0/76 │   12/76 │
  │ random   │   64 │ 224.92 │   50/76 │    1/76 │   56/76 │
  └──────────┴──────┴────────┴─────────┴─────────┴─────────┘
   layer-14 residual norm 47.0 · "broke" is a size-matched set of CORRECT UD digits
```

`t19` fixes 9 where a random direction of the same norm fixes 0 — but `t14`, the
**UB** specialist applied to **UD** errors, fixes 7. A UD-specific signal would
beat the wrong specialist; 9 against 7 does not. What separates both from random
is being a *trained* direction, which keeps the perturbed state nearer the
model's distribution and so likelier to emit a valid digit. That is an
in-distribution effect, not a sub-task one.

There is also no operating point worth having: at every dose where the knob
bites, damage exceeds repair — 36 correct digits destroyed for 9 repaired at
32×, 65 for 4 at 64×.

**Surgical single-code repair does not replicate here.** The causal effect in
Finding #2 is real but *distributed*: it appears when all positions are
perturbed together (scramble −13.08pp) and vanishes when one is. Codes can be
individually interpretable and jointly load-bearing without any one of them
being an individually actionable control — which is the honest scaling story,
and it matches the earlier sweeps (0/40 targeted; 48/289 vs 32/289, p=0.07)
rather than adding a separate negative.

Raw data: `results/arith_ud_repair.json` (reach control),
`results/arith_ud_repair_dose.json` (dose-response). Reproduce:
`python -m amir_interp_rebuttal.ud_repair --code 19 --label UD [--dose]`.

## Summary

On a real pretrained LLM — 596M parameters against the original study's ~0.1M
(`2L/1H/128d`), roughly 6,000× — on a task the base model cannot do at all
(0.0% untrained), and on **one checkpoint** so the causal and interpretability
claims describe the same model:

- **the codes are causally necessary** — scrambling identities costs 13–15
  points, more than deleting the steering table, replicated across two
  independent retrains
- **sub-task-specialised codes emerge** — a borrow-propagation detector at 78.1%
  purity and 9.35× lift, plus carry-out and borrow-critical codes at 3.86× and
  2.75×
- **specialists and polysemantic generalists coexist** — three sharp detectors
  alongside four high-traffic fallbacks
- **an independent model, blind, names all three specialists**, calls the four
  near-chance codes position tags, and derives the codebook's
  fallback-plus-exceptions structure from firing counts alone

Reproduce: `repro/r1_purity.sh`, `repro/f6_polysemanticity.sh`,
`repro/f7_autointerp.sh`, `repro/verify_claims.sh`. Raw results in
`results/gated/arithmetic_r1r2.json`,
`results/arith_s0.5_i10_z1_u8_knockout4.json`,
`results/arith_s0.5_REPLICATE_knockout4.json`,
`results/arith_gated_autointerp_rawfirings.json`,
`results/arith_untrained_baseline.json`.
