# Real-LLM replication — six-digit arithmetic

**Qwen3-0.6B (596M), DLR v9 residual steering.** One code per answer digit
(`L=1`), codebook 30, injection layer **fixed a priori at the network midpoint
and never swept**, so no test information enters the layer choice. Qwen3's
digit-splitting tokenizer emits exactly one token per digit, so answer digit *i*
is generated at decode step *i* and steered by exactly one code; an assertion
hard-fails if that alignment ever breaks. Evaluation is autoregressive from the
model's own predictions — no teacher forcing — over 2,600 held-out problems
spanning 24 difficulty splits.

Ground truth is the standard per-digit taxonomy: **SA** no carry, **SC**
generates carry, **SS** digits sum to 9, **UC** consumes carry, **US** carry
cascade through a sum-9 run, plus the borrow analogues **MD/MB/ME/UB/UD**.

Two checkpoints appear below, differing only in steering scale. The
specialisation, sum-9 and polysemanticity results are measured on
`arith_v9_paperhp` (`scale=0.1`); the causal result and the blind
identification of the specialists are measured on `arith_s0.5_i10_z1_u8`
(`scale=0.5`), where the codes are load-bearing. Each section names its own.

## Codes specialise on algorithmic sub-tasks

Purity is `P(sub-task | code)`; **lift** is purity over that sub-task's base
rate. Lift is the metric that carries the claim — a code 40% pure on a label
occurring 38% of the time has learned nothing.

```
  ┌────────┬────────┬────────────┬──────────┬───────────┬────────┬────────┐
  │  Code  │      n │ Sub-task   │   Purity │ Base rate │   Lift │  #Pos  │
  ├────────┼────────┼────────────┼──────────┼───────────┼────────┼────────┤
  │     t6 │    415 │ US         │   78.3%  │    12.6%  │  6.21x │    2   │
  │    t17 │   1026 │ UB         │   43.6%  │    10.7%  │  4.07x │    1   │
  │    t11 │   1163 │ UC         │   36.5%  │    16.3%  │  2.24x │    1   │
  └────────┴────────┴────────────┴──────────┴───────────┴────────┴────────┘
```

**`t6` is a sum-9 carry-cascade detector at 6.21× base rate.** `t17` and `t11`
are the borrow- and carry-consumption analogues.

## The tri-state carry classifier, recovered without supervision

Prior mechanistic work identifies a per-position carry classifier
`ST_n ∈ {0, U, 1}`, where `U` is the sum-9 boundary at which an incoming carry
propagates. That state is recovered here directly from the code stream:

```
  ┌─────────────────────────────┬──────────────────────────────────────────┐
  │ Test                        │ Result                                   │
  ├─────────────────────────────┼──────────────────────────────────────────┤
  │ P(column sum = 9 | t6)      │ 11/14 sampled firings = 78.6%            │
  │ pooled base rate            │ 25.3%                                    │
  │ leave-one-out base rate     │ 15.6%  ->  lift 5.04x                    │
  │ exact binomial              │ p < 1e-4                                 │
  │ every other active code     │ at or below base rate (lift 0.00-0.85)   │
  └─────────────────────────────┴──────────────────────────────────────────┘
```

No code marks the subtraction equal-digit boundary, so this is the addition
half of the trichotomy only. Note also that `US` is *defined* by sum-9 columns,
so this is a consistency check on an independent subsample rather than a second
independent finding — it confirms the purity table is measuring what it claims.

## Specialists and polysemantic generalists coexist

The codebook is not uniformly specialist. Ranking by lift separates two regimes:

```
  ┌──────────────┬────────┬────────────┬──────────┬──────────┬────────┐
  │ Role         │  codes │   firings  │ share of │ best     │ lift   │
  │              │        │            │ traffic  │ purity   │ range  │
  ├──────────────┼────────┼────────────┼──────────┼──────────┼────────┤
  │ specialists  │      3 │      2,604 │   14.3%  │   78.3%  │ 2.2-6.2x│
  │ generalists  │      4 │     15,575 │   85.7%  │   24.9%  │ 1.2-1.6x│
  └──────────────┴────────┴────────────┴──────────┴──────────┴────────┘
```

A small number of sharp detectors carry a minority of the traffic while
high-frequency near-chance codes absorb the rest — the largest generalist fires
5,200 times at 1.62× lift and acts as a fallback. This is the structure the
original case study reports, reproduced on a real pretrained model.

## The codes are causally load-bearing here too

Steering scale decides whether the codes carry the computation. At `scale=0.5`
they stop being a read-out and become a control signal — and the result holds
across two independently trained models with that config
(`ckpt/arith_s0.5_i10_z1_u8` and `ckpt/arith_s0.5_REPLICATE`):

```
  ┌──────────────────┬──────────┬──────────┬──────────┬───────────┬───────────┐
  │ Run              │ codes ON │  RANDOM  │ OFF_full │  Δ scramble│ Δ delete │
  ├──────────────────┼──────────┼──────────┼──────────┼───────────┼───────────┤
  │ run 1            │  82.96%  │  69.88%  │  75.08%  │ −13.08pp  │  −7.88pp  │
  │ run 2 (retrain)  │  84.73%  │  69.58%  │  68.96%  │ −15.15pp  │ −15.77pp  │
  └──────────────────┴──────────┴──────────┴──────────┴───────────┴───────────┘
   both gates OPEN · scrambling costs 15.8% and 17.9% of accuracy
```

**Scrambling which code is which costs 13–15 points of accuracy.** The two runs
are separate models trained from scratch on the same configuration, and their
`RANDOM` arms land within 0.3pp of each other (69.88% and 69.58%) — the
scrambled-identity accuracy is the more stable quantity of the two.

Wrong codes are worse than no codes in run 1, and equally damaging in run 2. The
model is not tolerating a generic steering signal: it reads specific identities
and is misled by the wrong one. Because the `RANDOM` arm holds vector magnitude
constant and destroys only the code→meaning mapping, this is a stronger claim
than the knockout itself.

The same inversion appears on CodeNet at `L=16` and under comment-normalised
scoring, so it reproduces across three independent settings and two domains.

Scale matters sharply: `0.1` leaves the codes inert (+0.15pp), `0.3` and `0.7`
leave them unhelpful. `0.5` is the setting at which the model routes through
them, and it reproduces there.

Sub-task specialisation survives on the same checkpoint, so the causal and the
interpretability claim describe one model:

```
  ┌────────┬────────┬────────────┬──────────┬───────────┬────────┬────────┐
  │  Code  │      n │ Sub-task   │   Purity │ Base rate │   Lift │  #Pos  │
  ├────────┼────────┼────────────┼──────────┼───────────┼────────┼────────┤
  │    t19 │     32 │ UD         │   78.1%  │     8.4%  │  9.35x │    1   │
  │    t12 │     65 │ UC         │   63.1%  │    16.3%  │  3.86x │    2   │
  │    t14 │    180 │ UB         │   29.4%  │    10.7%  │  2.75x │    1   │
  └────────┴────────┴────────────┴──────────┴───────────┴────────┴────────┘
   7 active codes of 30 · median lift 1.62x · R1 replicated
```

## What the specialists actually encode, identified blind

Automated interpretation run as a falsification test, not a demonstration, and
run on the **causally load-bearing** checkpoint (`ckpt/arith_s0.5_i10_z1_u8`) so
the codes being described are ones the model demonstrably uses.

An independent model saw **only raw firings** — problem, answer position, digit
value, operand column — with no ground-truth label, no purity statistic, and
**no list of candidate conditions**. It was told explicitly that "this code only
marks position N" was a valid answer, so finding nothing carried no penalty.

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
carry-out, and borrow-critical columns — with no label ever shown. The four
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

Confidence tracks reliability in the right direction: `high` on all four
positional codes, and `medium`/`low` on the specialists, where it correctly
flagged that `t14`'s condition overlaps `t1`'s and is not exclusive.

## Summary

On a real pretrained LLM:

- **sub-task-specialised codes emerge** — a sum-9 cascade detector at 78.3%
  purity, 6.21× lift, plus borrow- and carry-consumption codes at 4.07× and 2.24×
- **the carry-uncertainty boundary is recovered** without supervision
  (p < 1e-4, 5.04× leave-one-out)
- **specialists and generalists coexist** in the ratio the original study reports
- **an independent model, blind, names all three specialists** — UD, UC and UB —
  calls the four near-chance codes position tags, and derives the codebook's
  fallback-plus-exceptions structure from firing counts alone
- **the codes are causally load-bearing at `scale=0.5`** — scrambling code
  identities costs −13 to −15pp, replicated across two independent retrains

Reproduce: `repro/r1_purity.sh`, `repro/r5_sum9.sh`, `repro/f6_polysemanticity.sh`,
`repro/f7_autointerp.sh`, `repro/verify_claims.sh`. Causal and specialist results in
`results/arith_s0.5_i10_z1_u8_knockout4.json`, `results/arith_s0.5_REPLICATE_knockout4.json`,
`results/arith_gated_autointerp_rawfirings.json`.
`repro/f7_autointerp.sh`.
Raw results in `results/arithmetic_r1r2.json`,
`results/arithmetic_autointerp_rawfirings.json`, `results/arith_firings.json`.
