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

## An independent model recovers the mechanism blind

Automated interpretation run as a falsification test, not a demonstration. A
separate model (Claude Sonnet 5) saw **only raw firings** — problem, answer-digit
index, digit value, operand column — with no ground-truth label, no purity
statistic, and **no list of candidate conditions**. It was told that "this code
only marks position N" was a valid answer, so finding nothing carried no penalty.

```
  ┌───────┬────────┬──────────┬───────────────────────────────────────────────┐
  │ Code  │  lift  │  conf.   │ blind description (verbatim, abridged)        │
  ├───────┼────────┼──────────┼───────────────────────────────────────────────┤
  │  t6   │  6.21x │  high    │ "...when that column generates a carry-out,   │
  │       │        │          │  i.e. A+G+carry_in >= 10 — equivalently,      │
  │       │        │          │  whenever the addition overflows into a 7th   │
  │       │        │          │  leading digit"                               │
  │  t17  │  4.07x │  high    │ "...when that column does NOT generate a      │
  │       │        │          │  carry-out/borrow-out"                        │
  │  t11  │  2.24x │ medium   │ "does not reduce to a clean carry rule"       │
  ├───────┼────────┼──────────┼───────────────────────────────────────────────┤
  │  t0   │  1.62x │  high    │ "two fixed positions ... regardless of        │
  │  t1   │  1.53x │  high    │  operand values, carry/borrow status, digit   │
  │  t7   │  1.60x │  high    │  value, or operation"                         │
  │  t2   │  1.24x │  high    │                                               │
  └───────┴────────┴──────────┴───────────────────────────────────────────────┘
   agreement with the purity table: 7/7
```

Three things make this more than a plausible-sounding label.

**It partitioned the codebook without being told there were two kinds.** The
three genuine specialists were described as arithmetic conditions; the four
near-chance codes were described as fixed-position tags carrying no arithmetic
information. That split is exactly the one the purity table makes, and nothing in
the prompt hinted that such a split existed.

**Its confidence tracks lift.** High on 6.21× and 4.07×, and it dropped itself to
*medium* on 2.24× — explicitly declining to state a clean rule for the weakest
specialist. A procedure that reports uncertainty where the signal is weak is one
whose confident claims mean something.

**The negative control held.** Inventing an arithmetic story for a code firing
5,200 times was the easy false positive. Instead it reasoned from the count —
5,200 is exactly 2× the 2,600-problem eval set, so the code fires twice per
problem regardless of content — and concluded "positional tag". It reached that
by arithmetic on the evidence, not by hedging.

For `t6` it produced a precise, executable rule — *carry-out at the leading
column, equivalently a 7th digit appearing* — that picks out the ground-truth
`US` firings. Notably it named the **observable signature** rather than the
textbook label "carry cascade through a sum-9 run": it described what it could
see in the data, and that description selects the same firings.

## The codes are causally load-bearing here too

Steering scale decides whether the codes carry the computation. At `scale=0.5`
they stop being a read-out and become a control signal.

```
  ┌───────────────┬──────────┬───────────┬──────────┬──────────────────────┐
  │ Arm           │ accuracy │   Δ abs   │   Δ rel  │ what it removes      │
  ├───────────────┼──────────┼───────────┼──────────┼──────────────────────┤
  │ codes ON      │  82.96%  │       —   │      —   │ nothing              │
  │ OFF_full      │  75.08%  │  −7.88pp  │   −9.5%  │ codes entirely       │
  │ RANDOM        │  69.88%  │ −13.08pp  │  −15.8%  │ code identity only   │
  └───────────────┴──────────┴───────────┴──────────┴──────────────────────┘
```

**Scrambling which code is which costs 13.08 points — nearly double the 7.88
points of deleting the steering table outright.** Wrong codes are worse than no
codes. The model is not tolerating a generic steering signal; it reads specific
identities and is actively misled by the wrong one.

That is a stronger statement than the knockout, because magnitude is held
constant: the `RANDOM` arm keeps steering vectors of the same norm and destroys
only the mapping from code to meaning. The same inversion appears on CodeNet at
`L=16` and under comment-normalised scoring, so it reproduces in three
independent settings.

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

Scale matters sharply: `0.1` leaves the codes inert (+0.15pp) and `0.3`/`0.7`
leave them unhelpful. `0.5` is the setting at which the model routes through
them.

## Summary

On a real pretrained LLM:

- **sub-task-specialised codes emerge** — a sum-9 cascade detector at 78.3%
  purity, 6.21× lift, plus borrow- and carry-consumption codes at 4.07× and 2.24×
- **the carry-uncertainty boundary is recovered** without supervision
  (p < 1e-4, 5.04× leave-one-out)
- **specialists and generalists coexist** in the ratio the original study reports
- **an independent model recovers the detector blind**, with a working negative
  control
- **the codes are causally load-bearing at `scale=0.5`** — −7.88pp on removal,
  −13.08pp when identities are scrambled (see the caveat on non-monotonicity)

Reproduce: `repro/r1_purity.sh`, `repro/r5_sum9.sh`, `repro/f6_polysemanticity.sh`,
`repro/f7_autointerp.sh`.
Raw results in `results/arithmetic_r1r2.json`,
`results/arithmetic_autointerp_rawfirings.json`, `results/arith_firings.json`.
