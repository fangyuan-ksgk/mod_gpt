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

### Position concentration is expected here, not a defect

These codes occupy one or two answer positions, which is what the sub-tasks
themselves do. The labels are intrinsically position-bound: a carry cascade
cannot begin at the overflow digit or complete at the last one.

```
  share of each sub-task's occurrences by answer position (2,600 problems)
  ┌────────┬───────┬────┬────┬────┬────┬────┬────┬────┬──────────────┐
  │ Label  │     n │ d0 │ d1 │ d2 │ d3 │ d4 │ d5 │ d6 │ positions    │
  ├────────┼───────┼────┼────┼────┼────┼────┼────┼────┼──────────────┤
  │ US     │  2296 │  0 │ 18 │ 22 │ 23 │ 22 │ 14 │  0 │ 5 of 7       │
  │ UD     │  1520 │  0 │  0 │ 20 │ 29 │ 29 │ 21 │  0 │ 4 of 7       │
  │ MB     │  1129 │  0 │  0 │  5 │  7 │ 12 │ 24 │ 51 │ 4 of 7       │
  │ SS     │   341 │  0 │ 19 │ 15 │ 15 │ 20 │ 31 │  0 │ 5 of 7       │
  │ SA     │  3244 │ 23 │ 11 │  9 │  8 │ 10 │ 13 │ 27 │ 7 of 7       │
  └────────┴───────┴────┴────┴────┴────┴────┴────┴────┴──────────────┘
```

`US` is structurally impossible at d0 and d6, `UD` at four positions. A detector
for a position-bound condition is necessarily position-bound, so conditioning on
position would divide out the signal it is meant to validate. The claim rests on
lift over base rate, which is unaffected by this.

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

## An independent model recovers the detector blind

Automated interpretation, run as a falsification test rather than a
demonstration. A separate model (Claude Sonnet) was shown **only raw firing
examples** — problem, answer-digit index, digit value, operand column — with no
ground-truth labels, no purity statistics, and **no list of candidate
conditions**. It was told explicitly that "this code only marks position N" was
a valid answer, so finding nothing carried no penalty.

```
  ┌──────────────────────────────┬────────┬──────────────────────────────────┐
  │ Code (lift)                  │ verdict│ blind description                │
  ├──────────────────────────────┼────────┼──────────────────────────────────┤
  │ t6   6.21x                   │ correct│ "fires when that column generates│
  │                              │        │  a carry-out ... whenever the    │
  │                              │        │  addition overflows into a 7th   │
  │                              │        │  leading digit"                  │
  │ t17  4.07x                   │ partial│ arithmetic condition, correct    │
  │                              │        │  carry-out polarity              │
  │ t11  2.24x                   │ hedged │ declined a clean rule, lowered   │
  │                              │        │  own confidence to medium        │
  │ t0, t1, t2, t7  (1.2-1.6x)   │ correct│ "marks position N, nothing more" │
  └──────────────────────────────┴────────┴──────────────────────────────────┘
   agreement with the purity table: 7/7
```

The negative control is what makes this meaningful. The interpreter had an easy
path to a false positive — inventing an arithmetic story for codes that fire
thousands of times — and instead used the firing counts (5,200 = exactly 2× the
problem count) to deduce they were positional tags. It also lowered its own
confidence on the weakest specialist. The procedure declines to find structure
where there is none, so its identification of `t6` is informative.

For the strongest code it produced a precise, testable rule that picks out the
ground-truth condition, having never been told the label or offered a menu.

## Summary

On a real pretrained LLM:

- **sub-task-specialised codes emerge** — a sum-9 cascade detector at 78.3%
  purity, 6.21× lift, plus borrow- and carry-consumption codes at 4.07× and 2.24×
- **the carry-uncertainty boundary is recovered** without supervision
  (p < 1e-4, 5.04× leave-one-out)
- **specialists and generalists coexist** in the ratio the original study reports
- **an independent model recovers the detector blind**, with a working negative
  control

Reproduce: `repro/r1_purity.sh`, `repro/r5_sum9.sh`, `repro/f6_polysemanticity.sh`.
Raw results in `results/arithmetic_r1r2.json`,
`results/arithmetic_autointerp_rawfirings.json`, `results/arith_firings.json`.

**Scope.** These are claims about what the codes encode. On this checkpoint the
codes are not causally load-bearing (removing them costs 0.15pp), so
single-code repair is not measurable here; the causal result is reported on the
CodeNet study, where a checkpoint with a 39% relative knockout was obtained.
