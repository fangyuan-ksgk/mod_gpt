# Real-LLM replication — Python source (Project CodeNet)

**Qwen3-0.6B (596M), DLR v9 residual steering**, one code per 8-token chunk,
codebook 30, injection layer fixed a priori. 800 competitive-programming Python
files, **split by problem** so near-identical solutions cannot straddle
train/test. Ground truth is the innermost AST construct each chunk sits inside,
recovered from source with a parser.

```
   toy model :  answer digit  ->  carry/borrow sub-task  ->  code
   this study:  token chunk   ->  AST construct          ->  code
```

Two measurements on **one checkpoint**, so the causal and the interpretability
claim describe the same model rather than two conveniently different ones.

## Finding #2 — the codes are causally load-bearing

Four arms. `OFF_full` zeroes the steering table entirely; `RANDOM` keeps
steering vectors of the same magnitude but destroys which code is which.

```
  ┌───────────────┬──────────┬──────────┬──────────┬────────────┬──────────┐
  │ Arm           │ accuracy │  Δ abs   │  Δ rel   │ what it removes         │
  ├───────────────┼──────────┼──────────┼──────────┼────────────┴──────────┤
  │ codes ON      │   17.50% │        — │        — │ nothing                 │
  │ RANDOM        │   11.13% │  −6.37pp │  −36.4%  │ code identity only      │
  │ OFF_full      │   10.62% │  −6.87pp │  −39.3%  │ codes entirely          │
  └───────────────┴──────────┴──────────┴──────────┴───────────────────────┘
```

**Removing the codes costs 39% of task accuracy.** The `RANDOM` arm is the
control that matters: it accounts for **6.37 of the 6.87 points**, so the model
is not merely adapted to *a* steering vector of that magnitude — the specific
code identities carry the information.

For scale, the toy model in the original study loses 99.9% relative under the
same intervention. Ours is a smaller effect on a far more capable backbone, but
it is the same phenomenon measured the same way, and it is not a rounding error.

## R1 — codes specialise on syntactic structure

`Call` covers 28.5% of chunks, so purity alone is uninterpretable; the table is
ranked by lift over each construct's base rate. The **position-matched** column
compares each code against its construct's frequency *at the positions where the
code fires*.

```
  ┌────────┬────────┬─────────────┬──────────┬──────────┬──────────┬────────┐
  │  Code  │      n │ Construct   │   Purity │   Lift   │ Lift     │  #Pos  │
  │        │        │             │          │  global  │ pos-match│ of 32  │
  ├────────┼────────┼─────────────┼──────────┼──────────┼──────────┼────────┤
  │    t10 │     30 │ For         │   33.3%  │   6.57x  │  6.54x   │   17   │
  │     t5 │    591 │ If          │   29.9%  │   2.19x  │  1.88x   │   31   │
  │     t3 │    608 │ If          │   28.1%  │   2.06x  │  1.88x   │   31   │
  │     t6 │    291 │ BinOp       │   32.0%  │   2.34x  │  1.80x   │   31   │
  │     t7 │     58 │ BinOp       │   32.8%  │   2.40x  │  1.53x   │   22   │
  └────────┴────────┴─────────────┴──────────┴──────────┴──────────┴────────┘
   11 active codes of 30 · median lift 2.06x
```

`t5`, `t3` and `t6` are the load-bearing rows: each fires at **31 of 32 chunk
positions** with n in the hundreds, so their lift cannot be inherited from any
single position's construct distribution. They track conditionals and binary
expressions wherever those occur in a file. `t10` shows the largest effect
(6.54× position-matched, essentially undiminished by the control) but rests on
n=30 and should be read as suggestive.

## An independent model recovers the structure blind — and catches a confound

The same falsification test as the arithmetic study, in a second domain. An
independent model saw **only raw firings** — the 8-token source chunk plus
surrounding context — with no AST label, no purity statistic, no candidate list,
and no access to the repository. It was told a positional or null answer carried
no penalty.

```
  ┌───────┬──────────────┬──────────┬────────┬──────────────────────────────────┐
  │ Code  │ ground truth │ lift_pos │  conf. │ blind description                │
  ├───────┼──────────────┼──────────┼────────┼──────────────────────────────────┤
  │  t10  │ For          │   6.54x  │ medium │ "for <var> in <iterable>: loop   │
  │       │              │          │        │  header"                      ✓  │
  │  t3   │ If           │   1.88x  │  low   │ "bare `else:` / branch tail"  ✓  │
  │  t6   │ BinOp        │   1.80x  │  low   │ "multi-term arithmetic expr"  ✓  │
  │  t7   │ BinOp        │   1.53x  │  low   │ "operator-dense float expr"   ✓  │
  │  t5   │ If           │   1.88x  │  low   │ "indentation run"             ✗  │
  ├───────┼──────────────┼──────────┼────────┼──────────────────────────────────┤
  │  t0   │ Call         │   1.15x  │  high  │ "no content rule; the default │
  │       │              │          │        │  majority code"               ✓  │
  │  t1   │ Call         │   0.77x  │ medium │ "no discernible rule"         ✓  │
  │  t2,t4│ Call         │  ≤1.30x  │  low   │ "no clean rule"               ✓  │
  │  t8   │ Call         │   0.95x  │  low   │ "may be no better than chance"✓  │
  ├───────┼──────────────┼──────────┼────────┼──────────────────────────────────┤
  │  t9   │ FunctionDef  │   1.00x  │  high  │ "the FIRST chunk of a file,      │
  │       │ (4.40x glob) │          │        │  regardless of content"       ✓✓ │
  └───────┴──────────────┴──────────┴────────┴──────────────────────────────────┘
   agreement with the purity table: 10/11
```

It named the strongest detector — `t10`, the highest position-matched lift in
the codebook — as a `for` loop header, citing that 8 of 12 sampled chunks begin
with `for` against 1 of the other 120 firings. It described both `BinOp` codes as
arithmetic expressions and reached the `If` code `t3` through its `else:`
branches. It declined to find structure in all five near-chance `Call` codes.

**The `t9` row is the result worth the space.** `t9` has the second-highest
*global* lift in the table, 4.40×, and reads as the best `FunctionDef` detector
in the codebook. It is an artefact: it fires on chunk 0, and Python files open
with `def`/`import`, so its position-matched lift is exactly **1.00×** — knowing
the code adds nothing over knowing the position. We caught that with an explicit
position control. The interpreter caught it from raw firings with no statistics
at all, at high confidence, by arithmetic on the data: sampled file indices
spaced ≈67 apart, i.e. 809/12 — one firing per file — and across the other 120
sampled firings, not one had chunk index 0.

That is the negative control doing real work. The interpreter's two
high-confidence calls are both content-free codes and both correct, while every
genuinely ambiguous code is marked low. A procedure that reports uncertainty
where the signal is weak, and that rejects the most impressive-looking number in
the table on evidence, is one whose positive identifications carry weight.

## Specialists and polysemantic generalists coexist

Ranking by position-matched lift splits the codebook into two regimes, exactly
as in the arithmetic study:

```
  ┌──────────────┬────────┬───────────┬──────────┬──────────┬────────────┐
  │ Role         │  codes │  firings  │ share of │ best     │ lift range │
  │              │        │           │ traffic  │ purity   │ (pos-match)│
  ├──────────────┼────────┼───────────┼──────────┼──────────┼────────────┤
  │ specialists  │      5 │     1,578 │   13.7%  │   33.3%  │ 1.53-6.54x │
  │ generalists  │      6 │     9,960 │   86.3%  │   40.8%  │ 0.77-1.30x │
  └──────────────┴────────┴───────────┴──────────┴──────────┴────────────┘
```

A handful of sharp detectors carry a small minority of the traffic while
high-frequency near-chance codes absorb the rest — `t0` alone fires 4,978 times
at 1.15× and acts as a fallback.

The cross-domain agreement is the interesting part. Arithmetic splits 3
specialists over **14.3%** of firings against 4 generalists over 85.7%; CodeNet
splits 5 over **13.7%** against 6 over 86.3%. Two entirely different label sets
— carry/borrow sub-tasks versus Python AST constructs — and the same ~14/86
division of labour.

Note also that the highest *purity* in the table (40.8%, `t9`) belongs to a
generalist, not a specialist: `t9` is the chunk-0 position artefact. Purity
alone would rank it first; position-matched lift puts it where it belongs.

## Summary

On a real pretrained LLM, in a syntactic domain with an entirely different label
set from the arithmetic case study:

- **codes are causally necessary** — 39% relative accuracy loss on removal, with
  identity (not magnitude) accounting for 93% of that loss
- **codes specialise on ground-truth structure** — conditional and
  binary-expression detectors at 1.8–1.9× position-matched lift, distributed
  across nearly every chunk position
- **an independent model recovers that structure blind** — 10/11 agreement on
  raw source chunks, including rejecting the codebook's second-highest global
  lift as a position artefact
- all measured on the **same checkpoint**

Reproduce: `repro/knockout.sh`, `repro/f3_codenet_purity.sh`,
`repro/f7_autointerp.sh`. Raw results in
`results/codenet_autointerp_rawfirings.json`,
`results/codenet_r1r2.json`, `results/codenet_s0.5_i10_z1_L8_n4000_knockout4.json`,
`results/codenet_gated_confound_nopad.json`.
