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
ranked by lift over each construct's base rate.

```
  ┌────────┬────────┬──────────────┬─────────┬─────────┐
  │  Code  │      n │ Construct    │  Purity │    Lift │
  ├────────┼────────┼──────────────┼─────────┼─────────┤
  │    t10 │     30 │ For          │   33.3% │   6.57x │
  │     t7 │     58 │ BinOp        │   32.8% │   2.40x │
  │     t6 │    291 │ BinOp        │   32.0% │   2.34x │
  │     t5 │    591 │ If           │   29.9% │   2.19x │
  │     t3 │    608 │ If           │   28.1% │   2.06x │
  └────────┴────────┴──────────────┴─────────┴─────────┘
   11 active codes of 30 · median lift 2.06x
```

`t5`, `t3` and `t6` carry the claim: n in the hundreds, tracking conditionals
and binary expressions throughout the corpus. `t10` shows the largest effect but
rests on n=30 and is suggestive only. One code is excluded — `t9`, nominally
`FunctionDef` at 4.40× — because it fires only on a file's opening chunk, where
Python files begin with `def`/`import`; it encodes nothing beyond that.

## An independent model recovers the structure blind — and catches a confound

The same falsification test as the arithmetic study, in a second domain. An
independent model saw **only raw firings** — the 8-token source chunk plus
surrounding context — with no AST label, no purity statistic, no candidate list,
and no access to the repository. It was told a positional or null answer carried
no penalty.

```
  ┌───────┬──────────────┬────────┬────────┬──────────────────────────────────┐
  │ Code  │ ground truth │  lift  │  conf. │ blind description                │
  ├───────┼──────────────┼────────┼────────┼──────────────────────────────────┤
  │  t10  │ For          │ 6.57x  │ medium │ "for <var> in <iterable>: loop   │
  │       │              │        │        │  header"                      ✓  │
  │  t7   │ BinOp        │ 2.40x  │  low   │ "operator-dense float expr"   ✓  │
  │  t6   │ BinOp        │ 2.34x  │  low   │ "multi-term arithmetic expr"  ✓  │
  │  t5   │ If           │ 2.19x  │  low   │ "indentation run"             ✗  │
  │  t3   │ If           │ 2.06x  │  low   │ "bare `else:` / branch tail"  ✓  │
  ├───────┼──────────────┼────────┼────────┼──────────────────────────────────┤
  │  t4   │ Call         │ 1.24x  │  low   │ "no clean rule"               ✓  │
  │  t0   │ Call         │ 1.18x  │  high  │ "no content rule; the default    │
  │       │              │        │        │  majority code"               ✓  │
  │  t2   │ Call         │ 0.98x  │  low   │ "no clean rule"               ✓  │
  │  t8   │ Call         │ 0.88x  │  low   │ "may be no better than chance"✓  │
  │  t1   │ Call         │ 0.75x  │ medium │ "no discernible rule"         ✓  │
  ├───────┼──────────────┼────────┼────────┼──────────────────────────────────┤
  │  t9   │ FunctionDef  │ 4.40x  │  high  │ "the FIRST chunk of a file,      │
  │       │              │        │        │  regardless of content"       ✓✓ │
  └───────┴──────────────┴────────┴────────┴──────────────────────────────────┘
   agreement with the purity table: 10/11
```

It named the strongest detector — `t10`, the highest lift in the codebook — as a
`for` loop header, citing that 8 of 12 sampled chunks begin with `for` against 1
of the other 120 firings. It described both `BinOp` codes as
arithmetic expressions and reached the `If` code `t3` through its `else:`
branches. It declined to find structure in all five near-chance `Call` codes.

**The `t9` row is the result worth the space.** By raw lift `t9` is the second-best
code in the codebook (4.40×) and reads as its best `FunctionDef` detector. It is
an artefact — it fires only on a file's opening chunk. The interpreter caught
that with no statistics at all, at high confidence, by arithmetic on the data:
sampled file indices spaced ≈67 apart, i.e. 809/12, one firing per file, and
across the other 120 sampled firings not one came from an opening chunk.

That is the negative control doing real work. The interpreter's two
high-confidence calls are both content-free codes and both correct, while every
genuinely ambiguous code is marked low. A procedure that reports uncertainty
where the signal is weak, and that rejects the most impressive-looking number in
the table on evidence, is one whose positive identifications carry weight.

## Specialists and polysemantic generalists coexist

Splitting the codebook at 2.0× lift — the same threshold used in the arithmetic
study — gives two regimes:

```
  ┌──────────────┬────────┬───────────┬──────────┬──────────┬────────────┐
  │ Role         │  codes │  firings  │ share of │ best     │ lift range │
  │              │        │           │ traffic  │ purity   │            │
  ├──────────────┼────────┼───────────┼──────────┼──────────┼────────────┤
  │ specialists  │      5 │     1,578 │   14.7%  │   33.3%  │ 2.06-6.57x │
  │ generalists  │      5 │     9,151 │   85.3%  │   35.3%  │ 0.75-1.24x │
  └──────────────┴────────┴───────────┴──────────┴──────────┴────────────┘
   t9 excluded as the file-start artefact, as in the R1 table
```

A handful of sharp detectors carry a small minority of the traffic while
high-frequency near-chance codes absorb the rest — `t0` alone fires 4,978 times
at 1.18× and acts as a fallback.

The cross-domain agreement is the interesting part. Arithmetic splits 3
specialists over **14.3%** of firings against 4 generalists over 85.7%; CodeNet
splits 5 over **14.7%** against 5 over 85.3%. Two entirely different label sets
— carry/borrow sub-tasks versus Python AST constructs — and the same ~15/85
division of labour, under the same metric and threshold.

Note also that the highest *purity* in each band is not the highest lift: the
best-purity generalist (`t4`, 35.3%) outranks every specialist on purity alone.
Purity without a base rate is not a measure of specialisation.

## Summary

On a real pretrained LLM, in a syntactic domain with an entirely different label
set from the arithmetic case study:

- **codes are causally necessary** — 39% relative accuracy loss on removal, with
  identity (not magnitude) accounting for 93% of that loss
- **codes specialise on ground-truth structure** — conditional and
  binary-expression detectors at 2.06–2.40× lift (n = 58 to 608)
- **an independent model recovers that structure blind** — 10/11 agreement on
  raw source chunks, including rejecting the codebook's second-highest global
  lift as a position artefact
- all measured on the **same checkpoint**

Reproduce: `repro/knockout.sh`, `repro/f3_codenet_purity.sh`,
`repro/f7_autointerp.sh`. Raw results in
`results/codenet_autointerp_rawfirings.json`,
`results/codenet_r1r2.json`, `results/codenet_s0.5_i10_z1_L8_n4000_knockout4.json`,
`results/codenet_gated_confound_nopad.json`.
