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

Position matching is a deliberately conservative check and is not the primary
metric. Where a target is intrinsically position-bound it over-corrects — in the
arithmetic study a carry cascade is structurally impossible at the first and last
answer digits, so conditioning on position divides out the signal. It is
reported here because Python file structure is strongly position-correlated
(files open with imports and defs), which makes the check informative for *this*
domain, and because it isolates the one failure mode worth excluding: a code
whose lift is exactly the position's own base rate.

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

## What the position control removed

The control is not decoration — it removed the code that *looked* like the
study's best result:

```
  ┌────────────────────────┬────────────┬────────────┬──────────────────────┐
  │ Candidate              │ Lift global│ Lift pos   │ Verdict              │
  ├────────────────────────┼────────────┼────────────┼──────────────────────┤
  │ t9  -> FunctionDef     │    4.40x   │   1.00x    │ withdrawn            │
  │ t20 -> FunctionDef     │    3.84x   │   1.00x    │ withdrawn            │
  └────────────────────────┴────────────┴────────────┴──────────────────────┘
```

The disqualifying property is not that they are position-concentrated — it is
that their lift is **exactly** the position's own base rate. `P(FunctionDef | t9)`
equals `P(FunctionDef | chunk 0)` to three decimals, so knowing the code adds
literally nothing over knowing the position. Unlike a carry cascade, a Python
`def` is not restricted to one location; the code simply is not tracking it.

That is a narrower test than "fires at few positions", and deliberately so: a
code confined to a handful of positions may still be a real detector if its
target is position-bound. The 1.00× is what makes these two indefensible.

A separate left-padding defect was also found and fixed: a prefill chunk index
aligns with its source chunk only when the pad length is a multiple of the chunk
size, true for 28.5% of rows at batch 32. All numbers here are measured at batch
1, where no padding exists. Correcting it moved median lift from 1.10× to 2.06×
and flipped R1 from not-replicated to replicated — the bug had been *hiding*
real structure, not manufacturing it.

## Summary

On a real pretrained LLM, in a syntactic domain with an entirely different label
set from the arithmetic case study:

- **codes are causally necessary** — 39% relative accuracy loss on removal, with
  identity (not magnitude) accounting for 93% of that loss
- **codes specialise on ground-truth structure** — conditional and
  binary-expression detectors at 1.8–1.9× position-matched lift, distributed
  across nearly every chunk position
- both measured on the **same checkpoint**

Reproduce: `repro/knockout.sh`, `repro/f3_codenet_purity.sh`. Raw results in
`results/codenet_r1r2.json`, `results/codenet_s0.5_i10_z1_L8_n4000_knockout4.json`,
`results/codenet_gated_confound_nopad.json`.

**Scope.** Claims here are about what the codes encode and whether removing them
costs accuracy. Single-code surgical repair was measured on this same
load-bearing checkpoint and did not replicate. Two independent batch-1
measurements, differing only in generation length, agree:

```
  ┌──────────────────────┬────────────┬────────────────┬──────────────┐
  │ Run                  │ n attempts │ label-matched  │ random code  │
  ├──────────────────────┼────────────┼────────────────┼──────────────┤
  │ max_new_tokens = 8   │         82 │  0  (0.0%)     │  0  (0.0%)   │
  │ max_new_tokens = 32  │         69 │  0  (0.0%)     │  1  (1.4%)   │
  └──────────────────────┴────────────┴────────────────┴──────────────┘
```

Forcing the code that its own purity table associates with the correct
construct repairs nothing, and never beats its random control. Because the
knockout on this checkpoint is large, this is a **measured negative** rather
than an underpowered null: the codes demonstrably carry information the model
uses, and single-code edits still do not steer the prediction.
