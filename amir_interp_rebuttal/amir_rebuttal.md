# Arithmetic study

We partially replicate our arithmetic findings on the residual stream of Qwen-3-0.6B (596M). This scales up the results of the paper on models under 1M parameters by a factor of over 600x.

We train DLR using a codebook allowed to be of size 30 over 100,000 examples in 1 epoch and batch size 32, taking the base model from an accuracy of 0% to 82.96%. Our hyperparameters are one abstract code step per answer digit, fix layer 14 (out of 28) at a steering strength of 0.5. We do autoregressive evaluations over 2600 examples.

Ground truth is the standard per-digit taxonomy we report in our paper in Table 19: **SA** no carry, **SC** generates carry, **SS** digits sum to 9, **UC** consumes carry, **US** carry cascade through a sum-9 run, plus the borrow analogues **MD/MB/ME/UB/UD**. In total, the model learns to generate 7 codes out of the 30 slots available to it.

## Finding 1: The codes remain causally significant.

For two independently trained models at the above settings, we find that scrambling all codes significantly diminishes the accuracy of the model, where we report both the absolute drop in performance, and the relative drop.

```
  ┌──────────────────┬──────────┬──────────┬────────────┬───────────┐
  │ Run              │ codes ON │  RANDOM  │ Δ scramble │   Δ rel   │
  ├──────────────────┼──────────┼──────────┼────────────┼───────────┤
  │ run 1            │  82.96%  │  69.88%  │  −13.08pp  │  −15.8%   │
  │ run 2 (retrain)  │  84.73%  │  69.58%  │  −15.15pp  │  −17.9%   │
  └──────────────────┴──────────┴──────────┴────────────┴───────────┘
```

For this particular hyperparameter set, we found the model converged on 7 codes being active out of 30, where the remainder were inactive. We focus on Run 1 for the remainder of our addition interp analysis.

## Finding 2: Select codes specialise on algorithmic sub-tasks

We define purity as `P(sub-task | code)`. Here **Lift** is the value of purity divided by that sub-task's base rate. I.e., a code that has 78% purity in sub-task UD, where its base occurrence rate is 8.4%, shows high specialization and a lift of 9.35x for this sub-task.

We found the model learnt to use a total of 7 codes, 3 of which specialized in the following tasks.

```
  ┌────────┬────────┬────────────┬──────────┬───────────┬────────┐
  │  Code  │      n │ Sub-task   │   Purity │ Base rate │   Lift │
  ├────────┼────────┼────────────┼──────────┼───────────┼────────┤
  │    t19 │     32 │ UD         │   78.1%  │     8.4%  │  9.35x │
  │    t12 │     65 │ UC         │   63.1%  │    16.3%  │  3.86x │
  │    t14 │    180 │ UB         │   29.4%  │    10.7%  │  2.75x │
  └────────┴────────┴────────────┴──────────┴───────────┴────────┘
```
`t19` is a borrow-propagation detector at 9.35× base rate.
`t12` marks leading-column carry-out and `t14` borrow-critical columns.
`t19` rests on n=32 examples only, because `UD` is rare at the position it fires on.

The remaining 4 used codes are much more generalist in nature, occurring to mark various answer positions.


## Finding 3: Auto-interp agrees with the distributional statistics of the specialized codes.

We apply the following auto-interp scheme. For each code, we give a Sonnet-5 model access to 14 generations, with the following information:

- `problem` — the literal string the model was given, plus the answer it produced
- `answer_pos` — which answer digit this code fired on (0 = leftmost)
- `answer_digit` — the digit value emitted at that position
- `operand_digits` — the two input digits in that column (null on the leading position, which has no column beneath it)
- `column_sum` — their sum

The Sonnet-5 model inspects the patterns and then outputs the role of this code. Importantly, the Sonnet-5 model is not given any access to the original Quirke et al. taxonomy when composing its interpretations. We find that the resulting products corroborate strongly with the distribution statistics of the code specializations of Finding 2.

```
  ┌───────┬──────┬────────┬──────────────────────────────────────────┬─────────┐
  │  Code │ true │  lift  │  blind description (verbatim, abridged)  │  match  │
  ├───────┼──────┼────────┼──────────────────────────────────────────┼─────────┤
  │  t19  │  UD  │ 9.35x  │ "equal digits consuming a borrow-in ...  │    ✓    │
  │       │      │        │ the digit is 9, produced by a borrow"    │         │
  │  t12  │  UC  │ 3.86x  │ "addition ... the two leading operand    │    ✓    │
  │       │      │        │ digits sum to >=15, i.e. the leading     │         │
  │       │      │        │ column carries out"                      │         │
  │  t14  │  UB  │ 2.75x  │ "subtraction columns where the digits    │    ✓    │
  │       │      │        │ are equal or differ by exactly one --    │         │
  │       │      │        │ borrow-critical columns"                 │         │
  └───────┴──────┴────────┴──────────────────────────────────────────┴─────────┘
```

We found 4 other codes fired primarily for generalist tasks such as position markers, and (for this particular set of hyperparameters) the remaining 23 other were not utilized by the model.

# Code study

Next to illustrate further that interpretability studies can be carried beyond toy tasks, we take up a study on a Python subset of CodeNet[1], a set of competitive programming python files. We allow one code per 8-token chunk, with a total vocabulary of 30 codes, and injection fixed at layer 14. We train on 4,000 Python files from the Python800 subset of CodeNet[1] (800 problem classes) and evaluate on 800 held-out files, splitting by problem rather than by submission so near-identical solutions to the same problem cannot straddle train/test. Ground truth is the innermost AST construct each chunk sits inside. These are:`For`, `If`, `Return`, `Call`, `Subscript`, `Assign`, `FunctionDef` and nine others; recovered from source with Python's `ast`. In total, the model learns to utilize 11 out of the 30 vocab slots available to it.

We recover similar flavored findings to our addition study, namely:
## Finding 1: The codes remain causally significant.

As in the arithmetic study, scrambling all codes significantly diminishes the accuracy of the model.

```
  ┌──────────────────┬──────────┬──────────┬────────────┬───────────┐
  │ Run              │ codes ON │  RANDOM  │ Δ scramble │   Δ rel   │
  ├──────────────────┼──────────┼──────────┼────────────┼───────────┤
  │ run 1            │  17.50%  │  11.13%  │   −6.37pp  │  −36.4%   │
  └──────────────────┴──────────┴──────────┴────────────┴───────────┘
```

## Finding 2: Again, select codes specialise on syntactic structure
We show here the lift for various AST constructs, as well as the most specialized codes linked with each.

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

`t5`, `t3` and `t6` support the claim most strongly: with hundreds of examples tracking conditionals and binary expressions throughout the corpus. `t10` shows the largest effect but rests on n=30 examples and is suggestive only. The remaining codes seem more generalist in nature, and generally linked with the `Call` operation.

## Finding 3: Auto-interp again agrees with the distributional statistics.

We apply the same auto-interp scheme as in our arithmetic study: a Sonnet-5 model sees only raw firings (the source chunk, its position in the file, and its neighbours); with no AST label, no purity statistic, and no list of candidate constructs.

```
  ┌───────┬─────────────┬────────┬───────────────────────────────────────────┬─────────┐
  │  Code │     true    │  lift  │             blind description             │  match  │
  ├───────┼─────────────┼────────┼───────────────────────────────────────────┼─────────┤
  │  t10  │     For     │ 6.57x  │ "for <var> in <iterable>: loop header"    │    ✓    │
  │   t7  │    BinOp    │ 2.40x  │ "middle of a long floating-point / trig   │    ✓    │
  │       │             │        │ expression"                               │         │
  │   t6  │    BinOp    │ 2.34x  │ "inside a multi-term arithmetic /         │    ✓    │
  │       │             │        │ trigonometric expression"                 │         │
  │   t5  │      If     │ 2.19x  │ "an indentation run opening a statement   │    ✗    │
  │       │             │        │ inside a block"                           │         │
  │   t3  │      If     │ 2.06x  │ "a bare `else:` / branch tail followed    │    ✓    │
  │       │             │        │ by an assignment"                         │         │
  └───────┴─────────────┴────────┴───────────────────────────────────────────┴─────────┘
```

Four of the five specialists are named correctly from raw firings alone, including the two `BinOp` codes described as arithmetic expressions and the `For` code described as a loop header. `t5` is a miss: the interpreter reported a purely positional rule (an indentation run) where the ground truth is `If`.

One negative finding we would admit at this stage of our replication is that we did not yet find the targeted ``subtask-repair" by code swapping. We suspect this would emerge more naturally at a certain range of vocabulary size, steering magnitude, dataset size and hyperparameters where the model becomes more sensitive to live DLR code edits; but this particular exhaustive search remains out of scope of rebuttal.

We acknowledge in advance that these are light, proof of concept interpretability studies; suitable for the scope of rebuttal. But we hope they illustrate the potential of our methods for interpretability on tasks in non-toy models and datasets, and will include these in the camera ready version with more in-depth exploration deferred to future research.

[1] https://arxiv.org/abs/2105.12655