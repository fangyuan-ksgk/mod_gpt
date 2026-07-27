# Reproduction

Every table in `REBUTTAL_arithmetic.md` and `REBUTTAL_codenet.md` renders from a
JSON in `results/` via a script in `repro/`. **No GPU required** — the scripts
read static files and print the table.

Run from the repository root.

## Check everything

```bash
bash amir_interp_rebuttal/repro/verify_claims.sh   # 242 assertions, every cell vs source
bash amir_interp_rebuttal/repro/determinism.sh     # each table twice, fails on any drift
bash amir_interp_rebuttal/repro/manifest.sh        # table -> checkpoint -> JSON -> sha256
```

`verify_claims.sh` asserts string cells as well as numbers, so a row cannot
drift onto the wrong code or checkpoint without failing.

## One table at a time

| Finding | Script | Checkpoint |
|---|---|---|
| #2 causal necessity (CodeNet) | `repro/knockout.sh` | `codenet_s0.5_i10_z1_L8_n4000` |
| #3 R1, syntactic (CodeNet) | `repro/f3_codenet_purity.sh` | `codenet_s0.5_i10_z1_L8_n4000` |
| #3 R1, sub-task (arithmetic) | `repro/r1_purity.sh` | `arith_v9_paperhp` |
| #5 tri-state carry | `repro/r5_sum9.sh` | `arith_v9_paperhp` |
| #6 specialists + generalists | `repro/f6_polysemanticity.sh` | both |
| #7 blind auto-interp | `repro/f7_autointerp.sh` | both |
| per-code ablation | `repro/f4_per_code_ablation.sh` | `codenet_s0.5_i10_z1_L8_n4000` |
| difficulty escalation | `repro/f2_escalation.sh` | sweep rungs |

## Weights

Published to `thoughtworks/dlr-rebuttal-interp` (private). Each checkpoint
folder carries its own `PROVENANCE.md` with the same finding → JSON → sha256 →
script mapping, so a downloaded model is self-describing.

Regenerating a results JSON from weights *does* need a GPU; see `PLAN.md` for
the commands.
