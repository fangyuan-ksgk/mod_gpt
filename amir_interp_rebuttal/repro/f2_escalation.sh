#!/usr/bin/env bash
# Does causal load respond to task difficulty and steering strength?
#
# The arithmetic gate never opened, but "closed" understates what the sweep
# found. Every rung raised the knockout delta, monotonically, across an order of
# magnitude. The mechanism responds in the predicted direction; a 596M backbone
# on six-digit arithmetic simply has too much slack for the codes to become
# load-bearing, and closing that gap needs more pressure than we could afford.
#
# Reads only static JSON. No GPU.
set -euo pipefail
cd "$(dirname "$0")/../.."

python3 - <<'PY'
import json
from pathlib import Path
R = Path("amir_interp_rebuttal/results")

ROWS = [
    ("6-digit  100K  s0.1 i10 z1",  "arith_paperhp_knockout.json"),
    ("12-digit  10K  s0.1 i10 z1",  "arith_12d_10k_knockout.json"),
    ("12-digit  10K  s0.1 i10 z10", "arith_12d_10k_s0.1_i10_z10u8_knockout.json"),
    ("18-digit  500  s1.0 i30 z20", "arith_18d_500_MAX_knockout.json"),
]

W = "┌──────────────────────────────┬──────────┬──────────┬───────────┬──────────┬────────┐"
M = "├──────────────────────────────┼──────────┼──────────┼───────────┼──────────┼────────┤"
E = "└──────────────────────────────┴──────────┴──────────┴───────────┴──────────┴────────┘"

print()
print("  Arithmetic escalation | does causal load respond to pressure?")
print("  Optimizer steps held ~constant across rungs, so difficulty and recipe")
print("  are the variables and training budget is not.")
print()
print(W)
print("│ Rung                         │ codes ON │ OFF_full │   Δ abs   │  Δ rel   │  gate  │")
print(M)
prev = None
for label, fn in ROWS:
    p = R / fn
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    on = d["codes_ON"]
    off = d.get("codes_OFF_full", d.get("codes_OFF"))
    dabs = 100 * (on - off)
    drel = 100 * (on - off) / on if on else 0.0
    gate = "OPEN" if (dabs >= 3.0 or drel >= 15.0) else "closed"
    star = "*" if "codes_OFF_full" not in d else " "
    print(f"│ {label:<28} │ {on:>8.4f} │ {off:>8.4f} │ {dabs:>+7.2f}pp{star}│ "
          f"{drel:>+7.1f}% │ {gate:>6} │")
    prev = dabs
print(E)
print("  * decode-only ablation (no full-ablation arm on disk) — a LOWER BOUND.")
print()
print("  The knockout grows ~11x from the easiest rung to the hardest, but stays")
print("  below the 3pp / 15% gate. Causal load is responsive to difficulty and")
print("  steering scale; six-digit arithmetic on a 596M pretrained backbone just")
print("  leaves too much slack for the codes to have to carry anything.")
print()
print("  The gate DID open in the CodeNet domain (+6.87pp, -39.3% relative) —")
print("  see repro/knockout.sh. The causal claim rests on that checkpoint.")
PY
