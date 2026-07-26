#!/usr/bin/env bash
# CodeNet R1 — do codes specialise on AST constructs, after controlling for position?
#
# Source is the BATCH-1 position-confound measurement of the gate-open (causally
# load-bearing) checkpoint. Two controls are baked in and both matter:
#
#   1. batch 1, so there is no left-padding. At batch 32 a prefill chunk index
#      aligns with its source chunk only when pad_len % L == 0 (28.5% of rows),
#      which silently reshuffles routing. Correcting it moved median lift
#      1.10x -> 2.06x and flipped R1 from not-replicated to replicated: the bug
#      was hiding real structure, not manufacturing it.
#   2. position-matched lift, comparing each code against its construct's
#      frequency AT THE POSITIONS THE CODE FIRES. Without this, a code that only
#      fires at chunk 0 inherits "Python files open with def/import" and looks
#      like a 4x detector while encoding nothing.
set -euo pipefail
cd "$(dirname "$0")/../.."

python3 - <<'PY'
import json
from pathlib import Path

d = json.loads(Path("amir_interp_rebuttal/results/codenet_gated_confound_nopad.json").read_text())
rows = sorted((r for r in d["code_rows"] if r.get("lift_pos") is not None),
              key=lambda r: -r["lift_pos"])

W = "┌────────┬────────┬─────────────┬──────────┬──────────┬──────────┬────────┬──────────────┐"
M = "├────────┼────────┼─────────────┼──────────┼──────────┼──────────┼────────┼──────────────┤"
E = "└────────┴────────┴─────────────┴──────────┴──────────┴──────────┴────────┴──────────────┘"

print()
print(f"  CodeNet R1 (position-controlled) | {d.get('ckpt','ckpt/codenet_s0.5_i10_z1_L8_n4000')}")
print(f"  Qwen3-0.6B | {d.get('n_eval')} held-out files | L={d.get('L')} | batch 1, no padding")
print()
print(W)
print("│  Code  │      n │ Construct   │   Purity │ Lift     │ Lift     │  #Pos  │ verdict      │")
print("│        │        │             │          │ global   │ pos-match│ of 32  │              │")
print(M)
for r in rows:
    lp, npos, n, lg = r["lift_pos"], r["n_positions"], r["n"], r["lift_global"]
    if lg >= 1.5 and lp <= 1.05:
        # Looked like a detector globally, then the effect vanished once we
        # compared against the construct's rate at the positions it fires.
        v = "CONFOUNDED"
    elif lp < 1.5:
        # Never a candidate — at or below chance on both baselines.
        v = "at chance"
    elif lp >= 1.5 and npos >= 20 and n >= 100:
        v = "robust"              # distributed + well-sampled + real effect
    else:
        v = "suggestive"          # real effect but thin n or few positions
    print(f"│ {('t%d' % r['code']):>6} │ {n:>6} │ {r['top_label']:<11} │ "
          f"{r['purity']:>7.1%} │ {r['lift_global']:>7.2f}x │ {lp:>7.2f}x │ "
          f"{npos:>6} │ {v:<12} │")
print(E)

robust = [r for r in rows if r["lift_pos"] >= 1.5 and r["n_positions"] >= 20 and r["n"] >= 100]
conf   = [r for r in rows if r["lift_global"] >= 1.5 and r["lift_pos"] <= 1.05]
lifts  = sorted(r["lift_global"] for r in rows)
print(f"\n  active codes: {len(rows)}   median global lift: {lifts[len(lifts)//2]:.2f}x")
print(f"  robust (pos-matched >=1.5x, >=20 positions, n>=100): {len(robust)}")
for r in robust:
    print(f"    t{r['code']:<3} -> {r['top_label']:<12} {r['lift_pos']:.2f}x pos-matched, "
          f"n={r['n']:<5} across {r['n_positions']}/32 positions")
print(f"\n  withdrawn as position-confounded: {len(conf)}")
for r in conf:
    print(f"    t{r['code']:<3} -> {r['top_label']:<12} {r['lift_global']:.2f}x global "
          f"collapses to {r['lift_pos']:.2f}x  ({r['n_positions']} position(s) — "
          f"this is P({r['top_label']} | that position))")
PY
