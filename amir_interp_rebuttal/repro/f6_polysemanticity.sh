#!/usr/bin/env bash
# Finding #6 — specialists and polysemantic generalists coexist in one codebook.
#
# The paper's claim: a trained codebook is not uniformly specialist. A few codes
# are sharp detectors; alongside them sit high-frequency generalists that carry
# most of the traffic at near-chance purity and act as fallbacks.
#
# Reads only static JSON. No GPU, no model load.
set -euo pipefail
cd "$(dirname "$0")/../.."

python3 - <<'PY'
import json
from pathlib import Path

R = Path("amir_interp_rebuttal/results")
arith = json.loads((R / "arithmetic_r1r2.json").read_text())["R1"]["rows"]

def band(r):
    """Specialist vs generalist by lift, not purity.

    Purity alone cannot separate the two: on a skewed label set a fallback code
    can post respectable purity purely by tracking the most common label. Lift
    over that label's base rate is what distinguishes a detector from a code
    that simply fires a lot.
    """
    return "specialist" if r["lift"] >= 2.0 else "generalist"

rows = sorted(arith, key=lambda r: -r["lift"])
W = "┌────────┬────────────┬────────┬─────────────┬──────────┬──────────┬────────┬────────┐"
M = "├────────┼────────────┼────────┼─────────────┼──────────┼──────────┼────────┼────────┤"
E = "└────────┴────────────┴────────┴─────────────┴──────────┴──────────┴────────┴────────┘"

print("  Finding #6 | specialists vs polysemantic generalists | ckpt/arith_v9_paperhp")
print("  Qwen3-0.6B | six-digit add/sub | 2,600 held-out problems | 7 active codes of 30")
print()
print(W)
print("│  Code  │ Role       │      n │ Top sub-task│   Purity │ Base rate│   Lift │  #Pos  │")
print(M)
for r in rows:
    print(f"│ {('t%d' % r['code']):>6} │ {band(r):<10} │ {r['n']:>6} │ "
          f"{r['top_subtask']:>11} │ {r['purity']:>7.1%} │ {r['marginal']:>7.1%} │ "
          f"{r['lift']:>5.2f}x │ {r['n_positions']:>6} │")
print(E)

spec = [r for r in rows if band(r) == "specialist"]
gen  = [r for r in rows if band(r) == "generalist"]
n_spec = sum(r["n"] for r in spec)
n_gen  = sum(r["n"] for r in gen)
tot = n_spec + n_gen

print()
print(f"  specialists (lift >= 2.0) : {len(spec)} codes, {n_spec:,} firings "
      f"({100*n_spec/tot:.1f}% of traffic)")
print(f"  generalists (lift <  2.0) : {len(gen)} codes, {n_gen:,} firings "
      f"({100*n_gen/tot:.1f}% of traffic)")
if gen:
    big = max(gen, key=lambda r: r["n"])
    print(f"  largest generalist        : t{big['code']}, {big['n']:,} firings at "
          f"{big['lift']:.2f}x lift — the fallback code")
if spec:
    b = spec[0]
    print(f"  sharpest specialist       : t{b['code']} -> {b['top_subtask']}, "
          f"{b['purity']:.1%} at {b['lift']:.2f}x lift, {b['n']:,} firings")
print()
print("  The two coexist: a small number of sharp detectors carry a minority of")
print("  the traffic, while high-frequency near-chance codes absorb the rest.")
print("  This is the structure the paper reports, reproduced on a real LLM.")
PY

python3 - <<'PY2'
import json
from pathlib import Path
R = Path("amir_interp_rebuttal/results")
d = json.loads((R/"codenet_gated_confound_nopad.json").read_text())
rows = [r for r in d["code_rows"] if r.get("lift_pos") is not None]
rows.sort(key=lambda r: -r["lift_pos"])
tot = sum(r["n"] for r in rows)
spec = [r for r in rows if r["lift_pos"] >= 1.5]
gen  = [r for r in rows if r["lift_pos"] <  1.5]
W="  ┌──────────────┬────────┬───────────┬──────────┬──────────┬────────────┐"
M="  ├──────────────┼────────┼───────────┼──────────┼──────────┼────────────┤"
E="  └──────────────┴────────┴───────────┴──────────┴──────────┴────────────┘"
print(); print("  CodeNet | specialists vs generalists | ranked by POSITION-MATCHED lift")
print(f"  {d.get('ckpt')} | {tot} firings"); print()
print(W)
print("  │ Role         │  codes │  firings  │ share of │ best     │ lift range │")
print(M)
for name, g in (("specialists", spec), ("generalists", gen)):
    n = sum(r["n"] for r in g); L=[r["lift_pos"] for r in g]
    print(f"  │ {name:<12} │ {len(g):>6} │ {n:>9,} │ {n/tot:>7.1%}  │ "
          f"{max(r['purity'] for r in g):>7.1%}  │ {min(L):.2f}-{max(L):.2f}x │")
print(E)
print()
print("  arithmetic splits 3 specialists / 14.3% of firings vs 4 generalists / 85.7%.")
print("  CodeNet splits 5 / 13.7% vs 6 / 86.3%. Different label sets entirely,")
print("  same ~14/86 division of labour.")
print()
print("  Note the highest PURITY in the table belongs to a generalist (t9, 40.8%),")
print("  the chunk-0 position artefact. Purity alone would rank it first.")
PY2
