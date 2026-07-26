#!/usr/bin/env bash
# Per-code ablation — is any INDIVIDUAL code load-bearing?
#
# The aggregate knockout (repro/knockout.sh) shows the code channel matters:
# removing it costs 39.3% relative accuracy, and a RANDOM arm shows it is code
# identity that carries the information. Neither result says anything about a
# single code.
#
# Here each code is zeroed on its own, with the rest of the codebook intact.
# The confound is exposure -- a code firing on 60% of examples costs more than
# one firing on 3% whatever either encodes -- so every code's eval set is split
# by its own firing pattern in the unablated run:
#
#     affected     examples where that code fires
#     control      examples where it never fires
#     localisation delta_affected - delta_control
#
# A code that carries something should damage `affected` and leave `control`
# flat. If both move together the damage is going through a downstream path
# rather than through that code's own contribution, and the table says so.
#
# Reads only static JSON. No GPU.
set -euo pipefail
cd "$(dirname "$0")/../.."

python3 - <<'PY'
import json
from pathlib import Path

p = Path("amir_interp_rebuttal/results/codenet_per_code_ablation.json")
if not p.exists():
    raise SystemExit("missing results/codenet_per_code_ablation.json — "
                     "run: python -m amir_interp_rebuttal.per_code_ablation "
                     "--ckpt ckpt/codenet_s0.5_i10_z1_L8_n4000 --study codenet")
d = json.loads(p.read_text())
rows = d["rows"]

W = "┌────────┬──────────┬──────────┬────────────┬────────────┬──────────────┬──────────────┐"
M = "├────────┼──────────┼──────────┼────────────┼────────────┼──────────────┼──────────────┤"
E = "└────────┴──────────┴──────────┴────────────┴────────────┴──────────────┴──────────────┘"

print()
print(f"  Per-code ablation | {d['ckpt']}")
print(f"  Qwen3-0.6B | {d['n_eval']} held-out files | batch {d['eval_batch_size']} | "
      f"baseline {d['accuracy_baseline']:.2%}")
print("  One code zeroed at a time; rest of the codebook intact.")
print()
print(W)
print("│  Code  │ fires on │  share   │ Δ affected │ Δ control  │ localisation │ reading      │")
print(M)
def isnan(x):
    return x != x

for r in rows:
    loc = r["localisation"]
    da, dc = r["delta_affected_pp"], r["delta_control_pp"]
    if isnan(loc):
        # Code fires on EVERY example, so there is no control set and the
        # ablation cost cannot be separated from a global effect. Reporting
        # this as "no effect" would be wrong -- the cost may be real.
        verdict = "no control"
    elif loc >= 3.0 and da >= 3.0:
        verdict = "load-bearing"
    elif loc >= 1.0 and da >= 1.0:
        verdict = "weak"
    elif da >= 3.0 and loc < 1.0:
        # hurts, but hurts everywhere -- not attributable to this code
        verdict = "diffuse"
    else:
        verdict = "no effect"
    dcs = "     n/a  " if isnan(dc) else f"{dc:>+9.2f}pp"
    locs = "       n/a  " if isnan(loc) else f"{loc:>+11.2f}pp"
    print(f"│ {('t%d' % r['code']):>6} │ {r['n_examples_firing']:>8} │ "
          f"{r['share_of_eval']:>7.1%}  │ {da:>+9.2f}pp │ {dcs} │ "
          f"{locs} │ {verdict:<12} │")
print(E)

ok = [r for r in rows if not isnan(r["localisation"])]
nocontrol = [r for r in rows if isnan(r["localisation"])]
lb = [r for r in ok if r["localisation"] >= 3.0 and r["delta_affected_pp"] >= 3.0]
weak = [r for r in ok if 1.0 <= r["localisation"] < 3.0 and r["delta_affected_pp"] >= 1.0]
diffuse = [r for r in ok if r["delta_affected_pp"] >= 3.0 and r["localisation"] < 1.0]

print()
print(f"  codes tested: {len(rows)}")
print(f"  load-bearing (localisation >= 3pp AND affected >= 3pp): {len(lb)}")
for r in lb:
    print(f"    t{r['code']:<3} ablating it costs {r['delta_affected_pp']:.2f}pp where it "
          f"fires ({r['n_examples_firing']} files) vs {r['delta_control_pp']:+.2f}pp where it does not")
print(f"  weak (localisation 1-3pp): {len(weak)}")
for r in weak:
    print(f"    t{r['code']:<3} {r['delta_affected_pp']:+.2f}pp affected / "
          f"{r['delta_control_pp']:+.2f}pp control")
print(f"  diffuse (hurts, but not localised to where it fires): {len(diffuse)}")
for r in diffuse:
    print(f"    t{r['code']:<3} {r['delta_affected_pp']:+.2f}pp affected but "
          f"{r['delta_control_pp']:+.2f}pp control -- damage is not attributable to this code")

if nocontrol:
    print(f"  no control set (code fires on every example): {len(nocontrol)}")
    for r in nocontrol:
        print(f"    t{r['code']:<3} fires on all {r['n_examples_firing']} files; "
              f"ablating it costs {r['delta_affected_pp']:+.2f}pp overall, but with no "
              f"unaffected examples that cost cannot be localised to this code")

if not lb and not weak:
    print()
    print("  No individual code is load-bearing by this test. The channel matters")
    print("  in aggregate (see repro/knockout.sh) but the effect does not localise")
    print("  to any single code -- consistent with a distributed code, and with")
    print("  single-code repair failing.")
PY
