#!/usr/bin/env bash
# R3 -- position locking of abstraction codes for ckpt/arith_v9_paperhp.
# Source: results/arithmetic_r1r2.json  (key R1.rows)
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/_common.sh"

"$PY" - "$RESULTS/arithmetic_r1r2.json" <<'PY'
import json, sys
from collections import Counter

d = json.load(open(sys.argv[1]))
rows = d["R1"]["rows"]
n_eval = d["n_eval"]
N_POS = 7

rows = sorted(rows, key=lambda r: (r["top_pos"], -r["n"]))

H = ["Code", "n fires", "Top pos", "#pos occupied", "of 7", "PosConc", "Cover@top_pos"]
W = [6, 8, 9, 14, 6, 8, 14]

def rule(l, m, r):
    return l + m.join("─" * (w + 2) for w in W) + r

def row(cells):
    return "│ " + " │ ".join(str(c).rjust(w) for c, w in zip(cells, W)) + " │"

print()
print("  R3 | position locking of active codes | %s | n_eval=%d problems x %d answer positions"
      % (d["ckpt"], n_eval, N_POS))
print("  " + rule("┌", "┬", "┐"))
print("  " + row(H))
print("  " + rule("├", "┼", "┤"))
for r in rows:
    cover = r["n"] * r["pos_concentration"] / n_eval
    print("  " + row([r["code"], r["n"], r["top_pos"], r["n_positions"], N_POS,
                      "%.3f" % r["pos_concentration"], "%.1f%%" % (100 * cover)]))
print("  " + rule("└", "┴", "┘"))

cnt = Counter(r["n_positions"] for r in rows)
n_codes = len(rows)
print()
print("  Codes occupying exactly 1 position : %d / %d" % (cnt.get(1, 0), n_codes))
print("  Codes occupying exactly 2 positions: %d / %d" % (cnt.get(2, 0), n_codes))
print("  Codes occupying 3+ positions       : %d / %d"
      % (sum(v for k, v in cnt.items() if k >= 3), n_codes))

# Inverse view: which codes peak at each answer position, and how much of that
# position's traffic they take. This is what decides whether position locking is
# informative or degenerate.
print()
print("  Inverse view -- codes whose top_pos is that position (share = fires_at_top_pos / n_eval):")
per_pos = {}
for r in rows:
    per_pos.setdefault(r["top_pos"], []).append(
        (r["code"], r["n"] * r["pos_concentration"] / n_eval))
W2 = [8, 7, 38, 11]
def rule2(l, m, r):
    return l + m.join("─" * (w + 2) for w in W2) + r
def row2(cells):
    return "│ " + " │ ".join(str(c).ljust(w) for c, w in zip(cells, W2)) + " │"
print("  " + rule2("┌", "┬", "┐"))
print("  " + row2(["Ans pos", "#codes", "codes peaking here (share of pos)", "accounted"]))
print("  " + rule2("├", "┼", "┤"))
unaccounted = []
for p in range(N_POS):
    lst = sorted(per_pos.get(p, []), key=lambda t: -t[1])
    acct = sum(s for _, s in lst)
    if acct < 0.5:
        unaccounted.append(p)
    desc = ", ".join("c%d %.0f%%" % (c, 100 * s) for c, s in lst) or "(no code peaks here)"
    print("  " + row2([p, len(lst), desc, "%.0f%%" % (100 * acct)]))
print("  " + rule2("└", "┴", "┘"))

single = [p for p in range(N_POS)
          if len(per_pos.get(p, [])) == 1 and sum(s for _, s in per_pos[p]) > 0.9]
print()
print("  Positions covered ~100% by a SINGLE code (code is a constant of position, not")
print("  a function of the input): %s" % (", ".join(str(p) for p in single) or "none"))
multi = [p for p in range(N_POS) if len(per_pos.get(p, [])) > 1]
print("  Positions with >1 competing code (code can carry input-dependent info): %s"
      % (", ".join(str(p) for p in multi) or "none"))
if unaccounted:
    two_pos = [r["code"] for r in rows
              if r["n_positions"] == 2 and r["pos_concentration"] <= 0.6]
    print("  Positions with no peaking code: %s" % ", ".join(str(p) for p in unaccounted))
    print("  -> these are the off-peak half of the 2-position codes %s (pos_concentration=0.5),"
          % ", ".join("c%d" % c for c in two_pos))
    print("     but this JSON stores only top_pos, so the exact code->second-position pairing")
    print("     is NOT recoverable from results/ alone. Not guessed here.")
print()
PY
