#!/usr/bin/env bash
# R1 -- per-code subtask purity / lift for ckpt/arith_v9_paperhp.
# Source: results/gated/arithmetic_r1r2.json  (key R1.rows)
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/_common.sh"

"$PY" - "$RESULTS/gated/arithmetic_r1r2.json" <<'PY'
import json, sys

d = json.load(open(sys.argv[1]))
rows = sorted(d["R1"]["rows"], key=lambda r: -r["lift"])
n_eval = d["n_eval"]

H = ["Code", "n fires", "Top subtask", "Purity", "Marginal", "Lift", "Recall", "F1"]
W = [6, 8, 13, 8, 9, 7, 8, 7]

def rule(l, m, r):
    return l + m.join("─" * (w + 2) for w in W) + r

def row(cells):
    return "│ " + " │ ".join(str(c).rjust(w) for c, w in zip(cells, W)) + " │"

print()
print("  R1 | code -> answer-digit-subtask purity | %s | n_eval=%d | acc=%.4f"
      % (d["ckpt"], n_eval, d["accuracy"]))
print("  active codes = %d / 30   median lift = %.3f   replicated = %s"
      % (d["R1"]["n_active_codes"], d["R1"]["median_lift"], d["R1"]["replicated"]))
print("  " + rule("┌", "┬", "┐"))
print("  " + row(H))
print("  " + rule("├", "┼", "┤"))
for r in rows:
    print("  " + row([r["code"], r["n"], r["top_subtask"],
                      "%.3f" % r["purity"], "%.3f" % r["marginal"],
                      "%.2f" % r["lift"], "%.3f" % r["recall"], "%.3f" % r["f1"]]))
print("  " + rule("└", "┴", "┘"))
print("  Purity = P(top_subtask | code fires).  Marginal = P(top_subtask) overall.  Lift = purity/marginal.")
print()
PY
