#!/usr/bin/env bash
# Finding #7 -- blind auto-interp agrees with the purity table.
# Source: results/arithmetic_autointerp_rawfirings.json
#
# A separate model was shown ONLY raw firing examples for each code -- no
# labels, no purity or lift statistics, no position distribution, and no list of
# candidate answers. It was told that "this code only marks a fixed position" was
# a valid answer, so declining to find structure carried no penalty.
#
# Agreement is the one comparison the table makes: did the interpreter sort each
# code onto the same side of the position-tag / genuine-specialist line as the
# purity table does? Ground truth (right-hand columns) never reached the model.
#
# Regenerate (needs a GPU for the dump, and an API key for the interpreter):
#     python -m amir_interp_rebuttal.dump_firings --study arithmetic
#     python -m amir_interp_rebuttal.autointerp  --study arithmetic --overwrite
# The second command leaves `verdict` unset; re-run the predicate-scoring pass
# afterwards. Without --overwrite it refuses to touch the reported file.
#
# Reads only static JSON. No GPU, no API calls.
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/_common.sh"

"$PY" - "$RESULTS/arithmetic_autointerp_rawfirings.json" "$RESULTS/arith_firings.json" <<'PY'
import json, sys

d = json.load(open(sys.argv[1]))
rows = d["results"]
s = d["summary"]
# Firing counts come from the dump the interpreter was shown, so the table
# reports the same n the model saw rather than a separately-derived number.
firings = json.load(open(sys.argv[2]))
firings = firings.get("codes", firings)
n_fires = {int(c): v["n_total"] for c, v in firings.items()}

H = ["Code", "n fires", "Interpreter call", "Conf", "GT label", "GT lift", "Agrees"]
W = [6, 8, 18, 7, 9, 8, 7]

def rule(l, m, r):
    return l + m.join("─" * (w + 2) for w in W) + r

def row(cells):
    return "│ " + " │ ".join(str(c).rjust(w) for c, w in zip(cells, W)) + " │"

# The scoring line: a code is a genuine specialist when its ground-truth lift
# clears 2.0x. Four of these sit at 1.24-1.62x, three at 2.24-6.21x.
SPECIALIST_LIFT = 2.0

print()
print("  Finding #7 | blind auto-interp from raw firings | %s" % d["ckpt"])
print("  interpreter: %s   method: %s" % (d["interpreter_model"], d["method"]))
print("  source: %s" % d["source_data"])
print("  " + rule("┌", "┬", "┐"))
print("  " + row(H))
print("  " + rule("├", "┼", "┤"))

n_agree = n_scored = 0
for r in sorted(rows, key=lambda r: -(r.get("gt_lift") or 0)):
    lift = r.get("gt_lift")
    call = "position tag" if r["is_positional_only"] else "real condition"
    if lift is None:
        agrees = "-"
    else:
        n_scored += 1
        ok = r["is_positional_only"] == (lift < SPECIALIST_LIFT)
        n_agree += ok
        agrees = "yes" if ok else "NO"
    print("  " + row(["t%d" % r["code"],
                      n_fires.get(r["code"], "-"),
                      call,
                      r["confidence"],
                      r.get("ground_truth_top_label", "-"),
                      "%.2fx" % lift if lift is not None else "-",
                      agrees]))
print("  " + rule("└", "┴", "┘"))
print("  Interpreter call = the model's own verdict from raw firings alone.")
print("  GT columns are withheld from the model and used only to score it.")
print("  Agrees = the call matches whether ground-truth lift clears %.1fx." % SPECIALIST_LIFT)
print()
print("  recomputed agreement : %d/%d" % (n_agree, n_scored))
print("  reported agreement   : %s" % s["agreement_with_purity_table"])
print("  flagged as position tags   : %d" % s["n_flagged_positional_only"])
print("  flagged as real conditions : %d" % s["n_flagged_arithmetic_condition"])

claimed = s["agreement_with_purity_table"]
if claimed != "%d/%d" % (n_agree, n_scored):
    print()
    print("  MISMATCH: the stored summary disagrees with the per-code rows.")
    sys.exit(1)
print()

print("  What each code was called, unprompted:")
for r in sorted(rows, key=lambda r: -(r.get("gt_lift") or 0)):
    print("    t%-3d %s" % (r["code"], r["fires_when"]))
print()
PY
