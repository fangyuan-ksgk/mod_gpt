#!/usr/bin/env bash
# R5 -- sum-9 / equal-digit selectivity of abstraction codes.
# Source: results/arith_firings.json
# column_sum is stored as (a+b) for BOTH ops; it is null at answer_pos 0,
# where the overflow place has no operand column.
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/_common.sh"

"$PY" - "$RESULTS/arith_firings.json" <<'PY'
import json, sys
from math import comb

f = json.load(open(sys.argv[1]))

def binom_p_ge(k, n, p):
    """One-sided exact binomial tail P(X >= k) under X ~ Bin(n, p)."""
    return sum(comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1))

stats = {}
for code, v in f.items():
    ex = [e for e in v["examples"] if e["column_sum"] is not None]
    stats[int(code)] = dict(
        n_total=v["n_total"],
        n_sampled=len(v["examples"]),
        n_usable=len(ex),
        n_sub=sum(1 for e in ex if e["op"] == "sub"),
        k9=sum(1 for e in ex if e["column_sum"] == 9),
        keq=sum(1 for e in ex if e["operand_digits"][0] == e["operand_digits"][1]),
    )

pool_n = sum(s["n_usable"] for s in stats.values())
pool_9 = sum(s["k9"] for s in stats.values())
pool_eq = sum(s["keq"] for s in stats.values())
base9 = pool_9 / pool_n
baseeq = pool_eq / pool_n

H = ["Code", "n fires", "sampled", "usable", "sum9 k/n", "P(sum9)", "lift", "p", "eq k/n", "P(eq)", "lift", "p"]
W = [5, 8, 8, 7, 9, 8, 6, 7, 8, 7, 6, 7]

def rule(l, m, r):
    return l + m.join("─" * (w + 2) for w in W) + r

def row(cells):
    return "│ " + " │ ".join(str(c).rjust(w) for c, w in zip(cells, W)) + " │"

def fmt_p(p):
    return "%.4f" % p if p >= 1e-4 else "<1e-4"

print()
print("  R5 | sum-9 (add) and equal-operand-digit (sub) selectivity | ckpt/arith_v9_paperhp")
print("  Base rates pooled over ALL sampled firings: P(sum9)=%d/%d=%.3f   P(eq)=%d/%d=%.3f"
      % (pool_9, pool_n, base9, pool_eq, pool_n, baseeq))
print("  " + rule("┌", "┬", "┐"))
print("  " + row(H))
print("  " + rule("├", "┼", "┤"))
for code in sorted(stats, key=lambda c: -(stats[c]["k9"] / max(stats[c]["n_usable"], 1))):
    s = stats[code]
    u = s["n_usable"]
    r9, req = s["k9"] / u, s["keq"] / u
    print("  " + row([code, s["n_total"], s["n_sampled"], u,
                      "%d/%d" % (s["k9"], u), "%.3f" % r9, "%.2f" % (r9 / base9),
                      fmt_p(binom_p_ge(s["k9"], u, base9)),
                      "%d/%d" % (s["keq"], u), "%.3f" % req, "%.2f" % (req / baseeq),
                      fmt_p(binom_p_ge(s["keq"], u, baseeq))]))
print("  " + rule("├", "┼", "┤"))
print("  " + row(["POOL", sum(s["n_total"] for s in stats.values()),
                  sum(s["n_sampled"] for s in stats.values()), pool_n,
                  "%d/%d" % (pool_9, pool_n), "%.3f" % base9, "1.00", "-",
                  "%d/%d" % (pool_eq, pool_n), "%.3f" % baseeq, "1.00", "-"]))
print("  " + rule("└", "┴", "┘"))
print("  sampled = firing examples dumped per code; usable = those with non-null column_sum")
print("  (answer_pos 0 has no operand column, so it is dropped). p = exact one-sided binomial")
print("  P(X >= k) against the pooled base rate; the pool INCLUDES the code being tested, so")
print("  these p-values are conservative for high-rate codes and anti-conservative nowhere.")

print()
print("  Leave-one-out base rates (pool with the tested code removed):")
W2 = [5, 7, 16, 8, 16, 8]
def rule2(l, m, r):
    return l + m.join("─" * (w + 2) for w in W2) + r
def row2(cells):
    return "│ " + " │ ".join(str(c).rjust(w) for c, w in zip(cells, W2)) + " │"
print("  " + rule2("┌", "┬", "┐"))
print("  " + row2(["Code", "n sub", "sum9 LOO base", "LOO lift", "eq LOO base", "LOO lift"]))
print("  " + rule2("├", "┼", "┤"))
for code in sorted(stats, key=lambda c: -(stats[c]["k9"] / max(stats[c]["n_usable"], 1))):
    s = stats[code]
    u = s["n_usable"]
    lb9 = (pool_9 - s["k9"]) / (pool_n - u)
    lbeq = (pool_eq - s["keq"]) / (pool_n - u)
    print("  " + row2([code, s["n_sub"], "%.3f" % lb9, "%.2f" % ((s["k9"] / u) / lb9),
                       "%.3f" % lbeq,
                       "%.2f" % ((s["keq"] / u) / lbeq) if lbeq > 0 else "inf"]))
print("  " + rule2("└", "┴", "┘"))
print("  n sub = usable examples that are subtraction problems. The equal-digit condition")
print("  (ME/UD) is only meaningful for those; every rate above rests on <= 14 examples.")
print()
PY
