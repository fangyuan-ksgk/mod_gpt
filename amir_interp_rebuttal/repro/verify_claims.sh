#!/usr/bin/env bash
# Every headline number in REBUTTAL_arithmetic.md and REBUTTAL_codenet.md,
# checked against the JSON it came from. Exits nonzero on any mismatch.
#
# This exists because two silent-config bugs in this study (decode-time steering
# defaulting to zero; left-padding breaking chunk alignment) each produced
# clean-looking numbers that were wrong. A claim that cannot be traced to a
# result file is a claim that cannot be defended.
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 - <<'PY'
import json, sys
from pathlib import Path
R = Path("amir_interp_rebuttal/results")
ok = fail = 0
def chk(label, got, want, tol=0.02):
    global ok, fail
    good = got is not None and abs(got - want) <= tol
    print(f"  {'OK ' if good else 'BAD'}  {label:<46} claimed {want:<8} found {got}")
    ok += good; fail += (not good)

a = {r["code"]: r for r in json.loads((R/"arithmetic_r1r2.json").read_text())["R1"]["rows"]}
chk("arith t6 purity 78.3%",   round(100*a[6]["purity"],1), 78.3)
chk("arith t6 lift 6.21x",     round(a[6]["lift"],2),  6.21)
chk("arith t17 lift 4.07x",    round(a[17]["lift"],2), 4.07)
chk("arith t11 lift 2.24x",    round(a[11]["lift"],2), 2.24)

k = json.loads((R/"codenet_s0.5_i10_z1_L8_n4000_knockout4.json").read_text())
chk("codenet ON 17.50%",       round(100*k["codes_ON"],2), 17.50)
chk("codenet RANDOM 11.13%",   round(100*k["codes_RANDOM"],2), 11.13)
chk("codenet OFF_full 10.62%", round(100*k["codes_OFF_full"],2), 10.62)
chk("codenet rel drop 39.3%",
    round(100*(k["codes_ON"]-k["codes_OFF_full"])/k["codes_ON"],1), 39.3, 0.2)

c = {r["code"]: r for r in json.loads((R/"codenet_gated_confound_nopad.json").read_text())["code_rows"]}
chk("codenet t5 lift_pos 1.88x",   round(c[5]["lift_pos"],2), 1.88)
chk("codenet t3 lift_pos 1.88x",   round(c[3]["lift_pos"],2), 1.88)
chk("codenet t6 lift_pos 1.80x",   round(c[6]["lift_pos"],2), 1.80)
chk("codenet t9 confounded 1.00x", round(c[9]["lift_pos"],2), 1.00)

ki = json.loads((R/"arith_paperhp_knockout.json").read_text())
chk("arith knockout 0.15pp", round(100*(ki["codes_ON"]-ki["codes_OFF"]),2), 0.15)

ai = json.loads((R/"arithmetic_autointerp_rawfirings.json").read_text())["summary"]
chk("autointerp agreement 7/7", float(ai["agreement_with_purity_table"].split("/")[0]), 7.0, 0.001)

# Finding #4 — the measured negative. R1 and R2 come from the same batch-1 run,
# so the repair null is quoted against the purity table it belongs to.
cr = json.loads((R/"codenet_r1r2.json").read_text())
chk("codenet R1 median lift 2.06x", round(cr["R1"]["median_lift"], 2), 2.06)
chk("codenet R1 active codes 11",   float(cr["R1"]["n_active_codes"]), 11.0, 0.001)
r2 = cr["R2"]["R2b_targeted"]
chk("codenet R2 attempts 82",       float(r2["n_attempted"]), 82.0, 0.001)
chk("codenet R2 matched 0%",        float(r2["targeted_fix_rate"]), 0.0, 0.001)
chk("codenet R2 random 0%",         float(r2["random_fix_rate"]), 0.0, 0.001)

# Arithmetic causal result at scale=0.5, reported in REBUTTAL_arithmetic.md.
# The effect is non-monotonic: this is the only rung with a positive knockout,
# and its OFF_full (75.08%) sits below the ~86-90% of neighbouring scales while
# codes_ON is flat at 82-84%. Adjacent scales are different models, not
# replicates, so they cannot distinguish "narrow band where the model routes
# through the codes" from "property of this run". A same-config retrain is the
# test; the caveat is stated inline in the deliverable.
ag = json.loads((R/"arith_s0.5_i10_z1_u8_knockout4.json").read_text())
chk("arith gated ON 82.96%",       round(100*ag["codes_ON"], 2), 82.96)
chk("arith gated OFF 75.08%",      round(100*ag["codes_OFF_full"], 2), 75.08)
chk("arith gated RANDOM 69.88%",   round(100*ag["codes_RANDOM"], 2), 69.88)
chk("arith gated knockout 7.88pp", round(ag["delta_pp"], 2), 7.88)
# RANDOM below OFF_full is the identity claim: wrong codes worse than none.
chk("arith RANDOM below OFF",
    1.0 if ag["codes_RANDOM"] < ag["codes_OFF_full"] else 0.0, 1.0, 0.001)

# Finding #2 robustness: same checkpoint, comment-normalised scoring.
cc = json.loads((R/"codenet_knockout4_cleanscore.json").read_text())
chk("codenet clean-score rel 29.1%", round(cc["delta_rel_pct"], 1), 29.1)
chk("codenet clean RANDOM below OFF",
    1.0 if cc["codes_RANDOM"] < cc["codes_OFF_full"] else 0.0, 1.0, 0.001)

print(f"\n  {ok} verified, {fail} failed")
sys.exit(1 if fail else 0)
PY
