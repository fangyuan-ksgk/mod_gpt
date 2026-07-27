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

def chks(label, got, want):
    """Same bookkeeping as chk, for the string-valued cells (checkpoint paths,
    ground-truth labels, confidence words) that a table also has to get right."""
    global ok, fail
    good = got == want
    print(f"  {'OK ' if good else 'BAD'}  {label:<46} claimed {str(want):<8} found {got}")
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
# REPLICATED: an independent retrain on the same config gives ON 84.73 /
# RANDOM 69.58 / OFF 68.96, i.e. -15.77pp and gate OPEN. The two runs' RANDOM
# arms agree to 0.3pp (69.88 vs 69.58), so scrambled-identity accuracy is the
# stable quantity. The earlier "per-run artefact" reading is refuted.
ag = json.loads((R/"arith_s0.5_i10_z1_u8_knockout4.json").read_text())
chk("arith gated ON 82.96%",       round(100*ag["codes_ON"], 2), 82.96)
chk("arith gated OFF 75.08%",      round(100*ag["codes_OFF_full"], 2), 75.08)
chk("arith gated RANDOM 69.88%",   round(100*ag["codes_RANDOM"], 2), 69.88)
chk("arith gated knockout 7.88pp", round(ag["delta_pp"], 2), 7.88)
# RANDOM below OFF_full is the identity claim: wrong codes worse than none.
chk("arith RANDOM below OFF",
    1.0 if ag["codes_RANDOM"] < ag["codes_OFF_full"] else 0.0, 1.0, 0.001)

# Independent retrain, same config -- the replication that settles it.
rp = json.loads((R/"arith_s0.5_REPLICATE_knockout4.json").read_text())
chk("arith replicate ON 84.73%",     round(100*rp["codes_ON"], 2), 84.73)
chk("arith replicate RANDOM 69.58%", round(100*rp["codes_RANDOM"], 2), 69.58)
chk("arith replicate knockout 15.77pp", round(rp["delta_pp"], 2), 15.77)
chk("arith replicate gate OPEN", 1.0 if rp["gate_open"] else 0.0, 1.0, 0.001)
# The two runs' RANDOM arms agree to within 0.5pp -- stability of the claim.
chk("RANDOM stable across retrains",
    round(abs(100*(ag["codes_RANDOM"] - rp["codes_RANDOM"])), 2), 0.30, 0.5)

# Finding #2 robustness: same checkpoint, comment-normalised scoring.
cc = json.loads((R/"codenet_knockout4_cleanscore.json").read_text())
chk("codenet clean-score rel 29.1%", round(cc["delta_rel_pct"], 1), 29.1)
chk("codenet clean RANDOM below OFF",
    1.0 if cc["codes_RANDOM"] < cc["codes_OFF_full"] else 0.0, 1.0, 0.001)

# ---------------------------------------------------------------------------
# Coverage extension: every remaining table cell in the two rebuttal documents.
# Everything below was traced by hand to a source JSON and is asserted here so
# that a stale copy-paste cannot survive a re-run.
# ---------------------------------------------------------------------------

print("\n  -- REBUTTAL_arithmetic.md | R1 purity table (ckpt/arith_v9_paperhp) --")
ar = json.loads((R/"arithmetic_r1r2.json").read_text())
chks("arith R1 ckpt", ar["ckpt"], "ckpt/arith_v9_paperhp")
chk("arith R1 n_eval 2600", float(ar["n_eval"]), 2600.0, 0.001)
for code, n, purity, base, lift, npos in [
        (6,   415, 78.3, 12.6, 6.21, 2),
        (17, 1026, 43.6, 10.7, 4.07, 1),
        (11, 1163, 36.5, 16.3, 2.24, 1)]:
    chk(f"arith t{code} n={n}",          float(a[code]["n"]), float(n), 0.001)
    chk(f"arith t{code} purity {purity}%", round(100*a[code]["purity"], 1), purity)
    chk(f"arith t{code} base rate {base}%", round(100*a[code]["marginal"], 1), base)
    chk(f"arith t{code} #Pos {npos}",     float(a[code]["n_positions"]), float(npos), 0.001)
chk("arith R1 active codes 7", float(ar["R1"]["n_active_codes"]), 7.0, 0.001)

print("\n  -- REBUTTAL_arithmetic.md | sum-9 / tri-state carry table --")
# Recomputed from the firing dump exactly as repro/r5_sum9.sh does it.
from math import comb
fir = json.loads((R/"arith_firings.json").read_text())
s9 = {}
for code, v in fir.items():
    ex = [e for e in v["examples"] if e["column_sum"] is not None]
    s9[int(code)] = (len(ex), sum(1 for e in ex if e["column_sum"] == 9))
pool_n = sum(u for u, _ in s9.values())
pool_9 = sum(k for _, k in s9.values())
u6, k6 = s9[6]
chk("sum9 t6 11/14 firings",       float(k6), 11.0, 0.001)
chk("sum9 t6 sampled usable 14",   float(u6), 14.0, 0.001)
chk("sum9 P(sum9|t6) 78.6%",       round(100*k6/u6, 1), 78.6)
chk("sum9 pooled base 25.3%",      round(100*pool_9/pool_n, 1), 25.3)
loo = (pool_9 - k6) / (pool_n - u6)
chk("sum9 leave-one-out base 15.6%", round(100*loo, 1), 15.6)
chk("sum9 leave-one-out lift 5.04x", round((k6/u6)/loo, 2), 5.04)
p = sum(comb(u6, i) * (pool_9/pool_n)**i * (1-pool_9/pool_n)**(u6-i) for i in range(k6, u6+1))
chk("sum9 exact binomial p < 1e-4", 1.0 if p < 1e-4 else 0.0, 1.0, 0.001)
others = [(k/u)/(pool_9/pool_n) for c, (u, k) in s9.items() if c != 6]
chk("sum9 other codes lift min 0.00", round(min(others), 2), 0.00)
chk("sum9 other codes lift max 0.85", round(max(others), 2), 0.85)

print("\n  -- REBUTTAL_arithmetic.md | specialists vs generalists (lift >= 2.0) --")
rows_a = ar["R1"]["rows"]
spec_a = [r for r in rows_a if r["lift"] >= 2.0]
gen_a  = [r for r in rows_a if r["lift"] <  2.0]
na, ng = sum(r["n"] for r in spec_a), sum(r["n"] for r in gen_a)
chk("arith specialists 3 codes",    float(len(spec_a)), 3.0, 0.001)
chk("arith specialists 2,604 fires", float(na), 2604.0, 0.001)
chk("arith specialists 14.3% share", round(100*na/(na+ng), 1), 14.3)
chk("arith specialists best purity 78.3%", round(100*max(r["purity"] for r in spec_a), 1), 78.3)
chk("arith specialists lift lo 2.2x", round(min(r["lift"] for r in spec_a), 1), 2.2)
chk("arith specialists lift hi 6.2x", round(max(r["lift"] for r in spec_a), 1), 6.2)
chk("arith generalists 4 codes",     float(len(gen_a)), 4.0, 0.001)
chk("arith generalists 15,575 fires", float(ng), 15575.0, 0.001)
chk("arith generalists 85.7% share", round(100*ng/(na+ng), 1), 85.7)
chk("arith generalists best purity 24.9%", round(100*max(r["purity"] for r in gen_a), 1), 24.9)
chk("arith generalists lift lo 1.2x", round(min(r["lift"] for r in gen_a), 1), 1.2)
chk("arith generalists lift hi 1.6x", round(max(r["lift"] for r in gen_a), 1), 1.6)
big = max(gen_a, key=lambda r: r["n"])
chk("arith largest generalist n=5,200", float(big["n"]), 5200.0, 0.001)
chk("arith largest generalist 1.62x",   round(big["lift"], 2), 1.62)

print("\n  -- REBUTTAL_arithmetic.md | blind autointerp, paperhp checkpoint --")
aij = json.loads((R/"arithmetic_autointerp_rawfirings.json").read_text())
air = {r["code"]: r for r in aij["results"]}
chks("arith autointerp ckpt", aij["ckpt"], "ckpt/arith_v9_paperhp")
for code, lift, conf in [(6, 6.21, "high"), (17, 4.07, "high"), (11, 2.24, "medium"),
                         (0, 1.62, "high"), (1, 1.53, "high"),
                         (7, 1.60, "high"), (2, 1.24, "high")]:
    chk(f"arith autointerp t{code} lift {lift}x", round(air[code]["gt_lift"], 2), lift)
    chks(f"arith autointerp t{code} conf {conf}", air[code]["confidence"], conf)
chk("arith autointerp 4 positional", float(aij["summary"]["n_flagged_positional_only"]), 4.0, 0.001)
chk("arith autointerp 3 arithmetic", float(aij["summary"]["n_flagged_arithmetic_condition"]), 3.0, 0.001)

print("\n  -- REBUTTAL_arithmetic.md | R1 on the load-bearing ckpt (results/gated/) --")
# The second purity table is measured on ckpt/arith_s0.5_i10_z1_u8 -- the SAME
# checkpoint as the causal result -- not on arith_v9_paperhp. Different file.
gr = json.loads((R/"gated"/"arithmetic_r1r2.json").read_text())
g = {r["code"]: r for r in gr["R1"]["rows"]}
chks("arith gated R1 ckpt", gr["ckpt"], "ckpt/arith_s0.5_i10_z1_u8")
for code, n, purity, base, lift, npos in [
        (19,  32, 78.1,  8.4, 9.35, 1),
        (12,  65, 63.1, 16.3, 3.86, 2),
        (14, 180, 29.4, 10.7, 2.75, 1)]:
    chk(f"arith gated t{code} n={n}",          float(g[code]["n"]), float(n), 0.001)
    chk(f"arith gated t{code} purity {purity}%", round(100*g[code]["purity"], 1), purity)
    chk(f"arith gated t{code} base rate {base}%", round(100*g[code]["marginal"], 1), base)
    chk(f"arith gated t{code} lift {lift}x",   round(g[code]["lift"], 2), lift)
    chk(f"arith gated t{code} #Pos {npos}",    float(g[code]["n_positions"]), float(npos), 0.001)
chk("arith gated active codes 7",     float(gr["R1"]["n_active_codes"]), 7.0, 0.001)
chk("arith gated median lift 1.62x",  round(gr["R1"]["median_lift"], 2), 1.62)
chk("arith gated R1 replicated",      1.0 if gr["R1"]["replicated"] else 0.0, 1.0, 0.001)

print("\n  -- REBUTTAL_arithmetic.md | blind autointerp, load-bearing ckpt --")
gij = json.loads((R/"arith_gated_autointerp_rawfirings.json").read_text())
gir = {r["code"]: r for r in gij["results"]}
chks("arith gated autointerp ckpt", gij["ckpt"], "ckpt/arith_s0.5_i10_z1_u8")
for code, label, lift, conf in [(19, "UD", 9.35, "medium"), (12, "UC", 3.86, "medium"),
                                (14, "UB", 2.75, "low"),    (3,  "US", 1.62, "high"),
                                (1,  "US", 1.46, "high"),   (0,  "SA", 1.35, "high"),
                                (2,  "SA", 0.91, "high")]:
    chks(f"arith gated t{code} true label {label}", gir[code]["ground_truth_top_label"], label)
    chk(f"arith gated t{code} lift {lift}x",  round(gir[code]["gt_lift"], 2), lift)
    chks(f"arith gated t{code} conf {conf}",  gir[code]["confidence"], conf)
chk("arith gated autointerp agreement 7/7",
    float(gij["summary"]["agreement_with_purity_table"].split("/")[0]), 7.0, 0.001)
# The structural identity the interpreter derived from firing counts alone.
chk("arith 4923+65+180+32 = 5200",
    float(g[1]["n"] + g[12]["n"] + g[14]["n"] + g[19]["n"]), 5200.0, 0.001)
chk("arith t1 n=4,923", float(g[1]["n"]), 4923.0, 0.001)
chk("arith 5200 = 2 x 2600 eval set", float(2*gr["n_eval"]), 5200.0, 0.001)

print("\n  -- REBUTTAL_arithmetic.md | causal table, both runs --")
chk("arith run1 scramble 13.08pp", round(100*(ag["codes_ON"]-ag["codes_RANDOM"]), 2), 13.08)
chk("arith run2 OFF_full 68.96%",  round(100*rp["codes_OFF_full"], 2), 68.96)
chk("arith run2 scramble 15.15pp", round(100*(rp["codes_ON"]-rp["codes_RANDOM"]), 2), 15.15)
chk("arith run1 gate OPEN",        1.0 if ag["gate_open"] else 0.0, 1.0, 0.001)
# CAVEAT for the caption "relative loss 15.8% and 18.6%": the two figures are
# NOT the same arm. 15.8% is run 1's SCRAMBLE-relative loss (13.08/82.96);
# run 1's delete-relative loss is 9.50% (delta_rel_pct in the JSON). 18.6% is
# run 2's DELETE-relative loss (delta_rel_pct); its scramble-relative is 17.9%.
chk("arith run1 scramble-rel 15.8%",
    round(100*(ag["codes_ON"]-ag["codes_RANDOM"])/ag["codes_ON"], 1), 15.8, 0.05)
chk("arith run1 delete-rel (JSON) 9.50%", round(ag["delta_rel_pct"], 2), 9.50)
chk("arith run2 delete-rel 18.6%",  round(rp["delta_rel_pct"], 1), 18.6, 0.05)
chk("arith run2 scramble-rel 17.9%",
    round(100*(rp["codes_ON"]-rp["codes_RANDOM"])/rp["codes_ON"], 1), 17.9, 0.05)

print("\n  -- REBUTTAL_arithmetic.md | steering-scale sweep --")
for s, fn, dpp in [(0.3, "arith_s0.3_i10_z1_u8_knockout4.json", -6.88),
                   (0.7, "arith_s0.7_i10_z1_u8_knockout4.json", -8.12)]:
    j = json.loads((R/fn).read_text())
    chk(f"arith scale {s} gate CLOSED", 0.0 if j["gate_open"] else 1.0, 1.0, 0.001)
    chk(f"arith scale {s} delta {dpp}pp", round(j["delta_pp"], 2), dpp)
# "the same inversion appears on CodeNet at L=16" -- RANDOM below OFF_full there too.
l16 = json.loads((R/"codenet_L16_knockout4.json").read_text())
chk("codenet L=16 RANDOM below OFF",
    1.0 if l16["codes_RANDOM"] < l16["codes_OFF_full"] else 0.0, 1.0, 0.001)

print("\n  -- REBUTTAL_codenet.md | Finding #2 knockout table --")
chk("codenet RANDOM delta 6.37pp", round(100*(k["codes_ON"]-k["codes_RANDOM"]), 2), 6.37)
chk("codenet OFF_full delta 6.87pp", round(100*(k["codes_ON"]-k["codes_OFF_full"]), 2), 6.87)
chk("codenet RANDOM rel 36.4%",
    round(100*(k["codes_ON"]-k["codes_RANDOM"])/k["codes_ON"], 1), 36.4, 0.05)
chk("codenet identity = 93% of loss",
    round(100*(k["codes_ON"]-k["codes_RANDOM"])/(k["codes_ON"]-k["codes_OFF_full"])), 93.0, 0.5)

print("\n  -- REBUTTAL_codenet.md | R1 purity table --")
# Guard against the superseded ckpt/codenet_v9: R1 must come from the gated run.
chks("codenet R1 ckpt", cr["ckpt"], "ckpt/codenet_s0.5_i10_z1_L8_n4000")
chk("codenet R1 n_eval 800", float(cr["n_eval"]), 800.0, 0.001)
cw = {r["code"]: r for r in cr["R1"]["rows"]}
for code, n, construct, purity, lift in [
        (10,  30, "For",   33.3, 6.57),
        (7,   58, "BinOp", 32.8, 2.40),
        (6,  291, "BinOp", 32.0, 2.34),
        (5,  591, "If",    29.9, 2.19),
        (3,  608, "If",    28.1, 2.06)]:
    chk(f"codenet t{code} n={n}",             float(cw[code]["n"]), float(n), 0.001)
    chks(f"codenet t{code} construct {construct}", cw[code]["top_subtask"], construct)
    chk(f"codenet t{code} purity {purity}%",  round(100*cw[code]["purity"], 1), purity)
    chk(f"codenet t{code} lift {lift}x",      round(cw[code]["lift"], 2), lift)
chk("codenet Call base rate 28.5%", round(100*cr["R1"]["marginal"]["Call"], 1), 28.5)
chk("codenet t9 excluded, lift 4.40x", round(cw[9]["lift"], 2), 4.40)
chk("codenet t9 n=809 (one per file)", float(cw[9]["n"]), 809.0, 0.001)

print("\n  -- REBUTTAL_codenet.md | blind autointerp table --")
cij = json.loads((R/"codenet_autointerp_rawfirings.json").read_text())
cir = {r["code"]: r for r in cij["results"]}
chks("codenet autointerp ckpt", cij["ckpt"], "ckpt/codenet_s0.5_i10_z1_L8_n4000")
for code, label, lift, conf in [
        (10, "For",         6.57, "medium"), (7,  "BinOp",       2.40, "low"),
        (6,  "BinOp",       2.34, "low"),    (5,  "If",          2.19, "low"),
        (3,  "If",          2.06, "low"),    (4,  "Call",        1.24, "low"),
        (0,  "Call",        1.18, "high"),   (2,  "Call",        0.98, "low"),
        (8,  "Call",        0.88, "low"),    (1,  "Call",        0.75, "medium"),
        (9,  "FunctionDef", 4.40, "high")]:
    chks(f"codenet ai t{code} truth {label}", cir[code]["ground_truth_top_label"], label)
    chk(f"codenet ai t{code} lift {lift}x",  round(cir[code]["gt_lift_global"], 2), lift)
    chks(f"codenet ai t{code} conf {conf}",  cir[code]["confidence"], conf)
chk("codenet autointerp agreement 10/11",
    float(cij["summary"]["agreement_with_purity_table"].split("/")[0]), 10.0, 0.001)
chks("codenet t5 is the one miss", cir[5]["verdict"], "miss")
chk("codenet t9 spacing 809/12 = 67", round(cw[9]["n"]/12), 67.0, 0.5)

print("\n  -- REBUTTAL_codenet.md | specialists vs generalists --")
# NOTE: this table's lift RANGE column is position-matched lift (lift_pos), not
# the global lift used by every other table in the same document. Asserted here
# against lift_pos so the provenance is explicit rather than implied.
cd_ = json.loads((R/"codenet_gated_confound_nopad.json").read_text())
chks("codenet confound ckpt", cd_["ckpt"], "ckpt/codenet_s0.5_i10_z1_L8_n4000")
chk("codenet confound n_eval 800", float(cd_["n_eval"]), 800.0, 0.001)
chk("codenet confound C=30",       float(cd_["C_SIZE"]), 30.0, 0.001)
chk("codenet confound L=8",        float(cd_["L"]), 8.0, 0.001)
crow = cd_["code_rows"]
sp = [r for r in crow if r["lift_pos"] >= 1.5]
ge = [r for r in crow if r["lift_pos"] <  1.5]
ns, nge = sum(r["n"] for r in sp), sum(r["n"] for r in ge)
chk("codenet specialists 5 codes",     float(len(sp)), 5.0, 0.001)
chk("codenet specialists 1,578 fires", float(ns), 1578.0, 0.001)
chk("codenet specialists 13.7% share", round(100*ns/(ns+nge), 1), 13.7)
chk("codenet specialists best purity 33.3%", round(100*max(r["purity"] for r in sp), 1), 33.3)
chk("codenet specialists lift_pos lo 1.53x", round(min(r["lift_pos"] for r in sp), 2), 1.53)
chk("codenet specialists lift_pos hi 6.54x", round(max(r["lift_pos"] for r in sp), 2), 6.54)
chk("codenet generalists 6 codes",     float(len(ge)), 6.0, 0.001)
chk("codenet generalists 9,960 fires", float(nge), 9960.0, 0.001)
chk("codenet generalists 86.3% share", round(100*nge/(ns+nge), 1), 86.3)
chk("codenet generalists best purity 40.8%", round(100*max(r["purity"] for r in ge), 1), 40.8)
chk("codenet generalists lift_pos lo 0.77x", round(min(r["lift_pos"] for r in ge), 2), 0.77)
chk("codenet generalists lift_pos hi 1.30x", round(max(r["lift_pos"] for r in ge), 2), 1.30)
c0 = next(r for r in crow if r["code"] == 0)
chk("codenet t0 fires 4,978",  float(c0["n"]), 4978.0, 0.001)
# The prose quotes t0 at "1.15x" -- that is lift_pos. Its GLOBAL lift is 1.18x,
# which is what the autointerp table in the same document shows for t0.
chk("codenet t0 lift_pos 1.15x",    round(c0["lift_pos"], 2), 1.15)
chk("codenet t0 lift_global 1.18x", round(c0["lift_global"], 2), 1.18)

print("\n  -- sampling arithmetic quoted in the autointerp prose --")
# "8 of 12 sampled chunks ... against 1 of the other 120 firings": 11 codes x 12
# samples = 132, minus t10's own 12 = 120. Both docs quote these counts inline.
cp = json.loads((R/"codenet_autointerp_rawfirings_prompts.json").read_text())
chk("codenet 11 codes sampled",       float(len(cp)), 11.0, 0.001)
chk("codenet 12 samples per code",    float(max(e["n_examples"] for e in cp)), 12.0, 0.001)
chk("codenet 12 samples per code (min)", float(min(e["n_examples"] for e in cp)), 12.0, 0.001)
chk("codenet 'other 120 firings'",
    float(sum(e["n_examples"] for e in cp) - 12), 120.0, 0.001)
chk("arith 14 sampled firings per code",
    float(max(len(v["examples"]) for v in fir.values())), 14.0, 0.001)
chk("arith 98 sampled firings total",
    float(sum(len(v["examples"]) for v in fir.values())), 98.0, 0.001)
# Summary bullet: "conditional and binary-expression detectors at 2.1-2.3x lift".
# The four If/BinOp codes actually span 2.06x (t3) to 2.40x (t7); t7 sits above
# the quoted range, and t7's n is 58, not "hundreds".
det = [cw[c]["lift"] for c in (3, 5, 6, 7)]
chk("codenet If/BinOp detector lift lo 2.06x", round(min(det), 2), 2.06)
chk("codenet If/BinOp detector lift hi 2.40x", round(max(det), 2), 2.40)

print("\n  -- cross-document consistency --")
chk("arith 14.3% vs codenet 13.7% split (both ~14%)",
    round(100*na/(na+ng), 1) - round(100*ns/(ns+nge), 1), 0.6, 0.05)

print(f"\n  {ok} verified, {fail} failed")
sys.exit(1 if fail else 0)
PY
