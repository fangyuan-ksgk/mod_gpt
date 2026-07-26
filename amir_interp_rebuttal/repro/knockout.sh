#!/usr/bin/env bash
# Finding #2 — are the abstraction codes causally load-bearing?
#
# Four arms, because the naive one is misleading:
#
#   ON          steering active everywhere
#   OFF_decode  decode_scale=0 -> silences steering during GENERATION only.
#               Prefill stays steered. For CodeNet the source lives in the
#               prompt, so this is almost no intervention at all; it is why an
#               earlier pass reported the codes as inert.
#   OFF_full    steering_emb zeroed -> no code anywhere. THE REAL ABLATION.
#   RANDOM      code identities replaced uniformly, vectors still injected.
#               Separates "the codes carry information" from "the model adapted
#               to a steering vector of roughly this magnitude".
#
# Relative delta is reported next to absolute: on a 17% task 6.9pp is a 39%
# loss, and the toy-model figure this is compared against (95.5% -> 0.1%) is
# itself a relative statement.
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/_common.sh"

"$PY" - "$RESULTS" <<'PY'
import json, os, sys
R = sys.argv[1]

ROWS = [
    ("arithmetic 6-digit",  "ckpt/arith_v9_paperhp",             "arith_paperhp_knockout.json"),
    ("arithmetic 12-digit", "ckpt/arith_12d_10k",                "arith_12d_10k_knockout.json"),
    ("arithmetic 12d z10",  "ckpt/arith_12d_10k_s0.1_i10_z10u8", "arith_12d_10k_s0.1_i10_z10u8_knockout.json"),
    ("codenet baseline",    "ckpt/codenet_v9",                   "codenet_v9_knockout4.json"),
    ("codenet gate-open",   "ckpt/codenet_s0.5_i10_z1_L8_n4000", "codenet_s0.5_i10_z1_L8_n4000_knockout4.json"),
]

def g(d, *names):
    """Tolerate the three knockout schemas this study produced."""
    for n in names:
        if n in d:
            return d[n]
    return None

W = "┌──────────────────────┬──────────┬──────────┬──────────┬──────────┬────────────┬──────────┬────────┐"
M = "├──────────────────────┼──────────┼──────────┼──────────┼──────────┼────────────┼──────────┼────────┤"
E = "└──────────────────────┴──────────┴──────────┴──────────┴──────────┴────────────┴──────────┴────────┘"

print()
print("  Finding #2 | are the codes causally load-bearing? | Qwen3-0.6B + DLR v9")
print("  OFF_full = steering_emb zeroed (the real ablation). rel = (ON-OFF_full)/ON")
print()
print(W)
print("│ Model                │ codes ON │ OFF_dec  │ RANDOM   │ OFF_full │   Δ abs    │  Δ rel   │  gate  │")
print(M)
partial = False
for name, ckpt, fn in ROWS:
    p = os.path.join(R, fn)
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    on   = g(d, "codes_ON")
    dec  = g(d, "codes_OFF_dec", "codes_OFF_decode", "codes_OFF")
    rnd  = g(d, "codes_RANDOM")
    full = g(d, "codes_OFF_full")
    ref  = full if full is not None else dec        # fall back if no full arm
    if full is None:
        partial = True
    dabs = 100 * (on - ref)
    drel = 100 * (on - ref) / on if on else 0.0
    gate = "OPEN" if (dabs >= 3.0 or drel >= 15.0) else "closed"
    f = lambda v: ("%8.4f" % v) if v is not None else "       -"
    mark = " " if full is not None else "*"
    print(f"│ {name:<20} │ {f(on)} │ {f(dec)} │ {f(rnd)} │ {f(full)} │ "
          f"{dabs:>+9.2f}pp{mark}│ {drel:>+7.1f}% │ {gate:>6} │")
print(E)
if partial:
    print("  * no full-ablation arm on disk; delta uses OFF_decode and is a LOWER BOUND.")
print()
print("  Reference — the paper's toy model: 95.5% -> 0.1% knockout (-99.9% rel),")
print("  random-code replacement 12.3%.")
print()
PY
