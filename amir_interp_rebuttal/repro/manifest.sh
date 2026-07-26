#!/usr/bin/env bash
# Provenance manifest: for each table, its source JSON, sha256, git commit and checkpoint.
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/_common.sh"

GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="clean"
if ! git -C "$REPO_ROOT" diff --quiet 2>/dev/null || \
   ! git -C "$REPO_ROOT" diff --cached --quiet 2>/dev/null; then
  GIT_DIRTY="DIRTY (uncommitted changes in tree)"
fi
export GIT_COMMIT GIT_DIRTY REPO_ROOT

"$PY" - "$RESULTS" <<'PY'
import hashlib, json, os, subprocess, sys

R = sys.argv[1]
commit = os.environ.get("GIT_COMMIT", "unknown")
dirty = os.environ.get("GIT_DIRTY", "unknown")

# table -> (script, [source json files], checkpoint)
TABLES = [
    ("R1 purity",     "repro/r1_purity.sh",  ["arithmetic_r1r2.json"], "ckpt/arith_v9_paperhp"),
    ("R3 position",   "repro/r3_position.sh", ["arithmetic_r1r2.json"], "ckpt/arith_v9_paperhp"),
    ("R5 sum9/eq",    "repro/r5_sum9.sh",    ["arith_firings.json"],   "ckpt/arith_v9_paperhp"),
    ("knockout",      "repro/knockout.sh",   ["arith_paperhp_knockout.json"], "ckpt/arith_v9_paperhp"),
    ("knockout",      "repro/knockout.sh",   ["codenet_v9_knockout4.json"], "ckpt/codenet_v9"),
    ("knockout",      "repro/knockout.sh",   ["codenet_s0.5_i10_z1_L8_n4000_knockout4.json"], "ckpt/codenet_s0.5_i10_z1_L8_n4000"),
]

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()

H = ["Table", "Script", "Source JSON (results/)", "sha256 (first 16)", "Checkpoint"]
W = [12, 22, 30, 18, 22]

def rule(l, m, r):
    return l + m.join("─" * (w + 2) for w in W) + r

print()
print("  Provenance manifest")
print("  repo   : %s" % os.environ.get("REPO_ROOT", "unknown"))
print("  commit : %s  [%s]" % (commit, dirty))
print("  " + rule("┌", "┬", "┐"))
print("  │ " + " │ ".join(h.ljust(w) for h, w in zip(H, W)) + " │")
print("  " + rule("├", "┼", "┤"))
for name, script, srcs, ckpt in TABLES:
    for s in srcs:
        p = os.path.join(R, s)
        digest = sha256(p)[:16] if os.path.exists(p) else "MISSING"
        cells = [name, script, "results/" + s, digest, ckpt]
        print("  │ " + " │ ".join(c.ljust(w) for c, w in zip(cells, W)) + " │")
print("  " + rule("└", "┴", "┘"))
print()
print("  Full sha256:")
seen = []
for _, _, srcs, _ in TABLES:
    for s in srcs:
        if s in seen:
            continue
        seen.append(s)
        p = os.path.join(R, s)
        print("    %-34s %s" % ("results/" + s, sha256(p) if os.path.exists(p) else "MISSING"))
print()
PY
