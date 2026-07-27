#!/usr/bin/env bash
# Determinism check: run every table script twice and diff the two outputs.
# These scripts read static JSON only, so the outputs MUST be byte-identical.
# Any diff is real nondeterminism (dict ordering, set iteration, float format,
# locale) and is worth chasing down. Exits nonzero if anything differs.
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/_common.sh"

# f4_per_code_ablation.sh was excluded while its results JSON did not exist --
# it exits nonzero in that state, which under `set -e` would abort the whole
# run rather than report one failure. The JSON exists now, so it is included.
SCRIPTS=(r1_purity.sh knockout.sh manifest.sh f2_escalation.sh \
         f3_codenet_purity.sh f4_per_code_ablation.sh f6_polysemanticity.sh \
         f7_autointerp.sh verify_claims.sh)

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

fail=0
for s in "${SCRIPTS[@]}"; do
  # Run from the repo root both times, exactly as a reader would.
  ( cd "$REPO_ROOT" && "$REPRO_DIR/$s" ) > "$TMP/$s.a" 2>&1
  ( cd "$REPO_ROOT" && "$REPRO_DIR/$s" ) > "$TMP/$s.b" 2>&1
  if diff -u "$TMP/$s.a" "$TMP/$s.b" > "$TMP/$s.diff"; then
    printf '  PASS  %-16s (%s bytes, identical across 2 runs)\n' \
      "$s" "$(wc -c < "$TMP/$s.a" | tr -d ' ')"
  else
    printf '  FAIL  %-16s outputs differ:\n' "$s"
    sed 's/^/        /' "$TMP/$s.diff"
    fail=1
  fi
done

# Also check the scripts are insensitive to the cwd they are invoked from.
for s in "${SCRIPTS[@]}"; do
  ( cd / && "$REPRO_DIR/$s" ) > "$TMP/$s.c" 2>&1 || true
  if ! diff -q "$TMP/$s.a" "$TMP/$s.c" > /dev/null; then
    printf '  FAIL  %-16s output depends on cwd (repo root vs /)\n' "$s"
    diff -u "$TMP/$s.a" "$TMP/$s.c" | sed 's/^/        /'
    fail=1
  fi
done

echo
if [ "$fail" -eq 0 ]; then
  echo "  determinism: PASS -- all ${#SCRIPTS[@]} tables byte-identical across runs and cwds."
else
  echo "  determinism: FAIL -- see diffs above."
fi
exit "$fail"
