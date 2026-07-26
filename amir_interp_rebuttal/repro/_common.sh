# shellcheck shell=bash
# Shared path resolution for the rebuttal repro scripts.
# Every script sources this, so all of them can be run from any cwd
# (including the repo root) and still find amir_interp_rebuttal/results/.
set -euo pipefail
export LC_ALL=C
REPRO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REBUTTAL_DIR="$(dirname "$REPRO_DIR")"
REPO_ROOT="$(dirname "$REBUTTAL_DIR")"
RESULTS="$REBUTTAL_DIR/results"
PY="${PYTHON:-python3}"
export REPRO_DIR REBUTTAL_DIR REPO_ROOT RESULTS PY
