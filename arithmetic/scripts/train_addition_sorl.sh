#!/bin/bash
# ──────────────────────────────────────────────────────────────
# Train addition with SoRL (abstraction tokens)
# Run AFTER baseline to compare
# ──────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/../.."

SAVE_DIR=${1:-ckpt/addition_sorl_$(date +%Y%m%d_%H%M)}

echo "TODO: implement SoRL addition training script"
echo "Will use same architecture + SoRL search + abstraction tokens"
echo "Save dir: $SAVE_DIR"
