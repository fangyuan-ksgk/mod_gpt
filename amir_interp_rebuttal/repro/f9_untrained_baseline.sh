#!/usr/bin/env bash
# Reproduces the "0.0% untrained" claim AND the control that makes it safe to
# state. Without the control, 0.0% is open to the obvious objection: the eval
# scores exact-match on a zero-padded 7-digit string, so a base model answering
# "948083" instead of "0948083" is wrong on formatting, not on arithmetic.
#
# Expected (verified 2026-07-28, n=200):
#   bare      exact 0.0%   lenient 0.0%   pred None       (emits no digits)
#   fewshot   exact 0.0%   lenient 0.0%   pred '0698883'  (right format, wrong answer)
#   instruct  exact 0.0%   lenient 1.0%   pred '417080'   (echoes an operand)
#
# The fewshot row is the one that matters: given four solved examples the base
# model produces correctly formatted 7-digit zero-padded answers and still gets
# 0/200 right under both exact and integer-lenient scoring. The 0.0% is a
# capability result, not a formatting artefact.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=/lambda/nfs/Amir-steering/codes/dlr/bin/python3

# Full-eval bare baseline (2600 examples, all 24 splits) -> arith_untrained_baseline.json
# is produced by the original baseline path; this script covers the prompted controls.
$PY -u -W ignore -m amir_interp_rebuttal.untrained_prompted --n 200 --shots 4
