#!/usr/bin/env bash
# Reproduces the single-code repair negative in REBUTTAL_arithmetic.md
# ("Single-code repair: a null about reach, not about t19").
#
# Two stages, both on ckpt/arith_s0.5_i10_z1_u8:
#   1. repair sweep  -> results/arith_ud_repair.json
#   2. dose response -> results/arith_ud_repair_dose.json
#
# Expected (verified 2026-07-28):
#   baseline 0.8296 · UD digits 1520 · routed to t19 25 (0 wrong) · 76 repairable
#   unforced resume sanity 0/76 · t19 0/76 · every control 0/76
#   dose: t19 x32 -> 72/76 moved, 9 fixed, 36 broken
#         t14 x32 -> 41/76 moved, 7 fixed, 40 broken   (wrong specialist, same effect)
#         rand x32 -> 34/76 moved, 0 fixed, 12 broken
#
# The controls are the point. `0/76` alone cannot distinguish "t19 carries no
# repair signal" from "no single-code edit can move this model at all"; the
# never-trained-code arm and the dose sweep settle that it is the latter.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=/lambda/nfs/Amir-steering/codes/dlr/bin/python3
CK=ckpt/arith_s0.5_i10_z1_u8

$PY -u -W ignore -m amir_interp_rebuttal.ud_repair \
    --ckpt "$CK" --code 19 --label UD --eval_n 2600 --n_control 5

$PY -u -W ignore -m amir_interp_rebuttal.ud_repair \
    --ckpt "$CK" --code 19 --label UD --eval_n 2600 \
    --dose --dose_mults 1,2,4,8,16,32,64
