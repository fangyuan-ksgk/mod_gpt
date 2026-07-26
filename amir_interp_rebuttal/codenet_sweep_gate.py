"""Escalate the CodeNet v9 recipe until the routing codes become causally
load-bearing — the precondition for R2 (surgical swap) meaning anything.

WHY A SWEEP
-----------
R2 asks "does swapping one code repair a wrong prediction?". That is only a
question if the codes affect the computation at all. On ckpt/codenet_v9 the
knockout delta was -0.6pp: deleting every code made the model *better*, within
noise. Any R2 number measured there describes a no-op.

A MEASUREMENT BUG IN THE PRIOR CODENET KNOCKOUT
-----------------------------------------------
The earlier CodeNet knockout compared decode_scale=scale against
decode_scale=0.0. But `StackedAbstractionWrapperV9.generate` documents
decode_scale as "leaves prefill untouched" — it only silences steering during
*generation*. For arithmetic that is the whole intervention, because the labelled
structure is the generated answer. For CodeNet it is nearly none of it: the
source sits in the prompt, so 32 prefill chunks stay fully steered in BOTH arms
while at most `max_new_tokens/L` decode chunks differ. The prior -0.6pp therefore
bounds a much weaker claim than "the codes are not load-bearing".

So the gate is measured with four arms:

  ON           decode_scale = scale                  steering everywhere
  OFF_decode   decode_scale = 0.0                    prefill steered, decode not
                                                     (comparable to the old number)
  OFF_full     steering_emb zeroed, decode_scale=s   no steering anywhere
  RANDOM       codes replaced uniformly, both phases steering vectors present but
                                                     identities destroyed

OFF_full sets the gate: it is the only arm that actually removes the codes.
RANDOM is the control that separates "the codes carry information" from "the
model merely adapted to a steering vector of some magnitude" — if OFF_full drops
accuracy but RANDOM does not, the magnitude matters and the identity does not,
and the codes are still not carrying task structure.

THE GATE
--------
    knockout_delta = acc(ON) - acc(OFF_full)
passes when knockout_delta >= 3.0pp AND 0.10 <= acc(ON) <= 0.80.

Optimizer steps are held at ~250 across every rung by scaling epochs to data
size, so the recipe is the variable and training budget is not — the confound
that made the existing 125-step vs 625-step CodeNet comparison uninterpretable.

The published loss weights carry no authority here: the paper never ran CodeNet.
They are rung 0, not a constraint. Every rung tried, passing or not, lands in
results/codenet_sweep_summary.json with its full config, and the config is baked
into the checkpoint tag so any number can name the recipe that produced it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

PY = "/lambda/nfs/Amir-steering/codes/dlr/bin/python3"
MODEL = "Qwen/Qwen3-0.6B"
LOGS = Path("amir_interp_rebuttal/logs")
RESULTS = Path("amir_interp_rebuttal/results")
SUMMARY = RESULTS / "codenet_sweep_summary.json"

GATE_DELTA_PP = 3.0
ACC_FLOOR = 0.10
ACC_CEIL = 0.80

TARGET_STEPS = 250          # effective batch 32 -> epochs = TARGET_STEPS*32/size
EFF_BATCH = 32

# (scale, alpha_info, alpha_zipf, L, size)
#
# Escalated cheapest-first, exactly the arithmetic sweep's logic:
#   scale      — multiplies the steering vector itself. The most direct lever:
#                a model that can ignore codes at 0.1 has less room at 1.0.
#   alpha_info — the loss term paying codes for improving prediction. Raising it
#                pressures the router to make codes useful rather than decorative.
#   L          — chunk size. L=4 doubles the number of routing decisions per file
#                and halves how much text one code has to summarise.
#   size       — less data to memorise from, so the model cannot coast on weights.
#
# rung 0 is the published recipe and trains nothing: ckpt/codenet_v9 already
# exists, and re-measuring it with the four-arm knockout is free and is the
# control every later rung is read against.
LADDER = [
    (0.1, 10.0, 1.0, 8, 4000),   # rung 0 — published recipe (reuses ckpt/codenet_v9)
    (0.5, 10.0, 1.0, 8, 4000),   # stronger steering
    (1.0, 10.0, 1.0, 8, 4000),   # stronger still
    (1.0, 30.0, 1.0, 8, 4000),   # + heavier info-gain
    (1.0, 30.0, 5.0, 4, 4000),   # + finer routing, more codebook pressure
    (1.0, 30.0, 5.0, 4, 1000),   # + data-starved
    (2.0, 30.0, 5.0, 4, 1000),   # max pressure
]


def tag_for(scale, a_info, a_zipf, L, size):
    if (scale, a_info, a_zipf, L, size) == (0.1, 10.0, 1.0, 8, 4000):
        return "codenet_v9"          # the existing published-recipe checkpoint
    return (f"codenet_s{scale}_i{int(a_info)}_z{int(a_zipf)}"
            f"_L{L}_n{size}")


def run(cmd, log_path, env=None):
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as fh:
        return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                              env={**os.environ, **(env or {})}).returncode


def train(tag, scale, a_info, a_zipf, L, size):
    ckpt = Path(f"ckpt/{tag}")
    if (ckpt / "final.pt").exists():
        print(f"[sweep] {tag}: checkpoint exists, skipping training", flush=True)
        return ckpt
    epochs = max(1, round(TARGET_STEPS * EFF_BATCH / size))
    steps = epochs * size // EFF_BATCH
    print(f"[sweep] {tag}: scale={scale} a_info={a_info} a_zipf={a_zipf} "
          f"L={L} size={size} epochs={epochs} (~{steps} steps)", flush=True)
    t0 = time.time()
    rc = run([
        PY, "-u", "train_steer_pt.py",
        "--mode", "v9", "--dataset", "codenet", "--model_name", MODEL,
        "--L", str(L), "--C_SIZE", "30", "--scale", str(scale),
        "--inject_layers", "14",
        "--alpha_info", str(a_info), "--alpha_abs", "0.1",
        "--alpha_zipf", str(a_zipf),
        "--max_length", "256", "--batch_size", "8",
        "--gradient_accumulation_steps", "4",
        "--num_epochs", str(epochs), "--lr", "1e-5", "--steer_lr", "1e-3",
        "--num_rollouts", "4", "--search_temp", "1.0",
        "--eval_samples", "200", "--eval_batch_size", "16",
        "--eval_every", "200", "--max_new_tokens", "32",
        # Training-time evals must use the run's own steering scale, else they
        # report the unsteered model and tell us nothing about the thing we train.
        "--eval_decode_scale", str(scale),
        "--log_every", "25", "--output_dir", str(ckpt),
    ], LOGS / f"{tag}_train.log", env={"CODENET_SIZE": str(size)})
    print(f"[sweep] {tag}: training rc={rc} in {(time.time()-t0)/60:.1f} min",
          flush=True)
    if rc != 0 or not (ckpt / "final.pt").exists():
        return None
    return ckpt


# Run out-of-process so each rung gets a clean CUDA context and a crash in one
# knockout cannot take the sweep down with it.
KNOCKOUT_SRC = r'''
import json, sys, random
from collections import Counter, defaultdict
import torch
from amir_interp_rebuttal.load_local import load_local_steered
from amir_interp_rebuttal.codenet_dataset import CodeNetDataset
from amir_interp_rebuttal.runner import batched_generate

ckpt, out = sys.argv[1], sys.argv[2]
w, tok, a = load_local_steered(ckpt, device="cuda")
scale, C = float(a["scale"]), int(a["C_SIZE"])
ds = CodeNetDataset(split="test", tokenizer=tok, max_length=256, size=800)
idxs = list(range(len(ds)))
res = {"scale": scale, "L": a["L"]}

def acc(record=False, decode_scale=scale):
    # decode_scale is passed EXPLICITLY on every call. The V9 wrapper defaults
    # _decode_scale_override to 0.0, so omitting it silently disables decode-time
    # steering and makes every arm identical.
    recs = batched_generate(w, tok, ds, "cuda", idxs, eval_batch_size=32,
                            max_new_tokens=32, record_codes=record,
                            decode_scale=decode_scale)
    return sum(r["correct"] for r in recs) / len(recs), recs

# ── ON ───────────────────────────────────────────────────────────
res["codes_ON"], recs = acc(record=True)
used = Counter()
for r in recs:
    for c in (r.get("prompt_codes") or []) + (r.get("codes") or []):
        if int(c) >= 0:
            used[int(c)] += 1
tot = sum(used.values()) or 1
res["n_codes_used"] = len(used)
res["n_codes_active"] = len([c for c, k in used.items() if k / tot >= 0.01])
res["top_code_share"] = round(used.most_common(1)[0][1] / tot, 4) if used else None

# ── OFF_decode: the arm the previous CodeNet knockout used. Prefill stays
#    steered, so for a prompt-heavy task this is a partial ablation only.
res["codes_OFF_decode"], _ = acc(decode_scale=0.0)

# ── RANDOM: steering vectors still injected at full magnitude, identities
#    destroyed, in BOTH phases. Separates "codes carry information" from
#    "the model adapted to a vector of roughly this size".
rng = random.Random(0)
orig = w._ablate_patch_codes
def rand_patch(codes, phase):
    return torch.randint(0, C, codes.shape, device=codes.device, dtype=codes.dtype)
w._ablate_patch_codes = rand_patch
try:
    res["codes_RANDOM"], _ = acc()
finally:
    w._ablate_patch_codes = orig

# ── OFF_full: steering_emb zeroed -> no steering in prefill OR decode.
#    This is the real knockout and it sets the gate.
emb = w.steering_emb
saved = [e.weight.data.clone() for e in emb] if isinstance(emb, torch.nn.ModuleList) \
        else emb.weight.data.clone()
try:
    if isinstance(emb, torch.nn.ModuleList):
        for e in emb:
            e.weight.data.zero_()
    else:
        emb.weight.data.zero_()
    res["codes_OFF_full"], _ = acc()
finally:
    if isinstance(emb, torch.nn.ModuleList):
        for e, s in zip(emb, saved):
            e.weight.data.copy_(s)
    else:
        emb.weight.data.copy_(saved)

res["delta_pp_full"] = 100 * (res["codes_ON"] - res["codes_OFF_full"])
res["delta_pp_decode"] = 100 * (res["codes_ON"] - res["codes_OFF_decode"])
res["delta_pp_random"] = 100 * (res["codes_ON"] - res["codes_RANDOM"])
for k in ("codes_ON", "codes_OFF_decode", "codes_RANDOM", "codes_OFF_full"):
    print(f"{k:>18} = {res[k]:.4f}")
print(f"DELTA_full={res['delta_pp_full']:+.2f}pp  "
      f"DELTA_decode={res['delta_pp_decode']:+.2f}pp  "
      f"DELTA_random={res['delta_pp_random']:+.2f}pp")
json.dump(res, open(out, "w"), indent=2)
'''


def knockout(tag, ckpt):
    out = RESULTS / f"{tag}_knockout4.json"
    RESULTS.mkdir(parents=True, exist_ok=True)
    rc = run([PY, "-u", "-W", "ignore", "-c", KNOCKOUT_SRC, str(ckpt), str(out)],
             LOGS / f"{tag}_knockout4.log")
    if rc != 0 or not out.exists():
        print(f"[sweep] {tag}: knockout failed — see {tag}_knockout4.log", flush=True)
        return None
    return json.loads(out.read_text())


def main():
    summary = []
    if SUMMARY.exists():
        summary = json.loads(SUMMARY.read_text())
    done = {s["tag"] for s in summary if s.get("status") == "ok"}

    for scale, a_info, a_zipf, L, size in LADDER:
        tag = tag_for(scale, a_info, a_zipf, L, size)
        if tag in done:
            print(f"[sweep] {tag}: already measured, skipping", flush=True)
            continue
        rec = {"tag": tag, "scale": scale, "alpha_info": a_info,
               "alpha_zipf": a_zipf, "L": L, "size": size,
               "target_steps": TARGET_STEPS}

        ckpt = train(tag, scale, a_info, a_zipf, L, size)
        if ckpt is None:
            rec["status"] = "train_failed"
            summary.append(rec)
            SUMMARY.write_text(json.dumps(summary, indent=2))
            continue

        k = knockout(tag, ckpt)
        if k is None:
            rec["status"] = "knockout_failed"
            summary.append(rec)
            SUMMARY.write_text(json.dumps(summary, indent=2))
            continue

        acc = k["codes_ON"]
        delta = k["delta_pp_full"]
        passed = (delta >= GATE_DELTA_PP) and (ACC_FLOOR <= acc <= ACC_CEIL)
        rec.update({"status": "ok", **k, "gate_open": passed})
        summary.append(rec)
        SUMMARY.write_text(json.dumps(summary, indent=2))
        print(f"[sweep] {tag}: acc={acc:.1%} delta_full={delta:+.2f}pp "
              f"delta_decode={k['delta_pp_decode']:+.2f}pp "
              f"delta_random={k['delta_pp_random']:+.2f}pp "
              f"active={k['n_codes_active']}/30 "
              f"gate={'OPEN' if passed else 'closed'}", flush=True)

        if passed:
            print(f"[sweep] GATE OPEN at {tag} — running R1/R2", flush=True)
            run([PY, "-u", "-W", "ignore", "-m", "amir_interp_rebuttal.analyze",
                 "--study", "codenet", "--ckpt", str(ckpt),
                 "--model_name", MODEL, "--eval_n", "800",
                 "--max_new_tokens", "32", "--max_swap_examples", "200"],
                LOGS / f"{tag}_analyze.log")
            # Position-confound audit too: a purity number that survives the
            # gate still has to survive the control that killed t20.
            run([PY, "-u", "-W", "ignore", "-m",
                 "amir_interp_rebuttal.codenet_confound", "--ckpt", str(ckpt),
                 "--out", f"amir_interp_rebuttal/results/{tag}_position_confound.json"],
                LOGS / f"{tag}_confound.log")
            print(f"[sweep] DONE — R1/R2 measured on a load-bearing model ({tag})",
                  flush=True)
            return

    print("[sweep] ladder exhausted, gate never opened.", flush=True)


if __name__ == "__main__":
    main()
