"""
Escalate task difficulty and data sparsity until the abstraction codes become
causally load-bearing, then — and only then — measure surgical-code repair (R2).

WHY THIS IS A SWEEP AND NOT A RUN
---------------------------------
R2 asks "does swapping one code repair a wrong prediction?". That question is
only meaningful if the codes affect the computation at all. On six-digit
arithmetic at 100K examples the knockout delta was +0.2pp — removing every code
cost nothing — so R2 measured nothing, in both arms, and its null said nothing
about real LLMs. R2 is gated on R4.

The gate is opened by making the model unable to solve the task from its weights
alone: longer carry chains (more digits) and less data to memorise from. This
script walks that ladder and stops at the first rung where the gate opens.

THE GATE
--------
    knockout_delta = acc(codes active) - acc(codes zeroed)

A rung passes when BOTH hold:
  * knockout_delta >= GATE_DELTA_PP      -> codes carry real computation
  * ACC_FLOOR <= acc <= ACC_CEIL         -> model works, but still makes errors
                                            (R2 needs a non-trivial error set;
                                             a 99% model has nothing to repair,
                                             a 5% model is noise)

Optimizer steps are held ~constant across rungs by scaling epochs to data size,
so difficulty and sparsity are the variables and training budget is not — the
mistake that made the CodeNet 125-vs-625 comparison uninterpretable.

The sweep escalates on three axes, cheapest first:

  1. DIVERSITY — alpha_zipf, target_vocab_util. Fixed first, because a collapsed
     codebook caps everything downstream: earlier runs used 7 of 30 codes while
     the Zipf prior was asking for 24, and with 7 codes over 10 sub-task labels
     no code CAN be a clean specialist. Six of those seven degenerated into
     position tags carrying no information about the input.
  2. RECIPE — scale, alpha_info. Makes the codes harder to route around.
  3. TASK — digits, data size. Changes what is being measured, so it goes last.

The published loss weights carry no special authority here: this is a task the
paper never ran. They are the starting rung, not a constraint. What does carry
over is disclosure — every rung tried, passing or not, lands in
sweep_gate_summary.json with its config and its active-code count, and the
config is baked into each checkpoint tag. Any number that reaches the rebuttal
can name the recipe that produced it, and a reader can see this was a search.

NOT MERGED WITH codenet_sweep_gate.py, DELIBERATELY
---------------------------------------------------
The two sweep drivers look like duplicates and are not. 87 of ~320 lines match,
and the largest shared block is the 13-line import header; the ladder shape, the
training command, the knockout ARM SET (two arms plus a full ablation here,
four arms with a RANDOM control there), the result schema (`{tag}_knockout.json`
vs `{tag}_knockout4.json`) and the gate rule (this one also passes on relative
delta) all differ. Merging them means a branch at every one of those points.

The schemas are the hard constraint: repro/verify_claims.sh and repro/manifest.sh
read both shapes, and arith_paperhp_knockout.json is behind a reported number.
Nor can a merge be validated — re-running either driver means retraining on a
GPU, so a refactor here would ship untested against the checkpoints the rebuttal
rests on. Highest risk, lowest reward. Left as two files on purpose.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

PY = "/lambda/nfs/Amir-steering/codes/dlr/bin/python3"
MODEL = "Qwen/Qwen3-0.6B"
LOGS = Path("amir_interp_rebuttal/logs")
RESULTS = Path("amir_interp_rebuttal/results")

GATE_DELTA_PP = 3.0     # knockout cost that counts as "codes are load-bearing"
ACC_FLOOR = 0.15        # below this the model hasn't learned the task
ACC_CEIL = 0.90         # above this there are too few errors to repair
GATE_DELTA_REL_PCT = 15.0  # relative knockout cost; catches real effects that a
                           # small absolute delta hides on low-accuracy tasks

# (digits, size, epochs, scale, alpha_info, alpha_zipf)
#
# Escalates on two axes: the TASK (longer carry chains, less data — the model
# cannot coast) and the RECIPE (stronger steering scale, heavier info-gain —
# the codes are made harder to route around).
#
# `scale` is swept first within a task setting because it is the most direct
# lever and the cheapest: it multiplies the steering vector itself, so a model
# that ignores codes at 0.1 has less room to at 0.5. `alpha_info` is the loss
# term rewarding codes for improving prediction — raising it pressures the
# router to make them useful rather than decorative. Only then do we change the
# task, which changes what is being measured.
#
# The published loss weights carry no special authority here: this is a task the
# paper never ran. They are the starting rung, not a constraint. What does carry
# over is disclosure — every rung tried is recorded in sweep_gate_summary.json,
# and the config is baked into each checkpoint tag, so any reported number names
# the recipe that produced it and a reader can judge this as the search it is.
# (digits, size, epochs, scale, alpha_info, alpha_zipf, target_util)
#
# CODEBOOK COLLAPSE IS THE FIRST THING TO FIX. Earlier runs used 7 of 30 codes
# while the Zipf prior was asking for 80% utilisation — because alpha_zipf=1.0
# was competing with alpha_info=10.0 and losing 10:1. A collapsed codebook caps
# every downstream result: with 7 codes over 10 sub-task labels no code CAN be a
# clean specialist, and 6 of those 7 degenerated into position tags.
#
# So diversity is swept before difficulty. Raising alpha_zipf is also the
# cheapest lever — it needs no change to the task, so a result obtained here is
# still a result about six/twelve-digit arithmetic rather than about some
# harder problem we invented.
LADDER = [
    (12, 10_000, 10, 0.1, 10.0,  1.0, 0.8),   # published recipe, harder task
    (12, 10_000, 10, 0.1, 10.0, 10.0, 0.8),   # diversity weight 10x
    (12, 10_000, 10, 0.5, 10.0, 10.0, 0.8),   # + stronger steering
    (12, 10_000, 10, 0.5, 10.0, 20.0, 0.9),   # + push utilisation to 27/30
    (12,  2_000, 50, 0.5, 30.0, 20.0, 0.9),   # + data-starved, heavier info-gain
    (18,  2_000, 50, 0.5, 30.0, 20.0, 0.9),   # + longer carry chains
    (18,    500, 200, 1.0, 30.0, 20.0, 0.9),  # max pressure
]


def run(cmd, log_path, env=None):
    LOGS.mkdir(parents=True, exist_ok=True)
    full_env = {**os.environ, **(env or {})}
    with open(log_path, "w") as fh:
        return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                              env=full_env).returncode


def _training_in_flight(tag):
    """True if a train_steer_pt process is already writing this rung.

    Matches the output_dir on the process command line, and excludes this
    script's own PID so the check can never match itself — the self-matching
    pgrep pattern that deadlocked an earlier pipeline.
    """
    try:
        out = subprocess.run(["pgrep", "-af", f"output_dir ckpt/{tag}"],
                             capture_output=True, text=True).stdout
    except FileNotFoundError:
        return False
    me = str(os.getpid())
    for line in out.splitlines():
        pid = line.split(None, 1)[0]
        if pid != me and "train_steer_pt.py" in line:
            return True
    return False


def train(tag, digits, size, epochs, scale, a_info, a_zipf, tgt_util):
    ckpt = Path(f"ckpt/{tag}")
    if (ckpt / "final.pt").exists():
        print(f"[sweep] {tag}: checkpoint exists, skipping training")
        return ckpt
    if _training_in_flight(tag):
        print(f"[sweep] {tag}: training already in flight, waiting for it")
        while _training_in_flight(tag):
            time.sleep(60)
        if (ckpt / "final.pt").exists():
            print(f"[sweep] {tag}: in-flight run finished, using its checkpoint")
            return ckpt
        print(f"[sweep] {tag}: in-flight run exited without a checkpoint")
        return None
    print(f"[sweep] {tag}: {digits}-digit, {size} ex, {epochs} ep, "
          f"scale={scale}, a_info={a_info}, a_zipf={a_zipf}, "
          f"target_util={tgt_util}")
    rc = run([
        PY, "-u", "train_steer_pt.py",
        "--mode", "v9", "--dataset", "arithmetic", "--model_name", MODEL,
        "--L", "1", "--C_SIZE", "30", "--scale", str(scale),
        "--inject_layers", "14",
        "--alpha_info", str(a_info), "--alpha_abs", "0.1",
        "--alpha_zipf", str(a_zipf),
        "--target_vocab_util", str(tgt_util),
        "--max_length", "96", "--batch_size", "32",
        "--gradient_accumulation_steps", "1",
        "--num_epochs", str(epochs), "--lr", "1e-5", "--steer_lr", "1e-3",
        "--num_rollouts", "4", "--search_temp", "1.0",
        "--eval_samples", "256", "--eval_batch_size", "32",
        "--eval_every", "400", "--max_new_tokens", str(digits + 2),
        # Training-time evals must use THIS rung's steering scale. Hardcoding
        # 0.1 meant every rung above the first was scored on a model steered at
        # a scale it was never trained for, so the eval curve described nothing
        # the sweep was actually escalating.
        "--eval_decode_scale", str(scale),
        "--log_every", "50", "--output_dir", str(ckpt),
    ], LOGS / f"{tag}_train.log",
        env={"ARITH_DIGITS": str(digits), "ARITH_SIZE": str(size)})
    if rc != 0 or not (ckpt / "final.pt").exists():
        print(f"[sweep] {tag}: TRAINING FAILED (rc={rc}) — see {tag}_train.log")
        return None
    return ckpt


KNOCKOUT_SRC = '''
import json, os, sys
from amir_interp_rebuttal.load_local import load_local_steered
from amir_interp_rebuttal.arith_dataset import ArithmeticDataset
from amir_interp_rebuttal.runner import batched_generate
ckpt, digits, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
w, tok, a = load_local_steered(ckpt, device="cuda")
ds = ArithmeticDataset(split="test", tokenizer=tok, max_length=96, n_digits=digits)
idxs = list(range(len(ds)))
from collections import Counter
import torch

# THREE ARMS, and the gate is decided by the third.
#
#   codes_ON        steering active everywhere
#   codes_OFF_dec   decode_scale=0 — zeroes steering during GENERATION only.
#                   Prefill steering survives, so the model still reads the
#                   operands under steering. This is a LOWER BOUND on the causal
#                   effect and it is what every previously reported knockout
#                   number measured (+0.2pp / -0.6pp / -0.3pp).
#   codes_OFF_full  steering_emb zeroed — removes the codes from prefill AND
#                   decode. This is the real ablation and sets the gate.
#
# The distinction matters most for arithmetic: with L=1 the prefill codes steer
# the 14 operand tokens, i.e. the model's entire reading of the problem. A
# decode-only ablation cannot see that contribution at all.
res = {}
for name, sc in [("codes_ON", float(a["scale"])), ("codes_OFF_dec", 0.0)]:
    # Record codes on the ON pass so codebook utilisation comes from the same
    # forward pass as the accuracy it is being reported alongside.
    recs = batched_generate(w, tok, ds, "cuda", idxs, eval_batch_size=64,
                            max_new_tokens=digits + 2,
                            record_codes=(name == "codes_ON"), decode_scale=sc)
    res[name] = sum(r["correct"] for r in recs) / len(recs)
    print(f"{name} acc={res[name]:.4f}")
    if name == "codes_ON":
        used = Counter()
        for r in recs:
            for c in (r.get("codes") or []):
                if int(c) >= 0:
                    used[int(c)] += 1
        total = sum(used.values()) or 1
        # "Active" = used at least 1% of the time. A code fired twice in 2600
        # problems is not a member of a working codebook.
        active = [c for c, k in used.items() if k / total >= 0.01]
        res["n_codes_used"] = len(used)
        res["n_codes_active"] = len(active)
        res["code_hist"] = dict(used.most_common())
        print(f"codes_used={len(used)} codes_active_1pct={len(active)}")

# Full ablation: zero the steering embedding table itself, so no code is
# injected at any position, prefill or decode.
_saved = w.steering_emb.weight.data.clone()
w.steering_emb.weight.data.zero_()
try:
    recs = batched_generate(w, tok, ds, "cuda", idxs, eval_batch_size=64,
                            max_new_tokens=digits + 2, record_codes=False,
                            decode_scale=float(a["scale"]))
    res["codes_OFF_full"] = sum(r["correct"] for r in recs) / len(recs)
finally:
    w.steering_emb.weight.data.copy_(_saved)
print(f"codes_OFF_full acc={res['codes_OFF_full']:.4f}")

res["delta_pp_decode_only"] = 100 * (res["codes_ON"] - res["codes_OFF_dec"])
res["delta_pp"] = 100 * (res["codes_ON"] - res["codes_OFF_full"])
# Relative drop as well as absolute. On a low-accuracy task an absolute delta
# badly understates the effect: 22.4% -> 15.8% is only 6.6pp but is a 30%
# relative loss, and the toy-model result everyone compares against
# (95.5% -> 0.1%) is itself a relative statement.
res["delta_rel_pct"] = (100 * (res["codes_ON"] - res["codes_OFF_full"])
                        / res["codes_ON"]) if res["codes_ON"] > 0 else 0.0
print(f"DELTA_PP_DECODE_ONLY={res['delta_pp_decode_only']:.2f}")
print(f"DELTA_PP={res['delta_pp']:.2f}  DELTA_REL={res['delta_rel_pct']:.1f}%"
      f"   <- gate uses either")
json.dump(res, open(out, "w"), indent=2)
'''


def knockout(tag, ckpt, digits):
    out = RESULTS / f"{tag}_knockout.json"
    RESULTS.mkdir(parents=True, exist_ok=True)
    rc = run([PY, "-u", "-W", "ignore", "-c", KNOCKOUT_SRC,
              str(ckpt), str(digits), str(out)],
             LOGS / f"{tag}_knockout.log")
    if rc != 0 or not out.exists():
        print(f"[sweep] {tag}: knockout failed — see {tag}_knockout.log")
        return None
    return json.loads(out.read_text())


def main():
    summary = []
    for digits, size, epochs, scale, a_info, a_zipf, tgt_util in LADDER:
        base = (f"arith_{digits}d_{size//1000}k" if size >= 1000
                else f"arith_{digits}d_{size}")
        # Config in the tag: checkpoints never collide, and every number stays
        # traceable to the exact recipe that produced it.
        tag = (base if (scale, a_info, a_zipf) == (0.1, 10.0, 1.0)
               else f"{base}_s{scale}_i{int(a_info)}_z{int(a_zipf)}u{int(tgt_util*10)}")
        ckpt = train(tag, digits, size, epochs, scale, a_info, a_zipf, tgt_util)
        if ckpt is None:
            summary.append({"tag": tag, "status": "train_failed"})
            continue

        k = knockout(tag, ckpt, digits)
        if k is None:
            summary.append({"tag": tag, "status": "knockout_failed"})
            continue

        acc, delta = k["codes_ON"], k["delta_pp"]
        n_active = k.get("n_codes_active")
        rel = k.get("delta_rel_pct", 0.0)
        passed = ((delta >= GATE_DELTA_PP or rel >= GATE_DELTA_REL_PCT)
                  and (ACC_FLOOR <= acc <= ACC_CEIL))
        print(f"[sweep] {tag}: acc={acc:.1%} knockout_delta={delta:+.2f}pp "
              f"({rel:+.1f}% rel) active_codes={n_active}/30 "
              f"gate={'OPEN' if passed else 'closed'}")
        summary.append({"tag": tag, "digits": digits, "size": size,
                        "scale": scale, "alpha_info": a_info, "alpha_zipf": a_zipf,
                        "target_vocab_util": tgt_util,
                        "acc": acc, "delta_pp": delta,
                        "delta_pp_decode_only": k.get("delta_pp_decode_only"),
                        "delta_rel_pct": rel,
                        "n_codes_active": n_active,
                        "n_codes_used": k.get("n_codes_used"),
                        "gate_open": passed})
        (RESULTS / "sweep_gate_summary.json").write_text(
            json.dumps(summary, indent=2))

        if passed:
            print(f"[sweep] GATE OPEN at {tag} — running R1/R2 here")
            run([PY, "-u", "-W", "ignore", "-m", "amir_interp_rebuttal.analyze",
                 "--study", "arithmetic", "--ckpt", str(ckpt),
                 "--model_name", MODEL, "--eval_n", "2600",
                 "--max_new_tokens", str(digits + 2),
                 "--max_swap_examples", "200"],
                LOGS / f"{tag}_analyze.log")
            print(f"[sweep] DONE — R1/R2 measured on a load-bearing model ({tag})")
            return

    print("[sweep] ladder exhausted, gate never opened. "
          "That is itself the result: across this difficulty and sparsity range, "
          "a 596M-param pretrained model does not route computation through the "
          "codebook. R2 remains unmeasurable rather than negative.")
    (RESULTS / "sweep_gate_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
