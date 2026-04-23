"""Per-code ablation sweep restricted to *failure cases* of fully-steered mode.

Motivation
----------
Fully-steered decode sometimes collapses into degenerate repetitive loops
(e.g. codes 5/15 on Qwen3-0.6B) that tank accuracy on specific samples.
Rather than sweep ablations over the full eval set (expensive and noisy),
we first identify which samples the steered run gets *wrong*, then for
each such sample try ablating each code c in {0..C-1} individually and
measure:

    - does the prediction flip to correct?
    - does the repetition-loop pattern go away?

This isolates "bad codes" whose removal rescues generation without
requiring per-sample n-gram discovery.

Pipeline
--------
1. Load (or compute) a steered baseline for RUN over the first N samples.
   Accepts `--baseline-jsonl` to skip recomputation.
2. Select failures (correct == 0). Optionally cap with `--max-failures`.
3. For each code c in `--codes` (default: 0..C-1), run steered generation
   on the failure set with `ablate_router_ngrams([(c,)], seed=SEED+c)`.
4. Stream per-(sample, code) records to JSONL; emit a leaderboard.

Usage
-----
    python eval_failure_per_code_ablation.py \
        --run q17_sciqa_v9_C32_detach_az0.1_aa0.5 \
        --baseline-jsonl analysis_out/steering_modes/q17_..._steered.jsonl \
        --num-samples 2000 --batch-size 16 \
        --out-dir analysis_out/failure_abl/q17

    # or let it recompute the steered baseline from scratch:
    python eval_failure_per_code_ablation.py \
        --run q17_sciqa_v9_C32_detach_az0.1_aa0.5 \
        --num-samples 2000 --batch-size 16
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
from tqdm.auto import tqdm

from data.pt_dataset import ScienceQADataset
from sorl.analyze import ablate_router_ngrams, load_steered_model


_META_KEYS = ("topic", "subject", "category", "skill")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _left_pad_batch(val_ds, s_idxs, pad_id, device):
    iis, ams, plens = [], [], []
    for si in s_idxs:
        it = val_ds[si]
        p = int(it["prompt_len"])
        iis.append(it["input_ids"][:p])
        ams.append(it["attention_mask"][:p])
        plens.append(p)
    T = max(plens)
    bii = torch.full((len(iis), T), pad_id, dtype=iis[0].dtype)
    bam = torch.zeros((len(iis), T), dtype=ams[0].dtype)
    for b, (ii, am, p) in enumerate(zip(iis, ams, plens)):
        bii[b, T - p:] = ii
        bam[b, T - p:] = am
    return bii.to(device), bam.to(device), T


@torch.no_grad()
def _gen_batch(wrapper, tokenizer, bii, bam, T, decode_scale, max_new):
    out = wrapper.generate(
        input_ids=bii, attention_mask=bam,
        max_new_tokens=max_new, do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        decode_scale=decode_scale,
    )
    gen = out[:, T:]
    return [tokenizer.decode(row, skip_special_tokens=True) for row in gen]


def _repetition_score(text: str) -> float:
    """Heuristic degenerate-loop detector in [0, 1].

    Uses the fraction of *duplicate* 4-grams (token-level on whitespace-split
    words) in the generated text. 0.0 = no repetition, 1.0 = every 4-gram
    seen before. Correlates well with visible ``#### A / #### B / ...`` loops.
    """
    toks = re.findall(r"\S+", text)
    if len(toks) < 8:
        return 0.0
    grams = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    if not grams:
        return 0.0
    uniq = len(set(grams))
    return 1.0 - uniq / len(grams)


def _load_baseline_failures(path: Path, s_idxs):
    """Return {sample_idx: record} for correct==0 entries restricted to s_idxs."""
    keep = set(int(i) for i in s_idxs)
    failures = {}
    with path.open() as fh:
        for line in fh:
            rec = json.loads(line)
            si = int(rec["sample_idx"])
            if si not in keep:
                continue
            if int(rec.get("correct", 0)) == 0:
                failures[si] = rec
    return failures


def _run_steered_baseline(wrapper, tokenizer, val_ds, s_idxs, golds, metas,
                          args, device, out_path: Path):
    """Run fully-steered decode once, stream to JSONL, return list of failure idxs."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    decode_scale = float(wrapper.scale)
    extract = ScienceQADataset.extract_answer

    B = max(1, args.batch_size)
    chunks = [s_idxs[i:i + B] for i in range(0, len(s_idxs), B)]

    failures = {}
    correct = 0
    with out_path.open("w") as fh:
        for ch in tqdm(chunks, desc="baseline(steered)"):
            bii, bam, T = _left_pad_batch(val_ds, ch, pad_id, device)
            texts = _gen_batch(wrapper, tokenizer, bii, bam, T,
                               decode_scale, args.max_new_tokens)
            for si, txt in zip(ch, texts):
                pred = extract(txt)
                gd = golds[si]
                ok = int(pred is not None and gd is not None and pred == gd)
                correct += ok
                rec = {
                    "sample_idx": int(si), "gold": gd, "mode": "steered",
                    "pred": pred, "correct": ok,
                    "rep_score": _repetition_score(txt),
                    "text": txt,
                }
                rec.update(metas[si])
                fh.write(json.dumps(rec) + "\n")
                if ok == 0:
                    failures[si] = rec
    acc = correct / max(len(s_idxs), 1)
    print(f"[baseline] steered acc = {acc*100:.2f}%  "
          f"({correct}/{len(s_idxs)})  #failures={len(failures)}")
    return failures, acc


# ---------------------------------------------------------------------------
# main sweep
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Ksgk-fy/sciqa_ckpt_20260416_0942")
    ap.add_argument("--run", required=True)
    ap.add_argument("--baseline-jsonl", default=None,
                    help="Existing steered JSONL to reuse for failure selection. "
                         "If omitted, a fresh steered baseline is computed.")
    ap.add_argument("--num-samples", type=int, default=2000,
                    help="Restrict to first N test samples.")
    ap.add_argument("--max-failures", type=int, default=0,
                    help="If >0, cap the failure set at this many samples "
                         "(prioritising highest repetition score).")
    ap.add_argument("--codes", nargs="*", type=int, default=None,
                    help="Subset of codes to sweep (default: 0..C-1).")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="analysis_out/failure_per_code")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir) / args.run
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load model + data ----
    wrapper, tokenizer, wargs = load_steered_model(args.run, args.repo, device)
    wrapper.eval()
    C = int(wargs["C_SIZE"])
    codes = list(args.codes) if args.codes else list(range(C))

    val_ds = ScienceQADataset(split="test", tokenizer=tokenizer, max_length=512)
    N = min(args.num_samples, len(val_ds))
    s_idxs = list(range(N))
    golds = [chr(ord("A") + int(val_ds.dataset[i]["answer"])) for i in s_idxs]
    metas = {si: {k: val_ds.dataset[si].get(k) for k in _META_KEYS}
             for si in s_idxs}

    # ---- 1. failure set ----
    if args.baseline_jsonl:
        failures = _load_baseline_failures(Path(args.baseline_jsonl), s_idxs)
        base_acc = 1.0 - len(failures) / max(len(s_idxs), 1)
        print(f"[baseline] loaded {args.baseline_jsonl}  "
              f"acc≈{base_acc*100:.2f}%  #failures={len(failures)}")
        # Fill missing rep_score lazily.
        for r in failures.values():
            if "rep_score" not in r:
                r["rep_score"] = _repetition_score(r.get("text", ""))
    else:
        base_path = out_dir / "baseline_steered.jsonl"
        failures, base_acc = _run_steered_baseline(
            wrapper, tokenizer, val_ds, s_idxs, golds, metas, args, device,
            base_path)

    fail_idxs = sorted(failures.keys())
    if args.max_failures and len(fail_idxs) > args.max_failures:
        fail_idxs = sorted(
            fail_idxs,
            key=lambda i: -failures[i].get("rep_score", 0.0),
        )[:args.max_failures]
        print(f"[failures] capped to top-{args.max_failures} by rep_score")
    print(f"[sweep] sweeping {len(codes)} codes × {len(fail_idxs)} failures "
          f"= {len(codes) * len(fail_idxs)} generations")

    # ---- 2. per-code ablation sweep over failures ----
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    decode_scale = float(wrapper.scale)
    extract = ScienceQADataset.extract_answer

    B = max(1, args.batch_size)
    chunks = [fail_idxs[i:i + B] for i in range(0, len(fail_idxs), B)]

    abl_path = out_dir / "failure_ablations.jsonl"
    # per-code aggregates
    agg = {c: {"fixed": 0, "still_wrong": 0,
               "rep_drop_sum": 0.0, "rep_drop_count": 0,
               "n": 0}
           for c in codes}
    # per-(topic, code) aggregates for topic-level view
    topic_agg = defaultdict(lambda: {c: {"fixed": 0, "n": 0} for c in codes})

    t0 = time.time()
    with abl_path.open("w") as fh:
        for c in tqdm(codes, desc="codes"):
            for ch in tqdm(chunks, desc=f"  code={c}", leave=False):
                bii, bam, T = _left_pad_batch(val_ds, ch, pad_id, device)
                with ablate_router_ngrams(
                    wrapper, [(int(c),)], seed=int(args.seed) + int(c)
                ):
                    texts = _gen_batch(wrapper, tokenizer, bii, bam, T,
                                       decode_scale, args.max_new_tokens)
                for si, txt in zip(ch, texts):
                    pred = extract(txt)
                    gd = golds[si]
                    ok = int(pred is not None and gd is not None and pred == gd)
                    base_rep = failures[si].get("rep_score", 0.0)
                    new_rep = _repetition_score(txt)
                    rec = {
                        "sample_idx": int(si), "code": int(c), "gold": gd,
                        "pred": pred, "correct": ok,
                        "baseline_pred": failures[si].get("pred"),
                        "baseline_rep": base_rep,
                        "rep_score": new_rep,
                        "rep_drop": base_rep - new_rep,
                        "text": txt,
                    }
                    rec.update(metas[si])
                    fh.write(json.dumps(rec) + "\n")

                    a = agg[c]
                    a["n"] += 1
                    a["fixed" if ok else "still_wrong"] += 1
                    a["rep_drop_sum"] += (base_rep - new_rep)
                    a["rep_drop_count"] += 1
                    tp = metas[si].get("topic", "unknown")
                    topic_agg[tp][c]["n"] += 1
                    topic_agg[tp][c]["fixed"] += ok
            fh.flush()
    print(f"[sweep] wall={time.time() - t0:.0f}s  ->  {abl_path}")

    # ---- 3. leaderboard + summary ----
    ranked = sorted(
        agg.items(),
        key=lambda kv: (-kv[1]["fixed"],
                        -(kv[1]["rep_drop_sum"] / max(kv[1]["rep_drop_count"], 1))),
    )
    print(f"\n=== per-code leaderboard on failure set (N_fail={len(fail_idxs)},"
          f" baseline_acc={base_acc*100:.2f}%) ===")
    print(f"{'code':>5s}  {'fixed':>5s}  {'fix%':>6s}  "
          f"{'mean_Δrep':>10s}  {'n':>5s}")
    for c, v in ranked:
        fx_pct = v["fixed"] / max(v["n"], 1) * 100
        md = v["rep_drop_sum"] / max(v["rep_drop_count"], 1)
        print(f"{c:>5d}  {v['fixed']:>5d}  {fx_pct:>5.1f}%  "
              f"{md:>+10.3f}  {v['n']:>5d}")

    summary = {
        "run": args.run, "repo": args.repo,
        "C_SIZE": C, "num_samples": N,
        "baseline_acc": base_acc,
        "n_failures": len(fail_idxs),
        "codes_swept": codes,
        "by_code": {
            c: {
                "fixed": v["fixed"],
                "still_wrong": v["still_wrong"],
                "n": v["n"],
                "fix_rate": v["fixed"] / max(v["n"], 1),
                "mean_rep_drop":
                    v["rep_drop_sum"] / max(v["rep_drop_count"], 1),
            }
            for c, v in agg.items()
        },
        "by_topic": {
            tp: {
                "codes": {
                    c: {"fixed": d["fixed"], "n": d["n"]}
                    for c, d in cd.items() if d["n"] > 0
                }
            }
            for tp, cd in topic_agg.items()
        },
    }
    sp = out_dir / "summary.json"
    with sp.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved: {out_dir}")


if __name__ == "__main__":
    main()
