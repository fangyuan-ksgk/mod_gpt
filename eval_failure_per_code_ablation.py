"""Per-code ablation sweep restricted to *failure cases* of fully-steered mode.

Pipeline
--------
1. Fully-steered baseline over the first N samples (batched). Accepts
   `--baseline-jsonl` to skip recomputation.
2. Collect failures (correct == 0). Optional cap via `--max-failures`.
3. For each code c in `--codes` (default: 0..C-1), run `--n-runs` random
   ablations of the unigram (c,) on the failure set with different seeds.
4. Stream per-(sample, code, run) records to JSONL; emit a leaderboard by
   average fix-rate across the random runs.

Usage
-----
    python eval_failure_per_code_ablation.py \
        --run q17_sciqa_v9_C32_detach_az0.1_aa0.5 \
        --baseline-jsonl analysis_out/steering_modes/q17_..._steered.jsonl \
        --num-samples 2000 --batch-size 16 --n-runs 2 \
        --out-dir analysis_out/failure_abl/q17
"""
from __future__ import annotations

import argparse
import json
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
                         "(first-N in sample_idx order).")
    ap.add_argument("--n-runs", type=int, default=2,
                    help="Number of random ablation runs per code "
                         "(different seeds).")
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
    else:
        base_path = out_dir / "baseline_steered.jsonl"
        failures, base_acc = _run_steered_baseline(
            wrapper, tokenizer, val_ds, s_idxs, golds, metas, args, device,
            base_path)

    fail_idxs = sorted(failures.keys())
    if args.max_failures and len(fail_idxs) > args.max_failures:
        fail_idxs = fail_idxs[:args.max_failures]
        print(f"[failures] capped to first {args.max_failures}")
    N_RUNS = max(1, int(args.n_runs))
    print(f"[sweep] sweeping {len(codes)} codes × {len(fail_idxs)} failures "
          f"× {N_RUNS} runs = {len(codes) * len(fail_idxs) * N_RUNS} generations")

    # ---- 2. per-code ablation sweep over failures ----
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    decode_scale = float(wrapper.scale)
    extract = ScienceQADataset.extract_answer

    B = max(1, args.batch_size)
    chunks = [fail_idxs[i:i + B] for i in range(0, len(fail_idxs), B)]

    abl_path = out_dir / "failure_ablations.jsonl"
    # per-code aggregates (summed across all runs)
    agg = {c: {"fixed": 0, "still_wrong": 0, "n": 0} for c in codes}
    # per-(topic, code) aggregates for topic-level view
    topic_agg = defaultdict(lambda: {c: {"fixed": 0, "n": 0} for c in codes})

    t0 = time.time()
    with abl_path.open("w") as fh:
        for c in tqdm(codes, desc="codes"):
            for r in range(N_RUNS):
                seed_r = int(args.seed) + int(c) * 1000 + r
                for ch in tqdm(chunks, desc=f"  code={c} run={r}", leave=False):
                    bii, bam, T = _left_pad_batch(val_ds, ch, pad_id, device)
                    with ablate_router_ngrams(
                        wrapper, [(int(c),)], seed=seed_r
                    ):
                        texts = _gen_batch(wrapper, tokenizer, bii, bam, T,
                                           decode_scale, args.max_new_tokens)
                    for si, txt in zip(ch, texts):
                        pred = extract(txt)
                        gd = golds[si]
                        ok = int(pred is not None and gd is not None and pred == gd)
                        rec = {
                            "sample_idx": int(si), "code": int(c),
                            "run": r, "seed": seed_r,
                            "gold": gd, "pred": pred, "correct": ok,
                            "baseline_pred": failures[si].get("pred"),
                            "text": txt,
                        }
                        rec.update(metas[si])
                        fh.write(json.dumps(rec) + "\n")

                        a = agg[c]
                        a["n"] += 1
                        a["fixed" if ok else "still_wrong"] += 1
                        tp = metas[si].get("topic", "unknown")
                        topic_agg[tp][c]["n"] += 1
                        topic_agg[tp][c]["fixed"] += ok
                fh.flush()
    print(f"[sweep] wall={time.time() - t0:.0f}s  ->  {abl_path}")

    # ---- 3. leaderboard + summary ----
    ranked = sorted(
        agg.items(),
        key=lambda kv: -kv[1]["fixed"],
    )
    print(f"\n=== per-code leaderboard on failure set (N_fail={len(fail_idxs)},"
          f" runs={N_RUNS}, baseline_acc={base_acc*100:.2f}%) ===")
    print(f"{'code':>5s}  {'fixed':>6s}  {'fix%':>6s}  {'n':>6s}")
    for c, v in ranked:
        fx_pct = v["fixed"] / max(v["n"], 1) * 100
        print(f"{c:>5d}  {v['fixed']:>6d}  {fx_pct:>5.1f}%  {v['n']:>6d}")

    summary = {
        "run": args.run, "repo": args.repo,
        "C_SIZE": C, "num_samples": N,
        "baseline_acc": base_acc,
        "n_failures": len(fail_idxs),
        "codes_swept": codes,
        "n_runs": N_RUNS,
        "by_code": {
            c: {
                "fixed": v["fixed"],
                "still_wrong": v["still_wrong"],
                "n": v["n"],
                "fix_rate": v["fixed"] / max(v["n"], 1),
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
