"""Overnight n-gram causal ablation script.

For a trained SoRL wrapper, pick the top-K highest-purity N-grams (N in
{2, 3}) among the inner-monologue code sequences, and for each picked
n-gram:
    - ablate every code that participates in the n-gram
      (wrapper.steering_emb[code].zero_()) via `ablate_steering_codes`,
    - run greedy evaluation on the full SciQA val set,
    - record per-sample (idx, topic, gold, pred, correct) along with
      metadata about the ablated n-gram (tuple, focused topic, purity,
      global count, participating codes).

Results are cached per-label to disk so the script resumes cleanly.

Usage::

    python run_ngram_ablation.py \\
        --repo Ksgk-fy/sciqa_ckpt_20260416_1452 \\
        --run  l1_sciqa_v9_C32_detach_az0.5_aa0.5 \\
        --top-k 15 \\
        --n-grams 2 3 \\
        --out-dir log/analysis_out/ablate_ngram

Outputs under ``<out-dir>/<run>/``:
    baseline.json
    ngram_<N>_<rank>__<topic>.json      per picked n-gram
    manifest.json                       index of all runs with meta
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
from collections import Counter, defaultdict

import numpy as np
import torch
from tqdm.auto import tqdm

from data.pt_dataset import ScienceQADataset
from sorl.analyze import ablate_steering_codes, load_steered_model

ANS_RE = re.compile(r"\b([A-D])\b")


def parse_mc(text: str):
    m = ANS_RE.findall(text)
    return m[-1] if m else None


def ngrams_of(seq, n):
    return [tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)]


def pick_top_purity_ngrams(
    samples, codes, val_ds, *,
    n: int,
    top_k: int,
    min_topic_seqs: int = 5,
    min_gram_count_global: int = 10,
    min_purity: float = 0.0,
    src: str = "response",
):
    """Return a list of (ngram, focused_topic, purity, global_count,
    in_topic_count) sorted by purity desc, then count desc. Uses the
    same bookkeeping as purity_sweep_report."""
    topic_by_idx = {s["idx"]: val_ds.dataset[s["idx"]].get("topic", "unknown")
                    for s in samples}
    topic_seqs = defaultdict(list)
    for s, c in zip(samples, codes):
        t = topic_by_idx[s["idx"]]
        seq = []
        if src in ("prompt", "both"):   seq += list(c["prompt"])
        if src in ("response", "both"): seq += list(c["response"])
        if seq:
            topic_seqs[t].append(seq)
    topics = [t for t, seqs in topic_seqs.items() if len(seqs) >= min_topic_seqs]

    global_ct = Counter()
    topic_ct = {t: Counter() for t in topics}
    for t in topics:
        for seq in topic_seqs[t]:
            g = ngrams_of(seq, n)
            topic_ct[t].update(g)
            global_ct.update(g)

    best_topic_of, best_count_of = {}, {}
    for t in topics:
        for g, c_t in topic_ct[t].items():
            if c_t > best_count_of.get(g, -1):
                best_count_of[g] = c_t
                best_topic_of[g] = t

    rows = []
    for g, cg in global_ct.items():
        if cg < min_gram_count_global: continue
        t = best_topic_of[g]
        c_t = best_count_of[g]
        purity = c_t / cg
        if purity < min_purity: continue
        rows.append((g, t, purity, int(cg), int(c_t)))
    rows.sort(key=lambda r: (-r[2], -r[3]))
    return rows[:top_k]


@torch.no_grad()
def eval_full(wrapper, tokenizer, val_ds, device, *,
              eval_indices, gold_by_idx, topic_by_idx_full,
              cache_path: str, max_new_tokens: int, desc: str):
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        if cached.get("n") == len(eval_indices):
            return cached

    rows = []
    correct_total = 0
    attempted = 0
    wrapper.eval()
    for s_idx in tqdm(eval_indices, desc=desc, leave=False):
        item = val_ds[s_idx]
        plen = int(item["prompt_len"])
        ii = item["input_ids"][:plen].unsqueeze(0).to(device)
        am = item["attention_mask"][:plen].unsqueeze(0).to(device)
        out = wrapper.generate(
            input_ids=ii, attention_mask=am,
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        text = tokenizer.decode(out[0, plen:], skip_special_tokens=True)
        pred = parse_mc(text)
        gold = gold_by_idx.get(s_idx)
        ok = int(pred is not None and gold is not None and pred == gold)
        correct_total += ok
        attempted += 1
        rows.append({
            "idx": int(s_idx),
            "topic": topic_by_idx_full[s_idx],
            "gold": gold,
            "pred": pred,
            "correct": ok,
        })

    res = {
        "n": len(eval_indices),
        "attempted": attempted,
        "correct": correct_total,
        "accuracy": correct_total / max(attempted, 1),
        "rows": rows,
    }
    tmp = cache_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, cache_path)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Ksgk-fy/sciqa_ckpt_20260416_1452")
    ap.add_argument("--run",  default="l1_sciqa_v9_C32_detach_az0.5_aa0.5")
    ap.add_argument("--n-grams", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--min-count", type=int, default=10,
                    help="min global count for an n-gram to be eligible")
    ap.add_argument("--min-purity", type=float, default=0.0)
    ap.add_argument("--min-topic-seqs", type=int, default=5)
    ap.add_argument("--src", default="response", choices=["response", "prompt", "both"])
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--eval-n", type=int, default=None,
                    help="truncate eval set (debug). default: full val set")
    ap.add_argument("--out-dir", default="log/analysis_out/ablate_ngram")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  run={args.run}  repo={args.repo}")

    # ---- Load model + codes -------------------------------------------------
    wrapper, tokenizer, wargs = load_steered_model(args.run, args.repo, device)
    code_pt_file = (
        glob.glob(f"*/analysis_out/decode_scale_*/{args.run}_*ps{wargs['scale']}*_respprompt.pt")
        + glob.glob(f"analysis_out/decode_scale_*/{args.run}_*ps{wargs['scale']}*_respprompt.pt")
    )[0]
    print(f"codes: {code_pt_file}")
    blob = torch.load(code_pt_file, map_location="cpu", weights_only=False)
    samples = blob["samples"]
    codes   = blob["codes"]

    val_ds = ScienceQADataset(split="test", tokenizer=tokenizer, max_length=512)

    gold_by_idx = {s["idx"]: s["gold"] for s in samples}
    eval_n = len(val_ds) if args.eval_n is None else min(args.eval_n, len(val_ds))
    eval_indices = list(range(eval_n))
    topic_by_idx_full = {i: val_ds.dataset[i].get("topic", "unknown")
                         for i in range(len(val_ds))}

    out_dir = os.path.join(args.out_dir, args.run)
    os.makedirs(out_dir, exist_ok=True)
    print(f"out_dir: {out_dir}")
    print(f"eval set: {eval_n} samples")

    # ---- Pick n-grams -------------------------------------------------------
    picks = []   # list of dicts
    for N in args.n_grams:
        rows = pick_top_purity_ngrams(
            samples, codes, val_ds,
            n=N, top_k=args.top_k,
            min_topic_seqs=args.min_topic_seqs,
            min_gram_count_global=args.min_count,
            min_purity=args.min_purity,
            src=args.src,
        )
        print(f"\n== top-{args.top_k} {N}-grams by purity  (|rows|={len(rows)}) ==")
        print(f"  {'rank':>4s}  {'ngram':<22s}  {'purity':>7s}  {'#in':>5s}  {'#global':>8s}  topic")
        for r, (g, t, pur, cg, ct) in enumerate(rows):
            print(f"  {r:>4d}  {str(g):<22s}  {pur*100:>6.1f}%  {ct:>5d}  {cg:>8d}  {t}")
        for r, (g, t, pur, cg, ct) in enumerate(rows):
            picks.append({
                "N": N, "rank": r, "ngram": list(g), "topic": t,
                "purity": float(pur), "in_topic_count": ct, "global_count": cg,
                "codes": sorted({int(x) for x in g}),
            })

    # ---- Baseline -----------------------------------------------------------
    print("\n== baseline ==")
    t0 = time.time()
    base_res = eval_full(
        wrapper, tokenizer, val_ds, device,
        eval_indices=eval_indices,
        gold_by_idx=gold_by_idx,
        topic_by_idx_full=topic_by_idx_full,
        cache_path=os.path.join(out_dir, "baseline.json"),
        max_new_tokens=args.max_new_tokens,
        desc="baseline",
    )
    print(f"  acc={base_res['accuracy']*100:.2f}%  ({time.time()-t0:.0f}s)")

    # ---- Per-n-gram ablations ----------------------------------------------
    manifest = {
        "run": args.run,
        "repo": args.repo,
        "eval_n": eval_n,
        "max_new_tokens": args.max_new_tokens,
        "baseline_acc": base_res["accuracy"],
        "picks": [],
    }

    for pk in picks:
        N, rank, t = pk["N"], pk["rank"], pk["topic"]
        safe_t = re.sub(r"\W+", "_", t)[:40]
        label = f"ngram_N{N}_r{rank:02d}__{safe_t}"
        cache_path = os.path.join(out_dir, label + ".json")
        desc = f"N={N} r={rank:02d} {t[:18]} g={pk['ngram']}"
        tt0 = time.time()
        with ablate_steering_codes(wrapper, pk["codes"]):
            res = eval_full(
                wrapper, tokenizer, val_ds, device,
                eval_indices=eval_indices,
                gold_by_idx=gold_by_idx,
                topic_by_idx_full=topic_by_idx_full,
                cache_path=cache_path,
                max_new_tokens=args.max_new_tokens,
                desc=desc,
            )
        # Tag rows with the ablation metadata (stored in manifest; rows stay lean)
        dacc = (res["accuracy"] - base_res["accuracy"]) * 100
        print(f"  {label:<58s} acc={res['accuracy']*100:.2f}% "
              f"(Δ={dacc:+.2f}pp)  ({time.time()-tt0:.0f}s)")
        manifest["picks"].append({
            **pk,
            "label": label,
            "accuracy": res["accuracy"],
            "delta_acc_pp": dacc,
        })
        # Write manifest each iteration so we always have a partial index.
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

    print(f"\ndone. manifest → {os.path.join(out_dir, 'manifest.json')}")


if __name__ == "__main__":
    main()
