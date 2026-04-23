"""Ablate high-purity n-grams on every sequence they appear in.

For each high-purity n-gram harvested by `purity_sweep_report`:
  1. find every sequence in which it appears (via `find_ngram_occurrences`)
  2. per sequence, run:
       - 1 baseline decode (no ablation) at decode_scale
       - K random-swap ablations (different seeds) at decode_scale
  3. parse the MC answer (A/B/C/D) from each decode and compare to gold
  4. stream results to JSONL + print per-pattern accuracy summary

Usage:
    python scripts/ablate_pure_ngrams_on_occurrences.py \
        --run l1_sciqa_v9_C32_detach_az0.5_aa0.5 \
        --repo Ksgk-fy/sciqa_ckpt_20260416_1452 \
        --decode-scale 0.3 --n-random 5
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
from tqdm.auto import tqdm

from data.pt_dataset import ScienceQADataset
from sorl.analyze import (
    ablate_router_ngrams,
    find_ngram_occurrences,
    load_steered_model,
    purity_sweep_report,
)


ANS_RE = re.compile(r"\b([A-D])\b")


def parse_mc(text: str) -> str | None:
    m = ANS_RE.findall(text)
    return m[-1] if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="l1_sciqa_v9_C32_detach_az0.5_aa0.5")
    ap.add_argument("--repo", default="Ksgk-fy/sciqa_ckpt_20260416_1452")
    ap.add_argument("--decode-scale", type=float, default=0.3)
    ap.add_argument("--n-random", type=int, default=5,
                    help="Number of random-swap seeds per sequence.")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--harvest-purity", type=float, default=0.9)
    ap.add_argument("--harvest-min-count", type=int, default=10)
    ap.add_argument("--harvest-N", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--src", default="response", choices=["prompt", "response", "both"])
    ap.add_argument("--max-seqs-per-pattern", type=int, default=None,
                    help="Cap sequences per pattern for quick runs (default: all).")
    ap.add_argument("--max-patterns", type=int, default=None)
    ap.add_argument("--out", default=None,
                    help="Output JSONL path (default: analysis_out/ablate_occurrences/<run>.jsonl)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------- load wrapper + codes + val dataset --------
    wrapper, tokenizer, wargs = load_steered_model(args.run, args.repo, device)
    wrapper.eval()

    code_pt_candidates = (
        glob.glob(f"*/analysis_out/decode_scale_*/{args.run}_*ps{wargs['scale']}*_respprompt.pt")
        + glob.glob(f"analysis_out/decode_scale_*/{args.run}_*ps{wargs['scale']}*_respprompt.pt")
    )
    if not code_pt_candidates:
        raise FileNotFoundError(f"No cached codes .pt found for {args.run}")
    blob = torch.load(code_pt_candidates[0], map_location="cpu", weights_only=False)
    samples = blob["samples"]
    codes = blob["codes"]

    val_ds = ScienceQADataset(split="test", tokenizer=tokenizer, max_length=512)
    gold_by_idx = {s["idx"]: s["gold"] for s in samples}

    # -------- purity report -> harvest high-purity n-grams --------
    purity_report = purity_sweep_report(
        samples, codes, val_ds,
        src=args.src,
        n_grams=tuple(range(1, max(args.harvest_N) + 1)),
        top_k=8,
        min_topic_seqs=5,
        min_gram_count_in_topic=5,
        min_gram_count_global=30,
        purity_thresholds=(0.5, 0.75, 0.9, 1.0),
        harvest_purity=args.harvest_purity,
        harvest_min_count=args.harvest_min_count,
        harvest_N=tuple(args.harvest_N),
        run_label=blob.get("run"),
        accuracy=blob.get("accuracy"),
        verbose=True,
        plot=False,
    )

    # flatten harvested (pattern -> topic, purity, count)
    per_N = purity_report["per_N"]
    flat = []
    for t, ngset in purity_report["topic_ngrams"].items():
        for g in ngset:
            N = len(g)
            if N not in per_N:
                continue
            global_ct, _, _, best_count = per_N[N]
            cg = global_ct.get(g, 0)
            if cg == 0:
                continue
            flat.append({
                "pattern": tuple(int(x) for x in g),
                "N": N,
                "topic": t,
                "purity": float(best_count[g] / cg),
                "count": int(cg),
            })
    flat.sort(key=lambda r: (-r["purity"], -r["count"]))
    if args.max_patterns:
        flat = flat[: args.max_patterns]
    print(f"\n[ablate-occ] harvested {len(flat)} high-purity patterns")

    # -------- output path --------
    out_path = Path(args.out) if args.out else Path(
        f"analysis_out/ablate_occurrences/{args.run}_ds{args.decode_scale}_k{args.n_random}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[ablate-occ] writing to {out_path}")

    gen_kw_common = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

    @torch.no_grad()
    def _decode(item) -> tuple[str, str | None]:
        plen = int(item["prompt_len"])
        ii = item["input_ids"][:plen].unsqueeze(0).to(device)
        am = item["attention_mask"][:plen].unsqueeze(0).to(device)
        out = wrapper.generate(
            input_ids=ii, attention_mask=am,
            decode_scale=args.decode_scale,
            **gen_kw_common,
        )
        text = tokenizer.decode(out[0, plen:], skip_special_tokens=True)
        return text, parse_mc(text)

    # -------- per-pattern: iterate its sequences, run baseline + K ablations --------
    per_pattern_summary = []
    n_records = 0
    t0 = time.time()

    with out_path.open("w") as fh:
        for pi, pmeta in enumerate(flat):
            pat = pmeta["pattern"]
            # sequences containing this pattern
            occ = find_ngram_occurrences(samples, codes, pat, src=args.src)
            seq_list_idxs = sorted({i for i, _ in occ})
            if args.max_seqs_per_pattern:
                seq_list_idxs = seq_list_idxs[: args.max_seqs_per_pattern]
            if not seq_list_idxs:
                continue

            print(f"\n[{pi+1}/{len(flat)}] pattern={pat} topic={pmeta['topic']} "
                  f"purity={pmeta['purity']:.2f} count={pmeta['count']} "
                  f"n_seqs={len(seq_list_idxs)}")

            base_c = base_tot = 0
            abl_c = abl_tot = 0

            for list_idx in tqdm(seq_list_idxs, desc="  seqs", leave=False):
                s = samples[list_idx]
                s_idx = int(s["idx"])
                gold = gold_by_idx.get(s_idx)
                item = val_ds[s_idx]

                # baseline (no ablation), at the same decode_scale
                base_text, base_pred = _decode(item)
                base_correct = int(base_pred is not None and gold is not None
                                   and base_pred == gold)
                base_c += base_correct
                base_tot += 1
                fh.write(json.dumps({
                    "pattern": list(pat), "N": pmeta["N"],
                    "pattern_topic": pmeta["topic"],
                    "pattern_purity": pmeta["purity"],
                    "pattern_count": pmeta["count"],
                    "decode_scale": args.decode_scale,
                    "sample_idx": s_idx,
                    "gold": gold,
                    "mode": "baseline",
                    "seed": None,
                    "pred": base_pred,
                    "correct": base_correct,
                    "text": base_text,
                }) + "\n")
                n_records += 1

                # K random-swap ablations
                for seed in range(args.n_random):
                    with ablate_router_ngrams(wrapper, [pat], seed=seed) as tr:
                        abl_text, abl_pred = _decode(item)
                    abl_correct = int(abl_pred is not None and gold is not None
                                      and abl_pred == gold)
                    abl_c += abl_correct
                    abl_tot += 1
                    fh.write(json.dumps({
                        "pattern": list(pat), "N": pmeta["N"],
                        "pattern_topic": pmeta["topic"],
                        "pattern_purity": pmeta["purity"],
                        "pattern_count": pmeta["count"],
                        "decode_scale": args.decode_scale,
                        "sample_idx": s_idx,
                        "gold": gold,
                        "mode": "ablate_random",
                        "seed": seed,
                        "pred": abl_pred,
                        "correct": abl_correct,
                        "n_hits": len(tr.hits),
                        "hits": tr.hits,
                        "text": abl_text,
                    }) + "\n")
                    n_records += 1
                fh.flush()

            base_acc = base_c / max(base_tot, 1)
            abl_acc = abl_c / max(abl_tot, 1)
            per_pattern_summary.append({
                **pmeta,
                "n_seqs": len(seq_list_idxs),
                "base_acc": base_acc,
                "abl_acc": abl_acc,
                "delta": abl_acc - base_acc,
            })
            print(f"    baseline acc={base_acc*100:5.1f}%  "
                  f"ablate acc={abl_acc*100:5.1f}%  "
                  f"Δ={(abl_acc-base_acc)*100:+5.1f}pp  "
                  f"(n_seqs={len(seq_list_idxs)}, k={args.n_random})")

    dt = time.time() - t0
    print(f"\n[ablate-occ] wrote {n_records} records in {dt:.0f}s to {out_path}")

    # -------- aggregate summary --------
    print("\n" + "=" * 90)
    print(f"  Per-pattern accuracy under random ablation  (decode_scale={args.decode_scale})")
    print("=" * 90)
    print(f"  {'pattern':<22s} {'N':>2s} {'topic':<22s} {'purity':>6s} "
          f"{'#seq':>5s} {'base%':>6s} {'abl%':>6s} {'Δpp':>7s}")
    print("  " + "-" * 86)
    for row in sorted(per_pattern_summary, key=lambda r: r["delta"]):
        print(f"  {str(row['pattern']):<22s} {row['N']:>2d} "
              f"{row['topic'][:22]:<22s} {row['purity']*100:>5.1f}% "
              f"{row['n_seqs']:>5d} {row['base_acc']*100:>5.1f} "
              f"{row['abl_acc']*100:>5.1f} {(row['delta'])*100:>+6.1f}")

    # also dump the summary as JSON next to the JSONL
    summary_path = out_path.with_suffix(".summary.json")
    with summary_path.open("w") as f:
        json.dump({
            "run": args.run,
            "decode_scale": args.decode_scale,
            "n_random": args.n_random,
            "per_pattern": per_pattern_summary,
        }, f, indent=2)
    print(f"[ablate-occ] summary -> {summary_path}")


if __name__ == "__main__":
    main()
