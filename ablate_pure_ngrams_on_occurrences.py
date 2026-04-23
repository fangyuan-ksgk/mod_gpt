"""Ablate high-purity n-grams on every sequence they appear in.

For each high-purity n-gram harvested by `purity_sweep_report`:
  1. find every sequence in which it appears (via `find_ngram_occurrences`)
  2. per sequence, run:
       - 1 baseline decode (no ablation) at decode_scale
       - K random-swap ablations (different seeds) at decode_scale
  3. parse the MC answer (A/B/C/D) from each decode and compare to gold
  4. stream results to JSONL + print per-pattern accuracy summary

Usage:
python ablate_pure_ngrams_on_occurrences.py \
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
    ap.add_argument("--per-N-top-k", type=int, default=10,
                    help="For each N in --harvest-N, keep the top-K patterns by "
                         "descending purity (tie-break: descending count).")
    ap.add_argument("--min-count", type=int, default=5,
                    help="Global-count floor applied before ranking (per N).")
    ap.add_argument("--harvest-N", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--src", default="response", choices=["prompt", "response", "both"])
    ap.add_argument("--max-seqs-per-pattern", type=int, default=None,
                    help="Cap sequences per pattern for quick runs (default: all).")
    ap.add_argument("--max-patterns", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Batch sequences together when decoding.")
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

    # -------- purity report -> top-K per N by descending purity --------
    purity_report = purity_sweep_report(
        samples, codes, val_ds,
        src=args.src,
        n_grams=tuple(args.harvest_N),
        top_k=8,
        min_topic_seqs=5,
        min_gram_count_in_topic=5,
        min_gram_count_global=30,
        purity_thresholds=(0.5, 0.75, 0.9, 1.0),
        # harvest_* only affect topic_ngrams/topic_codes (unused here); leave loose.
        harvest_purity=0.0, harvest_min_count=1,
        harvest_N=tuple(args.harvest_N),
        run_label=blob.get("run"), accuracy=blob.get("accuracy"),
        verbose=True, plot=False,
    )
    per_N = purity_report["per_N"]

    # Per N, rank every gram with count >= min_count by (-purity, -count), keep top-K.
    flat = []
    for N in sorted(args.harvest_N):
        if N not in per_N:
            continue
        global_ct, _, best_topic, best_count = per_N[N]
        ranked = sorted(
            [(g, best_topic[g], best_count[g] / cg, cg)
             for g, cg in global_ct.items() if cg >= args.min_count],
            key=lambda r: (-r[2], -r[3]),
        )[: args.per_N_top_k]
        for g, t, p, c in ranked:
            flat.append({
                "pattern": tuple(int(x) for x in g),
                "N": N, "topic": t,
                "purity": float(p), "count": int(c),
            })
    if args.max_patterns:
        flat = flat[: args.max_patterns]
    print(f"\n[ablate-occ] selected {len(flat)} patterns "
          f"({args.per_N_top_k}/N across N={args.harvest_N})")
    for r in flat:
        print(f"  N={r['N']}  {str(r['pattern']):<18s}  "
              f"topic={r['topic'][:20]:<20s}  "
              f"purity={r['purity']*100:5.1f}%  count={r['count']}")

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
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def _left_pad_batch(list_idxs):
        """Build left-padded (B, T) input_ids/attention_mask for a list of val_ds idxs."""
        iis, ams, plens = [], [], []
        for s_idx in list_idxs:
            it = val_ds[s_idx]
            p = int(it["prompt_len"])
            iis.append(it["input_ids"][:p])
            ams.append(it["attention_mask"][:p])
            plens.append(p)
        T = max(plens)
        B = len(iis)
        bii = torch.full((B, T), pad_id, dtype=iis[0].dtype)
        bam = torch.zeros((B, T), dtype=ams[0].dtype)
        for b, (ii, am, p) in enumerate(zip(iis, ams, plens)):
            bii[b, T - p:] = ii
            bam[b, T - p:] = am
        return bii.to(device), bam.to(device), T, plens

    @torch.no_grad()
    def _decode_batch(list_idxs):
        """Return list[(text, pred)] for each sample in list_idxs (order preserved)."""
        bii, bam, T, _ = _left_pad_batch(list_idxs)
        out = wrapper.generate(
            input_ids=bii, attention_mask=bam,
            decode_scale=args.decode_scale,
            **gen_kw_common,
        )
        gen = out[:, T:]  # generated tail only (left-padded prompts all end at T)
        results = []
        for row in gen:
            text = tokenizer.decode(row, skip_special_tokens=True)
            results.append((text, parse_mc(text)))
        return results

    def _write_record(fh, pmeta, s_idx, gold, mode, seed, pred, correct, text,
                      n_hits=None):
        rec = {
            "pattern": list(pmeta["pattern"]), "N": pmeta["N"],
            "pattern_topic": pmeta["topic"],
            "pattern_purity": pmeta["purity"],
            "pattern_count": pmeta["count"],
            "decode_scale": args.decode_scale,
            "sample_idx": s_idx, "gold": gold,
            "mode": mode, "seed": seed,
            "pred": pred, "correct": correct,
            "text": text,
        }
        if n_hits is not None:
            rec["n_hits"] = n_hits
        fh.write(json.dumps(rec) + "\n")

    # -------- per-pattern: batched baseline + K batched ablations --------
    per_pattern_summary = []
    n_records = 0
    t0 = time.time()
    B = max(1, args.batch_size)

    with out_path.open("w") as fh:
        for pi, pmeta in enumerate(flat):
            pat = pmeta["pattern"]
            occ = find_ngram_occurrences(samples, codes, pat, src=args.src)
            seq_list_idxs = sorted({i for i, _ in occ})
            if args.max_seqs_per_pattern:
                seq_list_idxs = seq_list_idxs[: args.max_seqs_per_pattern]
            if not seq_list_idxs:
                continue

            s_idxs = [int(samples[i]["idx"]) for i in seq_list_idxs]
            golds = [gold_by_idx.get(si) for si in s_idxs]
            chunks = [s_idxs[i:i + B] for i in range(0, len(s_idxs), B)]
            gold_chunks = [golds[i:i + B] for i in range(0, len(golds), B)]

            print(f"\n[{pi+1}/{len(flat)}] pattern={pat} N={pmeta['N']} "
                  f"topic={pmeta['topic']} purity={pmeta['purity']:.2f} "
                  f"count={pmeta['count']} n_seqs={len(s_idxs)}")

            base_c = base_tot = 0
            abl_c = abl_tot = 0

            # ---- baseline (batched) ----
            for ch_idxs, ch_golds in tqdm(list(zip(chunks, gold_chunks)),
                                          desc="  baseline", leave=False):
                res = _decode_batch(ch_idxs)
                for si, gd, (txt, pred) in zip(ch_idxs, ch_golds, res):
                    ok = int(pred is not None and gd is not None and pred == gd)
                    base_c += ok; base_tot += 1
                    _write_record(fh, pmeta, si, gd, "baseline", None, pred, ok, txt)
                    n_records += 1
                fh.flush()

            # ---- K random-swap ablations (batched per chunk) ----
            for seed in range(args.n_random):
                for ch_idxs, ch_golds in tqdm(list(zip(chunks, gold_chunks)),
                                              desc=f"  ablate seed={seed}", leave=False):
                    with ablate_router_ngrams(wrapper, [pat], seed=seed) as tr:
                        res = _decode_batch(ch_idxs)
                        hits_per_b = [0] * len(ch_idxs)
                        for (_, b, *_rest) in tr.hits:
                            if b < len(hits_per_b):
                                hits_per_b[b] += 1
                    for si, gd, (txt, pred), nh in zip(ch_idxs, ch_golds, res, hits_per_b):
                        ok = int(pred is not None and gd is not None and pred == gd)
                        abl_c += ok; abl_tot += 1
                        _write_record(fh, pmeta, si, gd, "ablate_random",
                                      seed, pred, ok, txt, n_hits=nh)
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
