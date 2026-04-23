"""Evaluate three steering modes on ScienceQA for SoRL checkpoints.

For each RUN we save a JSONL with one record per sample:
    1. plain       -> wrapper.scale = 0.0,  decode_scale = 0.0
    2. steered     -> wrapper.scale = orig, decode_scale = orig
    3. random_codes-> wrapper.scale = orig, decode_scale = orig, but every
                      committed router code is randomly swapped
                      (via ablate_router_ngrams with all unigram patterns).

Usage:
    python eval_steering_modes.py \
        --repo Ksgk-fy/sciqa_ckpt_20260416_0942 \
        --runs q06_sciqa_v9_C32_detach_az0.1_aa0.5 \
               q17_sciqa_v9_C32_detach_az0.1_aa0.5 \
               q4b_sciqa_v9_C32_detach_az0.1_aa0.1 \
        --num-samples 1000 --batch-size 8 --seed 0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

from data.pt_dataset import ScienceQADataset
from sorl.analyze import ablate_router_ngrams, load_steered_model


DEFAULT_RUNS = [
    "q06_sciqa_v9_C32_detach_az0.1_aa0.5",
    "q17_sciqa_v9_C32_detach_az0.1_aa0.5",
    "q4b_sciqa_v9_C32_detach_az0.1_aa0.1",
]


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
    return bii.to(device), bam.to(device), T, plens


@torch.no_grad()
def _generate(wrapper, tokenizer, bii, bam, T, decode_scale, max_new_tokens):
    out = wrapper.generate(
        input_ids=bii, attention_mask=bam,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        decode_scale=decode_scale,
    )
    gen = out[:, T:]
    return [tokenizer.decode(row, skip_special_tokens=True) for row in gen]


def run_mode(mode, wrapper, tokenizer, val_ds, golds, s_idxs, args, out_fh, device):
    """Run one mode over all samples. Streams records to out_fh."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    orig_scale = float(wrapper.scale)

    # ---- configure wrapper state for this mode ----
    if mode == "plain":
        wrapper.scale = 0.0
        decode_scale = 0.0
        ctx = None
    elif mode == "prompt_steered":
        # Prefill-time steering only (on the prompt); decode unsteered.
        wrapper.scale = orig_scale
        decode_scale = 0.0
        ctx = None
    elif mode == "steered":
        wrapper.scale = orig_scale
        decode_scale = orig_scale
        ctx = None
    elif mode == "random_codes":
        wrapper.scale = orig_scale
        decode_scale = orig_scale
        # unigram patterns: match EVERY committed code -> random swap.
        patterns = [(int(c),) for c in range(wrapper.C_SIZE)]
        ctx = ablate_router_ngrams(wrapper, patterns, seed=int(args.seed))
    else:
        raise ValueError(mode)

    extract = ScienceQADataset.extract_answer
    B = max(1, args.batch_size)
    chunks = [s_idxs[i:i + B] for i in range(0, len(s_idxs), B)]
    gold_chunks = [golds[i:i + B] for i in range(0, len(golds), B)]

    correct = total = 0
    try:
        if ctx is not None:
            ctx.__enter__()
        for ch_idxs, ch_golds in tqdm(list(zip(chunks, gold_chunks)),
                                       desc=f"  {mode}", leave=False):
            bii, bam, T, _ = _left_pad_batch(val_ds, ch_idxs, pad_id, device)
            texts = _generate(wrapper, tokenizer, bii, bam, T,
                              decode_scale, args.max_new_tokens)
            for si, gd, txt in zip(ch_idxs, ch_golds, texts):
                pred = extract(txt)
                ok = int(pred is not None and gd is not None and pred == gd)
                correct += ok
                total += 1
                out_fh.write(json.dumps({
                    "sample_idx": int(si),
                    "gold": gd,
                    "mode": mode,
                    "pred": pred,
                    "correct": ok,
                    "text": txt,
                }) + "\n")
            out_fh.flush()
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)
        wrapper.scale = orig_scale

    return correct, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Ksgk-fy/sciqa_ckpt_20260416_0942")
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    ap.add_argument("--modes", nargs="+",
                    default=["plain", "prompt_steered", "steered", "random_codes"],
                    choices=["plain", "prompt_steered", "steered", "random_codes"])
    ap.add_argument("--num-samples", type=int, default=1000,
                    help="Number of ScienceQA test samples to evaluate (from start).")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed used for random-code replacement in mode=random_codes.")
    ap.add_argument("--out-dir", default="analysis_out/steering_modes")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for run in args.runs:
        print(f"\n=== {run} ===")
        wrapper, tokenizer, wargs = load_steered_model(run, args.repo, device)
        wrapper.eval()

        val_ds = ScienceQADataset(split="test", tokenizer=tokenizer, max_length=512)
        N = min(args.num_samples, len(val_ds))
        s_idxs = list(range(N))
        # cache gold letters once (avoids rebuilding full text per sample)
        golds = []
        for si in s_idxs:
            ex = val_ds.dataset[si]
            golds.append(chr(ord("A") + int(ex["answer"])))

        run_accs = {}
        for mode in args.modes:
            fp = out_dir / f"{run}__{mode}.jsonl"
            print(f"  -> {fp}")
            t0 = time.time()
            with fp.open("w") as fh:
                c, t = run_mode(mode, wrapper, tokenizer, val_ds, golds,
                                s_idxs, args, fh, device)
            acc = c / max(t, 1)
            dt = time.time() - t0
            print(f"     mode={mode:<13s}  acc={acc*100:5.2f}%  "
                  f"({c}/{t}, {dt:.0f}s)")
            run_accs[mode] = dict(correct=c, total=t, acc=acc, path=str(fp))

        summary.append({"run": run, "modes": run_accs,
                        "orig_scale": float(wargs["scale"]),
                        "C_SIZE": int(wargs["C_SIZE"])})

        # free before next run
        del wrapper, tokenizer, val_ds
        torch.cuda.empty_cache()

    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump({"repo": args.repo, "num_samples": args.num_samples,
                   "seed": args.seed, "results": summary}, f, indent=2)

    # ---- print compact table ----
    print("\n" + "=" * 90)
    print(f"{'run':<44s} {'plain':>9s} {'prompt':>9s} {'steered':>9s} {'random':>9s}")
    print("-" * 90)
    for r in summary:
        m = r["modes"]
        def _a(k): return f"{m[k]['acc']*100:5.2f}%" if k in m else "   -   "
        print(f"{r['run']:<44s} {_a('plain'):>9s} {_a('prompt_steered'):>9s} "
              f"{_a('steered'):>9s} {_a('random_codes'):>9s}")
    print(f"\n[steering-modes] summary -> {summary_path}")


if __name__ == "__main__":
    main()
