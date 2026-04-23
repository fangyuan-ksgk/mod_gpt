"""Per-code unigram ablation sweep on ScienceQA.

For each code c in [0, C_SIZE) we run the whole eval set with
`ablate_router_ngrams(wrapper, [(c,)], seed=...)`, plus one steered baseline
(no ablation). Compared to the baseline, each sample × code falls into one of:

    help        baseline wrong, ablate correct         (code c hurts this sample)
    hurt        baseline correct, ablate wrong         (code c helps this sample)
    same_ok     both correct
    same_bad    both wrong (either same pred or both unparsable)

Aggregated over samples (and optionally over topics), this tells us which
codes are consistently destructive vs. consistently beneficial to swap out
— the "super codes" hypothesis from the notebook case studies.

Outputs (under --out-dir, default `analysis_out/per_code_ablation/<run>/`):
  - `baseline.jsonl`      : one record per sample (pred/correct/topic/gold)
  - `ablations.jsonl`     : one record per (sample, code): pred, correct,
                            outcome in {help, hurt, same_ok, same_bad}
  - `summary_by_code.json`: aggregated help/hurt counts per code
  - `summary_by_topic_code.json`: {topic: {code: counts}}

Usage:
python eval_per_code_ablation.py \
    --repo Ksgk-fy/sciqa_ckpt_20260416_0942 \
    --run q06_sciqa_v9_C32_detach_az0.1_aa0.5 \
    --num-samples 200 --batch-size 8 --seed 0
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
        max_new_tokens=max_new,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        decode_scale=decode_scale,
    )
    gen = out[:, T:]
    return [tokenizer.decode(row, skip_special_tokens=True) for row in gen]


def _classify(base_ok, abl_ok):
    if base_ok and abl_ok:     return "same_ok"
    if not base_ok and not abl_ok: return "same_bad"
    if abl_ok and not base_ok: return "help"
    return "hurt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Ksgk-fy/sciqa_ckpt_20260416_0942")
    ap.add_argument("--run", default="q06_sciqa_v9_C32_detach_az0.1_aa0.5")
    ap.add_argument("--num-samples", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--codes", nargs="*", type=int, default=None,
                    help="Optional subset of codes to sweep (default: all).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="analysis_out/per_code_ablation")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir) / args.run
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- model + data ----
    wrapper, tokenizer, wargs = load_steered_model(args.run, args.repo, device)
    wrapper.eval()
    C = int(wargs["C_SIZE"])
    codes = list(args.codes) if args.codes else list(range(C))
    decode_scale = float(wargs["scale"])

    val_ds = ScienceQADataset(split="test", tokenizer=tokenizer, max_length=512)
    N = min(args.num_samples, len(val_ds))
    s_idxs = list(range(N))
    golds = [chr(ord("A") + int(val_ds.dataset[i]["answer"])) for i in s_idxs]
    topics = [val_ds.dataset[i].get("topic", "unknown") for i in s_idxs]

    extract = ScienceQADataset.extract_answer
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    B = max(1, args.batch_size)
    chunks = [s_idxs[i:i + B] for i in range(0, N, B)]

    # ---- 1. baseline (steered, no ablation) ----
    print(f"\n[baseline] {args.run}  N={N}  C={C}  decode_scale={decode_scale}")
    base_path = out_dir / "baseline.jsonl"
    base_preds, base_correct = [], []
    with base_path.open("w") as fh:
        for ch in tqdm(chunks, desc="baseline"):
            bii, bam, T = _left_pad_batch(val_ds, ch, pad_id, device)
            texts = _gen_batch(wrapper, tokenizer, bii, bam, T,
                               decode_scale, args.max_new_tokens)
            for si, txt in zip(ch, texts):
                pred = extract(txt)
                gd = golds[si]
                ok = int(pred is not None and pred == gd)
                base_preds.append(pred)
                base_correct.append(ok)
                fh.write(json.dumps({
                    "sample_idx": si, "topic": topics[si], "gold": gd,
                    "pred": pred, "correct": ok, "text": txt,
                }) + "\n")
    base_acc = sum(base_correct) / N
    print(f"  baseline acc = {base_acc*100:.2f}%  ({sum(base_correct)}/{N})")

    # ---- 2. per-code ablation sweep ----
    abl_path = out_dir / "ablations.jsonl"
    # counts[code] = {help, hurt, same_ok, same_bad}
    counts = {c: defaultdict(int) for c in codes}
    # per-topic counts[topic][code]
    topic_counts = defaultdict(lambda: {c: defaultdict(int) for c in codes})

    t0 = time.time()
    with abl_path.open("w") as fh:
        for c in tqdm(codes, desc="codes"):
            for ch in tqdm(chunks, desc=f"  code={c}", leave=False):
                bii, bam, T = _left_pad_batch(val_ds, ch, pad_id, device)
                with ablate_router_ngrams(wrapper, [(int(c),)],
                                          seed=int(args.seed) + c):
                    texts = _gen_batch(wrapper, tokenizer, bii, bam, T,
                                       decode_scale, args.max_new_tokens)
                for si, txt in zip(ch, texts):
                    pred = extract(txt)
                    gd = golds[si]
                    ok = int(pred is not None and pred == gd)
                    outcome = _classify(bool(base_correct[si]), bool(ok))
                    counts[c][outcome] += 1
                    topic_counts[topics[si]][c][outcome] += 1
                    fh.write(json.dumps({
                        "sample_idx": si, "topic": topics[si], "gold": gd,
                        "code": c, "pred": pred, "correct": ok,
                        "baseline_pred": base_preds[si],
                        "baseline_correct": int(base_correct[si]),
                        "outcome": outcome,
                        "text": txt,
                    }) + "\n")
    print(f"[sweep] wall={time.time()-t0:.0f}s  ->  {abl_path}")

    # ---- 3. summaries ----
    by_code = {}
    for c in codes:
        k = counts[c]
        tot = sum(k.values())
        help_, hurt_ = k["help"], k["hurt"]
        abl_correct = k["help"] + k["same_ok"]
        by_code[c] = {
            "help": help_, "hurt": hurt_,
            "same_ok": k["same_ok"], "same_bad": k["same_bad"],
            "n": tot,
            "ablate_acc": abl_correct / tot if tot else 0.0,
            "delta_vs_baseline": (abl_correct / tot - base_acc) if tot else 0.0,
            "net_help_minus_hurt": help_ - hurt_,
        }

    with (out_dir / "summary_by_code.json").open("w") as f:
        json.dump({
            "run": args.run, "repo": args.repo,
            "num_samples": N, "C_SIZE": C,
            "baseline_acc": base_acc,
            "by_code": by_code,
        }, f, indent=2)

    by_topic = {}
    for topic, codedict in topic_counts.items():
        base_ok_topic = sum(base_correct[si] for si in s_idxs if topics[si] == topic)
        n_topic = sum(1 for si in s_idxs if topics[si] == topic)
        by_topic[topic] = {
            "n": n_topic,
            "baseline_correct": base_ok_topic,
            "codes": {
                c: {
                    "help": codedict[c]["help"], "hurt": codedict[c]["hurt"],
                    "same_ok": codedict[c]["same_ok"],
                    "same_bad": codedict[c]["same_bad"],
                } for c in codes
            },
        }
    with (out_dir / "summary_by_topic_code.json").open("w") as f:
        json.dump(by_topic, f, indent=2)

    # ---- 4. compact leaderboard ----
    ranked = sorted(by_code.items(),
                    key=lambda kv: kv[1]["delta_vs_baseline"], reverse=True)
    print(f"\n=== per-code leaderboard (N={N}, baseline={base_acc*100:.2f}%) ===")
    print(f"{'code':>5s}  {'ablate_acc':>10s}  {'Δ acc':>7s}  "
          f"{'help':>5s}  {'hurt':>5s}  {'net':>5s}")
    for c, v in ranked:
        print(f"{c:>5d}  {v['ablate_acc']*100:>9.2f}%  "
              f"{v['delta_vs_baseline']*100:>+6.2f}  "
              f"{v['help']:>5d}  {v['hurt']:>5d}  "
              f"{v['net_help_minus_hurt']:>+5d}")
    print(f"\nsaved: {out_dir}")


if __name__ == "__main__":
    main()
