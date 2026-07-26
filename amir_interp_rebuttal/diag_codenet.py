"""
Why did the matched-budget CodeNet model score 4.9% when the 125-step model scored
22.2% on the same eval set?

Both trained to a similar loss on the same data distribution, and the test split is
identical by construction (collection order is deterministic; CODENET_SIZE only
changes where it stops). So the accuracy gap is real and needs an explanation before
either model is reported as a measurement rather than an artefact.

Three hypotheses this separates:

  H1  degenerate output — the model emits empty / repeated / truncated text
  H2  length mismatch   — completions are reasonable but longer or shorter than the
                          single target line, so exact match fails on formatting
  H3  genuinely worse   — well-formed, correctly-shaped, simply wrong predictions

Prints side-by-side samples plus summary statistics for both checkpoints.
"""
from __future__ import annotations

import json
from pathlib import Path

from amir_interp_rebuttal.codenet_dataset import CodeNetDataset
from amir_interp_rebuttal.load_local import load_local_steered
from amir_interp_rebuttal.runner import batched_generate

CKPTS = ["ckpt/codenet_v9", "ckpt/codenet_v9_20k"]
N = 200


def summarise(recs):
    preds = [(r["pred"] or "") for r in recs]
    golds = [(r["gold"] or "") for r in recs]
    n = len(preds)
    empty = sum(1 for p in preds if not p.strip())
    exact = sum(1 for p, g in zip(preds, golds) if p == g)
    # A prediction that merely repeats a fragment is a classic collapse signature.
    repet = sum(1 for p in preds if len(p) > 8 and len(set(p.split())) <= 2)
    mean_len = sum(len(p) for p in preds) / max(n, 1)
    gold_len = sum(len(g) for g in golds) / max(n, 1)
    # Prefix agreement is metric-independent: it shows whether the model is heading
    # in the right direction even when exact match fails.
    def pref(a, b):
        k = 0
        for x, y in zip(a, b):
            if x != y:
                break
            k += 1
        return k
    mean_pref = sum(pref(p, g) for p, g in zip(preds, golds)) / max(n, 1)
    return {
        "n": n, "exact": exact, "exact_rate": exact / max(n, 1),
        "empty": empty, "empty_rate": empty / max(n, 1),
        "repetitive": repet,
        "mean_pred_len": round(mean_len, 1),
        "mean_gold_len": round(gold_len, 1),
        "mean_prefix_match_chars": round(mean_pref, 1),
    }


def main():
    out = {}
    samples = {}
    for ckpt in CKPTS:
        if not Path(ckpt, "final.pt").exists():
            print(f"skip {ckpt} (missing)")
            continue
        w, tok, a = load_local_steered(ckpt, device="cuda", verbose=False)
        ds = CodeNetDataset(split="test", tokenizer=tok, max_length=256, size=800)
        recs = batched_generate(w, tok, ds, "cuda", list(range(min(N, len(ds)))),
                                eval_batch_size=32, max_new_tokens=32,
                                record_codes=False, decode_scale=float(a["scale"]))
        out[ckpt] = summarise(recs)
        samples[ckpt] = [
            {"gold": r["gold"], "pred": r["pred"]} for r in recs[:6]
        ]
        print(f"\n=== {ckpt} ===")
        for k, v in out[ckpt].items():
            print(f"  {k:>24}: {v}")
        print("  --- samples ---")
        for s in samples[ckpt]:
            print(f"    gold: {s['gold']!r}")
            print(f"    pred: {s['pred']!r}")
        del w

    Path("amir_interp_rebuttal/results").mkdir(parents=True, exist_ok=True)
    Path("amir_interp_rebuttal/results/codenet_generation_diag.json").write_text(
        json.dumps({"summary": out, "samples": samples}, indent=2))

    if len(out) == 2:
        a, b = out[CKPTS[0]], out[CKPTS[1]]
        print("\n=== verdict ===")
        if b["empty_rate"] > 0.3 or b["repetitive"] > 0.3 * b["n"]:
            print("  H1 degenerate output — the 20K model emits empty/repetitive text.")
        elif abs(b["mean_pred_len"] - b["mean_gold_len"]) > 2 * abs(
                a["mean_pred_len"] - a["mean_gold_len"]) + 5:
            print("  H2 length mismatch — completions are mis-shaped, not mis-aimed.")
        else:
            print("  H3 genuinely worse — well-formed predictions, simply wrong.")


if __name__ == "__main__":
    main()
