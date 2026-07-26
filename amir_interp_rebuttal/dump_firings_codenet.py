"""Dump raw per-firing source chunks for CodeNet codes, for blind auto-interp.

The arithmetic study was upgraded from "describe this code from its firing
statistics" to "describe it from raw firings", because the first version lets a
model pattern-match a label off numbers instead of reading the data. CodeNet
auto-interp was never upgraded — `autointerp.build_prompts` still reads a
distribution summary, and it reads it from `codenet_r1r2_125step.json`, a file
for the superseded scale=0.1 checkpoint that no longer exists.

This dumps, for each code, the actual Python source each firing sits on: the
L-token chunk itself plus a window of context either side. The AST label is
recorded so the result can be scored, and is withheld from the interpreter.

Batch size 1 throughout. At batch 32 a prefill chunk index aligns with its
source chunk only when pad_len % L == 0, so any larger batch would silently
attach firings to the wrong source text.

Usage:
    python -m amir_interp_rebuttal.dump_firings_codenet \
        --ckpt ckpt/codenet_s0.5_i10_z1_L8_n4000
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from amir_interp_rebuttal.codenet_dataset import CodeNetDataset
from amir_interp_rebuttal.load_local import load_local_steered
from amir_interp_rebuttal.runner import batched_generate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="ckpt/codenet_s0.5_i10_z1_L8_n4000")
    p.add_argument("--eval_n", type=int, default=800)
    p.add_argument("--per_code", type=int, default=12)
    p.add_argument("--context_tokens", type=int, default=16,
                   help="tokens of source shown either side of the chunk")
    p.add_argument("--min_firings", type=int, default=30)
    p.add_argument("--out",
                   default="amir_interp_rebuttal/results/codenet_firings.json")
    args = p.parse_args()

    wrapper, tok, ckpt_args = load_local_steered(args.ckpt, device="cuda")
    L = int(ckpt_args["L"])
    ds = CodeNetDataset(split="test", tokenizer=tok, max_length=256,
                        size=args.eval_n)
    idxs = list(range(len(ds)))
    print(f"loaded {args.ckpt} | L={L} | {len(idxs)} files", flush=True)

    recs = batched_generate(wrapper, tok, ds, "cuda", idxs,
                            eval_batch_size=1, max_new_tokens=32,
                            record_codes=True,
                            decode_scale=float(ckpt_args["scale"]))

    by_code = defaultdict(list)
    for r in recs:
        i = r["ds_idx"]
        codes = r.get("prompt_codes") or []
        labels = ds.chunk_labels(i, L)
        ids = tok(ds.sources[i], add_special_tokens=False)["input_ids"]
        for m, c in enumerate(codes):
            c = int(c)
            if c < 0 or m >= len(labels):
                continue
            lo, hi = m * L, (m + 1) * L
            if lo >= len(ids):
                continue
            chunk = tok.decode(ids[lo:hi])
            before = tok.decode(ids[max(0, lo - args.context_tokens):lo])
            after = tok.decode(ids[hi:hi + args.context_tokens])
            by_code[c].append({
                "file_idx": i,
                "chunk_index": m,
                "chunk_text": chunk,
                "context_before": before,
                "context_after": after,
                "label": labels[m],        # withheld from the interpreter
            })

    out = {}
    for c, firings in by_code.items():
        if len(firings) < args.min_firings:
            continue
        # Even spread rather than the first N, which would all come from the
        # same few files (records are ordered by file).
        step = max(1, len(firings) // args.per_code)
        out[str(c)] = {
            "n_total": len(firings),
            "examples": firings[::step][:args.per_code],
        }

    payload = {"ckpt": args.ckpt, "L": L, "n_eval": len(idxs),
               "eval_batch_size": 1, "codes": out}
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}: {len(out)} codes, "
          f"{sum(v['n_total'] for v in out.values())} total firings")
    for c, v in sorted(out.items(), key=lambda kv: -kv[1]["n_total"]):
        print(f"  t{c}: {v['n_total']} firings, {len(v['examples'])} sampled")


if __name__ == "__main__":
    main()
