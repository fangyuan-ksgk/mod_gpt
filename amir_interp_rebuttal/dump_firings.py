"""Dump raw per-firing examples for the top codes, for both studies.

Auto-interp reads THESE files, not distribution summaries. The distinction is
the whole point of the test. Describing a code from its statistics ("78% of its
firings land on one category") lets a model pattern-match a label off numbers.
Describing it from raw firings ("here are 14 columns where it fired: 4+5, 7+2,
6+3, ...") is the real readability test — the pattern has to be visible in the
data itself.

Two studies, one module. They differ only in what a "firing" is attached to:

    arithmetic : one code per answer digit (L=1), so a firing is a digit column
                 -> results/arith_firings.json
    codenet    : one code per L-token chunk, so a firing is a span of source
                 -> results/codenet_firings.json

The two OUTPUT SCHEMAS differ and both are load-bearing, so they are kept
exactly as they were. `arith_firings.json` is a flat {code: {...}} mapping and
is consumed by repro/r5_sum9.sh; changing its shape breaks that table. The
CodeNet file wraps the same per-code payload in a header carrying the ckpt and
the eval batch size, because for CodeNet those two facts decide whether the
numbers mean anything at all (see below).

Batch size is 1 for CodeNet and 32 for arithmetic, and that is not a tuning
knob. A prefill chunk index lines up with its source chunk only when
pad_len % L == 0 (true for 28.5% of rows at batch 32), so any larger CodeNet
batch silently attaches firings to the wrong source text. Arithmetic is immune:
its prompt is a fixed length, so pad_len is constant across rows.

Usage:
    python -m amir_interp_rebuttal.dump_firings --study arithmetic
    python -m amir_interp_rebuttal.dump_firings --study codenet
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from amir_interp_rebuttal.load_local import load_local_steered
from amir_interp_rebuttal.runner import batched_generate

# Per-study defaults. Every one of these has a reason; see the module docstring
# for the batch-size and schema constraints in particular.
STUDIES = {
    "arithmetic": dict(
        ckpt="ckpt/arith_v9_paperhp",
        eval_n=2600,
        per_code=14,
        eval_batch_size=32,        # safe: fixed-length prompt, constant pad_len
        max_new_tokens=8,
        min_firings=30,
        out="amir_interp_rebuttal/results/arith_firings.json",
    ),
    "codenet": dict(
        ckpt="ckpt/codenet_s0.5_i10_z1_L8_n4000",
        eval_n=800,
        per_code=12,
        eval_batch_size=1,         # REQUIRED: see the padding note above
        max_new_tokens=32,
        min_firings=30,
        out="amir_interp_rebuttal/results/codenet_firings.json",
    ),
}


def _load_dataset(study, tok, eval_n):
    if study == "arithmetic":
        from amir_interp_rebuttal.arith_dataset import ArithmeticDataset
        ds = ArithmeticDataset(split="test", tokenizer=tok, max_length=64)
        return ds, list(range(min(eval_n, len(ds))))
    from amir_interp_rebuttal.codenet_dataset import CodeNetDataset
    ds = CodeNetDataset(split="test", tokenizer=tok, max_length=256, size=eval_n)
    return ds, list(range(len(ds)))


def _firings_arithmetic(recs, ds, _tok, _L, _ctx):
    """Decode-stream firings: answer digit d was steered by code c."""
    by_code = defaultdict(list)
    for r in recs:
        codes = r.get("codes") or []
        ex = ds.examples[r["ds_idx"]]
        labels = ex.labels
        x, y, z = ex.x_digits, ex.y_digits, ex.z_digits
        op = "+" if ex.op == "add" else "-"
        for d, c in enumerate(codes[:len(labels)]):
            c = int(c)
            if c < 0:
                continue
            # The answer has n_digits+1 places; answer digit d aligns with
            # operand column d-1 (d=0 is the overflow place, no operand column).
            col = d - 1
            a = x[col] if 0 <= col < len(x) else None
            b = y[col] if 0 <= col < len(y) else None
            by_code[c].append({
                "problem": f"{''.join(map(str,x))}{op}{''.join(map(str,y))}"
                           f"={''.join(map(str,z))}",
                "op": ex.op,
                "answer_pos": d,
                "answer_digit": z[d],
                "operand_digits": [a, b],
                "column_sum": (a + b) if (a is not None and b is not None) else None,
                "label": labels[d],          # withheld from the interpreter
                "split": ds.split_of[r["ds_idx"]],
            })
    return by_code


def _firings_codenet(recs, ds, tok, L, context_tokens):
    """Prefill-stream firings: source chunk m was routed to code c.

    The labelled structure is the whole source file, nearly all of which is
    prompt, so the prefill code stream is the one that indexes the same tokens
    as the labels. The decode stream covers only the generated last line.
    """
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
            by_code[c].append({
                "file_idx": i,
                "chunk_index": m,
                "chunk_text": tok.decode(ids[lo:hi]),
                "context_before": tok.decode(ids[max(0, lo - context_tokens):lo]),
                "context_after": tok.decode(ids[hi:hi + context_tokens]),
                "label": labels[m],          # withheld from the interpreter
            })
    return by_code


COLLECTORS = {"arithmetic": _firings_arithmetic, "codenet": _firings_codenet}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--study", choices=sorted(STUDIES), required=True)
    # Everything below defaults to the study's own value, so `--study X` alone
    # reproduces the reported file. An explicit flag always wins.
    p.add_argument("--ckpt")
    p.add_argument("--eval_n", type=int)
    p.add_argument("--per_code", type=int)
    p.add_argument("--eval_batch_size", type=int)
    p.add_argument("--max_new_tokens", type=int)
    p.add_argument("--min_firings", type=int)
    p.add_argument("--context_tokens", type=int, default=16,
                   help="codenet only: tokens of source shown either side")
    p.add_argument("--out")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    d = STUDIES[args.study]
    for k, v in d.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)

    if args.study == "codenet" and args.eval_batch_size != 1:
        raise SystemExit(
            f"--eval_batch_size {args.eval_batch_size} on codenet: prefill chunk "
            "k maps to source chunk k only when pad_len % L == 0, so any batch "
            "above 1 attaches firings to the wrong source text. Use 1.")

    wrapper, tok, ckpt_args = load_local_steered(args.ckpt, device=args.device)
    L = int(ckpt_args["L"])
    ds, idxs = _load_dataset(args.study, tok, args.eval_n)
    print(f"loaded {args.ckpt} | L={L} | {len(idxs)} examples "
          f"| batch {args.eval_batch_size}", flush=True)

    # decode_scale is passed EXPLICITLY. The V9 wrapper defaults
    # _decode_scale_override to 0.0, so omitting it silently disables steering
    # during generation and the recorded codes describe an unsteered model.
    recs = batched_generate(wrapper, tok, ds, args.device, idxs,
                            eval_batch_size=args.eval_batch_size,
                            max_new_tokens=args.max_new_tokens,
                            record_codes=True,
                            decode_scale=float(ckpt_args["scale"]))

    by_code = COLLECTORS[args.study](recs, ds, tok, L, args.context_tokens)

    codes = {}
    for c, firings in by_code.items():
        if len(firings) < args.min_firings:
            continue
        # Even spread across the code's firings rather than the first N, which
        # would all come from the same eval split / the same few files (records
        # are ordered by split for arithmetic and by file for codenet).
        step = max(1, len(firings) // args.per_code)
        codes[str(c)] = {
            "n_total": len(firings),
            "examples": firings[::step][:args.per_code],
        }

    # SCHEMAS DIFFER BY DESIGN — arithmetic is flat because repro/r5_sum9.sh
    # iterates it as {code: {...}} and that table must not move.
    payload = codes if args.study == "arithmetic" else {
        "ckpt": args.ckpt, "L": L, "n_eval": len(idxs),
        "eval_batch_size": args.eval_batch_size, "codes": codes,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}: {len(codes)} codes, "
          f"{sum(v['n_total'] for v in codes.values())} total firings")
    for c, v in sorted(codes.items(), key=lambda kv: -kv[1]["n_total"]):
        print(f"  t{c}: {v['n_total']} firings, {len(v['examples'])} sampled")


if __name__ == "__main__":
    main()
