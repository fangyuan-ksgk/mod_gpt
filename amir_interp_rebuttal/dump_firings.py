"""
Dump raw per-firing examples for the top codes, so auto-interp can read actual
firings instead of summary statistics.

The distinction matters. Describing a code from its distribution ("78% of
firings land on one category") lets a model pattern-match a label off numbers.
Describing it from raw firings ("here are 12 columns where it fired: 4+5, 7+2,
6+3, ...") is the real readability test — the pattern has to be visible in the
data itself.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from amir_interp_rebuttal.arith_dataset import ArithmeticDataset
from amir_interp_rebuttal.load_local import load_local_steered
from amir_interp_rebuttal.runner import batched_generate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="ckpt/arith_v9_paperhp")
    p.add_argument("--eval_n", type=int, default=2600)
    p.add_argument("--per_code", type=int, default=14)
    p.add_argument("--out", default="amir_interp_rebuttal/results/arith_firings.json")
    args = p.parse_args()

    wrapper, tok, ckpt_args = load_local_steered(args.ckpt, device="cuda")
    ds = ArithmeticDataset(split="test", tokenizer=tok, max_length=64)
    idxs = list(range(min(args.eval_n, len(ds))))

    recs = batched_generate(wrapper, tok, ds, "cuda", idxs, eval_batch_size=32,
                            max_new_tokens=8, record_codes=True,
                            decode_scale=float(ckpt_args["scale"]))

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

    out = {}
    for c, firings in by_code.items():
        if len(firings) < 30:
            continue
        # Even spread across the code's firings rather than the first N, which
        # would all come from the same eval split (the set is ordered by split).
        step = max(1, len(firings) // args.per_code)
        out[str(c)] = {
            "n_total": len(firings),
            "examples": firings[::step][:args.per_code],
        }

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}: {len(out)} codes, "
          f"{sum(v['n_total'] for v in out.values())} total firings")
    for c, v in sorted(out.items(), key=lambda kv: -kv[1]["n_total"])[:8]:
        print(f"  t{c}: {v['n_total']} firings, {len(v['examples'])} sampled")


if __name__ == "__main__":
    main()
