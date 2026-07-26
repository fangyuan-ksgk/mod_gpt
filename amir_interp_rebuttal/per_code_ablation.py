"""Per-code ablation: does zeroing ONE code cost accuracy, and only where it fires?

The aggregate knockout (all codes off, -6.87pp) shows the code channel matters.
It does not show that any INDIVIDUAL code matters. Surgical repair (Finding #4)
tried to show that by forcing a code on and failed. This is the same question
from the other direction: knock a single code OUT and see what breaks.

The confound to beat is exposure. A code firing on 60% of examples will cost
more accuracy than one firing on 3%, whatever either encodes. So for every code
we split the eval set by that code's own firing pattern in the unablated run:

    affected = examples where the code fires
    control  = examples where it never fires

Ablating code c should hurt `affected` and leave `control` alone. A code whose
control set moves as much as its affected set is not being localised by this
measurement -- the damage is going through some downstream path, not through
that code's contribution.

Usage:
    python -m amir_interp_rebuttal.per_code_ablation \
        --ckpt ckpt/codenet_s0.5_i10_z1_L8_n4000 --study codenet
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import torch


def build_dataset(study, tok, size):
    if study == "codenet":
        from amir_interp_rebuttal.codenet_dataset import CodeNetDataset
        return CodeNetDataset(split="test", tokenizer=tok, max_length=256, size=size)
    from amir_interp_rebuttal.arith_dataset import ArithmeticDataset
    return ArithmeticDataset(split="test", tokenizer=tok, size=size)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--study", default="codenet", choices=["codenet", "arithmetic"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--eval_n", type=int, default=800)
    p.add_argument("--eval_batch_size", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--min_fires", type=int, default=20,
                   help="skip codes firing on fewer than this many examples — "
                        "their affected set is too small to read")
    p.add_argument("--out_dir", default="amir_interp_rebuttal/results")
    args = p.parse_args()

    from amir_interp_rebuttal.load_local import load_local_steered
    from amir_interp_rebuttal.runner import batched_generate

    wrapper, tok, meta = load_local_steered(args.ckpt, device=args.device)
    scale = float(meta["scale"])
    ds = build_dataset(args.study, tok, args.eval_n)
    idxs = list(range(len(ds)))
    print(f"loaded {args.ckpt} | scale={scale} L={meta.get('L')} "
          f"C={meta.get('C_SIZE')} | {len(idxs)} eval examples", flush=True)

    # ---- baseline: everything on, and record which codes touch which example
    base = batched_generate(wrapper, tok, ds, args.device, idxs,
                            eval_batch_size=args.eval_batch_size,
                            max_new_tokens=args.max_new_tokens,
                            record_codes=True, decode_scale=scale)
    base_correct = [bool(r["correct"]) for r in base]
    acc_base = sum(base_correct) / len(base_correct)
    print(f"baseline accuracy {acc_base:.4f}", flush=True)

    # fires[c] = set of example indices where code c appears (prompt or decode)
    fires = {}
    for i, r in enumerate(base):
        seen = set()
        for k in list(r.get("prompt_codes") or []) + list(r.get("codes") or []):
            k = int(k)
            if k >= 0:
                seen.add(k)
        for k in seen:
            fires.setdefault(k, set()).add(i)

    counts = Counter({c: len(v) for c, v in fires.items()})
    todo = [c for c, n in counts.most_common() if n >= args.min_fires]
    print(f"codes firing on >= {args.min_fires} examples: {len(todo)} "
          f"-> {todo}", flush=True)

    saved = wrapper.steering_emb.weight.data.clone()
    rows = []
    for c in todo:
        wrapper.steering_emb.weight.data.copy_(saved)
        wrapper.steering_emb.weight.data[c].zero_()
        out = batched_generate(wrapper, tok, ds, args.device, idxs,
                               eval_batch_size=args.eval_batch_size,
                               max_new_tokens=args.max_new_tokens,
                               record_codes=False, decode_scale=scale)
        abl_correct = [bool(r["correct"]) for r in out]

        aff = sorted(fires[c])
        ctl = [i for i in idxs if i not in fires[c]]

        def acc(sel, arr):
            return (sum(arr[i] for i in sel) / len(sel)) if sel else float("nan")

        row = dict(
            code=int(c),
            n_examples_firing=len(aff),
            share_of_eval=len(aff) / len(idxs),
            acc_base_all=acc_base,
            acc_ablated_all=sum(abl_correct) / len(abl_correct),
            acc_base_affected=acc(aff, base_correct),
            acc_ablated_affected=acc(aff, abl_correct),
            acc_base_control=acc(ctl, base_correct),
            acc_ablated_control=acc(ctl, abl_correct),
            n_control=len(ctl),
        )
        row["delta_affected_pp"] = 100 * (row["acc_base_affected"] - row["acc_ablated_affected"])
        row["delta_control_pp"] = 100 * (row["acc_base_control"] - row["acc_ablated_control"])
        row["delta_all_pp"] = 100 * (row["acc_base_all"] - row["acc_ablated_all"])
        # localisation: how much of the damage lands where the code actually fires
        row["localisation"] = (row["delta_affected_pp"] - row["delta_control_pp"])
        rows.append(row)
        print(f"  t{c:<3} fires_on={len(aff):<4} "
              f"Δaffected={row['delta_affected_pp']:+6.2f}pp  "
              f"Δcontrol={row['delta_control_pp']:+6.2f}pp  "
              f"localisation={row['localisation']:+6.2f}pp", flush=True)

    wrapper.steering_emb.weight.data.copy_(saved)

    rows.sort(key=lambda r: -r["localisation"])
    report = dict(ckpt=args.ckpt, study=args.study, scale=scale,
                  L=meta.get("L"), C_SIZE=meta.get("C_SIZE"),
                  n_eval=len(idxs), eval_batch_size=args.eval_batch_size,
                  max_new_tokens=args.max_new_tokens,
                  accuracy_baseline=acc_base, rows=rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.study}_per_code_ablation.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
