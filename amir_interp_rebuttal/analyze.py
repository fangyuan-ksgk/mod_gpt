"""
Produce the two measurements reviewer yrxa asked for, on a trained v9 model.

  R1  subtask-pure codes   — P(label | code), scored by LIFT over the label's marginal
  R2  surgical-swap repair — fix rate on wrong predictions vs a random-code control

Works for both studies; they differ only in where the per-chunk ground-truth label
comes from:
    arithmetic : Quirke subtask per answer digit  (ArithmeticDataset.labels_at)
    codenet    : dominant AST construct per chunk (CodeNetDataset.chunk_labels)

Usage:
    python -m amir_interp_rebuttal.analyze --study arithmetic --ckpt ckpt/arith_v9
    python -m amir_interp_rebuttal.analyze --study codenet    --ckpt ckpt/codenet_v9
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from amir_interp_rebuttal.interp import (
    build_contingency, format_purity_table, purity_report, surgical_swap_sweep,
    targeted_swap_sweep,
)
from amir_interp_rebuttal.runner import batched_generate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--study", choices=["arithmetic", "codenet"], required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model_name", default="Qwen/Qwen3-0.6B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--C_SIZE", type=int, default=30)
    p.add_argument("--L", type=int, default=1)
    p.add_argument("--eval_n", type=int, default=1200)
    # Left-padding corrupts prefill chunk alignment: padded chunk k maps to
    # source chunk k only when pad_len % L == 0, which held for just 28.5% of
    # rows at batch 32. That silently reshuffled every CodeNet routing number.
    # Batch 1 has no padding, so alignment is exact. Arithmetic is immune (its
    # prompt is a fixed 14 tokens, so pad_len is constant across rows) and can
    # keep the fast path.
    p.add_argument("--eval_batch_size", type=int, default=None,
                   help="default: 32 for arithmetic, 1 for codenet")
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--max_swap_examples", type=int, default=150)
    p.add_argument("--min_code_n", type=int, default=30)
    p.add_argument("--out_dir", default="amir_interp_rebuttal/results")
    return p.parse_args()


def load_study(args, tok):
    if args.study == "arithmetic":
        from amir_interp_rebuttal.arith_dataset import ArithmeticDataset, verify_alignment
        verify_alignment(tok)
        ds = ArithmeticDataset(split="test", tokenizer=tok, max_length=64)
        n_chunks = ds.answer_len
        label_fn = lambda i: ds.labels_at(i)
        return ds, n_chunks, label_fn
    else:
        from amir_interp_rebuttal.codenet_dataset import CodeNetDataset
        ds = CodeNetDataset(split="test", tokenizer=tok, max_length=256, size=1500)
        n_chunks = None  # variable per example
        label_fn = lambda i: ds.chunk_labels(i, args.L)
        return ds, n_chunks, label_fn


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = args.device

    from amir_interp_rebuttal.load_local import load_local_steered
    wrapper, tok, ckpt_args = load_local_steered(args.ckpt, device=dev)
    # Trust the checkpoint over the CLI for structural params — a mismatch here
    # silently corrupts the chunk<->label alignment.
    args.C_SIZE = ckpt_args["C_SIZE"]
    args.L = ckpt_args["L"]
    # CRITICAL: the V9 wrapper sets `_decode_scale_override = 0.0`, so unless we
    # pass the trained scale explicitly, steering is disabled during decoding —
    # codes get routed and logged but do not influence a single generated token.
    # Every swap experiment then returns exactly 0 for all arms, which reads as a
    # clean null but is a no-op.
    decode_scale = float(ckpt_args["scale"])
    print(f"decode_scale = {decode_scale} (steering ACTIVE during generation)")

    ds, n_chunks, label_fn = load_study(args, tok)
    n_eval = min(args.eval_n, len(ds))
    idxs = list(range(n_eval))

    print(f"\n=== {args.study} | {args.model_name} | {n_eval} eval examples ===")

    # ── generate once, recording routed codes ───────────────────────
    if args.eval_batch_size is None:
        args.eval_batch_size = 32 if args.study == "arithmetic" else 1
    if args.study != "arithmetic" and args.eval_batch_size != 1:
        print(f"  WARNING: batch={args.eval_batch_size} on a prefill-aligned "
              f"study; chunk<->label alignment is only exact at batch 1")
    print(f"  eval_batch_size = {args.eval_batch_size}")

    recs = batched_generate(wrapper, tok, ds, dev, idxs,
                            eval_batch_size=args.eval_batch_size,
                            max_new_tokens=args.max_new_tokens,
                            record_codes=True, decode_scale=decode_scale)
    acc = sum(r["correct"] for r in recs) / len(recs)
    wrong = [r["ds_idx"] for r in recs if not r["correct"]]
    print(f"accuracy {acc:.1%}   ({len(wrong)} wrong of {len(recs)})")

    report = {"study": args.study, "ckpt": args.ckpt, "n_eval": n_eval,
              "accuracy": acc, "n_wrong": len(wrong)}

    # ── R1: subtask purity ──────────────────────────────────────────
    print("\n--- R1: subtask-pure codes ---")

    class _LabelAdapter:
        """Uniform label interface over both studies."""
        def labels_at(self, i):
            return label_fn(i)

    # Which recorded code stream lines up with the labels differs by study:
    #
    #   arithmetic — the labelled structure IS the generated answer, so the
    #                decode codes (one per answer digit) are the right stream.
    #   codenet    — the labelled structure is the source code, nearly all of
    #                which sits in the prompt. Decode codes cover only the ~4
    #                chunks of the generated final line, so scoring them against
    #                whole-source chunk labels would compare chunk k of the
    #                completion with chunk k of the file. Use the prefill codes,
    #                which start at the same token as the labels.
    code_key = "codes" if args.study == "arithmetic" else "prompt_codes"
    usable = [r for r in recs if code_key in r and r[code_key]]
    if not usable:
        raise RuntimeError(
            f"no records carry '{code_key}' — cannot align codes to labels. "
            "Check that record_codes=True and the wrapper exposes the stream.")
    for r in usable:
        r["codes"] = r[code_key]

    span = n_chunks if n_chunks is not None else max(
        (len(r["codes"]) for r in usable), default=0)
    print(f"  aligning on '{code_key}' | {len(usable)} records | span {span} chunks")
    counts, pos_counts = build_contingency(usable, _LabelAdapter(), span)
    rows, marginal = purity_report(counts, pos_counts, min_n=args.min_code_n)

    if rows:
        print(format_purity_table(rows, marginal,
                                  title=f"{args.study}: P(label | code)"))
        # rows are lift-ranked, so rows[0] is the strongest specialist
        best = rows[0]
        best_pur = max(rows, key=lambda r: r["purity"])
        med_lift = sorted(r["lift"] for r in rows)[len(rows) // 2]
        print(f"\n  active codes (n>={args.min_code_n}): {len(rows)}")
        print(f"  best lift   : t{best['code']} -> {best['top_subtask']} "
              f"{best['lift']:.2f}x (purity {best['purity']:.1%}, "
              f"marginal {best['marginal']:.1%}, n={best['n']})")
        print(f"  best purity : t{best_pur['code']} -> {best_pur['top_subtask']} "
              f"{best_pur['purity']:.1%} (lift {best_pur['lift']:.2f}x)"
              + ("  <- high purity but near-chance lift"
                 if best_pur["lift"] < 1.5 else ""))
        print(f"  median lift : {med_lift:.2f}x")
        # Pass on either a clean specialist by the paper's bar, or a strong
        # lift that a skewed label distribution keeps below 70% purity.
        r1_pass = ((best["purity"] >= 0.70 or best["lift"] >= 3.0)
                   and med_lift > 1.2)
        print(f"  R1 {'REPLICATED' if r1_pass else 'NOT replicated'}")
    else:
        r1_pass, med_lift = False, float("nan")
        print(f"  no code reached n>={args.min_code_n} — R1 not measurable")

    report["R1"] = {
        "n_active_codes": len(rows),
        "median_lift": med_lift,
        "rows": rows,
        "marginal": marginal,
        "replicated": bool(rows) and r1_pass,
    }

    # ── R2: surgical swap ───────────────────────────────────────────
    print("\n--- R2: surgical-swap repairs ---")
    if len(wrong) < 20:
        print(f"  only {len(wrong)} wrong predictions — too few to measure a fix "
              f"rate. Task is saturated; escalate difficulty (see notes/arithmetic.md §3).")
        report["R2"] = {"measurable": False, "n_wrong": len(wrong)}
    else:
        # R2b — the predictive test. Uses R1's purity table to pick the code the
        # ground-truth label says should apply, matched 1:1 against a random code.
        # This is the claim-supporting number; run it first so it lands even if
        # the exhaustive sweep gets cut short.
        print("\n  [R2b] label-matched code vs random control")
        # Resolve decode step -> labelled chunk. Arithmetic: the labels ARE the
        # answer digits, so decode step i is label i. CodeNet: labels index the
        # whole source and the decode stream begins after the prompt's chunks,
        # so the same decode step is label (n_prompt_chunks + i).
        def label_at(i, pos):
            labels = label_fn(i)
            if args.study == "arithmetic":
                return labels[pos] if pos < len(labels) else None
            item = ds[i]
            offset = -(-int(item["prompt_len"]) // args.L)   # ceil div
            k = offset + pos
            return labels[k] if k < len(labels) else None

        targeted = targeted_swap_sweep(
            wrapper, tok, ds, dev, wrong, rows, label_at,
            args.C_SIZE, span, max_examples=args.max_swap_examples,
            max_new_tokens=args.max_new_tokens, decode_scale=decode_scale,
        )

        # R2a — exhaustive best-of-C. Reported for comparability with the
        # published number, but it is an EXISTENCE measure over C x positions
        # interventions per example, not evidence of structure on its own.
        print("\n  [R2a] exhaustive best-of-C (existence only)")
        # Swaps are applied at DECODE steps, so the position range is bounded by
        # how many chunks are actually generated (max_new_tokens / L) — not by
        # `span`, which for CodeNet counts chunks across the whole source.
        n_decode_chunks = max(1, args.max_new_tokens // max(1, args.L))
        positions = (list(range(span)) if args.study == "arithmetic"
                     else list(range(min(n_decode_chunks, 8))))
        print(f"  swap positions: {positions}")
        swap = surgical_swap_sweep(
            wrapper, tok, ds, dev, wrong, args.C_SIZE, span,
            positions=positions, max_examples=args.max_swap_examples,
            max_new_tokens=args.max_new_tokens, decode_scale=decode_scale,
        )
        best_pos = max(swap["per_position"].items(),
                       key=lambda kv: kv[1]["fix_rate_best_of_C"])

        r2_pass = bool(targeted.get("measurable")
                       and targeted["targeted_fix_rate"] > 1.5 * max(targeted["random_fix_rate"], 0.005))
        print(f"\n  R2 verdict is set by R2b (predictive), not R2a (existence).")
        print(f"  R2 {'REPLICATED' if r2_pass else 'NOT replicated'}")
        report["R2"] = {
            "measurable": True,
            "R2b_targeted": targeted,
            "R2a_best_of_C": {
                "per_position": {str(k): v for k, v in swap["per_position"].items()},
                "control": {str(k): v for k, v in swap["control"].items()},
                "best_position": str(best_pos[0]),
                "note": "existence measure over C x positions attempts per example",
            },
            "replicated": r2_pass,
        }

    path = out / f"{args.study}_r1r2.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
