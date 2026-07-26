"""Error-class-restricted repair: what fraction of ONE kind of error is fixable?

The aggregate repair test (force the label-matched code, see if the prediction
flips) came back 0/82. That number is hard to read, because at 17.5% accuracy
most wrong outputs are wrong in ways no single steering code could plausibly
fix -- the model simply did not solve the problem. Averaging a repairable class
together with a hopeless one buries whatever signal exists.

So: classify the errors first, pick a class that is a priori repairable, and ask
the existence question inside it -- is there SOME code that fixes this, and does
it beat a random code on the same examples?

    stage 1  --dump      generate, classify every error, report the taxonomy
    stage 2  --repair    for one class, sweep all C codes x positions

Stage 2 is the expensive one, which is why the class is chosen from stage 1
output rather than swept blindly.

Usage:
    python -m amir_interp_rebuttal.error_repair --study codenet \
        --ckpt ckpt/codenet_s0.5_i10_z1_L8_n4000 --dump
    python -m amir_interp_rebuttal.error_repair --study codenet \
        --ckpt ckpt/codenet_s0.5_i10_z1_L8_n4000 --repair --error_class truncation
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def _norm(s):
    """Whitespace-insensitive comparison form."""
    return re.sub(r"\s+", " ", (s or "").strip())


def _toks(s):
    return re.findall(r"[A-Za-z_]\w*|\d+|\S", s or "")


def classify(pred, gold):
    """Bucket one wrong prediction. Ordered: first match wins.

    The classes are deliberately mechanical -- no judgement calls -- so the
    taxonomy is reproducible and a reviewer can check any row by hand.
    """
    if pred is None or not str(pred).strip():
        return "empty"
    p, g = _norm(pred), _norm(gold)
    if p == g:
        return "whitespace_only"          # scored wrong, differs only in spacing
    pt, gt = _toks(p), _toks(g)
    if not pt or not gt:
        return "other"
    if g.startswith(p) or p.startswith(g):
        return "truncation"               # right prefix, stopped early / ran on
    if sorted(pt) == sorted(gt):
        return "reordering"               # same tokens, different order
    if pt[0] == gt[0] and len(pt) == len(gt):
        # same shape and same leading construct, differing in operands/names
        diff = sum(1 for x, y in zip(pt, gt) if x != y)
        if diff <= 2:
            return "identifier_swap"
    if pt[0] == gt[0]:
        return "same_construct"           # right construct, wrong body
    return "different_construct"


CLASS_ORDER = ["empty", "whitespace_only", "truncation", "reordering",
               "identifier_swap", "same_construct", "different_construct", "other"]


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
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--dump", action="store_true", help="stage 1: classify errors")
    p.add_argument("--repair", action="store_true", help="stage 2: best-of-C sweep")
    p.add_argument("--error_class", default=None,
                   help="stage 2: which class to attempt (from stage 1)")
    p.add_argument("--max_examples", type=int, default=60,
                   help="stage 2: cap attempts; C x positions x N gets large")
    p.add_argument("--show", type=int, default=8,
                   help="stage 1: sample errors printed per class for eyeballing")
    p.add_argument("--out_dir", default="amir_interp_rebuttal/results")
    args = p.parse_args()
    if not (args.dump or args.repair):
        raise SystemExit("pick --dump or --repair")

    from amir_interp_rebuttal.load_local import load_local_steered
    from amir_interp_rebuttal.runner import batched_generate

    wrapper, tok, meta = load_local_steered(args.ckpt, device=args.device)
    scale, L = float(meta["scale"]), int(meta["L"])
    C = int(meta["C_SIZE"])
    ds = build_dataset(args.study, tok, args.eval_n)
    idxs = list(range(len(ds)))
    print(f"loaded {args.ckpt} | scale={scale} L={L} C={C} | {len(idxs)} examples",
          flush=True)

    recs = batched_generate(wrapper, tok, ds, args.device, idxs,
                            eval_batch_size=1, max_new_tokens=args.max_new_tokens,
                            record_codes=True, decode_scale=scale)
    acc = sum(r["correct"] for r in recs) / len(recs)
    wrong = [r for r in recs if not r["correct"]]
    print(f"accuracy {acc:.4f} | {len(wrong)} wrong of {len(recs)}", flush=True)

    for r in wrong:
        r["error_class"] = classify(r.get("pred"), r.get("gold"))
    counts = Counter(r["error_class"] for r in wrong)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.dump:
        print("\n  error taxonomy")
        print("  " + "-" * 58)
        for k in CLASS_ORDER:
            n = counts.get(k, 0)
            if n:
                print(f"  {k:<22} {n:>5}  {n/len(wrong):>6.1%} of errors")
        print("  " + "-" * 58)
        for k in CLASS_ORDER:
            sel = [r for r in wrong if r["error_class"] == k][:args.show]
            if not sel:
                continue
            print(f"\n  === {k} ===")
            for r in sel:
                print(f"    gold: {_norm(r.get('gold'))[:90]!r}")
                print(f"    pred: {_norm(r.get('pred'))[:90]!r}")
        payload = {
            "ckpt": args.ckpt, "study": args.study, "n_eval": len(recs),
            "accuracy": acc, "n_wrong": len(wrong),
            "counts": {k: counts.get(k, 0) for k in CLASS_ORDER if counts.get(k)},
            "errors": [{"ds_idx": r["ds_idx"], "error_class": r["error_class"],
                        "gold": r.get("gold"), "pred": r.get("pred")}
                       for r in wrong],
        }
        path = out / f"{args.study}_error_taxonomy.json"
        path.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {path}")

    if args.repair:
        from amir_interp_rebuttal.interp import force_code_at
        if not args.error_class:
            raise SystemExit("--repair needs --error_class (see --dump output)")
        target = [r for r in wrong if r["error_class"] == args.error_class]
        if not target:
            raise SystemExit(f"no errors in class {args.error_class!r}")
        target = target[:args.max_examples]
        n_chunks = max(1, args.max_new_tokens // L)
        print(f"\n  repairing class {args.error_class!r}: {len(target)} examples "
              f"x {C} codes x {n_chunks} positions", flush=True)

        import torch

        def try_one(i, pos, c):
            with force_code_at(wrapper, pos, c):
                o = batched_generate(wrapper, tok, ds, args.device, [i],
                                     eval_batch_size=1,
                                     max_new_tokens=args.max_new_tokens,
                                     record_codes=False, decode_scale=scale)
            return bool(o[0]["correct"])

        # MATCHED-EFFORT CONTROL.
        #
        # best-of-C searches C x n_chunks = many attempts. Comparing that against
        # a single random attempt is not a control -- with enough tries on a
        # 17%-accuracy model something flips by chance, and the result would look
        # positive whatever the codes mean. The null has to spend the SAME budget
        # on directions that carry no learned meaning: random vectors matched to
        # the codebook's own norm, swapped into a scratch slot and forced the
        # same way.
        saved_emb = wrapper.steering_emb.weight.data.clone()
        mean_norm = saved_emb.norm(dim=1).mean().item()
        gen = torch.Generator(device="cpu").manual_seed(0)
        SCRATCH = 0

        fixed_any, fixed_by, first_hit = 0, Counter(), []
        rand_fixed = 0
        budget = C * n_chunks
        for n, r in enumerate(target, 1):
            i = r["ds_idx"]

            got = False
            for pos in range(n_chunks):
                for c in range(C):
                    if try_one(i, pos, c):
                        got, fixed_by[c] = True, fixed_by[c] + 1
                        first_hit.append({"ds_idx": i, "code": c, "position": pos})
                        break
                if got:
                    break
            fixed_any += got

            # same budget, arbitrary directions
            rgot = False
            for _ in range(budget):
                v = torch.randn(saved_emb.shape[1], generator=gen)
                v = (v / v.norm() * mean_norm).to(saved_emb.dtype)
                wrapper.steering_emb.weight.data[SCRATCH] = v.to(saved_emb.device)
                pos = int(torch.randint(n_chunks, (1,), generator=gen).item())
                if try_one(i, pos, SCRATCH):
                    rgot = True
                    break
            wrapper.steering_emb.weight.data.copy_(saved_emb)
            rand_fixed += rgot

            if n % 5 == 0:
                print(f"    {n}/{len(target)}  best-of-{budget} real {fixed_any}  "
                      f"random-matched {rand_fixed}", flush=True)

        payload = {
            "ckpt": args.ckpt, "study": args.study,
            "error_class": args.error_class,
            "n_attempted": len(target), "C": C, "n_positions": n_chunks,
            "fixed_best_of_C": fixed_any,
            "fix_rate_best_of_C": fixed_any / len(target),
            "attempt_budget_per_example": C * n_chunks,
            "fixed_random_matched": rand_fixed,
            "fix_rate_random_matched": rand_fixed / len(target),
            "fixing_codes": dict(fixed_by.most_common()),
            "first_hits": first_hit,
        }
        path = out / f"{args.study}_error_repair_{args.error_class}.json"
        path.write_text(json.dumps(payload, indent=2))
        print(f"\n  best-of-{C}: {fixed_any}/{len(target)} "
              f"({fixed_any/len(target):.1%})")
        print(f"  random matched (same {C*n_chunks}-attempt budget): "
              f"{rand_fixed}/{len(target)} ({rand_fixed/len(target):.1%})")
        if fixed_by:
            print("  codes that produced a fix: " +
                  ", ".join(f"t{c}x{n}" for c, n in fixed_by.most_common(8)))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
