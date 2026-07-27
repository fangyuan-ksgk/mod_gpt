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


def classify_arithmetic(pred, gold):
    """Bucket one wrong arithmetic answer.

    The arithmetic model is 86% accurate, so its errors are mostly a correct
    answer with one digit wrong -- the case a per-digit steering code could
    plausibly repair, since the code at answer position d is exactly the
    intervention that governs digit d. `single_digit` also records WHICH
    position differs, which turns best-of-C into a targeted C-attempt sweep at
    one position instead of a blind search over all of them.
    """
    p, g = (pred or "").strip(), (gold or "").strip()
    if not p:
        return "empty", None
    if len(p) != len(g):
        return "length_mismatch", None
    diff = [i for i, (a, b) in enumerate(zip(p, g)) if a != b]
    if len(diff) == 1:
        return "single_digit", diff[0]
    if len(diff) == 2:
        return "two_digit", diff[0]
    return "many_digit", (diff[0] if diff else None)


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
    if g.startswith(p):
        return "stopped_early"            # correct prefix, generation ended too soon
    if p.startswith(g):
        return "ran_on"                   # correct answer, then kept generating
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


CLASS_ORDER = ["empty", "whitespace_only", "stopped_early", "ran_on", "reordering",
               "identifier_swap", "same_construct", "different_construct", "other",
               # arithmetic
               "single_digit", "two_digit", "many_digit", "length_mismatch"]


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
    p.add_argument("--mode", default="single", choices=["single", "all", "targeted"],
                   help="single: force one code at one decode step (weakest "
                        "intervention). all: force it at EVERY decode step -- "
                        "the right shape when a code's role is sustained, e.g. "
                        "'stop the line here' rather than 'emit this token'.")
    p.add_argument("--max_examples", type=int, default=60,
                   help="stage 2: cap attempts; C x positions x N gets large")
    p.add_argument("--show", type=int, default=8,
                   help="stage 1: sample errors printed per class for eyeballing")
    p.add_argument("--dump_batch_size", type=int, default=None,
                   help="batch size for the --dump baseline pass. CodeNet MUST "
                        "stay at 1 (left padding misaligns prefill chunks unless "
                        "pad_len %% L == 0). Arithmetic has a fixed-length prompt "
                        "so there is no pad variance and larger batches are safe.")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for the random-control arm. A positive cell must "
                        "survive a fresh seed; with ~18 cells swept, a single "
                        "unreplicated positive is what multiple comparisons "
                        "produces.")
    p.add_argument("--digit_level", action="store_true",
                   help="arithmetic only. Score a repair as successful when the "
                        "TARGET DIGIT becomes correct, rather than requiring the "
                        "whole answer. Forcing a code at position d changes digit "
                        "d and everything downstream autoregressively, so a code "
                        "can fix its own digit and still fail a whole-answer test "
                        "by disturbing a later one. This asks the narrower "
                        "question the codes are actually supposed to answer: does "
                        "this code control this digit?")
    p.add_argument("--max_gold_tokens", type=int, default=None,
                   help="restrict to examples whose gold answer is at most this "
                        "many tokens. At 17%% accuracy most CodeNet errors are "
                        "'the model did not solve it', which no code swap fixes. "
                        "Short-gold errors are the competent subset: right "
                        "content, wrong boundary.")
    p.add_argument("--repair_scale", type=float, default=None,
                   help="decode_scale to use DURING repair attempts. The "
                        "arithmetic checkpoint trains at scale=0.1, so swapping "
                        "one learned code for another barely perturbs the "
                        "residual stream. Amplifying asks whether the code "
                        "CARRIES the sub-task, separately from whether it does "
                        "so loudly enough at its native strength. Applied to "
                        "both arms equally.")
    p.add_argument("--from_taxonomy", action="store_true",
                   help="skip the 800-example baseline pass and read the error "
                        "list from {study}_error_taxonomy.json. The baseline is "
                        "identical for every class and mode, so re-running it "
                        "per job wastes ~15 min each and serialises the sweep.")
    p.add_argument("--out_dir", default="amir_interp_rebuttal/results")
    args = p.parse_args()
    if not (args.dump or args.repair):
        raise SystemExit("pick --dump or --repair")

    from amir_interp_rebuttal.load_local import build_dataset, load_local_steered
    from amir_interp_rebuttal.runner import batched_generate

    wrapper, tok, meta = load_local_steered(args.ckpt, device=args.device)
    scale, L = float(meta["scale"]), int(meta["L"])
    C = int(meta["C_SIZE"])
    ds = build_dataset(args.study, tok, args.eval_n)
    idxs = list(range(len(ds)))
    print(f"loaded {args.ckpt} | scale={scale} L={L} C={C} | {len(idxs)} examples",
          flush=True)

    tax_path = Path(args.out_dir) / f"{args.study}_error_taxonomy.json"
    if args.from_taxonomy:
        if not tax_path.exists():
            raise SystemExit(f"--from_taxonomy needs {tax_path}; run --dump first")
        tax = json.loads(tax_path.read_text())
        if tax.get("ckpt") != args.ckpt:
            raise SystemExit(
                f"taxonomy was built on {tax.get('ckpt')} but --ckpt is "
                f"{args.ckpt}. Error indices are checkpoint-specific; refusing "
                f"to reuse them.")
        wrong = [dict(r) for r in tax["errors"]]
        acc = tax["accuracy"]
        counts = Counter(r["error_class"] for r in wrong)
        print(f"accuracy {acc:.4f} (cached) | {len(wrong)} wrong, "
              f"baseline pass skipped", flush=True)
    else:
        dbs = args.dump_batch_size or 1
        if args.study == "codenet" and dbs != 1:
            raise SystemExit("CodeNet requires --dump_batch_size 1")
        recs = batched_generate(wrapper, tok, ds, args.device, idxs,
                                eval_batch_size=dbs,
                                max_new_tokens=args.max_new_tokens,
                                record_codes=True, decode_scale=scale)
        acc = sum(r["correct"] for r in recs) / len(recs)
        wrong = [r for r in recs if not r["correct"]]
        print(f"accuracy {acc:.4f} | {len(wrong)} wrong of {len(recs)}", flush=True)
        for r in wrong:
            if args.study == "arithmetic":
                cls, wpos = classify_arithmetic(r.get("pred"), r.get("gold"))
                r["error_class"], r["wrong_pos"] = cls, wpos
            else:
                r["error_class"], r["wrong_pos"] = classify(r.get("pred"),
                                                            r.get("gold")), None
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
            # len(idxs), not len(recs): `recs` is only bound on the branch that
            # regenerates. With --dump --from_taxonomy this raised NameError.
            # The two are equal — batched_generate returns one record per idx.
            "ckpt": args.ckpt, "study": args.study, "n_eval": len(idxs),
            "accuracy": acc, "n_wrong": len(wrong),
            "counts": {k: counts.get(k, 0) for k in CLASS_ORDER if counts.get(k)},
            "errors": [{"ds_idx": r["ds_idx"], "error_class": r["error_class"],
                        "wrong_pos": r.get("wrong_pos"),
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
        if args.max_gold_tokens:
            before = len(target)
            target = [r for r in target
                      if len(_toks(r.get("gold"))) <= args.max_gold_tokens]
            print(f"  gold-length filter <= {args.max_gold_tokens} tokens: "
                  f"{before} -> {len(target)} examples", flush=True)
        if not target:
            raise SystemExit(f"no errors in class {args.error_class!r}")
        target = target[:args.max_examples]
        n_chunks = max(1, args.max_new_tokens // L)
        # Each entry is what gets handed to force_code_at as `position`.
        if args.mode == "targeted":
            # one attempt-set per example, at the digit that is actually wrong
            posspecs = "PER_EXAMPLE"
        else:
            posspecs = list(range(n_chunks)) if args.mode == "single" else [None]
        budget = C * (1 if posspecs == "PER_EXAMPLE" else len(posspecs))
        print(f"\n  repairing class {args.error_class!r} [mode={args.mode}]: "
              f"{len(target)} examples x {C} codes x "
              f"{1 if posspecs == 'PER_EXAMPLE' else len(posspecs)} position-set(s) "
              f"= {budget} attempts/arm", flush=True)

        import torch

        rscale = args.repair_scale if args.repair_scale is not None else scale
        if rscale != scale:
            print(f"  intervention amplified: decode_scale {scale} -> {rscale} "
                  f"(both arms)", flush=True)

        def try_one(i, pos, c, want_digit=None, gold=None):
            with force_code_at(wrapper, pos, c):
                o = batched_generate(wrapper, tok, ds, args.device, [i],
                                     eval_batch_size=1,
                                     max_new_tokens=args.max_new_tokens,
                                     record_codes=False, decode_scale=rscale)
            if not args.digit_level or want_digit is None:
                return bool(o[0]["correct"])
            pr = (o[0].get("pred") or "").strip()
            g = (gold or "").strip()
            if len(pr) != len(g) or want_digit >= len(pr):
                return False
            return pr[want_digit] == g[want_digit]

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
        gen = torch.Generator(device="cpu").manual_seed(args.seed)
        SCRATCH = 0

        fixed_any, fixed_by, first_hit = 0, Counter(), []
        rand_fixed = 0
        for n, r in enumerate(target, 1):
            i = r["ds_idx"]

            if posspecs == "PER_EXAMPLE":
                wp = r.get("wrong_pos")
                if wp is None:
                    continue
                spec = [wp]
            else:
                spec = posspecs
            got = False
            for pos in spec:
                for c in range(C):
                    if try_one(i, pos, c, r.get("wrong_pos"), r.get("gold")):
                        got, fixed_by[c] = True, fixed_by[c] + 1
                        first_hit.append({"ds_idx": i, "code": c,
                                          "position": ("all" if pos is None else pos)})
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
                if args.mode == "all":
                    pos = None
                elif args.mode == "targeted":
                    pos = spec[0]
                else:
                    pos = int(torch.randint(n_chunks, (1,), generator=gen).item())
                if try_one(i, pos, SCRATCH, r.get("wrong_pos"), r.get("gold")):
                    rgot = True
                    break
            wrapper.steering_emb.weight.data.copy_(saved_emb)
            rand_fixed += rgot

            if n % 5 == 0:
                print(f"    {n}/{len(target)}  best-of-{budget} real {fixed_any}  "
                      f"random-matched {rand_fixed}", flush=True)

        payload = {
            "ckpt": args.ckpt, "study": args.study,
            "error_class": args.error_class, "mode": args.mode,
            "native_scale": scale, "repair_scale": rscale,
            "max_gold_tokens": args.max_gold_tokens,
            "seed": args.seed,
            "success_criterion": ("target digit correct" if args.digit_level
                                  else "whole answer correct"),
            "n_attempted": len(target), "C": C, "n_positions": n_chunks,
            "fixed_best_of_C": fixed_any,
            "fix_rate_best_of_C": fixed_any / len(target),
            "attempt_budget_per_example": budget,
            "fixed_random_matched": rand_fixed,
            "fix_rate_random_matched": rand_fixed / len(target),
            "fixing_codes": dict(fixed_by.most_common()),
            "first_hits": first_hit,
        }
        tag = f"{args.error_class}_{args.mode}"
        if rscale != scale:
            tag += f"_s{rscale}"
        if args.max_gold_tokens:
            tag += f"_g{args.max_gold_tokens}"
        if args.digit_level:
            tag += "_digit"
        if args.seed:
            tag += f"_seed{args.seed}"
        path = out / f"{args.study}_error_repair_{tag}.json"
        path.write_text(json.dumps(payload, indent=2))
        print(f"\n  best-of-{C}: {fixed_any}/{len(target)} "
              f"({fixed_any/len(target):.1%})")
        print(f"  random matched (same {budget}-attempt budget): "
              f"{rand_fixed}/{len(target)} ({rand_fixed/len(target):.1%})")
        if fixed_by:
            print("  codes that produced a fix: " +
                  ", ".join(f"t{c}x{n}" for c, n in fixed_by.most_common(8)))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
