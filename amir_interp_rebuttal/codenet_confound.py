"""Task 1: position-confound audit for the CodeNet R1 purity result.

The arithmetic study found that "position locking" was degenerate: at 6 of 7 answer
positions a single code covered ~100% of examples, so P(label|code) was really
P(label|position) and the reported lift was a property of the task, not the model.

This re-derives the CodeNet contingency with POSITION kept as an explicit variable:

    lift_global = P(label|code) / P(label)               <- what analyze.py reported
    lift_pos    = P(label|code) / P(label | code's positions)   <- the honest control

`lift_pos` asks: given that you already know WHERE in the file this chunk is, does
knowing the code tell you anything more? If lift_pos ~= 1.0 the code is redundant
with position.
"""
from __future__ import annotations

import argparse, json
from collections import Counter, defaultdict
from pathlib import Path

from amir_interp_rebuttal.codenet_dataset import CodeNetDataset
from amir_interp_rebuttal.load_local import load_local_steered
from amir_interp_rebuttal.runner import batched_generate


def main():
    p = argparse.ArgumentParser()
    # The reported CodeNet checkpoint. ckpt/codenet_v9 (scale=0.1) is SUPERSEDED
    # and defaulting to it silently audits the wrong model.
    p.add_argument("--ckpt", default="ckpt/codenet_s0.5_i10_z1_L8_n4000")
    p.add_argument("--eval_n", type=int, default=800)
    p.add_argument("--min_code_n", type=int, default=30)
    # Prefill chunks are cut from position 0 of the LEFT-PADDED row, so with pad
    # P the mapping "padded chunk P//L -> source chunk 0" is exact only when
    # P % L == 0 (28.5% of rows at batch 32). Running at batch size 1 removes
    # padding entirely and makes the chunk<->label alignment exact for every
    # example — the only way to score R1 without the padding artifact that
    # manufactured the t20 result. Hence the default, and the guard below.
    p.add_argument("--eval_batch_size", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--out", default="amir_interp_rebuttal/results/codenet_position_confound.json")
    args = p.parse_args()

    if args.eval_batch_size != 1:
        raise SystemExit(
            f"--eval_batch_size {args.eval_batch_size}: prefill chunk k maps to "
            "source chunk k only when pad_len % L == 0, so every purity number "
            "above batch 1 is scored against misaligned source. This is the bug "
            "that manufactured t20 -> FunctionDef 3.84x. Use 1.")

    wrapper, tok, ck = load_local_steered(args.ckpt, device="cuda")
    L, C_SIZE, scale = ck["L"], ck["C_SIZE"], float(ck["scale"])
    ds = CodeNetDataset(split="test", tokenizer=tok, max_length=256, size=1500)
    N = min(args.eval_n, len(ds))
    recs = batched_generate(wrapper, tok, ds, "cuda", list(range(N)),
                            eval_batch_size=args.eval_batch_size,
                            max_new_tokens=args.max_new_tokens,
                            record_codes=True, decode_scale=scale)
    acc = sum(r["correct"] for r in recs) / len(recs)
    print(f"accuracy {acc:.1%}")

    # Same alignment analyze.py uses for codenet: prefill (prompt) codes vs source
    # chunk labels, both indexed from token 0 of the source.
    joint = Counter()          # (code, pos, label)
    code_pos = defaultdict(Counter)
    pos_code = defaultdict(Counter)
    pos_label = defaultdict(Counter)
    code_label = defaultdict(Counter)
    glob = Counter()
    for r in recs:
        codes = r.get("prompt_codes") or []
        labels = ds.chunk_labels(r["ds_idx"], L)
        for pos in range(min(len(codes), len(labels))):
            c, lab = int(codes[pos]), labels[pos]
            if c < 0:
                continue
            joint[(c, pos, lab)] += 1
            code_pos[c][pos] += 1
            pos_code[pos][c] += 1
            pos_label[pos][lab] += 1
            code_label[c][lab] += 1
            glob[lab] += 1

    total = sum(glob.values())
    marg = {k: v / total for k, v in glob.items()}

    # ── per-code rows with the position-conditional control ────────
    rows = []
    for c, labc in code_label.items():
        n = sum(labc.values())
        if n < args.min_code_n:
            continue
        top_lab, top_n = labc.most_common(1)[0]
        purity = top_n / n
        # P(label | position), averaged over the positions this code actually fires
        # at, weighted the same way the code's own firings are distributed. This is
        # the expected purity of a position-matched random baseline.
        exp = sum(cnt * (pos_label[pos].get(top_lab, 0) / sum(pos_label[pos].values()))
                  for pos, cnt in code_pos[c].items()) / n
        tp, tp_n = code_pos[c].most_common(1)[0]
        rows.append({
            "code": c, "n": n, "top_label": top_lab,
            "purity": round(purity, 4),
            "marginal": round(marg.get(top_lab, 0), 4),
            "lift_global": round(purity / marg[top_lab], 3) if marg.get(top_lab) else None,
            "pos_matched_baseline": round(exp, 4),
            "lift_pos": round(purity / exp, 3) if exp else None,
            "top_pos": tp, "top_pos_share_of_code": round(tp_n / n, 4),
            "n_positions": len(code_pos[c]),
            "share_of_top_pos": round(tp_n / sum(pos_code[tp].values()), 4),
        })
    rows.sort(key=lambda r: -r["purity"])

    # ── per-position code concentration (the arithmetic-style degeneracy check) ──
    pos_rows = []
    for pos in sorted(pos_code):
        cc = pos_code[pos]
        n = sum(cc.values())
        top_c, top_n = cc.most_common(1)[0]
        peaking = [c for c in code_pos if code_pos[c] and code_pos[c].most_common(1)[0][0] == pos
                   and sum(code_label[c].values()) >= args.min_code_n]
        pl = pos_label[pos]
        pos_rows.append({
            "pos": pos, "n": n, "n_distinct_codes": len(cc),
            "top_code": top_c, "top_code_share": round(top_n / n, 4),
            "top3_codes": [(c, round(k / n, 4)) for c, k in cc.most_common(3)],
            "n_codes_peaking_here": len(peaking), "codes_peaking_here": sorted(peaking),
            "top_label": pl.most_common(1)[0][0],
            "top_label_share": round(pl.most_common(1)[0][1] / n, 4),
        })

    out = {"ckpt": args.ckpt, "n_eval": N, "L": L, "C_SIZE": C_SIZE,
           "accuracy": acc, "marginal": {k: round(v, 4) for k, v in
                                         sorted(marg.items(), key=lambda kv: -kv[1])},
           "code_rows": rows, "position_rows": pos_rows}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}: {len(rows)} active codes, {len(pos_rows)} positions")
    for r in rows[:6]:
        print(f"  t{r['code']:<3} {r['top_label']:<12} purity {r['purity']:.3f}  "
              f"lift_glob {r['lift_global']}x  pos-matched {r['pos_matched_baseline']:.3f}  "
              f"lift_pos {r['lift_pos']}x  (pos {r['top_pos']}, {r['n_positions']} pos)")


if __name__ == "__main__":
    main()
