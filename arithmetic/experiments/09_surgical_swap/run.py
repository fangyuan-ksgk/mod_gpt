"""
Experiment 09: Surgical Token Swap

For each wrong example on hard splits (C4-C6, B3-B5), try every possible
token replacement at each abs position. Find which swaps fix wrong answers
and which break correct ones.

Protocol:
  1. Run model on hard splits, identify wrong examples
  2. For each wrong example, try replacing each abs token with every other token
  3. Record which (position, old_token, new_token) fixes the answer
  4. Aggregate: which token→token swaps have the best fix:break ratio?

Outputs:
  results.json    — per-swap fix/break counts, example details
  summary.md      — human-readable report

Usage:
    python experiments/09_surgical_swap/run.py \
        --model add_sub_sorl_v1_abs30_K1_100K_2L1H128d --device cuda:1
"""
import argparse
import json
import sys
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from arithmetic.data.hub import load_model
from arithmetic.data.addition import get_eval_set
from arithmetic.training.train import QWEN3_TOKEN_MAP, QWEN3_INV_MAP
from sorl.sorl_trainer import infer_insert_mask, insert_tokens_with_padding, expand_prompt_len
from transformers import AutoTokenizer

PROMPT_LEN = 14
ANSWER_LEN = 7
OUT_DIR = Path(__file__).parent
DEFAULT_MODEL = "add_sub_sorl_v1_abs30_K1_100K_2L1H128d"

HARD_SPLITS = ["add_C4", "add_C5", "add_C6", "sub_B3", "sub_B4", "sub_B5"]


def run_recursion(model, qwen_ids, K, base_v, pad_id, device):
    seq = qwen_ids.unsqueeze(0)
    attn = torch.ones_like(seq)
    pl = torch.tensor([PROMPT_LEN], dtype=torch.long, device=device)

    with torch.no_grad():
        im = infer_insert_mask(seq, K, attn)
        ep = expand_prompt_len(pl, im)
        ed, ea = insert_tokens_with_padding(seq, attn, im, base_v, pad_id)
        data, ppt, logits = model.recursion(
            ed, ea, max_iterations=2,
            memory_span_abs=1792, memory_span_traj=1792,
            temperature=0.0, prompt_len=ep,
        )

    expanded = data[0]
    is_abs = expanded >= base_v
    is_traj = ~is_abs
    traj_indices = is_traj.nonzero(as_tuple=True)[0]
    abs_indices = is_abs.nonzero(as_tuple=True)[0]

    digit_abs = {}
    for abs_idx in abs_indices:
        traj_before = (traj_indices < abs_idx).sum().item() - 1
        if PROMPT_LEN <= traj_before < PROMPT_LEN + ANSWER_LEN:
            answer_pos = traj_before - PROMPT_LEN
            tok_id = (expanded[abs_idx] - base_v).item()
            digit_abs[answer_pos] = (abs_idx.item(), tok_id)

    return expanded.clone(), digit_abs


def get_answer(model, expanded, base_v):
    with torch.no_grad():
        logits = model(expanded.unsqueeze(0),
                       attention_mask=torch.ones(1, len(expanded), device=expanded.device)).logits
    is_traj = expanded < base_v
    traj_indices = is_traj.nonzero(as_tuple=True)[0]
    pred = []
    target = []
    for d in range(ANSWER_LEN):
        traj_pos = PROMPT_LEN + d
        if traj_pos < len(traj_indices):
            exp_idx = traj_indices[traj_pos].item()
            if exp_idx > 0:
                pred.append(logits[0, exp_idx - 1, :base_v].argmax().item())
                target.append(expanded[exp_idx].item())
    return pred == target, pred, target


def format_question(qwen_ids):
    orig = [QWEN3_INV_MAP.get(t.item(), -1) for t in qwen_ids]
    digits1 = "".join(str(d) for d in orig[:6])
    op = "+" if orig[6] == 10 else "-"
    digits2 = "".join(str(d) for d in orig[7:13])
    return f"{digits1}{op}{digits2}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--abs_vocab", type=int, default=30)
    p.add_argument("--splits", default=",".join(HARD_SPLITS))
    args = p.parse_args()

    splits = args.splits.split(",")

    print(f"Loading {args.model}...")
    model, config, _ = load_model(args.model, device=args.device)
    model.eval()

    base_v = model.vocab_sizes[0].item()
    K = config.get("K", 1)
    abs_vocab = config.get("abs_vocab", args.abs_vocab)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    pad_id = tokenizer.pad_token_id

    categories = get_eval_set()

    # ── Phase 1: Identify wrong and correct examples on hard splits ───

    print(f"\nPhase 1: Baseline accuracy on hard splits")
    wrong_examples = []  # (split, ex, expanded, digit_abs, pred, target)
    correct_examples = []

    for split in splits:
        if split not in categories:
            print(f"  {split}: not found in eval set")
            continue

        n_correct = 0
        for ex in categories[split]:
            qwen_ids = torch.tensor([QWEN3_TOKEN_MAP[t] for t in ex.tokens],
                                    dtype=torch.long, device=args.device)
            expanded, digit_abs = run_recursion(model, qwen_ids, K, base_v, pad_id, args.device)
            ok, pred, target = get_answer(model, expanded, base_v)

            if ok:
                correct_examples.append((split, ex, expanded, digit_abs, pred, target, qwen_ids))
                n_correct += 1
            else:
                wrong_examples.append((split, ex, expanded, digit_abs, pred, target, qwen_ids))

        n_total = len(categories[split])
        print(f"  {split}: {n_correct}/{n_total} correct, {n_total - n_correct} wrong")

    print(f"\nTotal: {len(wrong_examples)} wrong, {len(correct_examples)} correct")

    # ── Phase 2: For each wrong example, try all token replacements ───

    print(f"\nPhase 2: Exhaustive token search on {len(wrong_examples)} wrong examples")

    # Track: for each (position, old_tok, new_tok), how many fixes?
    swap_fixes = defaultdict(lambda: {"fixed": 0, "examples": []})
    # Track per-position stats
    position_fixes = defaultdict(lambda: {"any_fix_found": 0, "n_wrong": 0})

    for i, (split, ex, expanded, digit_abs, pred_wrong, target, qwen_ids) in enumerate(wrong_examples):
        q_str = format_question(qwen_ids)

        for d, (abs_idx, old_tok) in digit_abs.items():
            position_fixes[d]["n_wrong"] += 1
            found_fix = False

            for new_tok in range(abs_vocab):
                if new_tok == old_tok:
                    continue

                swapped = expanded.clone()
                swapped[abs_idx] = base_v + new_tok + 1  # +1 for placeholder

                ok_swap, pred_swap, _ = get_answer(model, swapped, base_v)

                if ok_swap:
                    key = (d, old_tok, new_tok)
                    swap_fixes[key]["fixed"] += 1
                    swap_fixes[key]["examples"].append({
                        "split": split,
                        "q": q_str,
                        "target": "".join(str(x) for x in target),
                        "pred_wrong": "".join(str(x) for x in pred_wrong),
                        "pred_fixed": "".join(str(x) for x in pred_swap),
                        "position": d,
                        "label": ex.labels[d] if d < len(ex.labels) else "?",
                    })
                    found_fix = True

            if found_fix:
                position_fixes[d]["any_fix_found"] += 1

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(wrong_examples)} wrong examples...")

    print(f"  Done. Found {len(swap_fixes)} unique (pos, old, new) swaps that fix at least one example.")

    # ── Phase 3: For top swaps, check how many correct examples they break ───

    top_swaps = sorted(swap_fixes.items(), key=lambda x: x[1]["fixed"], reverse=True)[:20]

    print(f"\nPhase 3: Checking break rate for top {len(top_swaps)} swaps on {len(correct_examples)} correct examples")

    swap_results = []
    for (d, old_tok, new_tok), fix_info in top_swaps:
        n_broke = 0
        n_checked = 0

        for split, ex, expanded, digit_abs, pred_ok, target, qwen_ids in correct_examples:
            if d not in digit_abs:
                continue
            abs_idx, actual_tok = digit_abs[d]
            if actual_tok != old_tok:
                continue

            # This correct example uses old_tok at position d — would swapping break it?
            n_checked += 1
            swapped = expanded.clone()
            swapped[abs_idx] = base_v + new_tok + 1

            ok_swap, _, _ = get_answer(model, swapped, base_v)
            if not ok_swap:
                n_broke += 1

        swap_results.append({
            "position": d,
            "old_token": old_tok,
            "new_token": new_tok,
            "fixed": fix_info["fixed"],
            "broke": n_broke,
            "checked_correct": n_checked,
            "examples": fix_info["examples"][:5],
            "old_label": fix_info["examples"][0]["label"] if fix_info["examples"] else "?",
        })

        ratio_str = f"{fix_info['fixed']}:{n_broke}"
        print(f"  d{d} t{old_tok}→t{new_tok}: fixed={fix_info['fixed']}, broke={n_broke}/{n_checked}, ratio={ratio_str}")

    # ── Phase 4: Per-position summary ─────────────────────────────────

    print(f"\nPhase 4: Per-position fixability")
    print(f"  {'Pos':>4s} {'Wrong':>6s} {'Fixable':>8s} {'Fix%':>6s}")
    print(f"  {'-'*26}")
    for d in sorted(position_fixes.keys()):
        pf = position_fixes[d]
        pct = pf["any_fix_found"] / max(pf["n_wrong"], 1)
        print(f"  d{d:<3d} {pf['n_wrong']:>6d} {pf['any_fix_found']:>8d} {pct:>6.1%}")

    # ── Save ──────────────────────────────────────────────────────────

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "model": args.model,
        "K": K,
        "abs_vocab": abs_vocab,
        "splits": splits,
        "n_wrong": len(wrong_examples),
        "n_correct": len(correct_examples),
        "per_position_fixability": {
            f"d{d}": {
                "n_wrong": pf["n_wrong"],
                "any_fix_found": pf["any_fix_found"],
                "fix_rate": round(pf["any_fix_found"] / max(pf["n_wrong"], 1), 3),
            }
            for d, pf in sorted(position_fixes.items())
        },
        "top_swaps": swap_results,
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'results.json'}")

    # ── Summary markdown ──────────────────────────────────────────────

    lines = [
        "# Surgical Token Swap — Hard Cases",
        "",
        f"**Model:** `{args.model}` (K={K}, abs_vocab={abs_vocab})",
        f"**Splits:** {', '.join(splits)}",
        f"**Wrong examples:** {len(wrong_examples)} | **Correct:** {len(correct_examples)}",
        "",
        "## Protocol",
        "",
        "For each wrong example on hard splits, try replacing each abs token with",
        "every other token in the vocabulary. Record which swaps fix the answer.",
        "Then check how many correct examples each swap breaks.",
        "",
        "## Per-Position Fixability",
        "",
        "How often can a wrong answer be fixed by changing the abs token at each position?",
        "",
        "| Position | Wrong | Fixable | Fix Rate |",
        "|----------|-------|---------|----------|",
    ]
    for d in sorted(position_fixes.keys()):
        pf = position_fixes[d]
        pct = pf["any_fix_found"] / max(pf["n_wrong"], 1)
        lines.append(f"| d{d} | {pf['n_wrong']} | {pf['any_fix_found']} | {pct:.1%} |")

    lines += [
        "",
        "## Top Swaps (by fix count)",
        "",
        "| Pos | Old→New | Subtask | Fixed | Broke | Ratio |",
        "|-----|---------|---------|-------|-------|-------|",
    ]
    for sr in swap_results:
        ratio = f"{sr['fixed']}:{sr['broke']}"
        lines.append(
            f"| d{sr['position']} | t{sr['old_token']}→t{sr['new_token']} | "
            f"{sr['old_label']} | {sr['fixed']} | {sr['broke']} | {ratio} |"
        )

    # Example fixes from best swap
    if swap_results and swap_results[0]["examples"]:
        best = swap_results[0]
        lines += [
            "",
            f"## Example Fixes (best swap: d{best['position']} t{best['old_token']}→t{best['new_token']})",
            "",
        ]
        for fx in best["examples"][:5]:
            lines.append(f"- **{fx['split']}:** `{fx['q']}` = `{fx['target']}`")
            lines.append(f"  - Wrong: `{fx['pred_wrong']}` → Fixed: `{fx['pred_fixed']}`")

    # Verdict
    lines += ["", "## Verdict", ""]
    good_swaps = [sr for sr in swap_results if sr["fixed"] >= 3 and sr["fixed"] > sr["broke"] * 2]
    if good_swaps:
        lines.append(f"**{len(good_swaps)} swaps with fix:break > 2:1 and >= 3 fixes.**")
        lines.append("")
        lines.append("Token identity causally determines accuracy on hard cascade examples.")
        lines.append("Specific tokens encode specific cascade computation — replacing them")
        lines.append("with the right alternative surgically fixes wrong answers.")
    else:
        lines.append("No swaps found with strong fix:break ratio. Token swaps at single")
        lines.append("positions may not be sufficient to fix hard cascade errors.")

    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
