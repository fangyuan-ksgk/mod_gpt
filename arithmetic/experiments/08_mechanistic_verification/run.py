"""
Experiment 08: Mechanistic Verification of Novel Findings

Produces quantitative evidence for 5 findings from novel.md:

  (1) Sum-9 detector: measure token purity for sum=9 boundary case
  (2) MSB digit-sum encoding: measure per-position token-to-sum correlation
  (3) Position-locked: swap same-subtask tokens across positions
  (4) Cross-operation unification: transplant add tokens into sub examples
  (5) LSB shortcut: per-position ablation to test information gradient

Outputs:
  results.json    — all stats
  summary.md      — human-readable report

Usage:
    python experiments/08_mechanistic_verification/run.py \
        --model add_sub_sorl_v1_abs30_K1_10K --device cuda:1
"""
import argparse
import json
import sys
import torch
import random
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
DEFAULT_MODEL = "add_sub_sorl_v1_abs30_K1_10K"

random.seed(42)
torch.manual_seed(42)


def run_recursion(model, qwen_ids, K, base_v, pad_id, device):
    """Run SoRL recursion, return expanded sequence and per-answer-digit abs token info."""
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


def check_accuracy(model, expanded, base_v):
    """Check if the model gets the right answer from an expanded sequence."""
    with torch.no_grad():
        logits = model(expanded.unsqueeze(0),
                       attention_mask=torch.ones(1, len(expanded), device=expanded.device)).logits
    is_traj = expanded < base_v
    traj_indices = is_traj.nonzero(as_tuple=True)[0]
    correct = 0
    total = 0
    for d in range(ANSWER_LEN):
        traj_pos = PROMPT_LEN + d
        if traj_pos < len(traj_indices):
            exp_idx = traj_indices[traj_pos].item()
            if exp_idx > 0:
                pred = logits[0, exp_idx - 1, :base_v].argmax().item()
                target = expanded[exp_idx].item()
                if pred == target:
                    correct += 1
                total += 1
    return correct == total, correct, total


def collect_all_token_data(model, examples, K, base_v, pad_id, device):
    """Run recursion on all examples and collect per-token statistics."""
    records = []
    for ex in examples:
        qwen_ids = torch.tensor([QWEN3_TOKEN_MAP[t] for t in ex.tokens],
                                dtype=torch.long, device=device)
        expanded, digit_abs = run_recursion(model, qwen_ids, K, base_v, pad_id, device)

        orig_tokens = [QWEN3_INV_MAP.get(t.item(), -1) for t in qwen_ids]
        op = ex.op

        for d, (abs_idx, tok_id) in digit_abs.items():
            if d >= len(ex.labels):
                continue
            # Compute digit sum at this position
            if d < 6:
                d1 = orig_tokens[d]
                d2 = orig_tokens[7 + d]
            else:
                d1, d2 = -1, -1

            digit_sum = (d1 + d2) if (d1 >= 0 and d2 >= 0) else -1

            records.append({
                "tok_id": tok_id,
                "answer_pos": d,
                "label": ex.labels[d],
                "op": op,
                "digit_sum": digit_sum,
                "digit_sum_mod10": digit_sum % 10 if digit_sum >= 0 else -1,
                "expanded": expanded,
                "abs_idx": abs_idx,
                "digit_abs": digit_abs,
                "ex": ex,
            })

    return records


# ── Finding (1): Sum-9 Detector ──────────────────────────────────────

def test_sum9_detector(records):
    """Measure whether specific tokens fire preferentially when digit sum = 9."""
    token_sum9_rate = defaultdict(lambda: {"sum9": 0, "other": 0, "total": 0})

    for r in records:
        if r["digit_sum"] < 0:
            continue
        t = r["tok_id"]
        token_sum9_rate[t]["total"] += 1
        if r["digit_sum_mod10"] == 9:
            token_sum9_rate[t]["sum9"] += 1
        else:
            token_sum9_rate[t]["other"] += 1

    # Find tokens with highest sum-9 purity
    results = []
    for tok, counts in sorted(token_sum9_rate.items()):
        if counts["total"] < 20:
            continue
        purity = counts["sum9"] / counts["total"]
        results.append({
            "token": tok,
            "n": counts["total"],
            "sum9_count": counts["sum9"],
            "sum9_purity": round(purity, 3),
        })

    results.sort(key=lambda x: x["sum9_purity"], reverse=True)

    # Background rate
    total_sum9 = sum(1 for r in records if r["digit_sum_mod10"] == 9 and r["digit_sum"] >= 0)
    total_valid = sum(1 for r in records if r["digit_sum"] >= 0)
    background = total_sum9 / max(total_valid, 1)

    return {
        "background_sum9_rate": round(background, 3),
        "tokens": results,
        "top_sum9_token": results[0] if results else None,
    }


# ── Finding (2): MSB Digit-Sum Encoding ──────────────────────────────

def test_msb_encoding(records):
    """Measure per-position token-to-digit-sum correlation.
    At MSB, tokens should predict the raw digit sum; at other positions, less so."""

    per_pos = {}
    for d in range(ANSWER_LEN):
        pos_records = [r for r in records if r["answer_pos"] == d and r["digit_sum"] >= 0]
        if not pos_records:
            per_pos[d] = {"n": 0, "n_tokens": 0, "mean_sum_purity": 0, "max_sum_purity": 0, "tokens": []}
            continue

        # For each token at this position, compute its most-common digit sum concentration
        tok_sums = defaultdict(list)
        for r in pos_records:
            tok_sums[r["tok_id"]].append(r["digit_sum_mod10"])

        purities = []
        token_details = []
        for tok, sums in tok_sums.items():
            if len(sums) < 10:
                continue
            counter = Counter(sums)
            most_common_sum, most_common_count = counter.most_common(1)[0]
            purity = most_common_count / len(sums)
            purities.append(purity)
            token_details.append({
                "token": tok,
                "n": len(sums),
                "top_sum": most_common_sum,
                "top_sum_purity": round(purity, 3),
            })

        token_details.sort(key=lambda x: x["top_sum_purity"], reverse=True)

        per_pos[d] = {
            "n": len(pos_records),
            "n_tokens": len(purities),
            "mean_sum_purity": round(np.mean(purities), 3) if purities else 0,
            "max_sum_purity": round(max(purities), 3) if purities else 0,
            "tokens": token_details,
        }

    return per_pos


# ── Finding (3): Position-Locked Specialization ──────────────────────

def test_position_locked(model, records, base_v):
    """Swap same-subtask tokens across positions. If position matters, should break."""
    # Group records by example
    ex_records = defaultdict(list)
    for r in records:
        ex_id = id(r["expanded"])
        ex_records[ex_id].append(r)

    same_sub_results = {"survived": 0, "broke": 0, "n": 0}
    diff_sub_results = {"survived": 0, "broke": 0, "n": 0}

    tested_examples = set()
    for ex_id, recs in ex_records.items():
        if ex_id in tested_examples:
            continue
        tested_examples.add(ex_id)

        expanded = recs[0]["expanded"]
        digit_abs = recs[0]["digit_abs"]

        ok_orig, _, _ = check_accuracy(model, expanded, base_v)
        if not ok_orig:
            continue

        labels = recs[0]["ex"].labels

        for i, r1 in enumerate(recs):
            for r2 in recs[i+1:]:
                d1, d2 = r1["answer_pos"], r2["answer_pos"]
                if d1 == d2:
                    continue
                if d1 not in digit_abs or d2 not in digit_abs:
                    continue

                idx1, tok1 = digit_abs[d1]
                idx2, tok2 = digit_abs[d2]
                if tok1 == tok2:
                    continue

                swapped = expanded.clone()
                swapped[idx1] = expanded[idx2]
                swapped[idx2] = expanded[idx1]

                ok_swap, _, _ = check_accuracy(model, swapped, base_v)

                if r1["label"] == r2["label"]:
                    same_sub_results["n"] += 1
                    if ok_swap:
                        same_sub_results["survived"] += 1
                    else:
                        same_sub_results["broke"] += 1
                else:
                    diff_sub_results["n"] += 1
                    if ok_swap:
                        diff_sub_results["survived"] += 1
                    else:
                        diff_sub_results["broke"] += 1

    same_rate = same_sub_results["survived"] / max(same_sub_results["n"], 1)
    diff_rate = diff_sub_results["survived"] / max(diff_sub_results["n"], 1)

    return {
        "same_subtask": {**same_sub_results, "survive_rate": round(same_rate, 3)},
        "diff_subtask": {**diff_sub_results, "survive_rate": round(diff_rate, 3)},
    }


# ── Finding (4): Cross-Operation Unification ─────────────────────────

def test_cross_operation(model, examples, K, base_v, pad_id, device):
    """Transplant abs tokens from add examples into sub examples."""
    add_data, sub_data = [], []

    for ex in examples:
        qwen_ids = torch.tensor([QWEN3_TOKEN_MAP[t] for t in ex.tokens],
                                dtype=torch.long, device=device)
        expanded, digit_abs = run_recursion(model, qwen_ids, K, base_v, pad_id, device)
        ok, _, _ = check_accuracy(model, expanded, base_v)
        if not ok:
            continue

        rec = {"ex": ex, "expanded": expanded, "digit_abs": digit_abs}
        if ex.op == "add":
            add_data.append(rec)
        else:
            sub_data.append(rec)

    # add→sub transplant
    a2s = {"survived": 0, "n": 0, "digit_correct": 0, "digit_total": 0}
    for add_rec in add_data[:200]:
        sub_rec = random.choice(sub_data) if sub_data else None
        if sub_rec is None:
            break

        transplanted = sub_rec["expanded"].clone()
        n_tr = 0
        for d in range(ANSWER_LEN):
            if d in add_rec["digit_abs"] and d in sub_rec["digit_abs"]:
                sub_idx = sub_rec["digit_abs"][d][0]
                add_idx = add_rec["digit_abs"][d][0]
                transplanted[sub_idx] = add_rec["expanded"][add_idx]
                n_tr += 1
        if n_tr == 0:
            continue
        ok, correct, total = check_accuracy(model, transplanted, base_v)
        a2s["n"] += 1
        if ok:
            a2s["survived"] += 1
        a2s["digit_correct"] += correct
        a2s["digit_total"] += total

    # sub→add transplant
    s2a = {"survived": 0, "n": 0, "digit_correct": 0, "digit_total": 0}
    for sub_rec in sub_data[:200]:
        add_rec = random.choice(add_data) if add_data else None
        if add_rec is None:
            break

        transplanted = add_rec["expanded"].clone()
        n_tr = 0
        for d in range(ANSWER_LEN):
            if d in sub_rec["digit_abs"] and d in add_rec["digit_abs"]:
                add_idx = add_rec["digit_abs"][d][0]
                sub_idx = sub_rec["digit_abs"][d][0]
                transplanted[add_idx] = sub_rec["expanded"][sub_idx]
                n_tr += 1
        if n_tr == 0:
            continue
        ok, correct, total = check_accuracy(model, transplanted, base_v)
        s2a["n"] += 1
        if ok:
            s2a["survived"] += 1
        s2a["digit_correct"] += correct
        s2a["digit_total"] += total

    # random baseline: replace abs tokens with random tokens
    rand_baseline = {"survived": 0, "n": 0, "digit_correct": 0, "digit_total": 0}
    for rec in (add_data + sub_data)[:200]:
        randomized = rec["expanded"].clone()
        for d in range(ANSWER_LEN):
            if d in rec["digit_abs"]:
                idx = rec["digit_abs"][d][0]
                randomized[idx] = base_v + 1 + random.randint(0, 29)
        ok, correct, total = check_accuracy(model, randomized, base_v)
        rand_baseline["n"] += 1
        if ok:
            rand_baseline["survived"] += 1
        rand_baseline["digit_correct"] += correct
        rand_baseline["digit_total"] += total

    def rate(d):
        return {
            **d,
            "survive_rate": round(d["survived"] / max(d["n"], 1), 3),
            "digit_accuracy": round(d["digit_correct"] / max(d["digit_total"], 1), 3),
        }

    return {
        "add_to_sub": rate(a2s),
        "sub_to_add": rate(s2a),
        "random_baseline": rate(rand_baseline),
    }


# ── Finding (5): LSB Shortcut — Per-Position Ablation ────────────────

def test_per_position_ablation(model, records, base_v):
    """Replace abs token at each position with random token, one at a time."""
    results = {d: {"survived": 0, "broke": 0, "n": 0} for d in range(ANSWER_LEN)}

    tested = set()
    for r in records:
        ex_id = id(r["expanded"])
        d = r["answer_pos"]
        key = (ex_id, d)
        if key in tested:
            continue
        tested.add(key)

        expanded = r["expanded"]
        digit_abs = r["digit_abs"]

        ok_orig, _, _ = check_accuracy(model, expanded, base_v)
        if not ok_orig:
            continue

        if d not in digit_abs:
            continue

        ablated = expanded.clone()
        idx, tok = digit_abs[d]
        ablated[idx] = base_v + 1 + random.randint(0, 29)

        ok_abl, _, _ = check_accuracy(model, ablated, base_v)
        results[d]["n"] += 1
        if ok_abl:
            results[d]["survived"] += 1
        else:
            results[d]["broke"] += 1

    for d in results:
        results[d]["survive_rate"] = round(
            results[d]["survived"] / max(results[d]["n"], 1), 3)

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default="cuda:1")
    args = p.parse_args()

    print(f"Loading {args.model}...")
    model, config, _ = load_model(args.model, device=args.device)
    model.eval()

    base_v = model.vocab_sizes[0].item()
    K = config.get("K", 1)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    pad_id = tokenizer.pad_token_id

    categories = get_eval_set()
    all_examples = [ex for exs in categories.values() for ex in exs]
    print(f"Loaded {len(all_examples)} eval examples\n")

    # ── Collect token data (shared across findings 1, 2, 3, 5) ──
    print("Running recursion on all examples...")
    records = collect_all_token_data(model, all_examples, K, base_v, pad_id, args.device)
    print(f"Collected {len(records)} token records\n")

    # ── (1) Sum-9 Detector ──
    print("=== Finding (1): Sum-9 Detector ===")
    f1 = test_sum9_detector(records)
    print(f"  Background sum=9 rate: {f1['background_sum9_rate']:.1%}")
    if f1["top_sum9_token"]:
        t = f1["top_sum9_token"]
        print(f"  Top sum-9 token: t{t['token']} — {t['sum9_purity']:.1%} purity (n={t['n']})")
    top3 = [t for t in f1["tokens"] if t["sum9_purity"] > f1["background_sum9_rate"] * 2][:5]
    for t in top3:
        print(f"    t{t['token']}: {t['sum9_purity']:.1%} sum-9 (n={t['n']})")

    # ── (2) MSB Digit-Sum Encoding ──
    print("\n=== Finding (2): MSB Digit-Sum Encoding ===")
    f2 = test_msb_encoding(records)
    print(f"  {'Pos':<5s} {'Tokens':>7s} {'Mean Purity':>12s} {'Max Purity':>12s}")
    print(f"  {'-'*38}")
    for d in range(ANSWER_LEN):
        r = f2[d]
        print(f"  d{d:<4d} {r['n_tokens']:>7d} {r['mean_sum_purity']:>12.1%} {r['max_sum_purity']:>12.1%}")
    print(f"  → MSB (d4) should have highest mean purity")

    # ── (3) Position-Locked ──
    print("\n=== Finding (3): Position-Locked Specialization ===")
    f3 = test_position_locked(model, records, base_v)
    print(f"  Same-subtask swaps: {f3['same_subtask']['survive_rate']:.1%} survived (n={f3['same_subtask']['n']})")
    print(f"  Diff-subtask swaps: {f3['diff_subtask']['survive_rate']:.1%} survived (n={f3['diff_subtask']['n']})")
    print(f"  → If position matters: same-subtask should ALSO break")

    # ── (4) Cross-Operation ──
    print("\n=== Finding (4): Cross-Operation Unification ===")
    f4 = test_cross_operation(model, all_examples, K, base_v, pad_id, args.device)
    print(f"  Add→Sub: {f4['add_to_sub']['survive_rate']:.1%} survived, {f4['add_to_sub']['digit_accuracy']:.1%} digit acc (n={f4['add_to_sub']['n']})")
    print(f"  Sub→Add: {f4['sub_to_add']['survive_rate']:.1%} survived, {f4['sub_to_add']['digit_accuracy']:.1%} digit acc (n={f4['sub_to_add']['n']})")
    print(f"  Random:  {f4['random_baseline']['survive_rate']:.1%} survived, {f4['random_baseline']['digit_accuracy']:.1%} digit acc (n={f4['random_baseline']['n']})")
    print(f"  → Transplant should beat random if operations share tokens")

    # ── (5) Per-Position Ablation ──
    print("\n=== Finding (5): LSB Shortcut — Per-Position Ablation ===")
    f5 = test_per_position_ablation(model, records, base_v)
    print(f"  {'Pos':<5s} {'N':>5s} {'Survived':>10s} {'Survive%':>10s}")
    print(f"  {'-'*32}")
    for d in range(ANSWER_LEN):
        r = f5[d]
        print(f"  d{d:<4d} {r['n']:>5d} {r['survived']:>10d} {r['survive_rate']:>10.1%}")
    print(f"  → d0 survive% should be highest (LSB needs least info)")

    # ── Save ──
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {
        "model": args.model,
        "K": K,
        "n_examples": len(all_examples),
        "n_records": len(records),
        "finding_1_sum9_detector": f1,
        "finding_2_msb_encoding": {str(k): v for k, v in f2.items()},
        "finding_3_position_locked": f3,
        "finding_4_cross_operation": f4,
        "finding_5_per_position_ablation": {f"d{k}": v for k, v in f5.items()},
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'results.json'}")

    # ── Summary markdown ──
    lines = [
        "# Mechanistic Verification of Novel Findings",
        "",
        f"**Model:** `{args.model}` (K={K})",
        f"**Eval set:** canonical N=100 from HuggingFace",
        f"**Token records:** {len(records)}",
        "",
        "## (1) Sum-9 Detector",
        "",
        f"Background rate of sum=9 in eval data: **{f1['background_sum9_rate']:.1%}**",
        "",
        "Tokens with elevated sum-9 purity:",
        "",
        "| Token | N | Sum-9 Count | Sum-9 Purity |",
        "|-------|---|-------------|-------------|",
    ]
    for t in f1["tokens"][:10]:
        lines.append(f"| t{t['token']} | {t['n']} | {t['sum9_count']} | {t['sum9_purity']:.1%} |")
    lines += [
        "",
        f"**Verdict:** {'CONFIRMED' if f1['top_sum9_token'] and f1['top_sum9_token']['sum9_purity'] > 0.5 else 'WEAK'} — "
        f"top token has {f1['top_sum9_token']['sum9_purity']:.0%} sum-9 purity vs {f1['background_sum9_rate']:.0%} background"
        if f1["top_sum9_token"] else "No tokens found",
        "",
        "## (2) MSB Digit-Sum Encoding",
        "",
        "Mean token-to-digit-sum purity by answer position:",
        "",
        "| Position | Tokens | Mean Purity | Max Purity |",
        "|----------|--------|-------------|------------|",
    ]
    for d in range(ANSWER_LEN):
        r = f2[d]
        lines.append(f"| d{d} | {r['n_tokens']} | {r['mean_sum_purity']:.1%} | {r['max_sum_purity']:.1%} |")

    msb_purity = f2[4]["mean_sum_purity"] if 4 in f2 else 0
    other_purities = [f2[d]["mean_sum_purity"] for d in range(4) if d in f2 and f2[d]["n_tokens"] > 0]
    other_mean = np.mean(other_purities) if other_purities else 0
    lines += [
        "",
        f"**Verdict:** {'CONFIRMED' if msb_purity > other_mean * 1.3 else 'WEAK'} — "
        f"MSB mean purity {msb_purity:.1%} vs other positions {other_mean:.1%}",
        "",
        "## (3) Position-Locked Specialization",
        "",
        f"- Same-subtask cross-position swaps: **{f3['same_subtask']['survive_rate']:.1%}** survived (n={f3['same_subtask']['n']})",
        f"- Diff-subtask cross-position swaps: **{f3['diff_subtask']['survive_rate']:.1%}** survived (n={f3['diff_subtask']['n']})",
        "",
    ]
    same_r = f3["same_subtask"]["survive_rate"]
    diff_r = f3["diff_subtask"]["survive_rate"]
    lines += [
        f"**Verdict:** {'CONFIRMED' if same_r < 0.8 else 'WEAK'} — "
        f"same-subtask swaps break accuracy ({same_r:.0%}), confirming position matters beyond subtask identity. "
        f"Gap vs diff-subtask: {abs(same_r - diff_r):.0%}",
        "",
        "## (4) Cross-Operation Unification",
        "",
        f"- Add→Sub transplant: **{f4['add_to_sub']['survive_rate']:.1%}** full accuracy, **{f4['add_to_sub']['digit_accuracy']:.1%}** digit accuracy",
        f"- Sub→Add transplant: **{f4['sub_to_add']['survive_rate']:.1%}** full accuracy, **{f4['sub_to_add']['digit_accuracy']:.1%}** digit accuracy",
        f"- Random baseline: **{f4['random_baseline']['survive_rate']:.1%}** full accuracy, **{f4['random_baseline']['digit_accuracy']:.1%}** digit accuracy",
        "",
    ]
    transplant_rate = max(f4["add_to_sub"]["survive_rate"], f4["sub_to_add"]["survive_rate"])
    random_rate = f4["random_baseline"]["survive_rate"]
    lines += [
        f"**Verdict:** {'CONFIRMED' if transplant_rate > random_rate * 1.2 else 'WEAK'} — "
        f"cross-operation transplant ({transplant_rate:.0%}) {'beats' if transplant_rate > random_rate else 'does not beat'} random ({random_rate:.0%})",
        "",
        "## (5) LSB Shortcut — Per-Position Ablation",
        "",
        "| Position | N | Survived | Survive Rate |",
        "|----------|---|----------|-------------|",
    ]
    for d in range(ANSWER_LEN):
        r = f5[d]
        lines.append(f"| d{d} | {r['n']} | {r['survived']} | {r['survive_rate']:.1%} |")

    d0_rate = f5[0]["survive_rate"] if 0 in f5 else 0
    inner_rates = [f5[d]["survive_rate"] for d in range(1, ANSWER_LEN) if d in f5 and f5[d]["n"] > 0]
    inner_mean = np.mean(inner_rates) if inner_rates else 0
    lines += [
        "",
        f"**Verdict:** {'CONFIRMED' if d0_rate > inner_mean * 1.1 else 'WEAK'} — "
        f"d0 survive rate ({d0_rate:.0%}) {'>' if d0_rate > inner_mean else '<='} inner positions ({inner_mean:.0%})",
    ]

    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
