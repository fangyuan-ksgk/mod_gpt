"""
Experiment 10: Blanket vs Surgical Token Swap

The strongest test of token semantics: find a pair (A, B) where:
  - Blanket A→B lowers global accuracy (B is wrong in A's contexts)
  - Blanket B→A lowers global accuracy (A is wrong in B's contexts)
  - Surgical A→B on wrong examples fixes some (model had wrong token)
  - Surgical B→A on wrong examples fixes some (model had wrong token)

This proves the tokens encode DISTINCT information, and the model sometimes
misassigns them. It rules out vocab redundancy (where blanket swap would be
harmless) and vocab noise (where surgical fixes wouldn't work).

Protocol:
  1. For every pair of tokens with n >= 50, run blanket swap in both directions
  2. Find pairs where BOTH directions lower accuracy
  3. For those pairs, run surgical fixes on wrong examples
  4. Report the pair with strongest evidence

Outputs:
  results.json    — all pair results
  summary.md      — human-readable report

Usage:
    python experiments/10_blanket_swap/run.py \
        --model add_sub_sorl_v1_abs30_K1_100K_2L1H128d --device cuda:1
"""
import argparse
import json
import sys
import torch
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from arithmetic.hub import load_model
from arithmetic.datasets.addition import get_eval_set
from arithmetic.train import QWEN3_TOKEN_MAP
from sorl.sorl_trainer import infer_insert_mask, insert_tokens_with_padding, expand_prompt_len
from transformers import AutoTokenizer

PROMPT_LEN = 14
ANSWER_LEN = 7
OUT_DIR = Path(__file__).parent
DEFAULT_MODEL = "add_sub_sorl_v1_abs30_K1_100K_2L1H128d"


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
    return data[0].clone()


def check_accuracy(model, expanded, base_v):
    with torch.no_grad():
        logits = model(expanded.unsqueeze(0),
                       attention_mask=torch.ones(1, len(expanded), device=expanded.device)).logits
    is_traj = expanded < base_v
    traj_indices = is_traj.nonzero(as_tuple=True)[0]
    for d in range(ANSWER_LEN):
        traj_pos = PROMPT_LEN + d
        if traj_pos < len(traj_indices):
            exp_idx = traj_indices[traj_pos].item()
            if exp_idx > 0:
                if logits[0, exp_idx - 1, :base_v].argmax().item() != expanded[exp_idx].item():
                    return False
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--min_count", type=int, default=50,
                   help="Minimum token frequency to consider")
    args = p.parse_args()

    print(f"Loading {args.model}...")
    model, config, _ = load_model(args.model, device=args.device)
    model.eval()

    base_v = model.vocab_sizes[0].item()
    K = config.get("K", 1)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    pad_id = tokenizer.pad_token_id

    categories = get_eval_set()
    all_examples = [ex for split_exs in categories.values() for ex in split_exs]

    # ── Phase 1: Run recursion on all examples, cache results ─────────
    # Cache (expanded, is_correct, abs_token_set) per example to avoid
    # rerunning recursion for every swap pair.

    print(f"\nPhase 1: Caching recursion results for {len(all_examples)} examples...")
    cached = []
    token_freq = Counter()

    for i, ex in enumerate(all_examples):
        qwen_ids = torch.tensor([QWEN3_TOKEN_MAP[t] for t in ex.tokens],
                                dtype=torch.long, device=args.device)
        expanded = run_recursion(model, qwen_ids, K, base_v, pad_id, args.device)
        ok = check_accuracy(model, expanded, base_v)

        is_abs = expanded >= base_v
        abs_toks = set((expanded[is_abs] - base_v).tolist())
        for t in abs_toks:
            token_freq[t] += 1

        cached.append({
            "expanded": expanded,
            "correct": ok,
            "abs_toks": abs_toks,
            "is_abs_mask": is_abs,
        })

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(all_examples)}...")

    n_correct = sum(1 for c in cached if c["correct"])
    print(f"  Baseline: {n_correct}/{len(cached)} correct ({n_correct*100//len(cached)}%)")

    # Filter to tokens with enough frequency
    frequent_tokens = sorted([t for t, n in token_freq.items() if n >= args.min_count])
    print(f"  Tokens with n >= {args.min_count}: {len(frequent_tokens)} — {frequent_tokens}")

    # ── Phase 2: Blanket swap for all pairs ───────────────────────────
    # For efficiency: only test pairs, not all permutations.
    # For each pair (A, B), test A→B and B→A.

    print(f"\nPhase 2: Testing {len(frequent_tokens) * (len(frequent_tokens)-1) // 2} pairs...")

    pair_results = []

    for i, tok_a in enumerate(frequent_tokens):
        for tok_b in frequent_tokens[i+1:]:
            # Test A→B
            ab_fixed, ab_broke = 0, 0
            ba_fixed, ba_broke = 0, 0

            for c in cached:
                expanded = c["expanded"]
                is_abs = c["is_abs_mask"]
                ok_orig = c["correct"]

                # A→B: replace all tok_a with tok_b
                if tok_a in c["abs_toks"]:
                    swapped = expanded.clone()
                    mask = is_abs & (expanded - base_v == tok_a)
                    swapped[mask] = base_v + tok_b
                    ok_swap = check_accuracy(model, swapped, base_v)

                    if ok_orig and not ok_swap:
                        ab_broke += 1
                    elif not ok_orig and ok_swap:
                        ab_fixed += 1

                # B→A: replace all tok_b with tok_a
                if tok_b in c["abs_toks"]:
                    swapped = expanded.clone()
                    mask = is_abs & (expanded - base_v == tok_b)
                    swapped[mask] = base_v + tok_a
                    ok_swap = check_accuracy(model, swapped, base_v)

                    if ok_orig and not ok_swap:
                        ba_broke += 1
                    elif not ok_orig and ok_swap:
                        ba_fixed += 1

            pair_results.append({
                "tok_a": tok_a,
                "tok_b": tok_b,
                "freq_a": token_freq[tok_a],
                "freq_b": token_freq[tok_b],
                # A→B blanket
                "ab_fixed": ab_fixed,
                "ab_broke": ab_broke,
                "ab_net": ab_fixed - ab_broke,
                # B→A blanket
                "ba_fixed": ba_fixed,
                "ba_broke": ba_broke,
                "ba_net": ba_fixed - ba_broke,
                # Key test: both directions hurt globally?
                "both_hurt": ab_broke > ab_fixed and ba_broke > ba_fixed,
                # But surgical fixes exist in both directions?
                "both_fix": ab_fixed > 0 and ba_fixed > 0,
                # The ideal pair: both hurt AND both fix
                "ideal": (ab_broke > ab_fixed and ba_broke > ba_fixed
                          and ab_fixed > 0 and ba_fixed > 0),
            })

    # Sort by "ideal" signal strength
    ideal_pairs = [p for p in pair_results if p["ideal"]]
    ideal_pairs.sort(key=lambda x: min(x["ab_fixed"], x["ba_fixed"]), reverse=True)

    # Also sort all pairs by mutual destructiveness
    pair_results.sort(key=lambda x: (x["both_hurt"], x["both_fix"],
                                      min(x["ab_broke"], x["ba_broke"])),
                      reverse=True)

    # ── Print results ─────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print(f"IDEAL PAIRS (both directions hurt globally, both fix surgically):")
    print(f"{'='*70}")

    if not ideal_pairs:
        print("  None found.")
    else:
        print(f"  {'A':>4s} {'B':>4s} | {'A→B fixed':>10s} {'A→B broke':>10s} {'net':>5s} | {'B→A fixed':>10s} {'B→A broke':>10s} {'net':>5s}")
        print(f"  {'-'*70}")
        for pr in ideal_pairs[:20]:
            print(f"  t{pr['tok_a']:>2d} t{pr['tok_b']:>2d} | "
                  f"{pr['ab_fixed']:>10d} {pr['ab_broke']:>10d} {pr['ab_net']:>5d} | "
                  f"{pr['ba_fixed']:>10d} {pr['ba_broke']:>10d} {pr['ba_net']:>5d}")

    print(f"\nAll pairs with both_hurt: {sum(1 for p in pair_results if p['both_hurt'])}")
    print(f"All pairs with both_fix: {sum(1 for p in pair_results if p['both_fix'])}")
    print(f"Ideal pairs (both): {len(ideal_pairs)}")

    # Top 10 by any metric
    print(f"\nTop 10 most mutually destructive pairs:")
    print(f"  {'A':>4s} {'B':>4s} | {'A→B':>12s} | {'B→A':>12s} | {'Ideal':>6s}")
    print(f"  {'-'*50}")
    for pr in pair_results[:10]:
        ab = f"+{pr['ab_fixed']}/-{pr['ab_broke']}"
        ba = f"+{pr['ba_fixed']}/-{pr['ba_broke']}"
        print(f"  t{pr['tok_a']:>2d} t{pr['tok_b']:>2d} | {ab:>12s} | {ba:>12s} | {'YES' if pr['ideal'] else 'no':>6s}")

    # ── Save ──────────────────────────────────────────────────────────

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "model": args.model,
        "K": K,
        "n_examples": len(cached),
        "n_correct": n_correct,
        "frequent_tokens": frequent_tokens,
        "n_pairs_tested": len(pair_results),
        "n_both_hurt": sum(1 for p in pair_results if p["both_hurt"]),
        "n_both_fix": sum(1 for p in pair_results if p["both_fix"]),
        "n_ideal": len(ideal_pairs),
        "ideal_pairs": ideal_pairs[:20],
        "top_pairs": pair_results[:30],
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'results.json'}")

    # ── Summary markdown ──────────────────────────────────────────────

    lines = [
        "# Blanket vs Surgical Token Swap",
        "",
        f"**Model:** `{args.model}` (K={K})",
        f"**Baseline accuracy:** {n_correct}/{len(cached)} ({n_correct*100//len(cached)}%)",
        f"**Tokens tested:** {len(frequent_tokens)} (freq >= {args.min_count})",
        f"**Pairs tested:** {len(pair_results)}",
        "",
        "## The Test",
        "",
        "For a pair (A, B) to prove tokens encode distinct information:",
        "1. Blanket A→B must **lower** accuracy (B is wrong in A's natural contexts)",
        "2. Blanket B→A must **lower** accuracy (A is wrong in B's natural contexts)",
        "3. Surgical A→B on wrong examples must **fix** some (model assigned wrong token)",
        "4. Surgical B→A on wrong examples must **fix** some (same, other direction)",
        "",
        "If all four hold: the tokens are genuinely different, and the model sometimes",
        "picks the wrong one. This rules out vocab redundancy AND vocab noise.",
        "",
        f"## Results: {len(ideal_pairs)} ideal pairs found",
        "",
    ]

    if ideal_pairs:
        lines += [
            "| Token A | Token B | A→B Fixed | A→B Broke | B→A Fixed | B→A Broke |",
            "|---------|---------|-----------|-----------|-----------|-----------|",
        ]
        for pr in ideal_pairs[:15]:
            lines.append(
                f"| t{pr['tok_a']} (n={pr['freq_a']}) | t{pr['tok_b']} (n={pr['freq_b']}) | "
                f"{pr['ab_fixed']} | {pr['ab_broke']} | {pr['ba_fixed']} | {pr['ba_broke']} |"
            )

        best = ideal_pairs[0]
        lines += [
            "",
            f"**Best pair: t{best['tok_a']} ↔ t{best['tok_b']}**",
            "",
            f"- Blanket t{best['tok_a']}→t{best['tok_b']}: "
            f"fixes {best['ab_fixed']}, breaks {best['ab_broke']} (net {best['ab_net']:+d})",
            f"- Blanket t{best['tok_b']}→t{best['tok_a']}: "
            f"fixes {best['ba_fixed']}, breaks {best['ba_broke']} (net {best['ba_net']:+d})",
            "",
            "Both directions are net-negative globally, but both fix some wrong examples.",
            "The tokens encode **distinct, non-redundant information** that the model",
            "sometimes misassigns.",
        ]
    else:
        lines += [
            "No ideal pairs found. This could mean:",
            "- The vocab is mostly redundant (blanket swaps are harmless)",
            "- The model rarely misassigns tokens (surgical fixes are rare)",
            "- Need to test on harder splits or larger eval set",
        ]

    lines += [
        "",
        "## Statistics",
        "",
        f"- Pairs where both directions hurt: **{sum(1 for p in pair_results if p['both_hurt'])}**",
        f"- Pairs where both directions fix: **{sum(1 for p in pair_results if p['both_fix'])}**",
        f"- Ideal pairs (both hurt AND both fix): **{len(ideal_pairs)}**",
    ]

    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
