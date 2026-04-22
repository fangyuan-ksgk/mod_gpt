"""
Experiment 06: Token Swap (Surgical Transplant)

Swaps abstraction token t9 (shallow carry) with t21 (deep cascade) and
measures the causal effect on hard carry chains (C3-C6).

Key finding: replacing t9→t21 fixes ~96 hard cases while only breaking ~2,
confirming that specific tokens encode specific carry computation depths.

Outputs:
  - results.json        — per-split swap results
  - summary.md          — markdown with examples

Usage:
    python experiments/06_token_swap/run.py [--model MODEL] [--device cuda:0]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from arithmetic.interp_utils.token_swap import run_swap_experiment, print_results

OUT_DIR = Path(__file__).parent
DEFAULT_MODEL = "add_sub_sorl_v1_abs30_K1_100K"


def write_summary(results, swap_from, swap_to, model_name, path):
    total = results["total"]
    lines = [
        "# Token Swap: Surgical Transplant",
        "",
        f"**Model:** `{model_name}`",
        f"**Swap:** t{swap_from} → t{swap_to}",
        f"**Splits:** C3-C6 (hard carry chains)",
        "",
        "## Results",
        "",
        f"- **Fixed:** {total['fixed']} examples (wrong → correct after swap)",
        f"- **Broke:** {total['broke']} examples (correct → wrong after swap)",
        f"- **Ratio:** {total['fixed']}:{total['broke']}",
        "",
        "| Split | N | Normal | Swap | Fixed | Broke |",
        "|-------|---|--------|------|-------|-------|",
    ]

    for split in sorted(results.keys()):
        if split == "total":
            continue
        r = results[split]
        n = r["n"]
        na = r["normal_correct"] * 100 // max(n, 1)
        sa = r["swap_correct"] * 100 // max(n, 1)
        lines.append(f"| {split} | {n} | {na}% | {sa}% | {r['fixed']} | {r['broke']} |")

    # Example fixes
    all_fixes = []
    for split, r in results.items():
        if split == "total":
            continue
        for ex in r.get("fix_examples", [])[:3]:
            all_fixes.append((split, ex))

    if all_fixes:
        lines += ["", "## Example Fixes", ""]
        for split, ex in all_fixes[:5]:
            lines.append(f"- **{split}:** `{ex['q']}` = `{ex['target']}`")
            lines.append(f"  - Normal: `{ex['normal']}` (wrong)")
            lines.append(f"  - Swapped: `{ex['swapped']}` (correct, {ex['n_swaps']} tokens swapped)")

    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--swap_from", type=int, default=9)
    p.add_argument("--swap_to", type=int, default=21)
    p.add_argument("--splits", default="add_C3,add_C4,add_C5,add_C6")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--K", type=int, default=1)
    args = p.parse_args()

    splits = args.splits.split(",")
    print(f"Running t{args.swap_from}→t{args.swap_to} swap on {args.model}...")
    results = run_swap_experiment(
        args.model, args.swap_from, args.swap_to, splits,
        device=args.device, K=args.K,
    )

    print_results(results, args.swap_from, args.swap_to)

    with open(OUT_DIR / "results.json", "w") as f:
        # Strip non-serializable fix_examples
        clean = {}
        for k, v in results.items():
            if isinstance(v, dict):
                clean[k] = {kk: vv for kk, vv in v.items() if kk != "fix_examples" or isinstance(vv, list)}
            else:
                clean[k] = v
        json.dump(clean, f, indent=2)
    print(f"Wrote {OUT_DIR / 'results.json'}")

    write_summary(results, args.swap_from, args.swap_to, args.model, OUT_DIR / "summary.md")


if __name__ == "__main__":
    main()
