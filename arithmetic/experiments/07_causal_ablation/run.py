"""
Experiment 07: Causal Ablation

Tests whether abstraction tokens causally encode carry/borrow information
by applying interventions and measuring accuracy drops:

1. Knockout: replace all abs tokens with a placeholder
2. Shuffle: permute abs token identities within each sequence
3. Random: replace with random abs tokens

Parallels Quirke's activation patching (their Section 4) but operates on
explicit tokens instead of hidden activations.

Outputs:
  - results.json           — per-split accuracy under each intervention
  - fig_causal_ablation.png — bar chart of accuracy by intervention
  - summary.md

Usage:
    python experiments/07_causal_ablation/run.py [--model MODEL] [--device cuda:0]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from arithmetic.scripts.run_causal_verification import run_causal_verification

OUT_DIR = Path(__file__).parent
DEFAULT_MODEL = "add_sub_sorl_v1_abs30_K1_100K"


def plot_causal(results, path):
    """Bar chart of accuracy under each intervention."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = ["baseline", "knockout", "shuffle", "random"]
    colors = ["#2a9d8f", "#e63946", "#457b9d", "#e9c46a"]

    # Overall
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    accs = [results[l].get("accuracy", results[l]["correct"] / max(results[l]["total"], 1))
            for l in labels]
    bars = ax.bar(labels, accs, color=colors)
    ax.set_ylabel("Accuracy")
    ax.set_title("Overall Accuracy by Intervention")
    ax.set_ylim(0, 1.05)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{acc:.0%}", ha="center", fontsize=10)

    # Per-split
    ax = axes[1]
    per_split = results.get("per_split", {})
    splits = sorted([s for s in per_split.keys() if s.startswith("add_C")])
    if splits:
        x = np.arange(len(splits))
        width = 0.2
        for i, (label, color) in enumerate(zip(labels, colors)):
            vals = [per_split[s][label].get("accuracy", 0) for s in splits]
            ax.bar(x + i * width, vals, width, label=label, color=color)
        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels(splits, rotation=45, ha="right")
        ax.set_ylabel("Accuracy")
        ax.set_title("Hard Carry Splits by Intervention")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {path}")


def write_summary(results, model_name, path):
    base_acc = results["baseline"].get("accuracy",
        results["baseline"]["correct"] / max(results["baseline"]["total"], 1))
    lines = [
        "# Causal Ablation",
        "",
        f"**Model:** `{model_name}`",
        f"**Eval set:** canonical N=100 from HuggingFace",
        "",
        "## Overall",
        "",
        "| Intervention | Accuracy | Delta |",
        "|--------------|----------|-------|",
    ]
    for label in ["baseline", "knockout", "shuffle", "random"]:
        acc = results[label].get("accuracy",
            results[label]["correct"] / max(results[label]["total"], 1))
        delta = acc - base_acc
        lines.append(f"| {label} | {acc:.1%} | {delta:+.1%} |")

    lines += [
        "",
        "![Causal Ablation](fig_causal_ablation.png)",
        "",
        "## Interpretation",
        "",
        "If abstraction tokens are causally encoding carry information:",
        "- Knockout should drop accuracy (model loses carry signal)",
        "- Shuffle should drop accuracy (wrong carry info at each position)",
        "- Random should drop accuracy (random noise instead of signal)",
    ]
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    print(f"Running causal verification on {args.model}...")
    results = run_causal_verification(args.model, args.K, args.device)

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {OUT_DIR / 'results.json'}")

    plot_causal(results, OUT_DIR / "fig_causal_ablation.png")
    write_summary(results, args.model, OUT_DIR / "summary.md")


if __name__ == "__main__":
    main()
