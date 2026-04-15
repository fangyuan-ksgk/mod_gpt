"""
Experiment 02: Vocabulary Scaling

Shows how accuracy changes with abstract vocabulary size (abs_vocab).
Compares abs={1,2,5,10,16,20,30,50,100} across architectures.

Outputs:
  - results.json           — per-model accuracy by vocab size
  - fig_vocab_scaling.png  — accuracy vs vocab size plot
  - summary.md             — markdown summary

Usage:
    python experiments/02_vocab_scaling/run.py
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from arithmetic.catalog import ModelCatalog

OUT_DIR = Path(__file__).parent


def collect_vocab_data(cat: ModelCatalog) -> dict:
    """Collect accuracy by vocab size, grouped by architecture."""
    groups = defaultdict(list)
    for e in cat.valid():
        if e.mode != "sorl":
            continue
        groups[e.arch].append({
            "name": e.name,
            "abs_vocab": e.abs_vocab,
            "K": e.K,
            "dataset_size": e.dataset_size,
            "overall_accuracy": e.accuracy,
            "sft_accuracy": e.sft_accuracy,
            "c4": e.split_accuracy("sorl_eval", "add_C4"),
            "c5": e.split_accuracy("sorl_eval", "add_C5"),
            "c6": e.split_accuracy("sorl_eval", "add_C6"),
        })

    # Also get baselines for reference
    baselines = {}
    for e in cat.valid():
        if e.mode == "baseline":
            baselines[e.arch] = {
                "overall": e.accuracy,
                "c4": e.split_accuracy("sft_eval", "add_C4"),
                "c6": e.split_accuracy("sft_eval", "add_C6"),
            }

    return {"groups": {k: v for k, v in groups.items()}, "baselines": baselines}


def plot_vocab_scaling(data: dict, path: Path):
    """Plot accuracy vs abstract vocab size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    colors = {"2L/3H/510d": "#e63946", "1L/3H/510d": "#457b9d",
              "1L/2H/256d": "#2a9d8f", "2L/1H/128d": "#e9c46a"}

    for metric_idx, (metric, title) in enumerate([("overall_accuracy", "Overall Accuracy"),
                                                    ("c6", "C6 (Hardest Carry)")]):
        ax = axes[metric_idx]

        for arch, models in sorted(data["groups"].items()):
            # Filter to K=1, 100K data for fair comparison
            subset = [m for m in models if m["K"] == 1 and m["dataset_size"] == 100000]
            if not subset:
                # Try K=4
                subset = [m for m in models if m["K"] == 4 and m["dataset_size"] == 100000]
            if not subset:
                continue

            subset.sort(key=lambda x: x["abs_vocab"])
            vocabs = [m["abs_vocab"] for m in subset]
            accs = [m[metric] or 0 for m in subset]
            color = colors.get(arch, "#666")
            ax.plot(vocabs, accs, "o-", color=color, label=f"{arch}", markersize=6)

        # Add baseline reference lines
        for arch, bl in data["baselines"].items():
            val = bl.get("overall" if metric == "overall_accuracy" else "c4" if metric == "c4" else "c6")
            if val is not None:
                color = colors.get(arch, "#666")
                ax.axhline(val, color=color, linestyle="--", alpha=0.4, linewidth=1)

        ax.set_xlabel("Abstract Vocabulary Size")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 20, 30, 50, 100])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    plt.suptitle("Accuracy vs Abstract Vocabulary Size (K=1, 100K data)", fontsize=12)
    plt.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {path}")


def write_summary(data: dict, path: Path):
    lines = [
        "# Vocab Scaling",
        "",
        "Accuracy vs abstract vocabulary size across architectures.",
        "All models: K=1, 100K training data, SoRL v1.",
        "",
        "![Vocab Scaling](fig_vocab_scaling.png)",
        "",
        "## Data",
        "",
        "| Arch | abs_vocab | Overall | C4 | C5 | C6 |",
        "|------|-----------|---------|----|----|----| ",
    ]
    for arch, models in sorted(data["groups"].items()):
        subset = sorted(
            [m for m in models if m["K"] == 1 and m["dataset_size"] == 100000],
            key=lambda x: x["abs_vocab"],
        )
        for m in subset:
            fmt = lambda v: f"{v:.0%}" if v is not None else "?"
            lines.append(
                f"| {arch} | {m['abs_vocab']} | {fmt(m['overall_accuracy'])} "
                f"| {fmt(m['c4'])} | {fmt(m['c5'])} | {fmt(m['c6'])} |"
            )
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {path}")


def main():
    print("Fetching model catalog from HF...")
    cat = ModelCatalog()
    cat.fetch(verbose=False)

    data = collect_vocab_data(cat)

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Wrote {OUT_DIR / 'results.json'}")

    plot_vocab_scaling(data, OUT_DIR / "fig_vocab_scaling.png")
    write_summary(data, OUT_DIR / "summary.md")


if __name__ == "__main__":
    main()
