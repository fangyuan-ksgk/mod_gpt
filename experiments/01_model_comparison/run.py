"""
Experiment 01: Model Comparison Table

Fetches all VALID models from HF, builds comparison tables showing
SoRL vs baseline accuracy across architectures, data sizes, and hard splits.

Outputs:
  - results.json    — structured data for all models
  - summary.md      — markdown table for dashboard
  - fig_data_efficiency.png — accuracy vs dataset size plot

Usage:
    python experiments/01_model_comparison/run.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from arithmetic.catalog import ModelCatalog

OUT_DIR = Path(__file__).parent


def build_comparison_data(cat: ModelCatalog) -> dict:
    """Build structured comparison data from catalog."""
    models = []
    for e in cat.valid():
        eval_key = "sorl_eval" if e.mode == "sorl" else "sft_eval"
        hard_splits = {}
        for split in ["add_C1", "add_C2", "add_C3", "add_C4", "add_C5", "add_C6"]:
            acc = e.split_accuracy(eval_key, split)
            if acc is not None:
                hard_splits[split] = acc

        models.append({
            "name": e.name,
            "mode": e.mode,
            "trainer": e.trainer,
            "arch": e.arch,
            "dataset_size": e.dataset_size,
            "abs_vocab": e.abs_vocab,
            "K": e.K,
            "overall_accuracy": e.accuracy,
            "sft_accuracy": e.sft_accuracy,
            "hard_splits": hard_splits,
            "eval_method": e.config.get("eval_method", "unknown"),
        })
    return {"models": models, "n_models": len(models)}


def write_summary_md(data: dict, path: Path):
    """Write markdown summary with comparison tables."""
    models = data["models"]

    lines = [
        "# Model Comparison",
        "",
        f"**{data['n_models']} VALID models** from `thoughtworks/arithmetic-sorl`",
        "",
        "## SoRL K=1 vs Baseline (Standard Architecture: 2L/3H/510d)",
        "",
        "| Model | Mode | Data | V | Overall | C3 | C4 | C5 | C6 |",
        "|-------|------|------|---|---------|----|----|----|----|",
    ]

    std_arch = [m for m in models if m["arch"] == "2L/3H/510d"]
    for m in sorted(std_arch, key=lambda x: (x["mode"], x["dataset_size"])):
        name = m["name"].split("/")[-1] if "/" in m["name"] else m["name"]
        ov = f"{m['overall_accuracy']:.0%}" if m["overall_accuracy"] else "?"
        c_splits = [m["hard_splits"].get(f"add_C{i}", None) for i in [3, 4, 5, 6]]
        c_strs = [f"{v:.0%}" if v is not None else "?" for v in c_splits]
        ds = f"{m['dataset_size']//1000}K" if m["dataset_size"] else "?"
        lines.append(f"| {name} | {m['mode']} | {ds} | {m['abs_vocab']} | {ov} | {' | '.join(c_strs)} |")

    lines += [
        "",
        "## Cross-Architecture Comparison (100K data, K=1 abs30)",
        "",
        "| Arch | Baseline | SoRL | Delta | C4 Base | C4 SoRL | C6 Base | C6 SoRL |",
        "|------|----------|------|-------|---------|---------|---------|---------|",
    ]

    # Group by arch for cross-comparison
    archs = sorted(set(m["arch"] for m in models))
    for arch in archs:
        arch_models = [m for m in models if m["arch"] == arch and m["dataset_size"] == 100000]
        baseline = [m for m in arch_models if m["mode"] == "baseline"]
        sorl_k1 = [m for m in arch_models if m["mode"] == "sorl" and m["K"] == 1 and m["abs_vocab"] == 30]
        if baseline and sorl_k1:
            b, s = baseline[0], sorl_k1[0]
            ba = b["overall_accuracy"] or 0
            sa = s["overall_accuracy"] or 0
            delta = sa - ba
            bc4 = b["hard_splits"].get("add_C4")
            sc4 = s["hard_splits"].get("add_C4")
            bc6 = b["hard_splits"].get("add_C6")
            sc6 = s["hard_splits"].get("add_C6")
            fmt = lambda v: f"{v:.0%}" if v is not None else "?"
            lines.append(f"| {arch} | {fmt(ba)} | {fmt(sa)} | {delta:+.0%} | {fmt(bc4)} | {fmt(sc4)} | {fmt(bc6)} | {fmt(sc6)} |")

    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {path}")


def plot_data_efficiency(data: dict, path: Path):
    """Plot accuracy vs dataset size for baseline and SoRL."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = data["models"]
    std = [m for m in models if m["arch"] == "2L/3H/510d"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Overall accuracy
    ax = axes[0]
    for mode, marker, color in [("baseline", "s", "#888"), ("sorl", "o", "#e63946")]:
        subset = [m for m in std if m["mode"] == mode and m.get("K", 4) in (1, 4)]
        if not subset:
            continue
        sizes = [m["dataset_size"] / 1000 for m in subset]
        accs = [m["overall_accuracy"] or 0 for m in subset]
        label = f"SoRL K={subset[0]['K']}" if mode == "sorl" else "Baseline"
        ax.scatter(sizes, accs, marker=marker, color=color, label=label, s=60)
    ax.set_xlabel("Dataset Size (K)")
    ax.set_ylabel("Overall Accuracy")
    ax.set_title("Data Efficiency: Overall")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # C6 accuracy
    ax = axes[1]
    for mode, marker, color in [("baseline", "s", "#888"), ("sorl", "o", "#e63946")]:
        subset = [m for m in std if m["mode"] == mode and m.get("K", 4) in (1, 4)]
        if not subset:
            continue
        sizes = [m["dataset_size"] / 1000 for m in subset]
        accs = [m["hard_splits"].get("add_C6", 0) for m in subset]
        label = f"SoRL K={subset[0]['K']}" if mode == "sorl" else "Baseline"
        ax.scatter(sizes, accs, marker=marker, color=color, label=label, s=60)
    ax.set_xlabel("Dataset Size (K)")
    ax.set_ylabel("C6 (Hardest Carry) Accuracy")
    ax.set_title("Data Efficiency: Hard Carries")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {path}")


def main():
    print("Fetching model catalog from HF...")
    cat = ModelCatalog()
    cat.fetch(verbose=False)

    data = build_comparison_data(cat)

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {OUT_DIR / 'results.json'} ({data['n_models']} models)")

    write_summary_md(data, OUT_DIR / "summary.md")
    plot_data_efficiency(data, OUT_DIR / "fig_data_efficiency.png")


if __name__ == "__main__":
    main()
