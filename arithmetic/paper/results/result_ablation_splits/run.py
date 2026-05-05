"""
result_ablation_splits: per-split causal ablation table for the paper appendix.

Reads causal_verification.json from interp_results and produces:
  - results.json  — structured per-split numbers
  - table.tex     — LaTeX booktabs table grouped by split family

Model: add_sub_sorl_v1_abs30_K1_100K_2L1H128d  (2L/1H/128d, 100K)

Usage:
  /opt/pytorch/bin/python3 paper/results/result_ablation_splits/run.py
"""
import json, os

INTERP_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../../interp_results/as_sorl_abs30_K1_100K_2L1H128d",
)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Split groups for the table ────────────────────────────────────────────────
SPLIT_GROUPS = {
    "Addition (easy)": ["add_S0", "add_S1", "add_S2", "add_random"],
    "Addition cascade (hard)": ["add_C3", "add_C4", "add_C5", "add_C6"],
    "Subtraction (easy)": ["sub_random"],
    "Subtraction cascade (hard)": ["sub_M3", "sub_M4", "sub_M5"],
}

SPLIT_LABELS = {
    "add_S0": "S0 (no carry)",
    "add_S1": "S1",
    "add_S2": "S2",
    "add_random": "random",
    "add_C3": "C3 (3-deep)",
    "add_C4": "C4 (4-deep)",
    "add_C5": "C5 (5-deep)",
    "add_C6": "C6 (6-deep)",
    "sub_random": "random",
    "sub_M3": "M3 (3-deep borrow)",
    "sub_M4": "M4 (4-deep borrow)",
    "sub_M5": "M5 (5-deep borrow)",
}


def load_per_split(interp_dir):
    path = os.path.join(interp_dir, "causal_verification.json")
    with open(path) as f:
        cv = json.load(f)
    return cv["per_split"], cv


def build_results(per_split, cv):
    def acc(v):
        return v["accuracy"] if isinstance(v, dict) else v / cv["n_examples"]

    results = {
        "model": cv["model"],
        "n_examples": cv["n_examples"],
        "overall": {
            "baseline": acc(cv["baseline"]),
            "shuffle":  acc(cv["shuffle"]),
            "random":   acc(cv["random"]),
            "knockout": acc(cv["knockout"]),
        },
        "per_split": {},
    }
    for split, data in per_split.items():
        results["per_split"][split] = {
            k: v["accuracy"] for k, v in data.items()
        }
    return results


def build_table(results):
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"  \centering\small")
    lines.append(r"  \begin{tabular}{llrrrr}")
    lines.append(r"    \toprule")
    lines.append(r"    Family & Split & Baseline & Shuffle & Random & Knockout \\")
    lines.append(r"    \midrule")

    per_split = results["per_split"]
    first_group = True
    for group_name, splits in SPLIT_GROUPS.items():
        if not first_group:
            lines.append(r"    \midrule")
        first_group = False
        first_row = True
        n_rows = sum(1 for s in splits if s in per_split)
        group_tex = r"\multirow{" + str(n_rows) + r"}{*}{\textit{" + group_name + r"}}"
        for split in splits:
            if split not in per_split:
                continue
            sp = per_split[split]
            b  = sp["baseline"];  s = sp["shuffle"]
            r_ = sp["random"];    k = sp["knockout"]
            label = SPLIT_LABELS.get(split, split)
            row_group = group_tex if first_row else ""
            lines.append(
                f"    {row_group} & {label} "
                f"& {b:.0%} & {s:.0%} & {r_:.0%} & {k:.0%} \\\\"
            )
            first_row = False

    lines.append(r"    \midrule")
    ov = results["overall"]
    lines.append(
        f"    \\multicolumn{{2}}{{l}}{{\\textbf{{Overall}}}} "
        f"& \\textbf{{{ov['baseline']:.1%}}} "
        f"& {ov['shuffle']:.1%} & {ov['random']:.1%} & {ov['knockout']:.1%} \\\\"
    )
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(
        r"  \caption{Per-split causal ablation on \texttt{2L/1H/128d} (100K)."
        "\n"
        r"    \textbf{Shuffle}: tokens permuted within the sequence (identity preserved,"
        "\n"
        r"    position destroyed). \textbf{Random}: each token replaced by a uniform"
        "\n"
        r"    draw from the codebook. \textbf{Knockout}: all tokens replaced by a fixed"
        "\n"
        r"    mask. Shuffle $>$ Random on cascade splits because wrong-position tokens"
        "\n"
        r"    cause systematic one-off carry errors; random tokens cause broader"
        "\n"
        r"    incoherence. Sub-M5 (5-deep borrow cascade) is hardest even at baseline"
        "\n"
        r"    (57\%), and all ablations collapse it to $\leq$3\%.}"
    )
    lines.append(r"  \label{tab:ablation-splits}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    per_split, cv = load_per_split(INTERP_DIR)
    results = build_results(per_split, cv)

    json_path = os.path.join(OUT_DIR, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {json_path}")

    table = build_table(results)
    tex_path = os.path.join(OUT_DIR, "table.tex")
    with open(tex_path, "w") as f:
        f.write(table)
    print(f"Wrote {tex_path}")
    print()
    print(table)


if __name__ == "__main__":
    main()
