"""
Experiment 04: Addition Hierarchy

Shows the hierarchical structure of abstraction tokens within addition:
SA (simple add) → SC (make carry) → SS (sum-9) → UC (use carry) → US (cascade).

Produces a focused heatmap showing only addition subtasks, ordered by
Quirke's complexity hierarchy, revealing the carry computation spectrum.

Outputs:
  - results.json                — token data filtered to addition
  - fig_addition_hierarchy.png  — hierarchy-ordered heatmap
  - summary.md

Usage:
    python experiments/04_addition_hierarchy/run.py [--model MODEL] [--device cuda:0]
"""
import argparse
import json
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

OUT_DIR = Path(__file__).parent
DEFAULT_MODEL = "add_sub_sorl_v1_abs30_K1_100K"

# Quirke's addition subtask hierarchy (easy → hard)
ADD_HIERARCHY = ["SA", "SC", "SS", "UC", "US"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    # Load token data from experiment 03 if available, otherwise regenerate
    exp03_results = Path(__file__).parent.parent / "03_token_subtask_heatmap" / "results.json"
    if exp03_results.exists():
        print(f"Loading token data from {exp03_results}...")
        with open(exp03_results) as f:
            data = json.load(f)
    else:
        print("No cached data from experiment 03, running analysis...")
        from arithmetic.data.hub import load_model
        from arithmetic.data.addition import get_eval_set
        from arithmetic.training.train import QWEN3_TOKEN_MAP, QWEN3_INV_MAP
        from sorl.sorl_trainer import infer_insert_mask, insert_tokens_with_padding, expand_prompt_len
        from transformers import AutoTokenizer
        from collections import defaultdict, Counter

        PROMPT_LEN = 14
        ANSWER_LEN = 7

        model, config, _ = load_model(args.model, device=args.device)
        base_v = model.vocab_sizes[0].item()
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        pad_id = tokenizer.pad_token_id
        K = config.get("K", 4)

        categories = get_eval_set()
        token_data = defaultdict(lambda: {"subtask": Counter(), "count": 0})

        model.eval()
        n = 0
        for split_name, examples in categories.items():
            if not split_name.startswith("add_"):
                continue
            for ex in examples:
                qwen_ids = torch.tensor(
                    [QWEN3_TOKEN_MAP[t] for t in ex.tokens],
                    dtype=torch.long, device=args.device,
                )
                seq = qwen_ids.unsqueeze(0)
                attn = torch.ones_like(seq)
                pl = torch.tensor([PROMPT_LEN], dtype=torch.long, device=args.device)

                with torch.no_grad():
                    im = infer_insert_mask(seq, K, attn)
                    ep = expand_prompt_len(pl, im)
                    ed, ea = insert_tokens_with_padding(seq, attn, im, model.vocab_sizes[0], pad_id)
                    result_data, ppt, logits = model.recursion(
                        ed, ea, max_iterations=2,
                        memory_span_abs=1792, memory_span_traj=1792,
                        temperature=0.0, prompt_len=ep,
                    )

                expanded = result_data[0]
                is_abs = expanded >= base_v
                is_traj = ~is_abs
                traj_indices = is_traj.nonzero(as_tuple=True)[0]
                abs_indices = is_abs.nonzero(as_tuple=True)[0]

                for abs_idx in abs_indices:
                    tok_id = (expanded[abs_idx] - base_v).item()
                    traj_before = (traj_indices < abs_idx).sum().item() - 1
                    if PROMPT_LEN <= traj_before < PROMPT_LEN + ANSWER_LEN:
                        answer_pos = traj_before - PROMPT_LEN
                        label = ex.labels[answer_pos] if answer_pos < len(ex.labels) else "?"
                        token_data[tok_id]["subtask"][label] += 1
                        token_data[tok_id]["count"] += 1
                n += 1

        data = {
            "model": args.model, "K": K, "abs_vocab": config.get("abs_vocab", 30),
            "n_examples": n,
            "tokens": {str(tid): {"count": td["count"], "subtask": dict(td["subtask"])}
                       for tid, td in token_data.items()},
        }
        del model
        torch.cuda.empty_cache()

    # Filter to addition subtasks only
    add_tokens = {}
    for tid_str, td in data["tokens"].items():
        add_subs = {k: v for k, v in td["subtask"].items() if k in ADD_HIERARCHY}
        if add_subs and td["count"] >= 10:
            add_tokens[tid_str] = {"count": td["count"], "subtask": add_subs}

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump({"model": data["model"], "add_tokens": add_tokens,
                    "hierarchy": ADD_HIERARCHY}, f, indent=2)

    # Plot hierarchy heatmap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tids = sorted(add_tokens.keys(), key=lambda t: int(t))
    matrix = np.zeros((len(tids), len(ADD_HIERARCHY)))
    for i, tid in enumerate(tids):
        cnt = add_tokens[tid]["count"]
        for j, label in enumerate(ADD_HIERARCHY):
            matrix[i, j] = add_tokens[tid]["subtask"].get(label, 0) / max(cnt, 1)

    # Sort tokens by their "complexity center of mass"
    com = np.array([np.dot(matrix[i], range(len(ADD_HIERARCHY))) / max(matrix[i].sum(), 1e-9)
                     for i in range(len(tids))])
    order = np.argsort(com)
    matrix = matrix[order]
    tids_sorted = [tids[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, max(4, len(tids) * 0.35)))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ADD_HIERARCHY)))
    ax.set_xticklabels(ADD_HIERARCHY, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(tids_sorted)))
    ax.set_yticklabels([f"t{t} (n={add_tokens[t]['count']})" for t in tids_sorted], fontsize=8)

    for i in range(len(tids_sorted)):
        for j in range(len(ADD_HIERARCHY)):
            v = matrix[i, j]
            if v >= 0.05:
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.5 else "black")

    plt.colorbar(im, label="P(subtask | token)")
    ax.set_title(f"Addition Carry Hierarchy — {data['model']}")
    ax.set_xlabel("Quirke Subtask (easy → hard)")
    ax.set_ylabel("Abstraction Token (sorted by complexity)")
    plt.tight_layout()
    fig.savefig(str(OUT_DIR / "fig_addition_hierarchy.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {OUT_DIR / 'fig_addition_hierarchy.png'}")

    # Summary
    lines = [
        "# Addition Hierarchy",
        "",
        f"**Model:** `{data['model']}`",
        f"**Hierarchy:** {' → '.join(ADD_HIERARCHY)} (easy → hard)",
        "",
        "Tokens sorted by complexity center-of-mass: tokens at the top",
        "specialize in simple addition, tokens at the bottom handle carry cascades.",
        "",
        "![Addition Hierarchy](fig_addition_hierarchy.png)",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
