"""
Experiment 03: Token-Subtask Heatmap

For a given SoRL model, runs recursion on the canonical eval set and records
which abstraction token appears at each Quirke subtask position. Produces a
P(token | subtask) heatmap showing token specialization.

Outputs:
  - results.json              — raw token-subtask counts
  - fig_token_subtask.png     — P(token|subtask) heatmap
  - fig_token_positions.png   — token position distribution
  - summary.md

Usage:
    python experiments/03_token_subtask_heatmap/run.py [--model MODEL] [--device cuda:0]
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
DEFAULT_MODEL = "add_sub_sorl_v1_abs30_K1_100K"


def collect_token_data(model, config, device):
    """Run recursion on canonical eval set, collect token-subtask statistics."""
    base_v = model.vocab_sizes[0].item()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    pad_id = tokenizer.pad_token_id
    K = config.get("K", 4)

    categories = get_eval_set()
    token_data = defaultdict(lambda: {
        "subtask": Counter(), "answer_position": Counter(),
        "operation": Counter(), "carry_state": Counter(),
        "input_sum_mod10": Counter(), "count": 0,
    })

    model.eval()
    n = 0
    for split_name, examples in categories.items():
        for ex in examples:
            qwen_ids = torch.tensor(
                [QWEN3_TOKEN_MAP[t] for t in ex.tokens],
                dtype=torch.long, device=device,
            )
            seq = qwen_ids.unsqueeze(0)
            attn = torch.ones_like(seq)
            pl = torch.tensor([PROMPT_LEN], dtype=torch.long, device=device)

            with torch.no_grad():
                im = infer_insert_mask(seq, K, attn)
                ep = expand_prompt_len(pl, im)
                ed, ea = insert_tokens_with_padding(seq, attn, im, model.vocab_sizes[0], pad_id)
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

            orig_tokens = [QWEN3_INV_MAP.get(t.item(), -1) for t in qwen_ids]
            op = "add" if orig_tokens[6] == 10 else "sub"

            for abs_idx in abs_indices:
                tok_id = (expanded[abs_idx] - base_v).item()
                traj_before = (traj_indices < abs_idx).sum().item() - 1

                if PROMPT_LEN <= traj_before < PROMPT_LEN + ANSWER_LEN:
                    answer_pos = traj_before - PROMPT_LEN
                    label = ex.labels[answer_pos] if answer_pos < len(ex.labels) else "?"

                    d1 = orig_tokens[answer_pos] if answer_pos < 6 else -1
                    d2 = orig_tokens[7 + answer_pos] if answer_pos < 6 else -1

                    td = token_data[tok_id]
                    td["subtask"][label] += 1
                    td["answer_position"][answer_pos] += 1
                    td["operation"][op] += 1
                    if d1 >= 0 and d2 >= 0:
                        s = d1 + d2
                        td["input_sum_mod10"][s % 10] += 1
                        td["carry_state"]["carry" if s >= 10 else "no_carry"] += 1
                    td["count"] += 1
            n += 1

    return token_data, n, K, config.get("abs_vocab", 30)


def plot_heatmap(token_data, model_name, K, abs_vocab, path):
    """Plot P(subtask | token) heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = sorted(set(l for td in token_data.values() for l in td["subtask"]))
    # Only show tokens with >= 10 occurrences
    tids = sorted([t for t, td in token_data.items() if td["count"] >= 10])

    matrix = np.zeros((len(tids), len(labels)))
    for i, tid in enumerate(tids):
        cnt = token_data[tid]["count"]
        if cnt == 0:
            continue
        for j, label in enumerate(labels):
            matrix[i, j] = token_data[tid]["subtask"].get(label, 0) / cnt

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), max(4, len(tids) * 0.4)))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(tids)))
    ax.set_yticklabels([f"t{t} (n={token_data[t]['count']})" for t in tids], fontsize=8)

    for i in range(len(tids)):
        for j in range(len(labels)):
            v = matrix[i, j]
            if v >= 0.05:
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=6,
                        color="white" if v > 0.5 else "black")

    plt.colorbar(im, label="P(subtask | token)")
    ax.set_title(f"{model_name} (K={K}, abs={abs_vocab})")
    ax.set_xlabel("Quirke Subtask")
    ax.set_ylabel("Abstraction Token")
    plt.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {path}")


def plot_positions(token_data, model_name, path):
    """Plot token position distributions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tids = sorted([t for t, td in token_data.items() if td["count"] >= 10])
    if not tids:
        return

    fig, ax = plt.subplots(figsize=(10, max(3, len(tids) * 0.3)))
    positions = list(range(ANSWER_LEN))

    for i, tid in enumerate(tids):
        td = token_data[tid]
        cnt = td["count"]
        dist = [td["answer_position"].get(p, 0) / max(cnt, 1) for p in positions]
        ax.barh([i + p * 0.12 for p in positions], dist, height=0.1,
                color=plt.cm.viridis(np.linspace(0, 1, ANSWER_LEN)))

    ax.set_yticks(range(len(tids)))
    ax.set_yticklabels([f"t{t}" for t in tids], fontsize=8)
    ax.set_xlabel("Fraction of occurrences")
    ax.set_title(f"Token Position Distribution — {model_name}")
    plt.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {path}")


def write_summary(token_data, model_name, n_examples, K, abs_vocab, path):
    lines = [
        "# Token-Subtask Heatmap",
        "",
        f"**Model:** `{model_name}` (K={K}, abs_vocab={abs_vocab})",
        f"**Eval set:** canonical N=100 from HuggingFace",
        f"**Examples processed:** {n_examples}",
        "",
        "![Token-Subtask Heatmap](fig_token_subtask.png)",
        "",
        "## Token Summary",
        "",
        "| Token | Count | Top Subtask | % | Top Carry | % |",
        "|-------|-------|-------------|---|-----------|---|",
    ]

    for tid in sorted(token_data.keys()):
        td = token_data[tid]
        if td["count"] < 10:
            continue
        top_sub = td["subtask"].most_common(1)
        top_carry = td["carry_state"].most_common(1)
        sub_str = f"{top_sub[0][0]}={top_sub[0][1]*100//td['count']}%" if top_sub else "?"
        carry_str = f"{top_carry[0][0]}={top_carry[0][1]*100//td['count']}%" if top_carry else "?"
        lines.append(f"| t{tid} | {td['count']} | {sub_str} | | {carry_str} | |")

    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    print(f"Loading {args.model}...")
    model, config, _ = load_model(args.model, device=args.device)

    print("Collecting token-subtask data from canonical eval set...")
    token_data, n_examples, K, abs_vocab = collect_token_data(model, config, args.device)

    # Save raw data
    json_data = {}
    for tok_id, td in token_data.items():
        json_data[str(tok_id)] = {
            "count": td["count"],
            "subtask": dict(td["subtask"]),
            "operation": dict(td["operation"]),
            "carry_state": dict(td["carry_state"]),
            "answer_position": {str(k): v for k, v in td["answer_position"].items()},
            "input_sum_mod10": {str(k): v for k, v in td["input_sum_mod10"].items()},
        }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump({"model": args.model, "K": K, "abs_vocab": abs_vocab,
                    "n_examples": n_examples, "tokens": json_data}, f, indent=2)
    print(f"Wrote {OUT_DIR / 'results.json'}")

    plot_heatmap(token_data, args.model, K, abs_vocab, OUT_DIR / "fig_token_subtask.png")
    plot_positions(token_data, args.model, OUT_DIR / "fig_token_positions.png")
    write_summary(token_data, args.model, n_examples, K, abs_vocab, OUT_DIR / "summary.md")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
