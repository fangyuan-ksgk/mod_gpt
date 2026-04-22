"""
Experiment 05: Token Vignettes

Deep-dive analysis of 3 specific abstraction tokens showing their role
across examples. For each token:
  - Which subtasks it appears at (and percentages)
  - Position distribution
  - Carry state correlation
  - 3-5 concrete example problems showing the token in context

Outputs:
  - results.json           — per-token detailed data
  - summary.md             — narrative vignettes for dashboard

Usage:
    python experiments/05_token_vignettes/run.py [--model MODEL] [--device cuda:0]
        [--tokens 2,7,3]
"""
import argparse
import json
import sys
import torch
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
DEFAULT_TOKENS = "2,7,3"


def collect_vignette_data(model, config, device, target_tokens):
    """Collect detailed per-token data including concrete examples."""
    base_v = model.vocab_sizes[0].item()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    pad_id = tokenizer.pad_token_id
    K = config.get("K", 4)

    categories = get_eval_set()
    token_data = {t: {
        "subtask": Counter(), "answer_position": Counter(),
        "operation": Counter(), "carry_state": Counter(),
        "count": 0, "examples": [],
    } for t in target_tokens}

    model.eval()
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

            orig = list(ex.tokens)
            op_char = "+" if orig[6] == 10 else "-"
            a_str = "".join(str(d) for d in orig[:6])
            b_str = "".join(str(d) for d in orig[7:13])
            ans_str = "".join(str(d) for d in orig[14:])
            problem = f"{a_str}{op_char}{b_str}={ans_str}"

            # Collect all abs tokens in this example
            abs_in_example = []
            for abs_idx in abs_indices:
                tok_id = (expanded[abs_idx] - base_v).item()
                traj_before = (traj_indices < abs_idx).sum().item() - 1
                if PROMPT_LEN <= traj_before < PROMPT_LEN + ANSWER_LEN:
                    answer_pos = traj_before - PROMPT_LEN
                    label = ex.labels[answer_pos] if answer_pos < len(ex.labels) else "?"
                    abs_in_example.append({
                        "tok_id": tok_id, "position": answer_pos, "subtask": label,
                    })

                    if tok_id in token_data:
                        td = token_data[tok_id]
                        td["subtask"][label] += 1
                        td["answer_position"][answer_pos] += 1
                        td["operation"]["add" if orig[6] == 10 else "sub"] += 1
                        d1 = orig[answer_pos] if answer_pos < 6 else -1
                        d2 = orig[7 + answer_pos] if answer_pos < 6 else -1
                        if d1 >= 0 and d2 >= 0:
                            s = d1 + d2
                            td["carry_state"]["carry" if s >= 10 else "no_carry"] += 1
                        td["count"] += 1

                        if len(td["examples"]) < 5:
                            td["examples"].append({
                                "problem": problem,
                                "split": split_name,
                                "labels": list(ex.labels),
                                "abs_tokens": abs_in_example.copy(),
                            })

    return token_data, K


def write_summary(token_data, model_name, K, path):
    lines = [
        "# Token Vignettes",
        "",
        f"**Model:** `{model_name}` (K={K})",
        f"**Eval set:** canonical N=100 from HuggingFace",
        "",
    ]

    for tok_id in sorted(token_data.keys()):
        td = token_data[tok_id]
        if td["count"] == 0:
            lines.append(f"## Token t{tok_id} — not observed")
            lines.append("")
            continue

        # Determine role
        top_sub = td["subtask"].most_common(3)
        top_carry = td["carry_state"].most_common(1)
        top_pos = td["answer_position"].most_common(2)
        top_op = td["operation"].most_common(1)

        role_parts = []
        if top_sub:
            role_parts.append(f"{top_sub[0][0]} ({top_sub[0][1]*100//td['count']}%)")
        if top_carry:
            role_parts.append(f"{top_carry[0][0]} ({top_carry[0][1]*100//td['count']}%)")

        lines.append(f"## Token t{tok_id}")
        lines.append("")
        lines.append(f"**Occurrences:** {td['count']}")
        lines.append(f"**Primary role:** {', '.join(role_parts)}")
        lines.append("")

        # Subtask breakdown
        lines.append("**Subtask distribution:**")
        for label, count in top_sub:
            pct = count * 100 // td["count"]
            lines.append(f"- {label}: {pct}% ({count}/{td['count']})")
        lines.append("")

        # Position distribution
        lines.append("**Position distribution:**")
        for pos, count in sorted(td["answer_position"].items()):
            pct = count * 100 // td["count"]
            if pct >= 5:
                lines.append(f"- d{pos}: {pct}%")
        lines.append("")

        # Examples
        if td["examples"]:
            lines.append("**Example problems:**")
            lines.append("")
            lines.append("| Problem | Split | Labels | t{0} position |".format(tok_id))
            lines.append("|---------|-------|--------|--------------|")
            for ex in td["examples"][:5]:
                tok_pos = [a["position"] for a in ex["abs_tokens"] if a["tok_id"] == tok_id]
                pos_str = ", ".join(f"d{p}" for p in tok_pos)
                lines.append(f"| `{ex['problem']}` | {ex['split']} | {ex['labels']} | {pos_str} |")
            lines.append("")

    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--tokens", default=DEFAULT_TOKENS,
                   help="Comma-separated token IDs to profile")
    args = p.parse_args()

    target_tokens = [int(t) for t in args.tokens.split(",")]
    print(f"Profiling tokens {target_tokens} on {args.model}...")

    model, config, _ = load_model(args.model, device=args.device)
    token_data, K = collect_vignette_data(model, config, args.device, target_tokens)

    # Save results
    json_data = {}
    for tok_id, td in token_data.items():
        json_data[str(tok_id)] = {
            "count": td["count"],
            "subtask": dict(td["subtask"]),
            "operation": dict(td["operation"]),
            "carry_state": dict(td["carry_state"]),
            "answer_position": {str(k): v for k, v in td["answer_position"].items()},
            "examples": td["examples"],
        }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump({"model": args.model, "K": K, "tokens": json_data}, f, indent=2)
    print(f"Wrote {OUT_DIR / 'results.json'}")

    write_summary(token_data, args.model, K, OUT_DIR / "summary.md")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
