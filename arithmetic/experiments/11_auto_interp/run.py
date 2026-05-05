"""
Experiment 11: Auto-Interpretation of SoRL Tokens (Bills et al. style)

For each active abstraction token in the codebook, find the N examples
where the model assigned it with highest confidence (softmax probability),
then ask Claude to describe in one sentence what role the token plays.

Outputs:
  results.json   — per-token: top examples + confidence + auto-interpretation
  table.tex      — LaTeX table for the paper appendix

Usage:
    python experiments/11_auto_interp/run.py \
        --model add_sub_sorl_v1_abs30_K1_100K_2L1H128d \
        --device cuda:0 \
        --top_n 10 \
        --n_table 8
"""
import argparse
import json
import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from arithmetic.data.hub import load_model
from arithmetic.data.addition import get_eval_set
from arithmetic.training.train import QWEN3_TOKEN_MAP, QWEN3_INV_MAP
from sorl.sorl_trainer import infer_insert_mask, insert_tokens_with_padding, expand_prompt_len
from transformers import AutoTokenizer

PROMPT_LEN = 14
ANSWER_LEN = 7
OUT_DIR = Path(__file__).parent
DEFAULT_MODEL = "add_sub_sorl_v1_abs30_K1_100K_2L1H128d"

SUBTASK_LABELS = {
    "SA": "simple add", "SC": "sum carries", "SS": "sum-9 (uncertain)",
    "UC": "use carry", "US": "cascade (add)",
    "MD": "simple diff", "MB": "makes borrow", "ME": "equal digits",
    "UB": "use borrow", "UD": "cascade (sub)",
}


# ── Model inference ───────────────────────────────────────────────────────────

def example_to_problem_str(ex):
    op_ch = "+" if ex.op == "add" else "-"
    a = "".join(str(d) for d in ex.x_digits)
    b = "".join(str(d) for d in ex.y_digits)
    return f"{a}{op_ch}{b}"


def run_with_confidence(model, qwen_ids, K, base_v, pad_id, device):
    """
    Run SoRL recursion; return:
      - digit_abs: {answer_pos: (abs_idx, token_id)}
      - token_probs: {answer_pos: float}  — P(assigned_token | context)
    """
    seq = qwen_ids.unsqueeze(0).to(device)
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
    abs_indices  = is_abs.nonzero(as_tuple=True)[0]

    # Get full-sequence logits for confidence extraction
    with torch.no_grad():
        full_logits = model(
            expanded.unsqueeze(0),
            attention_mask=torch.ones(1, len(expanded), device=device),
        ).logits[0]  # [seq_len, vocab]

    digit_abs   = {}
    token_probs = {}
    for abs_idx in abs_indices:
        traj_before = (traj_indices < abs_idx).sum().item() - 1
        if PROMPT_LEN <= traj_before < PROMPT_LEN + ANSWER_LEN:
            answer_pos = traj_before - PROMPT_LEN
            tok_id = (expanded[abs_idx] - base_v).item()
            digit_abs[answer_pos] = (abs_idx.item(), tok_id)
            # Probability of the assigned token at position abs_idx - 1
            context_logits = full_logits[abs_idx - 1, base_v:]  # over abs vocab
            probs = F.softmax(context_logits, dim=-1)
            token_probs[answer_pos] = probs[tok_id].item()

    return expanded.clone(), digit_abs, token_probs


# ── Example formatting ────────────────────────────────────────────────────────

def format_example_for_prompt(rec):
    """Format one token record as a line for the LLM prompt."""
    p = rec["problem"]
    # Parse problem: "AAAAAA+BBBBBB" or "AAAAAA-BBBBBB"
    op = "+" if "+" in p else "-"
    a, b = p.split(op)
    ans_pos = rec["answer_pos"]
    tok_id  = rec["token_id"]
    subtask = rec["subtask"]
    carry   = rec["carry"]
    s10     = rec["sum_mod10"]
    prob    = rec["confidence"]
    return (
        f"  {a} {op} {b}  |  answer pos d{ans_pos}  "
        f"|  subtask={subtask}  carry={carry}  sum%%10={s10}  "
        f"|  P(token)={prob:.2f}"
    )


def build_interp_prompt(token_id, examples):
    lines = [format_example_for_prompt(e) for e in examples]
    return (
        f"You are helping interpret routing tokens in a small transformer trained "
        f"to do 6-digit addition and subtraction.\n\n"
        f"Token t{token_id} was assigned with high confidence at these positions:\n\n"
        + "\n".join(lines)
        + "\n\n"
        "Subtask codes: SA=simple add, SC=makes carry, SS=sum-9 boundary, "
        "UC=uses carry, US=sum-9 cascade; "
        "MD=simple diff, MB=makes borrow, ME=equal digits, "
        "UB=uses borrow, UD=borrow cascade.\n\n"
        "In ONE sentence (max 20 words), describe what computational role "
        "this token seems to play. Be concrete and specific."
    )


# ── Claude API ────────────────────────────────────────────────────────────────

def interpret_token(client, token_id, examples):
    prompt = build_interp_prompt(token_id, examples)
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=80,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


# ── LaTeX table ───────────────────────────────────────────────────────────────

def build_latex_table(token_results, n_table):
    """Build paper-ready LaTeX table (n_table most interesting tokens)."""
    # Sort by confidence descending, then pick a mix of specialist + polysemantic
    rows = sorted(token_results, key=lambda r: -r["mean_confidence"])
    rows = rows[:n_table]

    lines = [
        r"\begin{table}[h]",
        r"  \centering\small",
        r"  \begin{tabular}{clrp{5.5cm}}",
        r"    \toprule",
        r"    Token & Top subtask & Conf. & Auto-interpretation \\",
        r"    \midrule",
    ]
    for r in rows:
        tid   = r["token_id"]
        top   = r["top_subtask"]
        purity = r["top_subtask_purity"]
        conf  = r["mean_confidence"]
        interp = r["interpretation"].replace("%", r"\%")
        lines.append(
            f"    \\texttt{{t{tid}}} & {top} ({purity:.0%}) "
            f"& {conf:.2f} & {interp} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{Auto-interpretation of \sorl{} abstraction tokens"
        r" (\`{a} la \citealt{bills2023language})."
        r" For each token, the 10 examples where the model assigned it"
        r" with highest softmax confidence are shown to \texttt{claude-haiku},"
        r" which produces a one-sentence role description."
        r" \textbf{Conf.}\ = mean softmax probability of the assigned token.}",
        r"  \label{tab:auto-interp}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--top_n",  type=int, default=10,
                        help="Examples per token for LLM prompt")
    parser.add_argument("--n_table", type=int, default=8,
                        help="Tokens to show in paper table")
    parser.add_argument("--splits", nargs="+",
                        default=["add_C4", "add_C5", "add_C6",
                                 "sub_M3", "sub_M4",
                                 "add_random", "sub_random"])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Loading model {args.model} on {device}...")
    model, cfg, _ = load_model(args.model, device=str(device))
    model.eval()

    base_v = model.full_vocab_size_list[0]
    K      = cfg.get("K", 1)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    pad_id = tokenizer.pad_token_id or 0

    # ── Collect high-confidence token occurrences ─────────────────────────────
    print("Collecting token occurrences across splits...")
    all_splits = get_eval_set()   # {split_name: [examples]}
    token_records = defaultdict(list)

    for split in args.splits:
        if split not in all_splits:
            print(f"  {split}: not in eval set, skipping")
            continue
        examples = all_splits[split]
        print(f"  {split} ({len(examples)} examples)...", end=" ", flush=True)
        for ex in examples:
            problem  = example_to_problem_str(ex)
            subtasks = ex.labels          # list[str], one per answer digit
            # ex.tokens = full 21-token sequence (prompt=14, answer=7)
            qwen_ids = torch.tensor(ex.tokens[:PROMPT_LEN], dtype=torch.long)

            try:
                expanded, digit_abs, token_probs = run_with_confidence(
                    model, qwen_ids, K, base_v, pad_id, device
                )
            except Exception:
                continue

            # st: Quirke carry state per answer digit (0/1/'U'), indexed d6→d0
            st_list = list(ex.st) if hasattr(ex, "st") else []

            for ans_pos, (abs_idx, tok_id) in digit_abs.items():
                subtask = subtasks[ans_pos] if ans_pos < len(subtasks) else "?"
                # carry: st == 1 or 'U' means carry is active or uncertain
                st_val  = st_list[ans_pos] if ans_pos < len(st_list) else 0
                carry   = "carry" if st_val in (1, "U") else "no_carry"
                # digit sum mod 10 for the operand digits at this position
                try:
                    a_dig = ex.x_digits[ans_pos - 1] if 0 < ans_pos <= 6 else 0
                    b_dig = ex.y_digits[ans_pos - 1] if 0 < ans_pos <= 6 else 0
                    s10   = (a_dig + b_dig) % 10
                except Exception:
                    s10 = -1

                token_records[tok_id].append({
                    "problem":    problem,
                    "split":      split,
                    "answer_pos": ans_pos,
                    "token_id":   tok_id,
                    "subtask":    subtask,
                    "carry":      carry,
                    "sum_mod10":  s10,
                    "confidence": token_probs.get(ans_pos, 0.0),
                })
        print(f"ok ({sum(len(v) for v in token_records.values())} records so far)")

    # ── Sort by confidence, take top-N per token ──────────────────────────────
    print(f"\nFound {len(token_records)} active tokens. Calling Claude for interpretations...")

    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    token_results = []
    for tok_id, recs in sorted(token_records.items()):
        recs_sorted = sorted(recs, key=lambda r: -r["confidence"])
        top_recs    = recs_sorted[: args.top_n]
        mean_conf   = sum(r["confidence"] for r in top_recs) / len(top_recs)

        # top subtask stats
        sub_counts = defaultdict(int)
        for r in recs:
            sub_counts[r["subtask"]] += 1
        total = sum(sub_counts.values())
        top_sub    = max(sub_counts, key=sub_counts.get)
        top_purity = sub_counts[top_sub] / total

        print(f"  t{tok_id:>2}: n={len(recs):3d}  top={top_sub} {top_purity:.0%}"
              f"  conf={mean_conf:.2f}", end="  → ", flush=True)
        interp = interpret_token(client, tok_id, top_recs)
        print(interp[:60])

        token_results.append({
            "token_id":          tok_id,
            "n_occurrences":     len(recs),
            "mean_confidence":   mean_conf,
            "top_subtask":       top_sub,
            "top_subtask_purity": top_purity,
            "subtask_counts":    dict(sub_counts),
            "top_examples":      top_recs,
            "interpretation":    interp,
        })

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "model":  args.model,
        "top_n":  args.top_n,
        "splits": args.splits,
        "tokens": token_results,
    }
    json_path = OUT_DIR / "results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {json_path}")

    table = build_latex_table(token_results, args.n_table)
    tex_path = OUT_DIR / "table.tex"
    with open(tex_path, "w") as f:
        f.write(table)
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
