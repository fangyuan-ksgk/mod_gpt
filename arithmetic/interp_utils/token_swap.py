"""
Token swap experiments for SoRL interpretability.

Tests whether swapping one abstraction token ID for another causally
changes model predictions. Used to establish that specific tokens
encode specific arithmetic operations (carry state, borrow state).

Usage:
    python -m arithmetic.interp_utils.token_swap \
        --model add_sub_sorl_v1_abs30_K1_100K_1L3H510d \
        --swap_from 9 --swap_to 21 --splits add_C3,add_C4,add_C5,add_C6
"""
import torch
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from arithmetic.hub import load_model
from arithmetic.data.addition import get_eval_set
from arithmetic.train import QWEN3_TOKEN_MAP, QWEN3_INV_MAP
from arithmetic.evaluate import ArithmeticEvaluator
from sorl.sorl_trainer import infer_insert_mask, insert_tokens_with_padding, expand_prompt_len
from transformers import AutoTokenizer

PROMPT_LEN = 14
ANSWER_LEN = 7


def predict_with_swap(model, qwen_ids, K, base_v, pad_id, device,
                      swap_from=None, swap_to=None):
    """
    Autoregressive prediction with optional global token swap.

    After each recursion step, replaces ALL occurrences of swap_from
    with swap_to in the expanded sequence, then re-forwards to get
    updated logits before predicting the next digit.

    Args:
        model: SorlModelWrapper
        qwen_ids: (seq_len,) tensor of Qwen3 token IDs
        K: abstraction insertion frequency
        base_v: base vocabulary size (tokens >= base_v are abstractions)
        pad_id: padding token ID
        device: torch device
        swap_from: abstraction token ID to replace (None = no swap)
        swap_to: replacement abstraction token ID

    Returns:
        pred: list of predicted digit values (length ANSWER_LEN)
        swap_count: number of tokens swapped across all AR steps
    """
    seq = torch.cat([
        qwen_ids[:PROMPT_LEN].clone(),
        torch.zeros(ANSWER_LEN, dtype=torch.long, device=device),
    ])
    total_swaps = 0

    with torch.no_grad():
        for d in range(ANSWER_LEN):
            ids = seq.unsqueeze(0)
            attn = torch.ones_like(ids)
            pl = torch.tensor([PROMPT_LEN], dtype=torch.long, device=device)

            im = infer_insert_mask(ids, K, attn)
            ep = expand_prompt_len(pl, im)
            ed, ea = insert_tokens_with_padding(
                ids, attn, im, model.vocab_sizes[0], pad_id)

            data, ppt, logits = model.recursion(
                ed, ea, max_iterations=2,
                memory_span_abs=1792, memory_span_traj=1792,
                temperature=0.0, prompt_len=ep,
            )

            # Apply swap after recursion, before reading logits
            if swap_from is not None:
                mask = data[0] == (base_v + swap_from)
                n_swapped = mask.sum().item()
                if n_swapped > 0:
                    total_swaps += n_swapped
                    data[0][mask] = base_v + swap_to
                    block_mask = model._create_sorl_block_mask(
                        data, 1792, 1792)
                    out = model.model.forward(
                        input_ids=data, attention_mask=ea,
                        block_mask=block_mask, use_cache=False)
                    logits = out.logits
                    if logits.dim() == 2:
                        logits = logits.unsqueeze(0)

            is_traj = data[0] < base_v
            traj_idx = is_traj.nonzero(as_tuple=True)[0]
            ap = traj_idx[PROMPT_LEN + d].item()
            seq[PROMPT_LEN + d] = logits[0, ap - 1, :base_v].argmax()

    pred = [QWEN3_INV_MAP.get(t.item(), -1) for t in seq[PROMPT_LEN:]]
    return pred, total_swaps


def run_swap_experiment(model_name, swap_from, swap_to, splits,
                        device="cuda:0", n_per_split=100, K=1):
    """
    Run a full swap experiment: compare normal predictions vs swapped
    predictions across multiple eval splits.

    Returns dict with per-split and aggregate results.
    """
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model, cfg, _ = load_model(model_name, device=device)
    base_v = model.vocab_sizes[0].item()
    pad_id = tokenizer.pad_token_id
    model.eval()

    evaluator = ArithmeticEvaluator(model, tokenizer, device=device)
    categories = get_eval_set()

    results = {}
    total = {"normal_correct": 0, "swap_correct": 0,
             "fixed": 0, "broke": 0, "n": 0}

    for split in splits:
        examples = categories.get(split, [])
        if not examples:
            continue

        split_result = {"normal_correct": 0, "swap_correct": 0,
                        "fixed": 0, "broke": 0, "n": len(examples),
                        "fix_examples": [], "break_examples": []}

        for ex in examples:
            qwen_ids = torch.tensor(
                [QWEN3_TOKEN_MAP[t] for t in ex.tokens],
                dtype=torch.long, device=device)
            target = list(ex.tokens[14:])  # internal token IDs (0-9)

            # Normal prediction (use evaluator for consistency)
            with torch.no_grad():
                pred_n = evaluator._predict_sorl(qwen_ids, K=K)
            pred_normal = [QWEN3_INV_MAP.get(t.item(), -1)
                           for t in pred_n]

            # Swap prediction
            pred_swap, n_swaps = predict_with_swap(
                model, qwen_ids, K, base_v, pad_id, device,
                swap_from=swap_from, swap_to=swap_to)

            n_ok = pred_normal == target
            s_ok = pred_swap == target

            if n_ok:
                split_result["normal_correct"] += 1
            if s_ok:
                split_result["swap_correct"] += 1
            if not n_ok and s_ok:
                split_result["fixed"] += 1
                orig = ex.tokens
                q = (''.join(str(t) for t in orig[:6]) +
                     ('+' if orig[6] == 10 else '-') +
                     ''.join(str(t) for t in orig[7:13]))
                split_result["fix_examples"].append({
                    "q": q,
                    "target": ''.join(str(d) for d in target),
                    "normal": ''.join(str(d) for d in pred_normal),
                    "swapped": ''.join(str(d) for d in pred_swap),
                    "n_swaps": n_swaps,
                })
            if n_ok and not s_ok:
                split_result["broke"] += 1

        results[split] = split_result
        for k in total:
            if k in split_result:
                total[k] += split_result[k]

    results["total"] = total
    del model
    torch.cuda.empty_cache()
    return results


def print_results(results, swap_from, swap_to):
    """Pretty-print swap experiment results."""
    print(f"\nt{swap_from} → t{swap_to} swap experiment:")
    print(f"{'Split':<10s} {'N':>4s} {'Normal':>8s} {'Swap':>8s} "
          f"{'Fixed':>6s} {'Broke':>6s}")
    print("-" * 44)

    for split in sorted(results.keys()):
        if split == "total":
            continue
        r = results[split]
        n = r["n"]
        na = r["normal_correct"] * 100 // n
        sa = r["swap_correct"] * 100 // n
        print(f"  {split:<8s} {n:>4d} {na:>6d}% {sa:>6d}% "
              f"{r['fixed']:>5d} {r['broke']:>5d}")

    t = results["total"]
    print(f"\n  Total: fixed={t['fixed']}, broke={t['broke']}, "
          f"ratio={t['fixed']}/{t['broke']} "
          f"({'inf' if t['broke'] == 0 else str(t['fixed']//t['broke'])}:1)")

    # Show example fixes
    all_fixes = []
    for split, r in results.items():
        if split == "total":
            continue
        for ex in r.get("fix_examples", [])[:2]:
            all_fixes.append((split, ex))

    if all_fixes:
        print(f"\nExample fixes:")
        for split, ex in all_fixes[:5]:
            print(f"  {split}: {ex['q']} = {ex['target']}")
            print(f"    Normal:  {ex['normal']} ✗")
            print(f"    Swapped: {ex['swapped']} ✓ "
                  f"({ex['n_swaps']} tokens swapped)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",
                    default="add_sub_sorl_v1_abs30_K1_100K_1L3H510d")
    p.add_argument("--swap_from", type=int, default=9)
    p.add_argument("--swap_to", type=int, default=21)
    p.add_argument("--splits", default="add_C3,add_C4,add_C5,add_C6")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n_per_split", type=int, default=100)
    p.add_argument("--K", type=int, default=1)
    args = p.parse_args()

    splits = args.splits.split(",")
    results = run_swap_experiment(
        args.model, args.swap_from, args.swap_to, splits,
        device=args.device, n_per_split=args.n_per_split, K=args.K)
    print_results(results, args.swap_from, args.swap_to)
