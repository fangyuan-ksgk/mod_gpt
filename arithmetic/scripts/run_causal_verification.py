"""
Causal verification of SoRL abstraction tokens.

For each abstraction token that correlates with a subtask (carry/borrow),
verify CAUSALLY that the token encodes that information:

1. KNOCKOUT: Replace token with placeholder → accuracy drops on that subtask
2. SWAP: Swap tokens between carry/no-carry pairs → answer changes predictably
3. SHUFFLE: Permute token IDs within sequence → accuracy drops (identity matters)

This parallels Quirke's activation patching (their §4) but operates on
explicit tokens instead of hidden activations.

Usage:
    python -m arithmetic.scripts.run_causal_verification --model add_sub_sorl_v1_abs30_100K --K 4
"""
import argparse
import torch
import sys
import json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from arithmetic.hub import load_model
from arithmetic.datasets.addition import get_eval_set
from arithmetic.train import QWEN3_TOKEN_MAP, QWEN3_INV_MAP
from arithmetic.interp_utils.interventions import (
    token_knockout, token_shuffle, token_replace_random, get_abs_positions,
)
from sorl.sorl_trainer import infer_insert_mask, insert_tokens_with_padding, expand_prompt_len
from transformers import AutoTokenizer

PROMPT_LEN = 14
ANSWER_LEN = 7


def expand_and_recurse(model, qwen_ids, K, device, base_v, pad_id):
    """Run expansion + recursion, return (expanded_data, expanded_attn, expanded_prompt_len)."""
    seq = qwen_ids.unsqueeze(0)
    attn = torch.ones_like(seq)
    pl = torch.tensor([PROMPT_LEN], dtype=torch.long, device=device)

    im = infer_insert_mask(seq, K, attn)
    ep = expand_prompt_len(pl, im)
    ed, ea = insert_tokens_with_padding(seq, attn, im, model.vocab_sizes[0], pad_id)

    data, ppt, logits = model.recursion(
        ed, ea, max_iterations=2,
        memory_span_abs=1792, memory_span_traj=1792,
        temperature=0.0, prompt_len=ep,
    )
    return data, ea, ep, logits


def predict_from_expanded(model, data, attn, ep, base_v):
    """Given expanded data with filled abstractions, predict answer digits autoregressively."""
    # Use the logits from the last recursion forward pass
    # to predict each answer digit
    is_traj = data[0] < base_v
    traj_indices = is_traj.nonzero(as_tuple=True)[0]

    block_mask = model._create_sorl_block_mask(data, 1792, 1792)
    out = model.model.forward(input_ids=data, attention_mask=attn, block_mask=block_mask, use_cache=False)
    logits = out.logits

    pred = []
    for d in range(ANSWER_LEN):
        answer_pos = traj_indices[PROMPT_LEN + d].item()
        pred_token = logits[0, answer_pos - 1, :base_v].argmax().item()
        pred.append(pred_token)
    return pred


def run_causal_verification(model_name, K, device, n_per_split=100):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model, config, metrics = load_model(model_name, device=device)
    base_v = model.vocab_sizes[0].item()
    pad_id = tokenizer.pad_token_id
    model.eval()

    categories = get_eval_set(6, "add_sub", N=n_per_split)
    all_examples = []
    for split_name, examples in categories.items():
        for ex in examples:
            all_examples.append((split_name, ex))

    results = {
        "model": model_name, "K": K, "n_examples": len(all_examples),
        "baseline": {"correct": 0, "total": 0},
        "knockout": {"correct": 0, "total": 0},
        "shuffle": {"correct": 0, "total": 0},
        "random": {"correct": 0, "total": 0},
        "per_split": {},
    }

    # Per-split tracking
    split_results = defaultdict(lambda: {
        "baseline": {"correct": 0, "total": 0},
        "knockout": {"correct": 0, "total": 0},
        "shuffle": {"correct": 0, "total": 0},
        "random": {"correct": 0, "total": 0},
    })

    print(f"Running causal verification on {model_name} (K={K})...")
    print(f"  {len(all_examples)} examples")

    for split_name, ex in all_examples:
        qwen_ids = torch.tensor(
            [QWEN3_TOKEN_MAP[t] for t in ex.tokens],
            dtype=torch.long, device=device,
        )
        target = qwen_ids[PROMPT_LEN:].tolist()

        with torch.no_grad():
            # 1. Baseline: normal recursion
            data, attn, ep, logits = expand_and_recurse(model, qwen_ids, K, device, base_v, pad_id)
            pred_baseline = predict_from_expanded(model, data, attn, ep, base_v)
            baseline_correct = (pred_baseline == target)

            # 2. Knockout: replace all abs tokens with placeholder
            data_ko = token_knockout(data, get_abs_positions(data, base_v), base_v)
            pred_ko = predict_from_expanded(model, data_ko, attn, ep, base_v)
            ko_correct = (pred_ko == target)

            # 3. Shuffle: permute abs token identities
            data_shuf = token_shuffle(data, base_v)
            pred_shuf = predict_from_expanded(model, data_shuf, attn, ep, base_v)
            shuf_correct = (pred_shuf == target)

            # 4. Random: replace with random abs tokens
            abs_vocab = config.get("abs_vocab", 10)
            data_rand = token_replace_random(data, base_v, abs_vocab)
            pred_rand = predict_from_expanded(model, data_rand, attn, ep, base_v)
            rand_correct = (pred_rand == target)

        for label, correct in [("baseline", baseline_correct), ("knockout", ko_correct),
                                ("shuffle", shuf_correct), ("random", rand_correct)]:
            if correct:
                results[label]["correct"] += 1
                split_results[split_name][label]["correct"] += 1
            results[label]["total"] += 1
            split_results[split_name][label]["total"] += 1

    # Compute accuracies
    print(f"\n{'Intervention':<12s} {'Accuracy':>10s} {'Delta':>8s}")
    print("-" * 32)
    base_acc = results["baseline"]["correct"] / max(results["baseline"]["total"], 1)
    for label in ["baseline", "knockout", "shuffle", "random"]:
        acc = results[label]["correct"] / max(results[label]["total"], 1)
        delta = acc - base_acc
        print(f"  {label:<10s} {acc:>9.1%} {delta:>+7.1%}")
        results[label]["accuracy"] = acc

    # Per-split
    print(f"\n{'Split':<15s} {'Baseline':>9s} {'Knockout':>9s} {'Shuffle':>9s} {'Random':>9s}")
    print("-" * 55)
    for split in sorted(split_results.keys()):
        sr = split_results[split]
        accs = {}
        for label in ["baseline", "knockout", "shuffle", "random"]:
            accs[label] = sr[label]["correct"] / max(sr[label]["total"], 1)
        print(f"  {split:<13s} {accs['baseline']:>8.0%} {accs['knockout']:>8.0%} {accs['shuffle']:>8.0%} {accs['random']:>8.0%}")

    results["per_split"] = {
        s: {l: {"accuracy": sr[l]["correct"] / max(sr[l]["total"], 1)}
            for l in ["baseline", "knockout", "shuffle", "random"]}
        for s, sr in split_results.items()
    }

    # Save
    out_dir = f"arithmetic/interp_results/{model_name}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{out_dir}/causal_verification.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_dir}/causal_verification.json")

    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="add_sub_sorl_v1_abs30_100K")
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n_per_split", type=int, default=100)
    args = p.parse_args()

    run_causal_verification(args.model, args.K, args.device, args.n_per_split)
