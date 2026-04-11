"""
Systematic token-level intervention analysis.

Runs knockout, swap, replace, shuffle interventions across complexity splits
and abstraction positions. Produces per-digit, per-subtask effect tables.

Builds on primitives in arithmetic.interp_utils.interventions.

Intervention types:
  1. Knockout: remove abs tokens → which digits break?
  2. Swap: exchange abs tokens between paired examples → does the answer follow the token?
  3. Replace: substitute specific token IDs → which tokens are load-bearing?
  4. Shuffle: permute token positions → does position matter or just presence?
  5. Targeted knockout: knock out only tokens near carry/borrow cascade positions

Usage:
    python -m arithmetic.analysis.token_interventions \
        --model add_sub_sorl_abs10_K1_500K \
        --K 1 \
        --output_dir arithmetic/analysis/results/interventions
"""
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import Optional

from arithmetic.interp_utils.interventions import (
    token_knockout, token_swap, token_replace, token_replace_random,
    token_shuffle, knockout_at_digit, swap_at_digit,
    get_abs_positions, measure_intervention_effect,
)


def save_results(results: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ── 1. Global interventions per split ─────────────────────────────


def knockout_by_split(
    model,
    tokenizer,
    eval_set: dict,
    K: int,
    n_digits: int = 6,
) -> dict:
    """
    For each complexity split, knock out ALL abs tokens and measure accuracy drop.
    Shows how much the model depends on abstractions per difficulty level.

    Returns:
        {"splits": {split_name: {"original_acc": float, "knockout_acc": float,
                                  "drop": float, "n_examples": int}}}
    """
    raise NotImplementedError("TODO")


def shuffle_by_split(
    model,
    tokenizer,
    eval_set: dict,
    K: int,
    n_digits: int = 6,
) -> dict:
    """
    For each split, shuffle abs token positions. Tests whether token IDENTITY
    at specific POSITIONS matters, or just token presence.

    If shuffle doesn't hurt: tokens are just "thinking slots" (position doesn't matter).
    If shuffle hurts: tokens carry position-specific information.

    Returns:
        {"splits": {split_name: {"original_acc": float, "shuffled_acc": float,
                                  "drop": float}}}
    """
    raise NotImplementedError("TODO")


def replace_by_token_id(
    model,
    tokenizer,
    eval_set: dict,
    K: int,
    n_digits: int = 6,
) -> dict:
    """
    For each unique abs token ID, replace ALL occurrences with the placeholder
    and measure which splits break. Identifies which tokens are load-bearing.

    Returns:
        {"token_id": {split_name: {"drop": float}},
         "most_critical_token": int, "least_critical_token": int}
    """
    raise NotImplementedError("TODO")


# ── 2. Per-digit interventions ────────────────────────────────────


def knockout_per_digit(
    model,
    tokenizer,
    eval_set: dict,
    K: int,
    n_digits: int = 6,
) -> dict:
    """
    For each answer digit position, knock out the abs tokens that precede it.
    Shows which digits depend on abstraction tokens.

    Cascade digits (UC/US/UB/UD) should depend more than base digits (SA/MD).

    Returns:
        {"digit_idx": {"original_acc": float, "knockout_acc": float,
                       "drop": float, "dominant_subtask": str}}
    """
    raise NotImplementedError("TODO")


def swap_per_digit(
    model,
    tokenizer,
    eval_set: dict,
    K: int,
    n_digits: int = 6,
    n_pairs: int = 200,
) -> dict:
    """
    For each answer digit, swap abs tokens near that digit between paired examples.
    Tests whether the swapped answer digit follows the donor's token.

    The "answer follows the token" rate is the causal evidence that
    the token encodes information FOR that digit.

    Returns:
        {"digit_idx": {"follows_donor_rate": float, "breaks_rate": float,
                       "n_pairs": int}}
    """
    raise NotImplementedError("TODO")


# ── 3. Targeted cascade interventions ─────────────────────────────


def knockout_cascade_tokens(
    model,
    tokenizer,
    eval_set: dict,
    K: int,
    n_digits: int = 6,
) -> dict:
    """
    Only knock out abs tokens at positions within a carry/borrow cascade.
    Uses ground-truth ST/MB labels to identify cascade positions.

    Tests the specific claim: "abstraction tokens at cascade positions
    encode carry/borrow propagation information."

    Returns:
        {"cascade_knockout_acc": float, "non_cascade_knockout_acc": float,
         "cascade_only_drop": float, "non_cascade_only_drop": float}
    """
    raise NotImplementedError("TODO")


def swap_cascade_vs_noncascade(
    model,
    tokenizer,
    eval_set: dict,
    K: int,
    n_digits: int = 6,
    n_pairs: int = 200,
) -> dict:
    """
    Swap abs tokens only at cascade positions vs only at non-cascade positions.
    If cascade swaps break accuracy more, tokens at cascade positions carry
    more information.

    Returns:
        {"cascade_swap_drop": float, "non_cascade_swap_drop": float,
         "ratio": float}
    """
    raise NotImplementedError("TODO")


# ── 4. Cross-complexity transfer ──────────────────────────────────


def cross_complexity_swap(
    model,
    tokenizer,
    eval_set: dict,
    K: int,
    n_digits: int = 6,
    n_pairs: int = 100,
) -> dict:
    """
    Swap abs tokens between examples of DIFFERENT complexity levels.
    E.g., swap tokens from an S0 (no cascade) example into an S5 example.

    If S5 accuracy drops when given S0 tokens: the tokens genuinely encode
    cascade-specific information, not just generic "thinking" signal.

    Returns:
        {"source_split → target_split": {"accuracy": float, "drop": float}}
    """
    raise NotImplementedError("TODO")


# ── Visualization ─────────────────────────────────────────────────


def plot_knockout_heatmap(results: dict, path: str):
    """Heatmap: digit position × complexity split → accuracy drop from knockout."""
    raise NotImplementedError("TODO")


def plot_swap_follows_donor(results: dict, path: str):
    """Bar chart: per-digit rate at which swapped answer follows the donor's token."""
    raise NotImplementedError("TODO")


def plot_cascade_vs_noncascade(results: dict, path: str):
    """Side-by-side: cascade vs non-cascade intervention effects."""
    raise NotImplementedError("TODO")


# ── Main ──────────────────────────────────────────────────────────


def run_all(
    model_name: str,
    K: int = 1,
    device: str = "cuda",
    output_dir: str = "arithmetic/analysis/results/interventions",
) -> dict:
    """Full intervention suite: knockout, shuffle, swap, targeted cascade."""
    raise NotImplementedError("TODO")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Token-level intervention analysis")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="arithmetic/analysis/results/interventions")
    args = parser.parse_args()

    results = run_all(args.model, K=args.K, device=args.device, output_dir=args.output_dir)
    print(json.dumps({k: v for k, v in results.items() if not isinstance(v, torch.Tensor)}, indent=2))
