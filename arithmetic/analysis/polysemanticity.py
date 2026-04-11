"""Check if abstract tokens map 1-to-1 or many-to-many with subtasks.

Measures the polysemanticity of each abstract token by computing the entropy
of its subtask distribution. Monosemantic tokens have low entropy (map to one
subtask); polysemantic tokens have high entropy (used across many subtasks).
Also analyzes token usage across complexity levels (digit counts).
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from arithmetic.hub import load_model
from arithmetic.datasets.addition import get_eval_set, ArithmeticExample
from arithmetic.evaluate import ArithmeticEvaluator


def save_results(results: dict, path: str) -> None:
    """Save results dict to JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(results, f, indent=2, default=str)


def load_results(path: str) -> dict:
    """Load results dict from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def compute_polysemanticity_score(token_subtask_matrix: dict) -> dict:
    """Compute polysemanticity score for each abstract token.

    For each token, computes entropy of P(subtask | token). Low entropy means
    monosemantic (token maps cleanly to one subtask), high entropy means
    polysemantic (token is used across many subtasks indiscriminately).

    Args:
        token_subtask_matrix: Output of token_subtask_correlation.compute_token_subtask_matrix.

    Returns:
        Dict with keys: "per_token" (list of dicts with token_id, entropy,
        dominant_subtask, dominant_fraction), "mean_entropy", "median_entropy",
        "n_monosemantic" (entropy < 0.5), "n_polysemantic" (entropy > 1.5).
    """
    raise NotImplementedError("TODO")


def per_complexity_usage(
    model, tokenizer, K: int, n_samples: int = 1000
) -> dict:
    """Analyze which abstract tokens appear at which complexity levels.

    Complexity levels correspond to operand digit counts: S0-S6 for addition
    (sum digit counts), M0-M5 for multiplication. Tracks token frequency
    distributions per complexity level.

    Args:
        model: SoRL-trained model.
        tokenizer: Tokenizer for the model.
        K: Number of abstract tokens per sequence.
        n_samples: Number of examples to sample.

    Returns:
        Dict with keys: "usage_matrix" (complexity x token_id counts),
        "complexity_levels", "token_ids", "complexity_conditional_entropy".
    """
    raise NotImplementedError("TODO")


def plot_polysemanticity(scores: dict, path: str) -> None:
    """Plot polysemanticity score distribution and per-token breakdown.

    Args:
        scores: Output of compute_polysemanticity_score.
        path: File path to save the figure.
    """
    import matplotlib.pyplot as plt

    raise NotImplementedError("TODO")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Polysemanticity analysis of abstract tokens"
    )
    parser.add_argument("--model", type=str, required=True, help="HF model subfolder")
    parser.add_argument("--K", type=int, default=4, help="Abstract tokens per sequence")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="arithmetic/analysis/results",
        help="Output directory",
    )
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)
    usage = per_complexity_usage(model, tokenizer, K=args.K)

    output_dir = Path(args.output_dir)
    save_results(usage, str(output_dir / "complexity_usage.json"))
