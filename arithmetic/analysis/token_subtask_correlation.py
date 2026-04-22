"""Map abstract tokens to Quirke subtask labels.

Computes the correspondence between SoRL abstract token assignments and
the ground-truth arithmetic subtask labels (BA, MC1, MS9, UC1, US9 for
addition; BS, MB1, MD9, UB1, UD9 for subtraction). Produces confusion-style
matrices and precision/recall metrics for each token as a subtask classifier.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from arithmetic.hub import load_model
from arithmetic.data.addition import get_eval_set, ArithmeticExample
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


def compute_token_subtask_matrix(
    model, tokenizer, K: int, n_samples: int = 2000
) -> dict:
    """Compute P(token | subtask) and P(subtask | token) matrices.

    Runs the model on n_samples arithmetic examples, collects the abstract
    token assigned at each position, and cross-tabulates with the ground-truth
    subtask label for that digit position.

    Args:
        model: SoRL-trained model.
        tokenizer: Tokenizer for the model.
        K: Number of abstract tokens per sequence.
        n_samples: Number of examples to sample.

    Returns:
        Dict with keys: "p_token_given_subtask", "p_subtask_given_token",
        "joint_counts", "token_ids", "subtask_names".
    """
    raise NotImplementedError("TODO")


def precision_recall_per_token(matrix: dict) -> dict:
    """Compute precision and recall of each token as a subtask classifier.

    For each abstract token, treats its most-assigned subtask as the
    positive class and computes precision (fraction of token usage on that
    subtask) and recall (fraction of that subtask captured by this token).

    Args:
        matrix: Output of compute_token_subtask_matrix.

    Returns:
        Dict with keys: "per_token" (list of dicts with token_id, best_subtask,
        precision, recall, f1), "macro_f1".
    """
    raise NotImplementedError("TODO")


def plot_confusion_matrix(matrix: dict, path: str) -> None:
    """Plot heatmap of token x subtask assignment matrix.

    Args:
        matrix: Output of compute_token_subtask_matrix.
        path: File path to save the figure.
    """
    import matplotlib.pyplot as plt

    raise NotImplementedError("TODO")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Token-subtask correlation analysis"
    )
    parser.add_argument("--model", type=str, required=True, help="HF model subfolder")
    parser.add_argument("--K", type=int, default=4, help="Abstract tokens per sequence")
    parser.add_argument("--n_samples", type=int, default=2000, help="Number of samples")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="arithmetic/analysis/results",
        help="Output directory",
    )
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)
    matrix = compute_token_subtask_matrix(model, tokenizer, K=args.K, n_samples=args.n_samples)
    pr = precision_recall_per_token(matrix)

    output_dir = Path(args.output_dir)
    save_results(matrix, str(output_dir / "token_subtask_matrix.json"))
    save_results(pr, str(output_dir / "token_subtask_precision_recall.json"))
    plot_confusion_matrix(matrix, str(output_dir / "token_subtask_heatmap.png"))
