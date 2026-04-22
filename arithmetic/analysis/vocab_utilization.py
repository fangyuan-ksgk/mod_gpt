"""Per-model vocabulary usage statistics for abstract tokens.

Computes statistics on how the abstract token vocabulary is utilized:
effective vocabulary size, top-k concentration, entropy, Zipf distribution
fit. Analyzes whether rare tokens map to rare subtasks and compares
vocabulary usage patterns across multiple models.
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


def compute_vocab_stats(
    model, tokenizer, K: int, n_samples: int = 1000
) -> dict:
    """Compute vocabulary utilization statistics for abstract tokens.

    Runs the model on n_samples examples, collects abstract token assignments,
    and computes: effective vocabulary (tokens used at least once), top-k
    concentration (fraction of usage by top-k tokens), Shannon entropy of
    the usage distribution, and Zipf distribution fit parameters.

    Args:
        model: SoRL-trained model.
        tokenizer: Tokenizer for the model.
        K: Number of abstract tokens per sequence.
        n_samples: Number of examples to sample.

    Returns:
        Dict with keys: "effective_vocab", "total_vocab", "top1_pct",
        "top3_pct", "top5_pct", "entropy", "zipf_alpha", "zipf_r2",
        "token_frequencies" (dict of token_id -> count).
    """
    raise NotImplementedError("TODO")


def frequency_vs_subtask(
    model, tokenizer, K: int, n_samples: int = 1000
) -> dict:
    """Analyze whether rare abstract tokens map to rare arithmetic subtasks.

    Cross-tabulates token frequency rank with subtask frequency rank to
    check if there is a correspondence between token rarity and subtask rarity.

    Args:
        model: SoRL-trained model.
        tokenizer: Tokenizer for the model.
        K: Number of abstract tokens per sequence.
        n_samples: Number of examples to sample.

    Returns:
        Dict with keys: "token_freq_rank", "subtask_freq_rank",
        "rank_correlation" (Spearman), "per_token_dominant_subtask".
    """
    raise NotImplementedError("TODO")


def compare_across_models(model_names: list[str], K: int) -> dict:
    """Compare vocabulary statistics across multiple models.

    Args:
        model_names: List of HF model subfolder paths.
        K: Number of abstract tokens per sequence.

    Returns:
        Dict with keys: "per_model" (dict of model_name -> vocab stats),
        "comparison_table" (list of dicts for tabular display).
    """
    raise NotImplementedError("TODO")


def plot_zipf(stats: dict, path: str) -> None:
    """Plot Zipf distribution of abstract token usage.

    Args:
        stats: Output of compute_vocab_stats.
        path: File path to save the figure.
    """
    import matplotlib.pyplot as plt

    raise NotImplementedError("TODO")


def plot_frequency_subtask(freq_subtask: dict, path: str) -> None:
    """Plot token frequency rank vs subtask frequency rank.

    Args:
        freq_subtask: Output of frequency_vs_subtask.
        path: File path to save the figure.
    """
    import matplotlib.pyplot as plt

    raise NotImplementedError("TODO")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vocabulary utilization analysis"
    )
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Comma-separated list of model paths",
    )
    parser.add_argument("--K", type=int, default=4, help="Abstract tokens per sequence")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="arithmetic/analysis/results",
        help="Output directory",
    )
    args = parser.parse_args()

    model_names = [m.strip() for m in args.models.split(",")]
    comparison = compare_across_models(model_names, K=args.K)

    output_dir = Path(args.output_dir)
    save_results(comparison, str(output_dir / "vocab_comparison.json"))

    for name in model_names:
        model, tokenizer = load_model(name)
        stats = compute_vocab_stats(model, tokenizer, K=args.K)
        safe_name = name.replace("/", "_")
        plot_zipf(stats, str(output_dir / f"zipf_{safe_name}.png"))
