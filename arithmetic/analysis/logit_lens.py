"""Logit lens and future lens at abstraction positions.

Projects intermediate hidden states through the language model head to inspect
what the model is "thinking" at abstraction positions. Logit lens measures
prediction accuracy of the current token at each layer; future lens measures
prediction accuracy of tokens at a future offset. Compares SoRL models
(abstraction positions) vs baseline models (all positions).
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from arithmetic.data.hub import load_model
from arithmetic.data.addition import get_eval_set, ArithmeticExample
from arithmetic.training.evaluate import ArithmeticEvaluator


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


def logit_lens_at_positions(
    model,
    tokenizer,
    dataset,
    positions: str = "abstraction",
    layer: int = -1,
    n_samples: int = 500,
) -> dict:
    """Project hidden states through lm_head and measure prediction accuracy.

    At specified positions (abstraction positions for SoRL, or all positions),
    projects the hidden state at each layer through the unembedding matrix and
    checks whether the top-1 prediction matches the actual next token.

    Args:
        model: Trained model.
        tokenizer: Tokenizer for the model.
        dataset: Dataset of arithmetic examples.
        positions: "abstraction" for SoRL abstract token positions, "all" for
            every position in the sequence.
        layer: Which layer to analyze (-1 for all layers).
        n_samples: Number of sequences to sample.

    Returns:
        Dict with keys: "per_layer_accuracy" (list of floats), "per_layer_top5",
        "per_layer_entropy", "n_positions", "n_samples".
    """
    raise NotImplementedError("TODO")


def future_lens(
    model,
    tokenizer,
    dataset,
    positions: str = "abstraction",
    future_offset: int = 2,
    n_samples: int = 500,
) -> dict:
    """Predict tokens at a future offset from hidden states at current positions.

    Trains a linear map from hidden states at position t to the token at
    position t + future_offset. Measures how much future information is
    accessible at abstraction positions vs arbitrary positions.

    Args:
        model: Trained model.
        tokenizer: Tokenizer for the model.
        dataset: Dataset of arithmetic examples.
        positions: "abstraction" or "all".
        future_offset: Number of positions ahead to predict.
        n_samples: Number of sequences to sample.

    Returns:
        Dict with keys: "per_layer_accuracy", "per_layer_top5",
        "future_offset", "n_positions", "n_samples".
    """
    raise NotImplementedError("TODO")


def compare_sorl_vs_baseline(
    sorl_model_name: str, baseline_model_name: str
) -> dict:
    """Run logit lens and future lens on both models and compare.

    Args:
        sorl_model_name: Path/name of the SoRL-trained model.
        baseline_model_name: Path/name of the baseline model.

    Returns:
        Dict with keys: "logit_lens_sorl", "logit_lens_baseline",
        "future_lens_sorl", "future_lens_baseline", "delta_summary".
    """
    raise NotImplementedError("TODO")


def plot_lens_comparison(results: dict, path: str) -> None:
    """Plot layer-by-layer logit lens and future lens comparison.

    Args:
        results: Output of compare_sorl_vs_baseline.
        path: File path to save the figure.
    """
    import matplotlib.pyplot as plt

    raise NotImplementedError("TODO")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Logit lens and future lens analysis"
    )
    parser.add_argument(
        "--sorl_model", type=str, required=True, help="SoRL model path"
    )
    parser.add_argument(
        "--baseline_model", type=str, required=True, help="Baseline model path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="arithmetic/analysis/results",
        help="Output directory",
    )
    args = parser.parse_args()

    results = compare_sorl_vs_baseline(
        sorl_model_name=args.sorl_model,
        baseline_model_name=args.baseline_model,
    )

    output_dir = Path(args.output_dir)
    save_results(results, str(output_dir / "logit_lens.json"))
    plot_lens_comparison(results, str(output_dir / "logit_lens_comparison.png"))
