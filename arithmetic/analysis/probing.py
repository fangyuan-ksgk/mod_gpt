"""Linear probes at abstraction positions vs baseline trajectory positions.

Trains linear and MLP probes on hidden states extracted from abstraction
positions (SoRL models) or all positions (baseline models) to predict
arithmetic features: resolved cascade state, future digits, cascade length
remaining. Compares probe accuracy between SoRL and baseline to measure
whether abstract tokens concentrate task-relevant information.
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


def collect_hidden_states(
    model,
    tokenizer,
    dataset,
    K: Optional[int] = None,
    layer: int = -1,
    n_samples: int = 1000,
) -> dict:
    """Collect hidden states and labels at abstraction or all positions.

    For SoRL models (K is not None), collects hidden states at the K abstraction
    positions per sequence. For baseline models (K is None), collects at all
    positions. Labels include subtask variant labels (SV[n]), future digit values,
    and cascade length remaining.

    Args:
        model: Trained model (SoRL or baseline).
        tokenizer: Tokenizer for the model.
        dataset: Dataset of arithmetic examples.
        K: Number of abstract tokens per sequence. None for baseline models.
        layer: Which transformer layer to extract from (-1 for last).
        n_samples: Number of sequences to sample.

    Returns:
        Dict with keys: "hidden_states" (list of arrays), "labels" (dict of
        label_name -> list), "positions" (list of position indices),
        "n_samples", "layer", "hidden_dim".
    """
    raise NotImplementedError("TODO")


def train_probe(
    hidden_states,
    labels,
    probe_type: str = "linear",
) -> dict:
    """Train a linear or MLP probe on hidden states to predict labels.

    Args:
        hidden_states: Array of hidden state vectors.
        labels: Array of target labels (categorical or continuous).
        probe_type: "linear" for logistic regression, "mlp" for 1-hidden-layer MLP.

    Returns:
        Dict with keys: "accuracy", "f1_macro", "f1_per_class",
        "confusion_matrix", "probe_type", "n_train", "n_test".
    """
    raise NotImplementedError("TODO")


def compare_sorl_vs_baseline(
    sorl_model_name: str,
    baseline_model_name: str,
    targets: list[str],
) -> dict:
    """Run probes on both SoRL and baseline models and compare.

    For each target variable, trains probes on hidden states from both models
    and reports the accuracy difference.

    Args:
        sorl_model_name: Path/name of the SoRL-trained model.
        baseline_model_name: Path/name of the baseline model.
        targets: List of target variable names to probe for. Options:
            "resolved_cascade", "future_digit_2", "future_digit_3",
            "cascade_length".

    Returns:
        Dict with keys: "per_target" (dict of target -> {sorl_acc, baseline_acc,
        delta}), "summary".
    """
    raise NotImplementedError("TODO")


def plot_probe_comparison(results: dict, path: str) -> None:
    """Plot side-by-side probe accuracy comparison between SoRL and baseline.

    Args:
        results: Output of compare_sorl_vs_baseline.
        path: File path to save the figure.
    """
    import matplotlib.pyplot as plt

    raise NotImplementedError("TODO")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Linear probing at abstraction positions"
    )
    parser.add_argument(
        "--sorl_model", type=str, required=True, help="SoRL model path"
    )
    parser.add_argument(
        "--baseline_model", type=str, required=True, help="Baseline model path"
    )
    parser.add_argument(
        "--targets",
        type=str,
        nargs="+",
        default=["resolved_cascade", "future_digit_2", "future_digit_3", "cascade_length"],
        help="Target variables to probe",
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
        targets=args.targets,
    )

    output_dir = Path(args.output_dir)
    save_results(results, str(output_dir / "probe_comparison.json"))
    plot_probe_comparison(results, str(output_dir / "probe_comparison.png"))
