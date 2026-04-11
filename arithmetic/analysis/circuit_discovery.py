"""EAP (Edge Attribution Patching) with SoRL tokens as circuit anchors.

Discovers computational circuits in arithmetic models using edge attribution
patching. For SoRL models, constrains the analysis to edges into/out of
abstraction positions and partitions by assigned abstract token to find
token-specific circuits. Compares circuit structure between SoRL and baseline.
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


def compute_eap_edges(
    model, tokenizer, dataset, n_samples: int = 200
) -> dict:
    """Standard EAP on all edges in the computational graph.

    Computes attribution scores for all edges (attention head -> attention head,
    attention head -> MLP, etc.) using activation patching with a corrupted
    input distribution.

    Args:
        model: Trained model.
        tokenizer: Tokenizer for the model.
        dataset: Dataset of arithmetic examples.
        n_samples: Number of clean/corrupted pairs to use.

    Returns:
        Dict with keys: "edges" (list of {src, dst, attribution}),
        "n_edges", "n_samples", "top_edges_by_attribution".
    """
    raise NotImplementedError("TODO")


def compute_eap_constrained(
    model, tokenizer, dataset, K: int, n_samples: int = 200
) -> dict:
    """EAP constrained to edges into/out of abstraction positions.

    Only computes attribution for edges where at least one endpoint is an
    abstraction position. This dramatically reduces the search space and
    focuses on the circuit through abstract tokens.

    Args:
        model: SoRL-trained model.
        tokenizer: Tokenizer for the model.
        dataset: Dataset of arithmetic examples.
        K: Number of abstract tokens per sequence.
        n_samples: Number of clean/corrupted pairs.

    Returns:
        Dict with keys: "edges" (list of {src, dst, attribution}),
        "n_edges_constrained", "n_edges_total", "compression_ratio",
        "top_edges_by_attribution".
    """
    raise NotImplementedError("TODO")


def token_conditioned_eap(
    model, tokenizer, dataset, K: int, n_samples: int = 200
) -> dict:
    """EAP partitioned by assigned abstract token.

    Runs separate EAP analyses for subsets of data where a specific abstract
    token is assigned, revealing whether different tokens activate different
    computational circuits.

    Args:
        model: SoRL-trained model.
        tokenizer: Tokenizer for the model.
        dataset: Dataset of arithmetic examples.
        K: Number of abstract tokens per sequence.
        n_samples: Number of examples per token partition.

    Returns:
        Dict with keys: "per_token_edges" (dict of token_id -> edge list),
        "circuit_similarity_matrix" (Jaccard between top-k edges per token),
        "token_ids".
    """
    raise NotImplementedError("TODO")


def compare_circuits(baseline_edges: dict, sorl_edges: dict) -> dict:
    """Compare circuit structure between baseline and SoRL models.

    Measures edge count, specificity (concentration of attribution on few edges),
    and subtask alignment of discovered circuits.

    Args:
        baseline_edges: Output of compute_eap_edges on baseline model.
        sorl_edges: Output of compute_eap_constrained on SoRL model.

    Returns:
        Dict with keys: "baseline_n_edges", "sorl_n_edges",
        "baseline_gini", "sorl_gini", "overlap_top50",
        "subtask_alignment_score".
    """
    raise NotImplementedError("TODO")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Circuit discovery via Edge Attribution Patching"
    )
    parser.add_argument(
        "--sorl_model", type=str, required=True, help="SoRL model path"
    )
    parser.add_argument(
        "--baseline_model", type=str, required=True, help="Baseline model path"
    )
    parser.add_argument("--K", type=int, default=4, help="Abstract tokens per sequence")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="arithmetic/analysis/results",
        help="Output directory",
    )
    args = parser.parse_args()

    sorl_model, sorl_tok = load_model(args.sorl_model)
    baseline_model, baseline_tok = load_model(args.baseline_model)

    # Placeholder dataset — will be replaced with actual loading
    dataset = get_eval_set()

    baseline_edges = compute_eap_edges(baseline_model, baseline_tok, dataset)
    sorl_edges = compute_eap_constrained(sorl_model, sorl_tok, dataset, K=args.K)
    token_edges = token_conditioned_eap(sorl_model, sorl_tok, dataset, K=args.K)
    comparison = compare_circuits(baseline_edges, sorl_edges)

    output_dir = Path(args.output_dir)
    save_results(baseline_edges, str(output_dir / "eap_baseline.json"))
    save_results(sorl_edges, str(output_dir / "eap_sorl_constrained.json"))
    save_results(token_edges, str(output_dir / "eap_token_conditioned.json"))
    save_results(comparison, str(output_dir / "circuit_comparison.json"))
