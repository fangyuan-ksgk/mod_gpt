"""Embedding geometry, co-occurrence, substitution, and CKA analysis of abstract tokens.

Analyzes the learned structure of SoRL abstract token embeddings:
- Cosine similarity and clustering of abstract embeddings
- Mutual information and co-occurrence patterns across sequences
- Functional substitution distances (swap token i->j, measure accuracy drop)
- CKA between hidden state distributions grouped by assigned token
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


def embedding_similarity(model, base_vocab: int) -> dict:
    """Cosine similarity matrix of abstract embeddings with clustering and projection.

    Computes pairwise cosine similarity between all abstract token embeddings,
    performs hierarchical clustering, and projects to 2D via PCA/UMAP colored
    by the most-frequent subtask assignment.

    Args:
        model: SoRL-trained model with abstract token embeddings.
        base_vocab: Size of the base vocabulary (abstract tokens start after this).

    Returns:
        Dict with keys: "similarity_matrix", "cluster_linkage", "pca_coords",
        "token_ids", "subtask_colors".
    """
    raise NotImplementedError("TODO")


def token_cooccurrence(
    model, tokenizer, dataset, K: int, n_samples: int = 1000
) -> dict:
    """Mutual information I(token_i; token_j) across sequences and co-occurrence graph.

    For each pair of abstract token IDs, computes how often they co-occur
    within the same sequence and estimates mutual information.

    Args:
        model: SoRL-trained model.
        tokenizer: Tokenizer for the model.
        dataset: Dataset of arithmetic examples.
        K: Number of abstract tokens per sequence.
        n_samples: Number of sequences to sample.

    Returns:
        Dict with keys: "mi_matrix", "cooccurrence_counts", "token_ids".
    """
    raise NotImplementedError("TODO")


def substitution_matrix(
    model, tokenizer, eval_set: dict, K: int
) -> dict:
    """Functional distance matrix via pairwise token substitution.

    For every pair (i, j) of abstract tokens, swaps all occurrences of i with j
    in the searched abstractions, then measures the accuracy drop. The resulting
    matrix encodes functional distance between tokens.

    Args:
        model: SoRL-trained model.
        tokenizer: Tokenizer for the model.
        eval_set: Evaluation set dict from get_eval_set.
        K: Number of abstract tokens per sequence.

    Returns:
        Dict with keys: "distance_matrix", "accuracy_baseline", "accuracy_swapped",
        "token_ids".
    """
    raise NotImplementedError("TODO")


def activation_cka(
    model,
    tokenizer,
    dataset,
    K: int,
    layer: int,
    n_samples: int = 500,
) -> dict:
    """CKA between hidden state distributions grouped by assigned abstract token.

    Collects hidden states at abstraction positions, partitions them by the
    abstract token assigned at that position, and computes centered kernel
    alignment (CKA) between each pair of token groups.

    Args:
        model: SoRL-trained model.
        tokenizer: Tokenizer for the model.
        dataset: Dataset of arithmetic examples.
        K: Number of abstract tokens per sequence.
        layer: Which transformer layer to extract hidden states from.
        n_samples: Number of sequences to sample.

    Returns:
        Dict with keys: "cka_matrix", "token_ids", "group_sizes".
    """
    raise NotImplementedError("TODO")


def run_all(model_name: str, device: str = "cuda") -> dict:
    """Run all representation structure analyses and save figures.

    Args:
        model_name: HF model subfolder or path.
        device: Torch device string.

    Returns:
        Dict aggregating results from all sub-analyses.
    """
    raise NotImplementedError("TODO")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Embedding geometry and representation structure analysis"
    )
    parser.add_argument("--model", type=str, required=True, help="HF model subfolder")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="arithmetic/analysis/results",
        help="Output directory for figures and results",
    )
    args = parser.parse_args()

    results = run_all(model_name=args.model, device=args.device)
    save_results(results, str(Path(args.output_dir) / "representation_structure.json"))
