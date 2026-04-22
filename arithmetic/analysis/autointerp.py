"""Automated interpretability pipeline adapted for SoRL abstract tokens.

Implements the Juang et al. auto-interpretability pipeline: collect top
activations for each abstract token, generate natural language explanations
via an LLM, then score those explanations using detection, fuzzing, and
embedding-based retrieval metrics. Validates LLM-generated explanations
against ground-truth Quirke subtask labels.
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


def collect_top_activations(
    model,
    tokenizer,
    K: int,
    n_examples: int = 40,
    n_samples: int = 5000,
) -> dict:
    """Collect top-N examples for each abstract token ranked by logit confidence.

    For each abstract token ID, finds the sequences where that token was
    assigned with highest confidence (logit score), and returns the surrounding
    context for each.

    Args:
        model: SoRL-trained model.
        tokenizer: Tokenizer for the model.
        K: Number of abstract tokens per sequence.
        n_examples: Number of top examples to collect per token.
        n_samples: Total number of sequences to scan.

    Returns:
        Dict with keys: "per_token" (dict of token_id -> list of
        {sequence, position, logit_score, context}), "token_ids".
    """
    raise NotImplementedError("TODO")


def generate_explanations(
    activations: dict,
    explainer_model: str = "meta-llama/Llama-3.1-70B-Instruct",
) -> dict:
    """Generate natural language explanations for each abstract token.

    Feeds top activation examples to an LLM and asks it to describe what
    computational role this token plays in arithmetic.

    Args:
        activations: Output of collect_top_activations.
        explainer_model: HF model name/path for the explainer LLM.

    Returns:
        Dict with keys: "per_token" (dict of token_id -> {explanation,
        confidence, key_features}), "explainer_model".
    """
    raise NotImplementedError("TODO")


def score_detection(
    explanations: dict, scorer_model: str, held_out: dict
) -> dict:
    """Score whether the scorer can identify which sequences use a given token.

    Given the explanation for a token and a mix of sequences (some using that
    token, some not), measures whether the scorer can correctly classify them.

    Args:
        explanations: Output of generate_explanations.
        scorer_model: HF model name/path for the scorer LLM.
        held_out: Held-out examples not used for explanation generation.

    Returns:
        Dict with keys: "per_token" (dict of token_id -> {auroc, accuracy,
        n_positive, n_negative}), "mean_auroc".
    """
    raise NotImplementedError("TODO")


def score_fuzzing(
    explanations: dict, scorer_model: str, held_out: dict
) -> dict:
    """Score whether the scorer can identify which position uses a given token.

    Given the explanation and a sequence known to use the token, measures
    whether the scorer can identify the exact position.

    Args:
        explanations: Output of generate_explanations.
        scorer_model: HF model name/path for the scorer LLM.
        held_out: Held-out examples not used for explanation generation.

    Returns:
        Dict with keys: "per_token" (dict of token_id -> {position_accuracy,
        position_mrr}), "mean_position_accuracy".
    """
    raise NotImplementedError("TODO")


def score_embedding(
    explanations: dict, held_out: dict, encoder_model: str
) -> dict:
    """Embedding retrieval AUROC for each token explanation.

    Encodes explanations and sequence contexts with a sentence encoder,
    then measures retrieval performance (can the explanation retrieve the
    correct sequences?).

    Args:
        explanations: Output of generate_explanations.
        held_out: Held-out examples.
        encoder_model: HF model name/path for the sentence encoder.

    Returns:
        Dict with keys: "per_token" (dict of token_id -> {auroc,
        mean_cosine_positive, mean_cosine_negative}), "mean_auroc".
    """
    raise NotImplementedError("TODO")


def validate_against_ground_truth(
    explanations: dict, token_subtask_matrix: dict
) -> dict:
    """Validate LLM-generated explanations against Quirke subtask labels.

    Checks whether the LLM's explanation for each token aligns with the
    dominant subtask assignment from the token-subtask correlation analysis.

    Args:
        explanations: Output of generate_explanations.
        token_subtask_matrix: Output of token_subtask_correlation.compute_token_subtask_matrix.

    Returns:
        Dict with keys: "per_token" (dict of token_id -> {predicted_subtask,
        ground_truth_subtask, match}), "precision", "recall", "f1".
    """
    raise NotImplementedError("TODO")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automated interpretability pipeline for SoRL tokens"
    )
    parser.add_argument("--model", type=str, required=True, help="SoRL model path")
    parser.add_argument("--K", type=int, default=4, help="Abstract tokens per sequence")
    parser.add_argument(
        "--explainer",
        type=str,
        default="meta-llama/Llama-3.1-70B-Instruct",
        help="Explainer LLM",
    )
    parser.add_argument(
        "--scorer",
        type=str,
        default="meta-llama/Llama-3.1-70B-Instruct",
        help="Scorer LLM",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="arithmetic/analysis/results",
        help="Output directory",
    )
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)
    activations = collect_top_activations(model, tokenizer, K=args.K)
    explanations = generate_explanations(activations, explainer_model=args.explainer)
    detection = score_detection(explanations, scorer_model=args.scorer, held_out={})
    fuzzing = score_fuzzing(explanations, scorer_model=args.scorer, held_out={})

    output_dir = Path(args.output_dir)
    save_results(activations, str(output_dir / "top_activations.json"))
    save_results(explanations, str(output_dir / "explanations.json"))
    save_results(detection, str(output_dir / "detection_scores.json"))
    save_results(fuzzing, str(output_dir / "fuzzing_scores.json"))
