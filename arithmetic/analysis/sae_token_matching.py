"""
SAE Feature ↔ SoRL Token Matching

Hungarian matching between top SAE features (baseline residual stream) and
SoRL abstraction tokens. Tests whether SoRL externalizes the same features
that SAEs discover from activations.

Three levels of evidence:
  1. Correlational: MI(feature_i, token_j) → Hungarian matching
  2. Causal (ablation): knock out matched pairs, check correlated accuracy drops
  3. Cross-model: SAE on SoRL model itself — do features align with tokens?

Usage:
    python -m arithmetic.analysis.sae_token_matching \
        --sorl_model add_sub_sorl_abs10_K1_500K \
        --baseline_model add_sub_baseline_500K \
        --sae_path path/to/trained/sae \
        --output_dir arithmetic/analysis/results/sae_matching
"""
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


def save_results(results: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ── 1. Correlational matching ──────────────────────────────────────


def collect_sae_activations(
    baseline_model,
    sae,
    tokenizer,
    dataset,
    layer: int = -1,
    n_samples: int = 1000,
) -> dict:
    """
    Run baseline model on dataset, extract SAE feature activations at
    answer positions (where SoRL would place abstraction tokens).

    Returns:
        {"activations": np.array [n_samples, n_positions, n_features],
         "examples": list of example metadata}
    """
    raise NotImplementedError("TODO")


def collect_token_assignments(
    sorl_model,
    tokenizer,
    dataset,
    K: int,
    n_samples: int = 1000,
) -> dict:
    """
    Run SoRL model with recursion, record which abstract token was assigned
    at each abstraction position, plus the logit confidence.

    Returns:
        {"assignments": np.array [n_samples, n_abs_positions] (token IDs),
         "confidences": np.array [n_samples, n_abs_positions] (logit margin),
         "examples": list of example metadata}
    """
    raise NotImplementedError("TODO")


def compute_mi_matrix(
    sae_activations: dict,
    token_assignments: dict,
    top_k_features: int = 50,
) -> dict:
    """
    Compute mutual information I(SAE_feature_i; SoRL_token_j) across examples.
    Uses top-K most active SAE features.

    Returns:
        {"mi_matrix": np.array [n_features, n_tokens],
         "feature_ids": list, "token_ids": list}
    """
    raise NotImplementedError("TODO")


def hungarian_matching(mi_matrix: dict) -> dict:
    """
    Optimal 1-to-1 assignment between SAE features and SoRL tokens
    using the Hungarian algorithm on -MI as cost.

    Returns:
        {"matches": list of (feature_id, token_id, mi_score),
         "total_mi": float,
         "unmatched_features": list, "unmatched_tokens": list}
    """
    raise NotImplementedError("TODO")


# ── 2. Causal validation ──────────────────────────────────────────


def ablate_sae_feature(
    baseline_model,
    sae,
    feature_id: int,
    tokenizer,
    eval_set: dict,
    layer: int = -1,
) -> dict:
    """
    Zero out a specific SAE feature in the baseline model's residual stream,
    measure per-split accuracy drop.

    Returns:
        {"feature_id": int, "accuracy_drop": dict of split → delta,
         "overall_drop": float}
    """
    raise NotImplementedError("TODO")


def ablate_sorl_token(
    sorl_model,
    token_id: int,
    tokenizer,
    eval_set: dict,
    K: int,
    replacement: str = "random",
) -> dict:
    """
    Replace a specific SoRL token with random/zero/other token everywhere,
    measure per-split accuracy drop.

    Args:
        replacement: "random" (sample uniformly from other tokens),
                     "zero" (use placeholder), "swap_N" (use token N)

    Returns:
        {"token_id": int, "replacement": str,
         "accuracy_drop": dict of split → delta, "overall_drop": float}
    """
    raise NotImplementedError("TODO")


def matched_ablation_correlation(
    matches: dict,
    sae_ablations: dict,
    token_ablations: dict,
) -> dict:
    """
    For each matched (feature, token) pair, compute correlation between
    their ablation effects across splits. High correlation = same mechanism.

    Returns:
        {"per_pair": list of {feature, token, pearson_r, splits_affected},
         "mean_correlation": float}
    """
    raise NotImplementedError("TODO")


# ── 3. Cross-model SAE analysis ───────────────────────────────────


def sae_on_sorl_model(
    sorl_model,
    sae,
    tokenizer,
    dataset,
    K: int,
    layer: int = -1,
    n_samples: int = 1000,
) -> dict:
    """
    Train/apply SAE on the SoRL model's own residual stream. Check if
    SAE features activate exclusively when specific tokens are assigned.

    This tests: does the SoRL model INTERNALLY represent the same features
    that the tokens externalize? If feature_k fires iff token_j is present,
    the token is a faithful externalization.

    Returns:
        {"feature_token_mi": np.array, "exclusive_pairs": list,
         "token_predicts_feature_accuracy": dict}
    """
    raise NotImplementedError("TODO")


def cross_model_patching(
    sorl_model,
    baseline_model,
    tokenizer,
    dataset,
    K: int,
    positions: str = "abstraction",
    n_samples: int = 200,
) -> dict:
    """
    At abstraction positions, patch baseline's residual stream with SoRL's.
    Measure if baseline's behavior changes in a way predicted by the token.

    This is the strongest causal test: if patching the residual stream at
    a position where token_j="carry" makes the baseline compute carry
    correctly, the representations are causally equivalent.

    Returns:
        {"per_position_effect": dict, "token_conditioned_effect": dict,
         "overall_transfer_accuracy": float}
    """
    raise NotImplementedError("TODO")


# ── Visualization ─────────────────────────────────────────────────


def plot_mi_matrix(mi_matrix: dict, path: str):
    """Heatmap of MI(feature, token) with Hungarian matching overlay."""
    raise NotImplementedError("TODO")


def plot_ablation_correlation(correlation: dict, path: str):
    """Scatter plot: SAE feature ablation effect vs SoRL token ablation effect."""
    raise NotImplementedError("TODO")


# ── Main ──────────────────────────────────────────────────────────


def run_all(
    sorl_model_name: str,
    baseline_model_name: str,
    sae_path: str,
    K: int = 1,
    device: str = "cuda",
    output_dir: str = "arithmetic/analysis/results/sae_matching",
) -> dict:
    """Full pipeline: collect → match → ablate → validate."""
    raise NotImplementedError("TODO")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAE Feature ↔ SoRL Token Matching")
    parser.add_argument("--sorl_model", type=str, required=True)
    parser.add_argument("--baseline_model", type=str, required=True)
    parser.add_argument("--sae_path", type=str, required=True)
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="arithmetic/analysis/results/sae_matching")
    args = parser.parse_args()

    results = run_all(
        args.sorl_model, args.baseline_model, args.sae_path,
        K=args.K, device=args.device, output_dir=args.output_dir,
    )
    print(json.dumps(results, indent=2))
