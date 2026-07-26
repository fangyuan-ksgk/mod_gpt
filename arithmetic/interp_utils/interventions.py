"""
Token-level interventions for SoRL models.

These parallel Quirke's activation-level interventions but operate directly
on abstraction tokens — no hooks, no activation caching, no patching code.

Each intervention takes a sequence with abstraction tokens and modifies them,
then measures the effect on accuracy by complexity class.

Usage:
    from arithmetic.interp_utils.interventions import (
        token_knockout, token_swap, token_replace, token_shuffle,
        measure_intervention_effect,
    )
"""
import torch
import copy
from typing import List, Optional, Tuple, Dict


def get_abs_positions(tokens: torch.Tensor, base_vocab_size: int) -> torch.Tensor:
    """Find positions of abstraction tokens in a sequence.
    Returns boolean mask of shape (batch, seq_len)."""
    return tokens >= base_vocab_size


# ── Intervention 1: Token Knockout ──────────────────────────────────
# Parallel to Quirke's mean ablation — remove information from a specific position.

def token_knockout(tokens: torch.Tensor, positions: torch.Tensor,
                   placeholder_id: int) -> torch.Tensor:
    """
    Replace abs tokens at specified positions with the placeholder token.
    Like Quirke's mean ablation: removes the node's contribution.

    Args:
        tokens: (batch, seq_len) — sequence with abs tokens
        positions: (batch, seq_len) bool mask — which abs tokens to knock out
        placeholder_id: the SoRL placeholder token ID (= base_vocab_size)
    """
    out = tokens.clone()
    out[positions] = placeholder_id
    return out


def knockout_at_digit(tokens: torch.Tensor, digit_idx: int,
                      base_vocab_size: int, n_digits: int) -> torch.Tensor:
    """
    Knock out all abs tokens that precede a specific answer digit.
    Tests whether those tokens carry information needed for that digit.

    digit_idx: 0 = leftmost answer digit (overflow/sign), n_digits = rightmost (units)
    """
    placeholder_id = base_vocab_size
    ans_start = 2 * n_digits + 2
    target_pos = ans_start + digit_idx

    # Find abs tokens between question end and target answer digit
    abs_mask = get_abs_positions(tokens, base_vocab_size)
    pos_indices = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
    knockout_mask = abs_mask & (pos_indices < target_pos)

    return token_knockout(tokens, knockout_mask, placeholder_id)


# ── Intervention 2: Token Swap ──────────────────────────────────────
# Parallel to Quirke's activation patching — swap information between paired questions.

def token_swap(tokens_a: torch.Tensor, tokens_b: torch.Tensor,
               positions: Optional[torch.Tensor] = None,
               base_vocab_size: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Swap abs tokens between two sequences at specified positions.
    Like Quirke's activation patching: if token encodes subtask X,
    swapping it should change the answer predictably.

    Args:
        tokens_a, tokens_b: (1, seq_len) — paired sequences
        positions: bool mask of positions to swap (default: all abs positions)
        base_vocab_size: threshold for abs tokens

    Returns: (tokens_a_patched, tokens_b_patched)
    """
    if positions is None:
        positions = get_abs_positions(tokens_a, base_vocab_size)

    a_out = tokens_a.clone()
    b_out = tokens_b.clone()
    a_out[positions] = tokens_b[positions]
    b_out[positions] = tokens_a[positions]
    return a_out, b_out


def swap_at_digit(tokens_a: torch.Tensor, tokens_b: torch.Tensor,
                  digit_idx: int, base_vocab_size: int,
                  n_digits: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Swap abs tokens near a specific answer digit between two sequences.
    For testing if the token carries carry/borrow info for that digit.
    """
    ans_start = 2 * n_digits + 2
    target_pos = ans_start + digit_idx

    abs_mask = get_abs_positions(tokens_a, base_vocab_size)
    pos_indices = torch.arange(tokens_a.shape[1], device=tokens_a.device).unsqueeze(0)

    # Swap abs tokens in the window before this answer digit
    # (the most recent abs token is likely the one carrying relevant info)
    window = abs_mask & (pos_indices < target_pos) & (pos_indices >= target_pos - 4)
    return token_swap(tokens_a, tokens_b, window, base_vocab_size)


# ── Intervention 3: Token Replace ───────────────────────────────────
# Replace abs tokens with a specific value (e.g., most common token, random token).

def token_replace(tokens: torch.Tensor, replacement_id: int,
                  positions: Optional[torch.Tensor] = None,
                  base_vocab_size: int = 0) -> torch.Tensor:
    """
    Replace abs tokens with a fixed value.
    - replacement_id = most common abs token → "mean ablation" analog
    - replacement_id = random token → noise injection
    - replacement_id = placeholder → knockout (same as token_knockout)
    """
    if positions is None:
        positions = get_abs_positions(tokens, base_vocab_size)
    out = tokens.clone()
    out[positions] = replacement_id
    return out


def token_replace_random(tokens: torch.Tensor, base_vocab_size: int,
                         abs_vocab_size: int) -> torch.Tensor:
    """Replace all abs tokens with random abs tokens."""
    abs_mask = get_abs_positions(tokens, base_vocab_size)
    out = tokens.clone()
    n_abs = abs_mask.sum().item()
    # Random tokens from abs vocab (offset by base_vocab + 1 to skip placeholder)
    random_tokens = torch.randint(
        base_vocab_size + 1, base_vocab_size + abs_vocab_size + 1,
        (n_abs,), device=tokens.device, dtype=tokens.dtype,
    )
    out[abs_mask] = random_tokens
    return out


# ── Intervention 4: Token Shuffle ───────────────────────────────────
# Permute abs tokens within a sequence — breaks positional correspondence.

def token_shuffle(tokens: torch.Tensor,
                  base_vocab_size: int) -> torch.Tensor:
    """
    Randomly permute abs token values within the sequence.
    Preserves which positions have abs tokens but scrambles their identity.
    Tests whether token IDENTITY matters (not just presence).
    """
    abs_mask = get_abs_positions(tokens, base_vocab_size)
    out = tokens.clone()
    abs_values = out[abs_mask]
    perm = torch.randperm(abs_values.shape[0], device=tokens.device)
    out[abs_mask] = abs_values[perm]
    return out


# ── Measurement utility ─────────────────────────────────────────────

def measure_intervention_effect(
    model, tokens_original: torch.Tensor, tokens_intervened: torch.Tensor,
    n_digits: int = 6, base_vocab_size: int = 13,
) -> Dict:
    """
    Compare model predictions before and after intervention.

    Returns dict with:
        - digits_changed: (batch, n_digits+1) bool — which answer digits changed
        - n_changed: total digits changed
        - original_correct: (batch,) bool — was original answer correct
        - intervened_correct: (batch,) bool — is intervened answer correct
        - accuracy_drop: float — fraction of correct answers that broke
    """
    ans_start = 2 * n_digits + 2
    ans_len = n_digits + 1

    model.eval()
    with torch.no_grad():
        logits_orig = model(tokens_original).logits
        logits_interv = model(tokens_intervened).logits

    pred_orig = logits_orig[:, ans_start - 1:-1, :base_vocab_size].argmax(dim=-1)
    pred_interv = logits_interv[:, ans_start - 1:-1, :base_vocab_size].argmax(dim=-1)
    targets = tokens_original[:, ans_start:]

    digits_changed = (pred_orig != pred_interv)
    original_correct = (pred_orig == targets).all(dim=1)
    intervened_correct = (pred_interv == targets).all(dim=1)

    n_originally_correct = original_correct.sum().item()
    n_broke = (original_correct & ~intervened_correct).sum().item()

    return {
        "digits_changed": digits_changed,
        "n_changed": digits_changed.sum().item(),
        "original_correct": original_correct,
        "intervened_correct": intervened_correct,
        "accuracy_drop": n_broke / max(n_originally_correct, 1),
        "n_originally_correct": n_originally_correct,
        "n_broke": n_broke,
    }
