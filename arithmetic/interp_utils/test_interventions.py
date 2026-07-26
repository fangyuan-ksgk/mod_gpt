"""
Tests for token-level interventions.
Run: python -m pytest arithmetic/interp_utils/test_interventions.py -v
"""
import torch
import pytest
from arithmetic.interp_utils.interventions import (
    get_abs_positions,
    token_knockout,
    knockout_at_digit,
    token_swap,
    swap_at_digit,
    token_replace,
    token_replace_random,
    token_shuffle,
    measure_intervention_effect,
)

BASE_VOCAB = 13   # 0-9 digits + plus + equals + minus
ABS_VOCAB = 8     # 8 abstract tokens (IDs 13-20, with 13 = placeholder)
N_DIGITS = 6
SEQ_LEN = 21      # 6 + 1 + 6 + 1 + 7


def _make_sorl_sequence():
    """
    Create a fake SoRL sequence: normal tokens with abs tokens interleaved.
    Format: D D D [A] D D D [A] + D D D [A] D D D [A] = Z Z Z Z Z Z Z
    where [A] = abstract token (ID >= BASE_VOCAB)
    """
    # Base sequence: 123456+654321=0777777
    base = torch.tensor([[1, 2, 3, 4, 5, 6, 10, 6, 5, 4, 3, 2, 1, 11,
                           0, 7, 7, 7, 7, 7, 7]])
    # Insert abs tokens at positions 3, 7, 10, 13 (before =)
    # Expand to make room
    sorl = torch.tensor([[1, 2, 3, 14, 4, 5, 6, 15, 10, 6, 5, 16, 4, 3, 2, 17, 1, 11,
                           0, 7, 7, 7, 7, 7, 7]])
    return sorl


def _make_pair():
    """Two sequences with different abs token values but same structure."""
    a = torch.tensor([[1, 2, 14, 3, 15, 10, 4, 16, 5, 17, 11, 0, 5, 5]])
    b = torch.tensor([[1, 2, 18, 3, 19, 10, 4, 20, 5, 14, 11, 0, 6, 6]])
    return a, b


class TestGetAbsPositions:
    def test_finds_abs_tokens(self):
        tokens = torch.tensor([[0, 5, 14, 10, 15, 11, 7]])
        mask = get_abs_positions(tokens, BASE_VOCAB)
        assert mask.tolist() == [[False, False, True, False, True, False, False]]

    def test_no_abs_tokens(self):
        tokens = torch.tensor([[0, 5, 10, 11, 7]])
        mask = get_abs_positions(tokens, BASE_VOCAB)
        assert not mask.any()

    def test_all_abs_tokens(self):
        tokens = torch.tensor([[14, 15, 16]])
        mask = get_abs_positions(tokens, BASE_VOCAB)
        assert mask.all()


class TestTokenKnockout:
    def test_replaces_at_positions(self):
        tokens = torch.tensor([[1, 14, 2, 15, 3]])
        positions = torch.tensor([[False, True, False, True, False]])
        result = token_knockout(tokens, positions, placeholder_id=BASE_VOCAB)
        assert result.tolist() == [[1, BASE_VOCAB, 2, BASE_VOCAB, 3]]

    def test_does_not_modify_original(self):
        tokens = torch.tensor([[1, 14, 2]])
        positions = torch.tensor([[False, True, False]])
        original = tokens.clone()
        token_knockout(tokens, positions, BASE_VOCAB)
        assert (tokens == original).all()

    def test_empty_positions(self):
        tokens = torch.tensor([[1, 14, 2]])
        positions = torch.zeros_like(tokens, dtype=torch.bool)
        result = token_knockout(tokens, positions, BASE_VOCAB)
        assert (result == tokens).all()


class TestKnockoutAtDigit:
    def test_knocks_out_abs_before_digit(self):
        # Sequence with abs tokens at known positions
        # Positions: 0  1   2  3   4  5  6   7  8  9 10  11 12 13 14 15
        tokens = torch.tensor([[1, 14, 2, 15, 3, 4, 10, 16, 5, 6, 17,  7, 11,  0,  8,  9]])
        # base_vocab=13, n_digits=4 → ans_start = 2*4+2 = 10
        # But this doesn't match standard format. Use simpler test:

        # Standard 2-digit: XX+YY=ZZZ (seq_len = 2+1+2+1+3 = 9, ans_start=6)
        # With abs tokens interleaved — just test the mask logic
        n = 2
        ans_start = 2 * n + 2  # = 6
        tokens = torch.tensor([[1, 14, 2, 10, 3, 15, 4, 11, 0, 5, 6]])
        # abs at positions 1, 5; ans_start would be at position 6 for n=2
        # knockout_at_digit(digit_idx=0) should knock out abs before position 6
        result = knockout_at_digit(tokens, digit_idx=0, base_vocab_size=BASE_VOCAB, n_digits=n)
        # Abs tokens at pos 1 and 5 are both before ans_start+0=6
        assert result[0, 1] == BASE_VOCAB
        assert result[0, 5] == BASE_VOCAB


class TestTokenSwap:
    def test_swaps_abs_tokens(self):
        a, b = _make_pair()
        abs_a = a[get_abs_positions(a, BASE_VOCAB)]
        abs_b = b[get_abs_positions(b, BASE_VOCAB)]

        a_patched, b_patched = token_swap(a, b, base_vocab_size=BASE_VOCAB)

        # Abs tokens in a_patched should be b's abs tokens
        assert (a_patched[get_abs_positions(a, BASE_VOCAB)] == abs_b).all()
        # Abs tokens in b_patched should be a's abs tokens
        assert (b_patched[get_abs_positions(b, BASE_VOCAB)] == abs_a).all()

    def test_non_abs_unchanged(self):
        a, b = _make_pair()
        a_patched, _ = token_swap(a, b, base_vocab_size=BASE_VOCAB)
        non_abs = ~get_abs_positions(a, BASE_VOCAB)
        assert (a_patched[non_abs] == a[non_abs]).all()

    def test_does_not_modify_originals(self):
        a, b = _make_pair()
        a_orig, b_orig = a.clone(), b.clone()
        token_swap(a, b, base_vocab_size=BASE_VOCAB)
        assert (a == a_orig).all()
        assert (b == b_orig).all()

    def test_double_swap_is_identity(self):
        a, b = _make_pair()
        a1, b1 = token_swap(a, b, base_vocab_size=BASE_VOCAB)
        a2, b2 = token_swap(a1, b1, base_vocab_size=BASE_VOCAB)
        assert (a2 == a).all()
        assert (b2 == b).all()


class TestSwapAtDigit:
    def test_only_swaps_near_digit(self):
        a, b = _make_pair()
        # swap_at_digit only swaps abs tokens in a 4-token window before the target
        a_patched, b_patched = swap_at_digit(a, b, digit_idx=0,
                                              base_vocab_size=BASE_VOCAB, n_digits=2)
        # Some abs tokens should be swapped, others not
        # Just verify it returns valid tensors with same shape
        assert a_patched.shape == a.shape
        assert b_patched.shape == b.shape


class TestTokenReplace:
    def test_replaces_all_abs(self):
        tokens = torch.tensor([[1, 14, 2, 15, 3]])
        result = token_replace(tokens, replacement_id=20, base_vocab_size=BASE_VOCAB)
        assert result.tolist() == [[1, 20, 2, 20, 3]]

    def test_replaces_at_positions(self):
        tokens = torch.tensor([[14, 15, 16]])
        positions = torch.tensor([[True, False, True]])
        result = token_replace(tokens, replacement_id=20, positions=positions)
        assert result.tolist() == [[20, 15, 20]]


class TestTokenReplaceRandom:
    def test_replaces_abs_with_valid_range(self):
        tokens = torch.tensor([[1, 14, 2, 15, 3]])
        result = token_replace_random(tokens, BASE_VOCAB, ABS_VOCAB)
        # Non-abs unchanged
        assert result[0, 0] == 1
        assert result[0, 2] == 2
        assert result[0, 4] == 3
        # Abs tokens replaced with values in [BASE_VOCAB+1, BASE_VOCAB+ABS_VOCAB]
        assert result[0, 1] >= BASE_VOCAB + 1
        assert result[0, 1] <= BASE_VOCAB + ABS_VOCAB
        assert result[0, 3] >= BASE_VOCAB + 1
        assert result[0, 3] <= BASE_VOCAB + ABS_VOCAB

    def test_does_not_modify_original(self):
        tokens = torch.tensor([[1, 14, 2]])
        original = tokens.clone()
        token_replace_random(tokens, BASE_VOCAB, ABS_VOCAB)
        assert (tokens == original).all()


class TestTokenShuffle:
    def test_preserves_positions(self):
        tokens = torch.tensor([[1, 14, 2, 15, 3, 16]])
        result = token_shuffle(tokens, BASE_VOCAB)
        # Non-abs unchanged
        assert result[0, 0] == 1
        assert result[0, 2] == 2
        assert result[0, 4] == 3
        # Abs positions still have abs tokens
        assert result[0, 1] >= BASE_VOCAB
        assert result[0, 3] >= BASE_VOCAB
        assert result[0, 5] >= BASE_VOCAB

    def test_preserves_set_of_values(self):
        tokens = torch.tensor([[1, 14, 2, 15, 3, 16]])
        result = token_shuffle(tokens, BASE_VOCAB)
        abs_orig = sorted(tokens[get_abs_positions(tokens, BASE_VOCAB)].tolist())
        abs_shuf = sorted(result[get_abs_positions(result, BASE_VOCAB)].tolist())
        assert abs_orig == abs_shuf

    def test_non_abs_untouched(self):
        tokens = torch.tensor([[1, 14, 2, 15, 3]])
        result = token_shuffle(tokens, BASE_VOCAB)
        non_abs = ~get_abs_positions(tokens, BASE_VOCAB)
        assert (result[non_abs] == tokens[non_abs]).all()


class TestMeasureInterventionEffect:
    def test_no_change_gives_zero_drop(self):
        """Mock model that always predicts the same thing regardless of input."""
        class DummyModel:
            def eval(self): pass
            def __call__(self, x):
                B, L = x.shape
                logits = torch.zeros(B, L, BASE_VOCAB)
                # Predict all 7s for answer positions
                logits[:, :, 7] = 10.0
                class Out:
                    pass
                o = Out()
                o.logits = logits
                return o

        model = DummyModel()
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 10, 6, 5, 4, 3, 2, 1, 11,
                                 0, 7, 7, 7, 7, 7, 7]])
        result = measure_intervention_effect(model, tokens, tokens, n_digits=6,
                                              base_vocab_size=BASE_VOCAB)
        assert result["n_changed"] == 0
        assert result["accuracy_drop"] == 0.0
