"""
SoRL token interpretability analysis.

Core question: do abstraction tokens encode carry/borrow circuits?

For each eval example:
  1. Run SoRL recursion to get the abstraction token assignments
  2. Map each abstraction position to its Quirke subtask label
  3. Compute P(token_id | subtask_label) — token-subtask correlation

Usage:
    from arithmetic.interp_utils.token_analysis import TokenAnalyzer
    analyzer = TokenAnalyzer(model, tokenizer, device="cuda")
    results = analyzer.analyze(ops="add_sub", K=4, n_per_split=50)
    analyzer.print_summary(results)
    analyzer.plot_heatmap(results, "token_heatmap.png")
"""
import torch
import numpy as np
import json
from collections import defaultdict
from typing import Optional
from pathlib import Path

from arithmetic.datasets.addition import get_eval_set, ALL_LABELS
from arithmetic.train import QWEN3_TOKEN_MAP
from sorl.sorl_trainer import infer_insert_mask, insert_tokens_with_padding, expand_prompt_len


PROMPT_LEN = 14
ANSWER_LEN = 7


class TokenAnalyzer:
    def __init__(self, model, tokenizer, device="cuda", n_digits=6):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.n_digits = n_digits
        self.base_v = model.vocab_sizes[0].item()
        self.pad_id = tokenizer.pad_token_id

    def _get_abstraction_tokens(self, qwen_ids, K, max_iterations=2):
        """Run recursion on a single example, return (abs_positions, abs_token_ids, traj_positions)."""
        seq = qwen_ids.unsqueeze(0)
        attn = torch.ones_like(seq)
        pl = torch.tensor([PROMPT_LEN], dtype=torch.long, device=self.device)

        im = infer_insert_mask(seq, K, attn)
        ep = expand_prompt_len(pl, im)
        ed, ea = insert_tokens_with_padding(seq, attn, im, self.model.vocab_sizes[0], self.pad_id)

        data, ppt, logits = self.model.recursion(
            ed, ea, max_iterations=max_iterations,
            memory_span_abs=1792, memory_span_traj=1792,
            temperature=0.0, prompt_len=ep,
        )

        expanded = data[0]
        is_abs = expanded >= self.base_v
        is_traj = ~is_abs

        # Map expanded positions back to trajectory positions
        traj_indices = is_traj.nonzero(as_tuple=True)[0]
        abs_indices = is_abs.nonzero(as_tuple=True)[0]
        abs_token_ids = expanded[abs_indices] - self.base_v  # 0-indexed within abs vocab

        # For each abs token, find which trajectory position it follows
        # (the traj position just before it in the expanded sequence)
        abs_after_traj = []
        for abs_idx in abs_indices:
            # Find the trajectory position count up to this point
            traj_before = (traj_indices < abs_idx).sum().item()
            abs_after_traj.append(traj_before - 1)  # 0-indexed traj position

        return abs_indices, abs_token_ids, abs_after_traj

    def analyze(self, ops="add_sub", K=4, n_per_split=50, max_iterations=2):
        """Run token-subtask correlation analysis across all eval splits."""
        self.model.eval()
        categories = get_eval_set(self.n_digits, ops, N=n_per_split)

        # Collect: for each (subtask_label, digit_position), count token_id occurrences
        token_counts = defaultdict(lambda: defaultdict(int))  # {label: {token_id: count}}
        position_tokens = defaultdict(lambda: defaultdict(int))  # {traj_pos: {token_id: count}}
        total_by_label = defaultdict(int)
        total_by_pos = defaultdict(int)

        n_examples = 0
        for split_name, examples in categories.items():
            for ex in examples:
                qwen_ids = torch.tensor(
                    [QWEN3_TOKEN_MAP[t] for t in ex.tokens],
                    dtype=torch.long, device=self.device,
                )

                with torch.no_grad():
                    abs_indices, abs_token_ids, abs_after_traj = self._get_abstraction_tokens(
                        qwen_ids, K, max_iterations
                    )

                # Map each abs token to its subtask label
                for abs_tok, traj_pos in zip(abs_token_ids, abs_after_traj):
                    tok_id = abs_tok.item()
                    pos = traj_pos

                    # Position in answer (0-6) or prompt
                    if pos >= PROMPT_LEN:
                        answer_digit = pos - PROMPT_LEN
                        if answer_digit < len(ex.labels):
                            label = ex.labels[answer_digit]
                            token_counts[label][tok_id] += 1
                            total_by_label[label] += 1

                    position_tokens[pos][tok_id] += 1
                    total_by_pos[pos] += 1

                n_examples += 1

        # Compute P(token | label)
        p_token_given_label = {}
        for label, counts in token_counts.items():
            total = total_by_label[label]
            p_token_given_label[label] = {
                tok: count / total for tok, count in sorted(counts.items())
            }

        # Compute P(token | position)
        p_token_given_pos = {}
        for pos, counts in position_tokens.items():
            total = total_by_pos[pos]
            p_token_given_pos[pos] = {
                tok: count / total for tok, count in sorted(counts.items())
            }

        return {
            "K": K,
            "ops": ops,
            "n_examples": n_examples,
            "n_per_split": n_per_split,
            "p_token_given_label": p_token_given_label,
            "p_token_given_pos": p_token_given_pos,
            "token_counts": {k: dict(v) for k, v in token_counts.items()},
            "total_by_label": dict(total_by_label),
            "labels": list(sorted(token_counts.keys())),
            "token_ids": sorted(set(
                tok for counts in token_counts.values() for tok in counts
            )),
        }

    def print_summary(self, results):
        """Print token-subtask correlation table."""
        labels = results["labels"]
        token_ids = results["token_ids"]
        p = results["p_token_given_label"]

        print(f"\nP(token | subtask) — K={results['K']}, {results['n_examples']} examples")
        print(f"{'Label':<6s}", end="")
        for tok in token_ids:
            print(f"  t{tok:>2d}", end="")
        print(f"  {'N':>5s}")
        print("-" * (6 + len(token_ids) * 5 + 7))

        for label in labels:
            print(f"{label:<6s}", end="")
            dist = p.get(label, {})
            for tok in token_ids:
                val = dist.get(tok, 0)
                if val >= 0.3:
                    print(f" {val:>.2f}", end="")
                elif val >= 0.05:
                    print(f" {val:>.2f}", end="")
                else:
                    print(f"    .", end="")

            print(f"  {results['total_by_label'].get(label, 0):>5d}")

    def plot_heatmap(self, results, path=None, show=False):
        """Heatmap of P(token | subtask_label)."""
        import matplotlib.pyplot as plt

        labels = results["labels"]
        token_ids = results["token_ids"]
        p = results["p_token_given_label"]

        matrix = np.zeros((len(labels), len(token_ids)))
        for i, label in enumerate(labels):
            for j, tok in enumerate(token_ids):
                matrix[i, j] = p.get(label, {}).get(tok, 0)

        fig, ax = plt.subplots(figsize=(max(6, len(token_ids) * 0.8), max(4, len(labels) * 0.5)))
        im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")

        ax.set_xticks(range(len(token_ids)))
        ax.set_xticklabels([f"t{t}" for t in token_ids])
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)

        for i in range(len(labels)):
            for j in range(len(token_ids)):
                val = matrix[i, j]
                if val >= 0.05:
                    ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                            fontsize=7, color="white" if val > 0.5 else "black")

        plt.colorbar(im, ax=ax, label="P(token | subtask)")
        ax.set_title(f"Token–Subtask Correlation (K={results['K']})")
        ax.set_xlabel("Abstraction Token ID")
        ax.set_ylabel("Quirke Subtask")
        plt.tight_layout()

        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

    def save(self, results, path):
        """Save results to JSON."""
        # Convert defaultdicts and numpy types for JSON
        serializable = {}
        for k, v in results.items():
            if isinstance(v, dict):
                serializable[k] = {str(kk): vv for kk, vv in v.items()}
            else:
                serializable[k] = v
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)
