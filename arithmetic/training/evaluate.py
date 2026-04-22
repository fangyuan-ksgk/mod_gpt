"""
Arithmetic model evaluator with visualization.

Usage:
    from arithmetic.evaluate import ArithmeticEvaluator

    evaluator = ArithmeticEvaluator(model, tokenizer, device="cuda")
    results = evaluator.run(ops="add")          # SFT eval
    results = evaluator.run(ops="add", K=4)     # SoRL eval (recursion)
    evaluator.save(results, "results.json")

    # Visualizations
    evaluator.print_table(results)                          # box-drawing table
    evaluator.plot_by_complexity(results, "complexity.png") # bar chart
    evaluator.plot_subtask_heatmap(results, "heatmap.png")  # subtask x split
"""
import json
import torch
from pathlib import Path
from typing import Optional, List

from arithmetic.data.addition import get_eval_set, ALL_LABELS
from arithmetic.training.train import QWEN3_TOKEN_MAP, QWEN3_INV_MAP


class ArithmeticEvaluator:
    """Evaluates a SorlModelWrapper on Quirke complexity splits."""

    def __init__(self, model, tokenizer, device: str = "cuda", n_digits: int = 6):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.n_digits = n_digits
        self.prompt_len = 2 * n_digits + 2
        self.answer_len = n_digits + 1
        self.base_v = model.vocab_sizes[0].item()
        self.pad_id = tokenizer.pad_token_id

    def _to_qwen3_ids(self, tokens):
        """Convert internal token IDs to Qwen3 token IDs."""
        return torch.tensor(
            [QWEN3_TOKEN_MAP[t] for t in tokens], dtype=torch.long, device=self.device,
        )

    def _predict_sft(self, qwen_ids):
        """SFT autoregressive eval: feed prompt, generate answer digit by digit."""
        # Start with just the prompt
        generated = qwen_ids[:self.prompt_len].clone()

        for _ in range(self.answer_len):
            ids = generated.unsqueeze(0)
            attn = torch.ones_like(ids)
            out = self.model(ids, attention_mask=attn, memory_span_abs=1792, memory_span_traj=1792)
            next_token = out.logits[0, -1, :self.base_v].argmax()
            generated = torch.cat([generated, next_token.unsqueeze(0)])

        return generated[self.prompt_len:]

    def _predict_sorl(self, qwen_ids, K):
        """SoRL autoregressive eval with fixed-length structure.
        Pads to full sequence length so abstraction pattern matches training exactly.
        Fills in answer digits one at a time (errors propagate)."""
        from sorl.sorl_trainer import infer_insert_mask, insert_tokens_with_padding, expand_prompt_len

        # Build full-length sequence: prompt + dummy trajectory tokens for answer
        # Use token 0 (a valid trajectory token) — NOT pad_id which equals the abstraction placeholder
        seq = qwen_ids[:self.prompt_len].clone()
        pad_answer = torch.zeros(self.answer_len, dtype=torch.long, device=self.device)
        seq = torch.cat([seq, pad_answer])  # [21] — same length as training

        for digit_idx in range(self.answer_len):
            ids = seq.unsqueeze(0)
            attn = torch.ones_like(ids)
            pl_t = torch.tensor([self.prompt_len], dtype=torch.long, device=self.device)

            # Insert abstractions into full-length sequence (same pattern as training)
            im = infer_insert_mask(ids, K, attn)
            ep = expand_prompt_len(pl_t, im)
            ed, ea = insert_tokens_with_padding(ids, attn, im, self.model.vocab_sizes[0], self.pad_id)
            data, ppt, logits = self.model.recursion(
                ed, ea, max_iterations=2,
                memory_span_abs=1792, memory_span_traj=1792,
                temperature=0.0, prompt_len=ep,
            )

            # Find expanded position of this answer digit and predict from logits[pos-1]
            is_traj = data[0] < self.base_v
            traj_indices = is_traj.nonzero(as_tuple=True)[0]
            answer_pos = traj_indices[self.prompt_len + digit_idx].item()
            pred_token = logits[0, answer_pos - 1, :self.base_v].argmax()

            # Fill it into the sequence for next iteration
            seq[self.prompt_len + digit_idx] = pred_token

        return seq[self.prompt_len:]

    def _eval_split(self, examples, K=None):
        """Evaluate a list of ArithmeticExample. Returns split-level results dict."""
        n_correct = 0
        total_digits = 0
        correct_digits_total = 0
        digit_correct_by_label = {t: [] for t in ALL_LABELS}

        for ex in examples:
            qwen_ids = self._to_qwen3_ids(ex.tokens)
            target = qwen_ids[self.prompt_len:]

            with torch.no_grad():
                if K is not None:
                    pred = self._predict_sorl(qwen_ids, K)
                else:
                    pred = self._predict_sft(qwen_ids)

            assert len(pred) == len(target) == self.answer_len, \
                f"Length mismatch: pred={len(pred)}, target={len(target)}, expected={self.answer_len}"

            correct_digits = (pred == target)
            if correct_digits.all():
                n_correct += 1
            total_digits += len(correct_digits)
            correct_digits_total += correct_digits.sum().item()

            # Per-digit subtask accuracy
            for d in range(self.answer_len):
                label = ex.labels[d]
                digit_correct_by_label[label].append(correct_digits[d].item())

        per_subtask = {}
        for label, vals in digit_correct_by_label.items():
            if vals:
                per_subtask[label] = {
                    "accuracy": sum(vals) / len(vals),
                    "count": len(vals),
                }

        return {
            "full_accuracy": n_correct / max(len(examples), 1),
            "digit_accuracy": correct_digits_total / max(total_digits, 1),
            "n_examples": len(examples),
            "per_subtask": per_subtask,
        }

    def run(self, K: Optional[int] = None, eval_set_path: str = None) -> dict:
        """
        Run full evaluation across all Quirke complexity splits.
        Loads eval set from HuggingFace (canonical N=100) or a specified path.

        Args:
            K: if set, use SoRL recursion eval with this K. None = SFT eval.
            eval_set_path: path to eval set JSON. None = canonical set from HF.

        Returns:
            dict with per-split results + summary.
        """
        self.model.eval()
        if eval_set_path == "epoch":
            from arithmetic.data.addition import get_epoch_eval_set
            categories = get_epoch_eval_set()
        else:
            categories = get_eval_set(path=eval_set_path)

        results = {
            "config": {
                "ops": "add_sub",
                "K": K,
                "mode": "sorl" if K is not None else "sft",
                "n_digits": self.n_digits,
                "n_per_split": 100,  # canonical eval set
            },
            "splits": {},
        }

        total_correct = 0
        total_examples = 0
        total_digits_correct = 0
        total_digits = 0

        for split_name, examples in categories.items():
            if not examples:
                continue
            split_result = self._eval_split(examples, K=K)
            results["splits"][split_name] = split_result
            total_correct += int(split_result["full_accuracy"] * split_result["n_examples"])
            total_examples += split_result["n_examples"]
            total_digits_correct += int(split_result["digit_accuracy"] * split_result["n_examples"] * self.answer_len)
            total_digits += split_result["n_examples"] * self.answer_len

        results["summary"] = {
            "overall_accuracy": total_correct / max(total_examples, 1),
            "digit_accuracy": total_digits_correct / max(total_digits, 1),
            "total_examples": total_examples,
            "n_splits": len(results["splits"]),
        }

        self.model.train()
        return results

    @staticmethod
    def save(results: dict, path: str):
        """Save results dict to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)

    @staticmethod
    def load(path: str) -> dict:
        """Load results dict from JSON."""
        with open(path) as f:
            return json.load(f)

    # ── Visualization ─────────────────────────────────────────────────

    @staticmethod
    def print_table(results: dict):
        """Print box-drawing table of results by complexity split."""
        cfg = results["config"]
        mode = cfg["mode"].upper()
        K_str = f" K={cfg['K']}" if cfg["K"] is not None else ""
        header = f"{mode}{K_str} | {cfg['ops']} | {cfg['n_digits']}-digit"

        splits = results["splits"]
        # Sort: add_S0..S6, add_random, sub_M0..M6, sub_random
        sorted_names = sorted(splits.keys(), key=lambda s: (
            0 if s.startswith("add_S") else 1 if s == "add_random" else
            2 if s.startswith("sub_M") else 3,
            int(s.split("S")[-1]) if "S" in s and s[-1].isdigit() else
            int(s.split("M")[-1]) if "M" in s and s[-1].isdigit() else 99,
        ))

        # Collect subtask labels that appear
        all_subtasks = []
        for name in sorted_names:
            for label in splits[name].get("per_subtask", {}):
                if label not in all_subtasks:
                    all_subtasks.append(label)

        print(f"  {header}")
        print(f"  ┌{'─' * 14}┬{'─' * 8}┬{'─' * 6}┬" + "┬".join(f"{'─' * 8}" for _ in all_subtasks) + "┐")
        sub_hdr = "│".join(f" {s:>6} " for s in all_subtasks)
        print(f"  │ {'Split':<12} │ {'Acc':>6} │ {'N':>4} │{sub_hdr}│")
        print(f"  ├{'─' * 14}┼{'─' * 8}┼{'─' * 6}┼" + "┼".join(f"{'─' * 8}" for _ in all_subtasks) + "┤")

        for name in sorted_names:
            s = splits[name]
            acc = f"{s['full_accuracy'] * 100:5.1f}%"
            n = str(s["n_examples"])
            sub_vals = []
            for label in all_subtasks:
                if label in s.get("per_subtask", {}):
                    v = s["per_subtask"][label]["accuracy"]
                    sub_vals.append(f" {v * 100:5.1f}% ")
                else:
                    sub_vals.append(f" {'—':>6} ")
            sub_str = "│".join(sub_vals)
            print(f"  │ {name:<12} │ {acc:>6} │ {n:>4} │{sub_str}│")

        print(f"  └{'─' * 14}┴{'─' * 8}┴{'─' * 6}┴" + "┴".join(f"{'─' * 8}" for _ in all_subtasks) + "┘")

        summary = results["summary"]
        digit_acc = summary.get('digit_accuracy', 0)
        print(f"  Overall: {summary['overall_accuracy'] * 100:.1f}% full-seq, {digit_acc * 100:.1f}% per-digit ({summary['total_examples']} examples)")

    @staticmethod
    def plot_by_complexity(results: dict, path: str = None, show: bool = False):
        """Bar chart of full accuracy by complexity split."""
        import matplotlib.pyplot as plt

        splits = results["splits"]
        cfg = results["config"]

        add_splits = {k: v for k, v in splits.items() if k.startswith("add_S")}
        sub_splits = {k: v for k, v in splits.items() if k.startswith("sub_M")}

        fig, axes = plt.subplots(1, max(1, int(bool(add_splits)) + int(bool(sub_splits))),
                                 figsize=(6 * max(1, int(bool(add_splits)) + int(bool(sub_splits))), 5))
        if not isinstance(axes, (list, type(plt.subplots()[1]))):
            axes = [axes]
        axes = list(axes) if hasattr(axes, '__iter__') else [axes]

        plot_idx = 0
        for label_prefix, split_group, title in [
            ("S", add_splits, "Addition by carry cascade depth"),
            ("M", sub_splits, "Subtraction by borrow cascade depth"),
        ]:
            if not split_group:
                continue
            ax = axes[plot_idx]
            names = sorted(split_group.keys(), key=lambda s: int(s.split(label_prefix)[-1]))
            accs = [split_group[n]["full_accuracy"] * 100 for n in names]
            short_names = [n.split("_")[-1] for n in names]

            bars = ax.bar(short_names, accs, color="#4C72B0", edgecolor="white")
            ax.set_ylim(0, 105)
            ax.set_ylabel("Full accuracy (%)")
            ax.set_xlabel("Complexity")
            ax.set_title(title)
            for bar, acc in zip(bars, accs):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{acc:.0f}", ha="center", va="bottom", fontsize=9)
            plot_idx += 1

        mode = cfg["mode"].upper()
        K_str = f" K={cfg['K']}" if cfg["K"] is not None else ""
        fig.suptitle(f"{mode}{K_str} | {cfg['ops']} | {cfg['n_digits']}-digit", fontsize=13)
        plt.tight_layout()

        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

    @staticmethod
    def plot_subtask_heatmap(results: dict, path: str = None, show: bool = False):
        """Heatmap of per-subtask digit accuracy across complexity splits."""
        import matplotlib.pyplot as plt
        import numpy as np

        splits = results["splits"]
        cfg = results["config"]

        # Collect all subtask labels and split names
        split_names = sorted(splits.keys(), key=lambda s: (
            0 if s.startswith("add_S") else 1 if s == "add_random" else
            2 if s.startswith("sub_M") else 3,
            int(s.split("S")[-1]) if "S" in s and s[-1].isdigit() else
            int(s.split("M")[-1]) if "M" in s and s[-1].isdigit() else 99,
        ))
        all_subtasks = []
        for name in split_names:
            for label in splits[name].get("per_subtask", {}):
                if label not in all_subtasks:
                    all_subtasks.append(label)

        matrix = np.full((len(all_subtasks), len(split_names)), np.nan)
        for j, name in enumerate(split_names):
            for i, label in enumerate(all_subtasks):
                if label in splits[name].get("per_subtask", {}):
                    matrix[i, j] = splits[name]["per_subtask"][label]["accuracy"] * 100

        fig, ax = plt.subplots(figsize=(max(8, len(split_names) * 0.9), max(4, len(all_subtasks) * 0.6)))
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

        ax.set_xticks(range(len(split_names)))
        ax.set_xticklabels([n.split("_", 1)[-1] for n in split_names], rotation=45, ha="right")
        ax.set_yticks(range(len(all_subtasks)))
        ax.set_yticklabels(all_subtasks)

        for i in range(len(all_subtasks)):
            for j in range(len(split_names)):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=8,
                            color="black" if 30 < val < 80 else "white")

        plt.colorbar(im, ax=ax, label="Digit accuracy (%)")
        mode = cfg["mode"].upper()
        K_str = f" K={cfg['K']}" if cfg["K"] is not None else ""
        ax.set_title(f"Subtask accuracy: {mode}{K_str} | {cfg['ops']} | {cfg['n_digits']}-digit")
        plt.tight_layout()

        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

    @staticmethod
    def compare_table(results_list: List[dict], labels: List[str] = None):
        """Print side-by-side comparison of multiple eval results."""
        if labels is None:
            labels = [r["config"]["mode"] + (f"_K{r['config']['K']}" if r["config"]["K"] else "")
                      for r in results_list]

        # Collect all split names across results
        all_splits = []
        for r in results_list:
            for s in r["splits"]:
                if s not in all_splits:
                    all_splits.append(s)
        all_splits.sort(key=lambda s: (
            0 if s.startswith("add_S") else 1 if s == "add_random" else
            2 if s.startswith("sub_M") else 3,
            int(s.split("S")[-1]) if "S" in s and s[-1].isdigit() else
            int(s.split("M")[-1]) if "M" in s and s[-1].isdigit() else 99,
        ))

        col_w = 8
        print(f"  ┌{'─' * 14}┬" + "┬".join(f"{'─' * col_w}" for _ in labels) + "┐")
        hdr = "│".join(f" {l:>{col_w - 2}} " for l in labels)
        print(f"  │ {'Split':<12} │{hdr}│")
        print(f"  ├{'─' * 14}┼" + "┼".join(f"{'─' * col_w}" for _ in labels) + "┤")

        for split in all_splits:
            vals = []
            for r in results_list:
                if split in r["splits"]:
                    acc = r["splits"][split]["full_accuracy"] * 100
                    vals.append(f" {acc:5.1f}% ")
                else:
                    vals.append(f" {'—':>6} ")
            row = "│".join(vals)
            print(f"  │ {split:<12} │{row}│")

        print(f"  ├{'─' * 14}┼" + "┼".join(f"{'─' * col_w}" for _ in labels) + "┤")
        totals = []
        for r in results_list:
            acc = r["summary"]["overall_accuracy"] * 100
            totals.append(f" {acc:5.1f}% ")
        print(f"  │ {'Overall':<12} │" + "│".join(totals) + "│")
        print(f"  └{'─' * 14}┴" + "┴".join(f"{'─' * col_w}" for _ in labels) + "┘")
