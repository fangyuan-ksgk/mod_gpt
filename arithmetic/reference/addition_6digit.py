"""
Reference implementation: 6-digit addition with GAT.
Reproduces the Integer Addition paper (Nanda et al., ICLR 2024) using mod_gpt's GAT.

Paper config:
  - 1 layer, 3 heads, d_model=510, d_mlp=2040, ReLU, LayerNorm
  - 12-token vocab (0-9, +, =)
  - Online data generation, batch_size=64, 5000 steps
  - AdamW lr=8e-5, weight_decay=0.1, warmup 10 steps
  - Loss only on answer digits (last n_digits+1 positions)
  - UseSum9 augmentation: 20% of batches, 20% of positions forced to sum-to-9

Our config (as close as possible with GAT):
  - 2 layers (GAT uses U-net: 1 encoder + 1 decoder = minimum 2)
  - 3 heads, d_model=510 (510/3=170 per head)
  - MLP = 4*510 = 2040 (matches paper)
  - RMS norm instead of LayerNorm (GAT default)
  - Squared ReLU instead of ReLU (GAT default)
  - RoPE instead of learned positional embeddings
  - Vocab: use single-level vocab_sizes=[12] (no abstraction tokens)
"""
import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sorl.gat_sim import GAT, GATConfig
from arithmetic.datasets.addition import (
    generate_batch, make_eval_set, classify_digits,
    NUM_TOKENS, PLUS_INDEX, EQUALS_INDEX,
)


# ── Model wrapper for clean eval interface ──────────────────────────

class AdditionGAT:
    """Wraps GAT for addition task: handles loss masking and eval."""

    def __init__(self, n_digits=6, n_layer=2, n_head=3, n_embd=510,
                 device="cuda", compile_model=True):
        self.n_digits = n_digits
        self.seq_len = 3 * n_digits + 3  # XXXXXX+YYYYYY=ZZZZZZZ
        self.ans_start = 2 * n_digits + 2
        self.ans_len = n_digits + 1
        self.device = device

        config = GATConfig(
            vocab_sizes=[NUM_TOKENS],  # no abstraction tokens — pure baseline
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            device=device,
            _compile=compile_model,
        )
        self.model = GAT(config)
        self.memory_span = self.seq_len
        self.attn_blocksize = self.seq_len

    def train_step(self, tokens):
        """
        tokens: (B, seq_len) — full sequence including answer.
        Returns loss (only on answer positions).
        """
        loss_all, logits = self.model.forward(
            tokens, self.memory_span, self.attn_blocksize
        )
        # loss_all shape: (B * (seq_len-1),)
        # Mask to only answer positions
        B = tokens.shape[0]
        loss_all = loss_all.view(B, -1)  # (B, seq_len-1)

        # Answer positions in the shifted target: ans_start-1 to seq_len-2
        # Because loss[i] = CE(logits[i], tokens[i+1])
        ans_loss = loss_all[:, self.ans_start - 1:self.seq_len - 1]
        return ans_loss.mean()

    def get_logits(self, tokens):
        """Get full logits for eval."""
        _, logits = self.model.forward(
            tokens, self.memory_span, self.attn_blocksize
        )
        return logits

    def get_hidden_states(self, tokens):
        """Get hidden states (for SAE training)."""
        x = self.model._forward_pass(
            tokens, self.memory_span, self.attn_blocksize
        )
        return x

    def predict(self, tokens):
        """
        Autoregressive prediction of answer digits.
        tokens: (B, ans_start) — query only (up to and including '=').
        Returns: predicted answer digits (B, ans_len).
        """
        B = tokens.shape[0]
        idx = tokens.clone()

        for _ in range(self.ans_len):
            _, logits = self.model.forward(
                idx, self.memory_span, self.attn_blocksize
            )
            next_logits = logits[:, -1, :NUM_TOKENS]
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, next_token], dim=1)

        return idx[:, self.ans_start:]

    def eval_accuracy(self, n_examples=64):
        """Quick eval: generate random problems, check full-sequence accuracy."""
        self.model.eval()
        tokens, labels, _ = generate_batch(
            n_examples, self.n_digits, use_sum9_aug=False, device=self.device
        )
        with torch.no_grad():
            logits = self.get_logits(tokens)
            pred_logits = logits[:, self.ans_start - 1:-1, :NUM_TOKENS]
            preds = pred_logits.argmax(dim=-1)

        targets = tokens[:, self.ans_start:]
        correct = (preds == targets).all(dim=1).float().mean().item()
        self.model.train()
        return correct

    def eval_by_subtask(self):
        """Detailed eval by sub-task category."""
        self.model.eval()
        categories = make_eval_set(self.n_digits, self.device)
        results = {}

        for cat_name, examples in categories.items():
            if not examples:
                continue
            tokens = torch.tensor(
                [e.tokens for e in examples], dtype=torch.long, device=self.device
            )
            all_labels = [e.labels for e in examples]

            with torch.no_grad():
                logits = self.get_logits(tokens)
                pred_logits = logits[:, self.ans_start - 1:-1, :NUM_TOKENS]
                preds = pred_logits.argmax(dim=-1)

            targets = tokens[:, self.ans_start:]
            correct_digits = (preds == targets)

            full_acc = correct_digits.all(dim=1).float().mean().item()

            subtask_correct = {t: [] for t in ["BA", "MC1", "MS9", "UC1", "US9"]}
            for b in range(len(examples)):
                for d in range(self.ans_len):
                    label = all_labels[b][d]
                    subtask_correct[label].append(correct_digits[b, d].item())

            per_subtask = {}
            for t, vals in subtask_correct.items():
                if vals:
                    per_subtask[t] = sum(vals) / len(vals)

            results[cat_name] = {"full_acc": full_acc, "per_subtask": per_subtask}

        self.model.train()
        return results


# ── Training loop ───────────────────────────────────────────────────

def train(args):
    device = args.device
    n_digits = args.n_digits

    print(f"Training addition baseline: {n_digits}-digit, device={device}")
    print(f"Architecture: n_layer={args.n_layer}, n_head={args.n_head}, n_embd={args.n_embd}")

    wrapper = AdditionGAT(
        n_digits=n_digits,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        device=device,
        compile_model=not args.no_compile,
    )
    model = wrapper.model

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(step / 10, 1.0)
    )

    history = []
    best_acc = 0.0

    for step in range(1, args.num_steps + 1):
        model.train()
        optimizer.zero_grad()

        tokens, labels, _ = generate_batch(
            args.batch_size, n_digits, use_sum9_aug=True, device=device
        )
        loss = wrapper.train_step(tokens)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # ── Logging ─────────────────────────────────────────────
        if step % args.log_every == 0:
            acc = wrapper.eval_accuracy(n_examples=128)
            record = {"step": step, "loss": loss.item(), "acc": acc}
            history.append(record)
            print(f"step {step:5d} | loss: {loss.item():.4f} | acc: {acc:.3f}")

            if acc > best_acc:
                best_acc = acc

        # ── Detailed eval ───────────────────────────────────────
        if step % args.eval_every == 0:
            results = wrapper.eval_by_subtask()
            print(f"\n{'─' * 60}")
            print(f"  Detailed eval at step {step}")
            print(f"{'─' * 60}")
            for cat, res in results.items():
                subtask_str = " | ".join(
                    f"{t}: {v:.3f}" for t, v in sorted(res["per_subtask"].items())
                )
                print(f"  {cat:15s} | full: {res['full_acc']:.3f} | {subtask_str}")
            print(f"{'─' * 60}\n")

    # ── Save ────────────────────────────────────────────────────
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")
        with open(save_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        with open(save_dir / "config.json", "w") as f:
            json.dump(vars(args), f, indent=2)
        print(f"Saved to {save_dir}")

    print(f"\nBest accuracy: {best_acc:.3f}")
    return wrapper, history


def main():
    parser = argparse.ArgumentParser(description="Train addition baseline (paper reproduction)")
    parser.add_argument("--n_digits", type=int, default=6)
    parser.add_argument("--n_layer", type=int, default=2, help="GAT minimum is 2 (U-net)")
    parser.add_argument("--n_head", type=int, default=3)
    parser.add_argument("--n_embd", type=int, default=510, help="510 = 3 heads * 170")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--save_dir", type=str, default="ckpt/addition_baseline")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_compile", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
