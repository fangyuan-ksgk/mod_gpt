"""
Model wrapper for arithmetic experiments.
Wraps GAT for both baseline (no abstraction) and SoRL (with abstraction tokens).
"""
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sorl.gat_sim import GAT, GATConfig
from arithmetic.datasets.addition import NUM_TOKENS


class ArithmeticModel:
    """
    Unified wrapper for baseline and SoRL arithmetic models.

    Args:
        n_digits: number of digits in operands
        n_abs_tokens: abstract vocab size (0 = baseline, >0 = SoRL)
        n_layer: number of transformer layers (GAT U-net: min 2)
        n_head: number of attention heads
        n_embd: embedding dimension
        device: cuda device string
        compile_model: use torch.compile
    """

    def __init__(self, n_digits=6, n_abs_tokens=0,
                 n_layer=4, n_head=4, n_embd=512,
                 device="cuda", compile_model=True):
        self.n_digits = n_digits
        self.n_abs_tokens = n_abs_tokens
        self.seq_len = 3 * n_digits + 3
        self.ans_start = 2 * n_digits + 2
        self.ans_len = n_digits + 1
        self.device = device

        vocab_sizes = [NUM_TOKENS]
        if n_abs_tokens > 0:
            vocab_sizes.append(n_abs_tokens)

        config = GATConfig(
            vocab_sizes=vocab_sizes,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            device=device,
            _compile=compile_model,
        )
        self.model = GAT(config).to(device)
        self.memory_span = self.seq_len
        self.attn_blocksize = self.seq_len

    @property
    def is_sorl(self):
        return self.n_abs_tokens > 0

    def train_loss(self, tokens):
        """Compute loss on answer positions only."""
        loss_all, logits = self.model.forward(
            tokens, self.memory_span, self.attn_blocksize
        )
        B = tokens.shape[0]
        loss_all = loss_all.view(B, -1)
        ans_loss = loss_all[:, self.ans_start - 1 : self.seq_len - 1]
        return ans_loss.mean()

    def get_logits(self, tokens):
        """Full logits for evaluation."""
        _, logits = self.model.forward(
            tokens, self.memory_span, self.attn_blocksize
        )
        return logits

    def get_hidden_states(self, tokens):
        """Hidden states for SAE training."""
        return self.model._forward_pass(
            tokens, self.memory_span, self.attn_blocksize
        )

    def predict_accuracy(self, tokens):
        """Check full-sequence accuracy on a batch."""
        with torch.no_grad():
            logits = self.get_logits(tokens)
            preds = logits[:, self.ans_start - 1:-1, :NUM_TOKENS].argmax(dim=-1)
        targets = tokens[:, self.ans_start:]
        return (preds == targets).all(dim=1).float().mean().item()

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "model.pt")

    def load(self, path):
        state = torch.load(Path(path) / "model.pt", map_location=self.device)
        self.model.load_state_dict(state)
