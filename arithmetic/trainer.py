"""
Trainers for GAT arithmetic experiments.
- BaselineTrainer: standard next-token prediction (no abstraction)
- SoRLTrainer: with abstraction token search and info-gain loss
"""
import torch
from sorl.gat_sim import GAT, GATConfig
from sorl.neo_utils import sorl_search, sorl_evaluate, generate
from sorl.info import SoRLLoss
from sorl.gapt import GatedPhaseTransition
from arithmetic.datasets.multiplication import (
    DigitTokenizer, MultiplicationDataLoader, process_query, check_answer,
)


class BaselineTrainer:
    def __init__(self, model, train_loader, val_loader,
                 lr=1e-3, weight_decay=0.1,
                 memory_span=1792, attn_blocksize=1792):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.memory_span = memory_span
        self.attn_blocksize = attn_blocksize
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_step(self, batch_size):
        self.optimizer.zero_grad()
        tokens, _ = self.train_loader.get_batch(batch_size)
        loss = self.model.forward(tokens, self.memory_span, self.attn_blocksize)[0].mean()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def val_step(self, batch_size):
        with torch.no_grad():
            tokens, _ = self.val_loader.get_batch(batch_size)
            loss = self.model.forward(tokens, self.memory_span, self.attn_blocksize)[0].mean()
        return loss.item()

    def train(self, num_steps, batch_size, log_every=2):
        for step in range(num_steps):
            self.train_step(batch_size)
            if step % log_every == 0:
                val_loss = self.val_step(batch_size)
                print(f"step {step:4d} | val_loss: {val_loss:.2f}")


class SoRLTrainer:
    def __init__(self, model, train_loader, val_loader,
                 lr=1e-3, weight_decay=0.1,
                 memory_span=1792, attn_blocksize=1792,
                 K=3, n=2, max_iterations=2,
                 alpha_abs=0.1, alpha_info_gain=10.0, alpha_soft_zipf=1.0,
                 n_eval=4, temperatures=None, temperatures_eval=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.memory_span = memory_span
        self.attn_blocksize = attn_blocksize
        self.K = K
        self.n = n
        self.max_iterations = max_iterations
        self.alpha_abs = alpha_abs
        self.alpha_info_gain = alpha_info_gain
        self.alpha_soft_zipf = alpha_soft_zipf
        self.n_eval = n_eval

        device = model.device
        self.temperatures = temperatures if temperatures is not None else torch.tensor([0.0, 5.0], device=device)
        self.temperatures_eval = temperatures_eval if temperatures_eval is not None else torch.tensor([0.0, 5.0, 5.0, 5.0], device=device)

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = SoRLLoss(model.vocab_sizes[1])
        self.gapt = GatedPhaseTransition(p_m=10)

    def train_step(self, batch_size):
        self.optimizer.zero_grad()
        tokens, _ = self.train_loader.get_batch(batch_size)

        with torch.no_grad():
            search_tokens, search_ppt, search_adv = sorl_search(
                tokens, self.model, n=self.n, K=self.K,
                max_iterations=self.max_iterations,
                memory_span=self.memory_span, attn_blocksize=self.attn_blocksize,
                temperature=self.temperatures, truncate_seq_len=False,
            )

        base_traj_loss = self.model.forward(tokens, self.memory_span, self.attn_blocksize)[0].mean()
        info_gain_loss, abs_loss, zipf_bigram_loss = self.loss_fn(
            search_tokens, self.model, base_traj_loss.detach(),
            self.memory_span, self.attn_blocksize,
        )
        loss = (base_traj_loss
                + self.alpha_info_gain * info_gain_loss
                + self.alpha_abs * abs_loss
                + self.alpha_soft_zipf * zipf_bigram_loss)

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def val_step(self, batch_size):
        with torch.no_grad():
            tokens, _ = self.val_loader.get_batch(batch_size)
            # reuse last search tokens for eval
            search_tokens, _, _ = sorl_search(
                tokens, self.model, n=self.n, K=self.K,
                max_iterations=self.max_iterations,
                memory_span=self.memory_span, attn_blocksize=self.attn_blocksize,
                temperature=self.temperatures, truncate_seq_len=False,
            )
            val_tokens, val_adv, traj_loss, abs_loss = sorl_evaluate(
                search_tokens, self.model, n=self.n_eval, K=self.K,
                max_iterations=self.max_iterations,
                memory_span=self.memory_span, attn_blocksize=self.attn_blocksize,
                temperature=self.temperatures_eval, truncate_seq_len=False,
            )
        return traj_loss.item(), abs_loss.item(), val_adv.item()

    def train(self, num_steps, batch_size, log_every=2):
        for step in range(num_steps):
            self.train_step(batch_size)
            if step % log_every == 0:
                traj_loss, abs_loss, search_adv = self.val_step(batch_size)
                print(f"step {step:4d} | traj_loss: {traj_loss:.2f} | abs_loss: {abs_loss:.2f} | search_adv: {search_adv * 100:.2f}%")

    def generate_answer(self, val_loader, tokenizer):
        tokens, _ = val_loader.get_batch(1)
        idx, answer_idx = process_query(tokens)
        print(f"question: {tokenizer.decode(idx[0].tolist()[1:-1])}")

        for _ in range(len(answer_idx[0]) * 2):
            idx = generate(
                self.model, idx, K=999, max_iterations=self.max_iterations,
                memory_span=self.memory_span, attn_blocksize=self.attn_blocksize,
                temperature=self.temperatures_eval[0],
            )
            idx_without_abstraction = idx[idx < self.model.vocab_sizes[0]]

        is_correct, pred_answer, true_answer = check_answer(idx_without_abstraction, answer_idx, tokenizer)
        print(f"is_correct: {is_correct} | pred: {pred_answer} | true: {true_answer}")
        return is_correct, pred_answer, true_answer
