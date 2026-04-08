#!/usr/bin/env python3
"""
Optuna hyperparameter sweep for SoRL v1 on arithmetic.

Searches over: alpha_info_gain, alpha_abs, alpha_soft_zipf, abs_vocab, K, lr
Uses pruning: kills bad trials early based on intermediate accuracy.

Usage:
    CUDA_VISIBLE_DEVICES=0 python arithmetic/scripts/optuna_sweep.py --n_trials 50 --ops add
"""
import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import wandb
import optuna
from optuna.trial import TrialState
from optuna_integration.wandb import WeightsAndBiasesCallback
from transformers import AutoTokenizer, Qwen3Config
from sorl.sorl_wrapper import SorlModelWrapper
from sorl.trainer_ablate import SoRLTrainer, SoRLConfig
from arithmetic.train import (
    Qwen3ArithmeticDataset, collate_fn,
    eval_with_recursion, TOKENIZER_NAME,
)


def make_model(abs_vocab, n_layer, n_head, n_embd, tokenizer):
    config = Qwen3Config(
        hidden_size=n_embd, num_hidden_layers=n_layer,
        num_attention_heads=n_head, num_key_value_heads=n_head,
        intermediate_size=n_embd * 4, vocab_size=tokenizer.vocab_size,
        max_position_embeddings=128,
    )
    return SorlModelWrapper.from_scratch(
        config, [tokenizer.vocab_size, abs_vocab], tokenizer.pad_token_id
    )


class PruningCallback:
    """Reports accuracy to Optuna for pruning."""
    def __init__(self, trial, model, val_ds, device, K):
        self.trial = trial
        self.model = model
        self.val_ds = val_ds
        self.device = device
        self.K = K
        self.step = 0

    def __call__(self, accuracy):
        self.trial.report(accuracy, self.step)
        self.step += 1
        if self.trial.should_prune():
            raise optuna.TrialPruned()


class PruningSoRLTrainer(SoRLTrainer):
    """SoRLTrainer that reports eval accuracy to Optuna."""
    pruning_callback = None

    def evaluate(self, eval_K=None):
        if self.compute_accuracy is None or self.val_dataset is None:
            return None
        self.raw_model.eval()
        # Use proper generation eval
        acc = eval_with_recursion(
            self.raw_model, self.val_dataset, self.device,
            K=self.config.K, num_samples=50,  # fewer samples for speed
        )
        self.raw_model.train()
        result = {"accuracy": acc}
        if self.pruning_callback:
            self.pruning_callback(acc)
        return result


def objective(trial, args, tokenizer):
    # Sample hyperparameters
    alpha_info_gain = trial.suggest_float("alpha_info_gain", 0.5, 30.0, log=True)
    alpha_abs = trial.suggest_float("alpha_abs", 0.01, 1.0, log=True)
    alpha_soft_zipf = trial.suggest_float("alpha_soft_zipf", 0.0, 2.0)
    abs_vocab = trial.suggest_int("abs_vocab", 2, 20)
    K = trial.suggest_int("K", 2, 6)
    lr = trial.suggest_float("lr", 1e-5, 3e-4, log=True)

    print(f"\n{'─' * 50}")
    print(f"  Trial {trial.number}: ig={alpha_info_gain:.2f} abs={alpha_abs:.3f} "
          f"zipf={alpha_soft_zipf:.2f} vocab={abs_vocab} K={K} lr={lr:.1e}")
    print(f"{'─' * 50}")

    model = make_model(abs_vocab, args.n_layer, args.n_head, args.n_embd, tokenizer)
    train_ds = Qwen3ArithmeticDataset(tokenizer, 6, args.ops, args.dataset_size)
    val_ds = Qwen3ArithmeticDataset(tokenizer, 6, args.ops, 1000)

    cfg = SoRLConfig(
        K=K, batch_size=args.batch_size,
        num_epochs=args.num_epochs, lr=lr,
        output_dir=f"ckpt/optuna/trial_{trial.number}",
        log_every=100, eval_every=args.eval_every,
        save_every=999999, eval_samples=50,
        alpha_info_gain=alpha_info_gain,
        alpha_abs=alpha_abs,
        alpha_soft_zipf=alpha_soft_zipf,
        alpha_traj=0.0,
    )

    trainer = PruningSoRLTrainer(
        model, tokenizer, train_ds, val_ds,
        compute_accuracy=lambda *a, **kw: {"accuracy": 0},  # unused, we override evaluate()
        collate_fn=collate_fn, config=cfg, device=args.device,
    )
    trainer.pruning_callback = PruningCallback(trial, model, val_ds, args.device, K)

    try:
        trainer.train()
    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"  Trial {trial.number} failed: {e}")
        return 0.0

    # Final eval with more samples
    acc = eval_with_recursion(model, val_ds, args.device, K=K, num_samples=100)
    print(f"  Trial {trial.number} final accuracy: {acc:.3f}")

    # Log to wandb
    try:
        run = wandb.init(
            project="sorl-arithmetic-optuna",
            entity="nlp_and_interpretability",
            name=f"trial_{trial.number}_acc{acc:.2f}",
            config=trial.params,
            reinit=True,
        )
        wandb.log({"final_accuracy": acc, **trial.params})
        wandb.finish()
    except:
        pass

    # Cleanup
    del model, trainer
    torch.cuda.empty_cache()
    import shutil
    shutil.rmtree(f"ckpt/optuna/trial_{trial.number}", ignore_errors=True)

    return acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ops", default="add")
    p.add_argument("--n_layer", type=int, default=2)
    p.add_argument("--n_head", type=int, default=3)
    p.add_argument("--n_embd", type=int, default=510)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_epochs", type=int, default=2)
    p.add_argument("--dataset_size", type=int, default=100000)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--n_trials", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--study_name", type=str, default="sorl_v1_arithmetic")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        storage=f"sqlite:///ckpt/optuna_{args.ops}.db",
        load_if_exists=True,
    )

    wandb_callback = WeightsAndBiasesCallback(
        metric_name="accuracy",
        wandb_kwargs={
            "project": "sorl-arithmetic-optuna",
            "entity": "nlp_and_interpretability",
        },
        as_multirun=True,
    )

    study.optimize(
        lambda trial: objective(trial, args, tokenizer),
        n_trials=args.n_trials,
        callbacks=[wandb_callback],
    )

    print(f"\n{'═' * 60}")
    print(f"  Best trial: {study.best_trial.number}")
    print(f"  Best accuracy: {study.best_trial.value:.3f}")
    print(f"  Best params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
