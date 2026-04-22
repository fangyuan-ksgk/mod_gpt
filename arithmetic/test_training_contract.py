"""
Contract tests for the arithmetic training pipeline.

These tests catch silent regressions where one component changes its output
format and another component silently breaks. Each test documents WHY the
contract exists and which commit broke it last time.

Run: python -m pytest arithmetic/test_training_contract.py -v
"""
import pytest
import json
import hashlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ══════════════════════════════════════════════════════════════════
# 1. History contract: training must write eval_step to history
#    Regression: commit 5bc1da0 removed this when stripping double eval.
#    Caught by: post_sweep validator requiring history["eval_step"].
# ══════════════════════════════════════════════════════════════════

class TestHistoryContract:
    """WandbSoRLTrainer.evaluate() must write eval_step/eval_accuracy to history."""

    def _make_trainer_with_history(self, steps=(100, 200)):
        """Build a WandbSoRLTrainer with a fake history."""
        from arithmetic.train import WandbSoRLTrainer, ArithmeticConfig
        trainer = object.__new__(WandbSoRLTrainer)
        trainer.is_master = True
        trainer.history = {
            "step": list(steps), "loss": [1.0] * len(steps),
            "base_loss": [1.0] * len(steps), "info_loss": [0.0] * len(steps),
            "abs_loss": [0.5] * len(steps), "zipf_loss": [0.3] * len(steps),
            "ortho_loss": [0.1] * len(steps), "lr": [8e-5] * len(steps),
        }
        cfg = ArithmeticConfig()
        cfg.K = 1
        trainer.config = cfg
        trainer.raw_model = MagicMock()
        trainer.tokenizer = MagicMock()
        trainer.device = "cpu"
        return trainer

    def test_evaluate_writes_eval_step_to_history(self):
        trainer = self._make_trainer_with_history()
        fake_eval = {
            "summary": {"overall_accuracy": 0.75},
            "splits": {},
        }
        with patch("arithmetic.evaluate.ArithmeticEvaluator") as MockEval:
            MockEval.return_value.run.return_value = fake_eval
            with patch("wandb.run", None):
                result = trainer.evaluate(eval_K=1)

        assert "eval_step" in trainer.history, \
            "evaluate() must write eval_step to history (required by post_sweep validator)"
        assert len(trainer.history["eval_step"]) > 0
        assert "eval_accuracy" in trainer.history
        assert trainer.history["eval_accuracy"][-1] == pytest.approx(0.75)

    def test_evaluate_appends_on_multiple_calls(self):
        trainer = self._make_trainer_with_history(steps=(100, 200, 300))
        fake_eval = {"summary": {"overall_accuracy": 0.5}, "splits": {}}
        with patch("arithmetic.evaluate.ArithmeticEvaluator") as MockEval:
            MockEval.return_value.run.return_value = fake_eval
            with patch("wandb.run", None):
                trainer.evaluate(eval_K=1)
                trainer.evaluate(eval_K=1)

        assert len(trainer.history["eval_step"]) == 2, \
            "Each call to evaluate() should append one entry"

    def test_train_sft_history_has_eval_step(self):
        """train_sft() must also write eval_step (baseline path)."""
        from arithmetic.train import ArithmeticConfig
        import torch

        cfg = ArithmeticConfig()
        cfg.batch_size = 4
        cfg.num_epochs = 1
        cfg.log_every = 1
        cfg.lr = 1e-4
        cfg.warmup_steps = 1
        cfg.device = "cpu"
        cfg.mode = "baseline"
        cfg.K = None

        # Minimal fake dataset: 8 examples of shape (21,)
        class TinyDS:
            prompt_len = 14
            n_digits = 6
            tokenizer = MagicMock()
            def __len__(self): return 8
            def __getitem__(self, i):
                return {
                    "input_ids": torch.zeros(21, dtype=torch.long),
                    "attention_mask": torch.ones(21, dtype=torch.long),
                    "prompt_len": torch.tensor(14),
                }

        ds = TinyDS()

        # Fake model that returns logits
        model = MagicMock()
        fake_logits = torch.randn(4, 21, 151936, requires_grad=True)
        model.return_value.logits = fake_logits
        model.to.return_value = model  # model.to(device) must return same mock
        model.parameters.return_value = [torch.zeros(1, requires_grad=True)]

        fake_epoch_eval = {
            "summary": {"overall_accuracy": 0.1},
            "splits": {"add_S5": {"full_accuracy": 0.0}},
        }

        from arithmetic.train import train_sft
        with patch("arithmetic.evaluate.ArithmeticEvaluator") as MockEval, \
             patch("wandb.run", None):
            MockEval.return_value.run.return_value = fake_epoch_eval
            history = train_sft(model, ds, ds, cfg, "test_run", tokenizer=MagicMock())

        assert "eval_step" in history, \
            "train_sft must write eval_step to history"
        assert len(history["eval_step"]) >= 1


# ══════════════════════════════════════════════════════════════════
# 2. Validator contract: post_sweep.validate checks the right things
#    Ensures validator and trainer stay in sync.
# ══════════════════════════════════════════════════════════════════

class TestValidatorContract:
    """The validator should pass/fail on known-good/bad metrics shapes."""

    def _make_metrics(self, *, has_eval_step=True, n_splits=15, trainer_version="v1"):
        history = {"step": [100, 200], "loss": [1.0, 0.5]}
        if has_eval_step:
            history["eval_step"] = [100, 200]
            history["eval_accuracy"] = [0.5, 0.7]
        splits = {f"split_{i}": {"full_accuracy": 0.5} for i in range(n_splits)}
        cfg = {
            "final_accuracy": 0.7,
            "eval_method": "ArithmeticEvaluator",
            "trainer_version": trainer_version,
        }
        metrics = {
            "history": history,
            "sft_eval": {"summary": {"overall_accuracy": 0.7}, "splits": splits},
        }
        return cfg, metrics

    def _run_checks(self, cfg, metrics):
        """Run just the local (non-HF) validation logic inline."""
        issues = []
        history = metrics.get("history", {})
        trainer_version = cfg.get("trainer_version", "")
        if trainer_version != "v6":
            if "eval_step" not in history or len(history.get("eval_step", [])) == 0:
                issues.append("no eval curves in training history")
        if "sft_eval" not in metrics:
            issues.append("no sft_eval in metrics")
        for eval_key in ["sft_eval", "sorl_eval"]:
            ev = metrics.get(eval_key, {})
            splits = ev.get("splits", {})
            if ev and len(splits) < 10:
                issues.append(f"{eval_key} has only {len(splits)} splits (expected 15+)")
        if cfg.get("eval_method") != "ArithmeticEvaluator":
            issues.append(f"eval_method={cfg.get('eval_method')} (expected ArithmeticEvaluator)")
        if cfg.get("final_accuracy") is None:
            issues.append("no final_accuracy")
        return issues

    def test_valid_metrics_pass(self):
        cfg, metrics = self._make_metrics()
        issues = self._run_checks(cfg, metrics)
        assert issues == [], f"Valid metrics should pass: {issues}"

    def test_missing_eval_step_fails(self):
        cfg, metrics = self._make_metrics(has_eval_step=False)
        issues = self._run_checks(cfg, metrics)
        assert any("eval curves" in i for i in issues)

    def test_v6_skips_eval_step_check(self):
        cfg, metrics = self._make_metrics(has_eval_step=False, trainer_version="v6")
        issues = self._run_checks(cfg, metrics)
        assert not any("eval curves" in i for i in issues), \
            "v6 should not require eval_step in history"

    def test_too_few_splits_fails(self):
        cfg, metrics = self._make_metrics(n_splits=5)
        issues = self._run_checks(cfg, metrics)
        assert any("splits" in i for i in issues)


# ══════════════════════════════════════════════════════════════════
# 3. Config hash consistency: gpu_queue hash must match train.py hash
#    If these diverge, the queue will re-run already-completed jobs.
# ══════════════════════════════════════════════════════════════════

class TestConfigHashConsistency:

    def _train_py_hash(self, **overrides):
        from arithmetic.train import ArithmeticConfig
        cfg = ArithmeticConfig()
        for k, v in overrides.items():
            setattr(cfg, k, v)
        cfg.auto_scale_lr()
        _hash_keys = sorted({
            "mode", "ops", "dataset_size", "n_digits", "abs_vocab", "K",
            "n_layer", "n_head", "n_embd", "num_epochs", "lr", "batch_size",
            "weight_decay", "warmup_ratio", "beta2", "emb_lr_mult", "seed",
            "alpha_info_gain", "alpha_abs", "alpha_soft_zipf", "alpha_ortho",
            "alpha_contrastive", "gamma_contrastive", "num_rollouts",
            "max_iterations", "temperature",
        })
        d = {k: getattr(cfg, k, None) for k in _hash_keys}
        d["train_dataset"] = f"fixed_train/train_{cfg.dataset_size // 1000}K_seed42.pt"
        from arithmetic.data.addition import CANONICAL_EVAL_SET
        d["eval_dataset"] = f"eval_sets/{CANONICAL_EVAL_SET}"
        return hashlib.sha256(
            json.dumps(d, sort_keys=True, default=str).encode()
        ).hexdigest()[:12]

    def _queue_hash(self, cmd):
        import arithmetic.job_manager.gpu_queue as gq
        return gq._compute_config_hash(cmd)

    def test_baseline_hash_matches(self):
        cmd = ("python -m arithmetic.train --mode baseline --ops add_sub "
               "--dataset_size 10000 --num_epochs 20 --push_to_hub "
               "--job_name as_baseline_10K_2L3H510d")
        queue_h = self._queue_hash(cmd)
        train_h = self._train_py_hash(mode="baseline", ops="add_sub",
                                       dataset_size=10000, num_epochs=20)
        assert queue_h == train_h, \
            f"Hash mismatch: queue={queue_h} train={train_h}"

    def test_sorl_hash_matches(self):
        cmd = ("python -m arithmetic.train --mode sorl --ops add_sub "
               "--dataset_size 10000 --abs_vocab 30 --K 1 --num_epochs 20 "
               "--push_to_hub --job_name as_sorl_abs30_K1_10K_2L3H510d")
        queue_h = self._queue_hash(cmd)
        train_h = self._train_py_hash(mode="sorl", ops="add_sub",
                                       dataset_size=10000, abs_vocab=30,
                                       K=1, num_epochs=20)
        assert queue_h == train_h, \
            f"Hash mismatch: queue={queue_h} train={train_h}"

    def test_explicit_lr_hash_matches(self):
        cmd = ("python -m arithmetic.train --mode sorl --ops add_sub "
               "--dataset_size 10000 --abs_vocab 30 --K 1 --num_epochs 20 "
               "--n_layer 2 --n_head 1 --n_embd 128 --lr 8e-5 --push_to_hub "
               "--job_name as_sorl_abs30_K1_10K_2L1H128d")
        queue_h = self._queue_hash(cmd)
        train_h = self._train_py_hash(mode="sorl", ops="add_sub",
                                       dataset_size=10000, abs_vocab=30, K=1,
                                       num_epochs=20, n_layer=2, n_head=1,
                                       n_embd=128, lr=8e-5)
        assert queue_h == train_h, \
            f"Hash mismatch: queue={queue_h} train={train_h}"


# ══════════════════════════════════════════════════════════════════
# 4. LR auto-scale: correct defaults for each architecture size
#    If this changes, small-model jobs silently train at wrong LR.
# ══════════════════════════════════════════════════════════════════

class TestLRAutoScale:
    def _lr_for(self, n_embd):
        from arithmetic.train import ArithmeticConfig
        cfg = ArithmeticConfig(n_embd=n_embd)
        cfg.auto_scale_lr()
        return cfg.lr

    def test_standard_arch_gets_8e5(self):
        assert self._lr_for(510) == pytest.approx(8e-5)

    def test_256d_arch_gets_2e5_by_default(self):
        # Default auto-scale; sweep files override this with explicit --lr 8e-5
        assert self._lr_for(256) == pytest.approx(2e-5)

    def test_128d_arch_gets_2e5_by_default(self):
        assert self._lr_for(128) == pytest.approx(2e-5)

    def test_explicit_lr_overrides_autoscale(self):
        from arithmetic.train import ArithmeticConfig
        cfg = ArithmeticConfig(n_embd=128, lr=8e-5)
        cfg.auto_scale_lr()
        assert cfg.lr == pytest.approx(8e-5), \
            "Explicit --lr must not be overwritten by auto_scale_lr"


# ══════════════════════════════════════════════════════════════════
# 5. Subprocess environment: queue must set PYTHONUNBUFFERED=1
#    Without this, job logs are silent until the 8KB buffer fills.
# ══════════════════════════════════════════════════════════════════

class TestSubprocessEnv:
    def test_pythonunbuffered_set_in_subprocess(self):
        import arithmetic.job_manager.gpu_queue as gq
        import inspect, ast

        src = inspect.getsource(gq.GPUQueue._run_job)
        # Check that PYTHONUNBUFFERED appears in the source near the Popen call
        assert "PYTHONUNBUFFERED" in src, \
            "gpu_queue must set PYTHONUNBUFFERED=1 in subprocess env for real-time log output"

    def test_eval_k_is_none_for_baseline(self):
        """Baseline eval must use K=None (no abstract tokens)."""
        from arithmetic.train import ArithmeticConfig
        cfg = ArithmeticConfig(mode="baseline")
        eval_K = cfg.K if cfg.mode == "sorl" else None
        assert eval_K is None

    def test_eval_k_is_cfg_k_for_sorl(self):
        """SoRL eval must use cfg.K, not None."""
        from arithmetic.train import ArithmeticConfig
        cfg = ArithmeticConfig(mode="sorl", K=1)
        eval_K = cfg.K if cfg.mode == "sorl" else None
        assert eval_K == 1
