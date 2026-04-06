"""
HuggingFace Hub utilities for arithmetic experiments.

Model repo:   amirali1985/arithmetic-sorl
Dataset repo: amirali1985/arithmetic-sorl-data

Structure (model repo):
    add_baseline/          # model checkpoint + config + metrics
    add_sorl_abs16/        # ...
    add_sub_sorl_abs8/     # ...

Structure (dataset repo):
    add_6digit/            # train.parquet, val.parquet, config.json
    add_sub_6digit/        # ...
"""
import os
import json
import torch
import tempfile
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

MODEL_REPO = "amirali1985/arithmetic-sorl"
DATASET_REPO = "amirali1985/arithmetic-sorl-data"


# ── Model save/load ────────────────────────────────────────────────

def save_model(model, config: dict, metrics: dict, subfolder: str,
               repo_id: str = MODEL_REPO):
    """
    Save model checkpoint + config + metrics to HF Hub.

    Args:
        model: SorlModelWrapper instance
        config: training config dict (from argparse)
        metrics: dict with accuracy, loss curves, etc.
        subfolder: e.g. "add_sorl_abs16"
    """
    api = HfApi()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # Save model weights
        model.save_pretrained(str(tmp))

        # Save config
        with open(tmp / "train_config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Save metrics
        with open(tmp / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Upload folder
        api.upload_folder(
            folder_path=str(tmp),
            repo_id=repo_id,
            path_in_repo=subfolder,
            commit_message=f"Upload {subfolder}",
        )

    print(f"Saved to {repo_id}/{subfolder}")


def load_model(subfolder: str, device: str = "cuda",
               repo_id: str = MODEL_REPO):
    """
    Load model + config + metrics from HF Hub.

    Returns: (model, config, metrics)
    """
    from transformers import Qwen3Config
    from sorl.sorl_wrapper import SorlModelWrapper

    local_dir = snapshot_download(
        repo_id=repo_id,
        allow_patterns=[f"{subfolder}/*"],
        local_dir=f"/tmp/hf_cache/{repo_id.split('/')[-1]}",
    )
    model_dir = Path(local_dir) / subfolder

    config = json.load(open(model_dir / "train_config.json"))
    metrics = json.load(open(model_dir / "metrics.json")) if (model_dir / "metrics.json").exists() else {}

    # Reconstruct model
    abs_vocab = config.get("abs_vocab", 0)
    abs_vocab = abs_vocab if abs_vocab > 0 else 1

    qwen_config = Qwen3Config(
        hidden_size=config["n_embd"],
        num_hidden_layers=config["n_layer"],
        num_attention_heads=config["n_head"],
        num_key_value_heads=config["n_head"],
        intermediate_size=config["n_embd"] * 4,
        vocab_size=config.get("base_vocab_size", 151936),
        max_position_embeddings=128,
    )

    model = SorlModelWrapper.from_scratch(
        qwen_config,
        full_vocab_size_list=[qwen_config.vocab_size, abs_vocab],
        pad_token_id=config.get("pad_token_id", 151643),
    )
    model.load_state_dict(torch.load(model_dir / "model.safetensors", map_location=device), strict=False)
    model = model.to(device)

    return model, config, metrics


def list_models(repo_id: str = MODEL_REPO):
    """List all experiment subfolders in the model repo."""
    api = HfApi()
    files = api.list_repo_files(repo_id)
    subfolders = sorted(set(f.split("/")[0] for f in files if "/" in f))
    return subfolders


# ── Dataset save/load ──────────────────────────────────────────────

def save_dataset(subfolder: str, n_digits: int = 6, ops: str = "add",
                 train_size: int = 500_000, val_size: int = 10_000,
                 repo_id: str = DATASET_REPO):
    """
    Generate and upload an arithmetic dataset to HF Hub.

    Saves as parquet files with columns: tokens, labels, op.
    """
    import pandas as pd
    from arithmetic.datasets.addition import (
        random_add_example, random_sub_example,
    )

    def generate_examples(n, ops, n_digits):
        rows = []
        for _ in range(n):
            if ops == "add":
                ex = random_add_example(n_digits, use_sum9_aug=True)
            elif ops == "add_sub":
                if torch.rand(1).item() < 0.5:
                    ex = random_add_example(n_digits, use_sum9_aug=True)
                else:
                    ex = random_sub_example(n_digits)
            rows.append({
                "tokens": ex.tokens,
                "labels": ex.labels,
                "op": ex.op,
                "x_digits": ex.x_digits,
                "y_digits": ex.y_digits,
                "z_digits": ex.z_digits,
            })
        return pd.DataFrame(rows)

    api = HfApi()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # Generate
        print(f"Generating train ({train_size})...")
        train_df = generate_examples(train_size, ops, n_digits)
        train_df.to_parquet(tmp / "train.parquet")

        print(f"Generating val ({val_size})...")
        val_df = generate_examples(val_size, ops, n_digits)
        val_df.to_parquet(tmp / "val.parquet")

        # Config
        ds_config = {
            "n_digits": n_digits,
            "ops": ops,
            "train_size": train_size,
            "val_size": val_size,
        }
        with open(tmp / "config.json", "w") as f:
            json.dump(ds_config, f, indent=2)

        # Upload
        api.upload_folder(
            folder_path=str(tmp),
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=subfolder,
            commit_message=f"Upload {subfolder}: {ops} {n_digits}-digit, {train_size} train / {val_size} val",
        )

    print(f"Saved to {repo_id}/{subfolder}")


def load_dataset(subfolder: str, split: str = "train",
                 repo_id: str = DATASET_REPO):
    """Load a dataset split from HF Hub. Returns DataFrame."""
    import pandas as pd

    path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{subfolder}/{split}.parquet",
        repo_type="dataset",
        local_dir=f"/tmp/hf_cache/{repo_id.split('/')[-1]}",
    )
    return pd.read_parquet(path)


def list_datasets(repo_id: str = DATASET_REPO):
    """List all dataset subfolders."""
    api = HfApi()
    files = api.list_repo_files(repo_id, repo_type="dataset")
    subfolders = sorted(set(f.split("/")[0] for f in files if "/" in f))
    return subfolders
