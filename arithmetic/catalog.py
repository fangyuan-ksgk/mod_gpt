"""
Model catalog — index all trained models on HF with their configs.

Usage:
    from arithmetic.catalog import ModelCatalog
    cat = ModelCatalog()
    cat.fetch()              # pull all train_config.json from HF
    cat.print_table()        # box-drawing summary
    cat.filter(mode="sorl", ops="add_sub")  # filtered list
    cat.save("catalog.json") # cache locally
    cat = ModelCatalog.load("catalog.json")  # reload without HF calls
"""
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from huggingface_hub import HfApi, hf_hub_download

MODEL_REPO = "thoughtworks/arithmetic-sorl"

# Keys to extract from train_config.json for the summary
SUMMARY_KEYS = [
    "mode", "trainer_version", "ops", "dataset_size", "abs_vocab", "K",
    "n_layer", "n_head", "n_embd", "num_epochs", "lr", "batch_size",
    "alpha_info_gain", "alpha_abs", "alpha_soft_zipf",
    "final_accuracy", "n_params", "timestamp",
]


@dataclass
class ModelEntry:
    name: str                          # subfolder path on HF (e.g. "non_enriched/add_baseline_100K")
    config: dict = field(default_factory=dict)  # full train_config.json
    enriched: bool = False             # whether trained with enriched data

    @property
    def mode(self) -> str:
        return self.config.get("mode", "unknown")

    @property
    def trainer(self) -> str:
        return self.config.get("trainer_version", "unknown")

    @property
    def ops(self) -> str:
        return self.config.get("ops", "unknown")

    @property
    def dataset_size(self) -> int:
        return self.config.get("dataset_size", 0)

    @property
    def abs_vocab(self) -> int:
        return self.config.get("abs_vocab", 0)

    @property
    def K(self) -> int:
        return self.config.get("K", 0)

    @property
    def accuracy(self) -> Optional[float]:
        return self.config.get("final_accuracy")

    @property
    def arch(self) -> str:
        L = self.config.get("n_layer", "?")
        H = self.config.get("n_head", "?")
        D = self.config.get("n_embd", "?")
        return f"{L}L/{H}H/{D}d"

    def summary(self) -> dict:
        """Subset of config for quick comparison."""
        out = {"name": self.name, "enriched": self.enriched}
        for k in SUMMARY_KEYS:
            if k in self.config:
                out[k] = self.config[k]
        return out


class ModelCatalog:
    def __init__(self, repo_id: str = MODEL_REPO):
        self.repo_id = repo_id
        self.entries: list[ModelEntry] = []

    def fetch(self, verbose: bool = True):
        """Pull all train_config.json files from HF and build the index."""
        api = HfApi()
        all_files = api.list_repo_files(self.repo_id)

        # Find all train_config.json paths
        config_files = [f for f in all_files if f.endswith("train_config.json")]
        if verbose:
            print(f"Found {len(config_files)} models in {self.repo_id}")

        self.entries = []
        for cf in sorted(config_files):
            subfolder = str(Path(cf).parent)
            enriched = not subfolder.startswith("non_enriched/")
            try:
                local = hf_hub_download(
                    self.repo_id, cf,
                    local_dir="/tmp/hf_cache/arithmetic-sorl",
                )
                config = json.load(open(local))
                self.entries.append(ModelEntry(name=subfolder, config=config, enriched=enriched))
                if verbose:
                    print(f"  {subfolder}")
            except Exception as e:
                if verbose:
                    print(f"  {subfolder} — FAILED: {e}")

        if verbose:
            print(f"Indexed {len(self.entries)} models")
        return self

    def filter(self, **kwargs) -> list[ModelEntry]:
        """
        Filter entries by config values.

        Examples:
            cat.filter(mode="sorl")
            cat.filter(ops="add_sub", enriched=True)
            cat.filter(trainer_version="v1", dataset_size=500000)
        """
        results = []
        for entry in self.entries:
            match = True
            for k, v in kwargs.items():
                if k == "enriched":
                    if entry.enriched != v:
                        match = False
                elif entry.config.get(k) != v:
                    match = False
            if match:
                results.append(entry)
        return results

    def print_table(self, entries: list[ModelEntry] | None = None):
        """Print box-drawing table of all (or filtered) models."""
        entries = entries or self.entries
        if not entries:
            print("No models to display.")
            return

        # Column definitions: (header, width, getter)
        cols = [
            ("Name",     40, lambda e: e.name.removeprefix("non_enriched/")),
            ("Mode",      5, lambda e: e.mode),
            ("Trainer",   7, lambda e: e.trainer),
            ("Ops",       7, lambda e: e.ops),
            ("Data",      6, lambda e: f"{e.dataset_size // 1000}K" if e.dataset_size else "?"),
            ("Vocab",     5, lambda e: str(e.abs_vocab)),
            ("K",         3, lambda e: str(e.K)),
            ("Arch",     12, lambda e: e.arch),
            ("Epochs",    6, lambda e: str(e.config.get("num_epochs", "?"))),
            ("Acc",       5, lambda e: f"{e.accuracy:.0%}" if e.accuracy is not None else "?"),
            ("Enrich",    6, lambda e: "yes" if e.enriched else "no"),
        ]

        # Build table
        header = "│".join(f" {h:<{w}} " for h, w, _ in cols)
        sep = "┼".join("─" * (w + 2) for _, w, _ in cols)

        print(f"┌{'┬'.join('─' * (w + 2) for _, w, _ in cols)}┐")
        print(f"│{header}│")
        print(f"├{sep}┤")

        for entry in sorted(entries, key=lambda e: (e.enriched, e.mode, e.ops, e.dataset_size)):
            row = "│".join(f" {g(entry):<{w}} " for _, w, g in cols)
            print(f"│{row}│")

        print(f"└{'┴'.join('─' * (w + 2) for _, w, _ in cols)}┘")
        print(f"  {len(entries)} models")

    def save(self, path: str):
        """Save catalog to JSON (avoids re-fetching)."""
        data = [{"name": e.name, "config": e.config, "enriched": e.enriched} for e in self.entries]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved catalog ({len(self.entries)} models) to {path}")

    @classmethod
    def load(cls, path: str, repo_id: str = MODEL_REPO) -> "ModelCatalog":
        """Load catalog from JSON."""
        cat = cls(repo_id=repo_id)
        data = json.load(open(path))
        cat.entries = [ModelEntry(name=d["name"], config=d["config"], enriched=d["enriched"]) for d in data]
        print(f"Loaded catalog ({len(cat.entries)} models) from {path}")
        return cat

    def __len__(self):
        return len(self.entries)

    def __repr__(self):
        n_enriched = sum(1 for e in self.entries if e.enriched)
        n_non = len(self.entries) - n_enriched
        return f"ModelCatalog({n_non} non-enriched, {n_enriched} enriched)"
