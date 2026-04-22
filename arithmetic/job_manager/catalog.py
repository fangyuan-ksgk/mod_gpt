from __future__ import annotations

"""
Model and data catalogs — persistent indexes on HuggingFace.

Model catalog: tracks all trained models with status (VALID/SUPERSEDED/DELETED).
Data catalog: tracks all eval sets and datasets with versioning.

Both catalogs live as JSON files at the root of their respective HF repos.
The catalog is the source of truth — new models/datasets discovered on HF
are added with status=VALID, but status changes are persisted back to HF.

Usage:
    # Models
    from arithmetic.catalog import ModelCatalog
    cat = ModelCatalog()
    cat.fetch()                          # pull catalog + configs from HF
    cat.valid()                          # only VALID models
    cat.set_status("old_model", "SUPERSEDED", "replaced by new_model")
    cat.push()                           # upload catalog back to HF

    # Data
    from arithmetic.catalog import DataCatalog
    dcat = DataCatalog()
    dcat.fetch()                         # pull catalog from HF
    dcat.register("eval_sets/my_new_eval.json", description="...", version="v2")
    dcat.push()                          # upload catalog back to HF
"""
import json
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from huggingface_hub import HfApi, hf_hub_download

MODEL_REPO = "thoughtworks/arithmetic-sorl"
DATASET_REPO = "thoughtworks/arithmetic-sorl-data"

SUMMARY_KEYS = [
    "mode", "trainer_version", "ops", "dataset_size", "abs_vocab", "K",
    "n_layer", "n_head", "n_embd", "num_epochs", "lr", "batch_size",
    "alpha_info_gain", "alpha_abs", "alpha_soft_zipf",
    "final_accuracy", "n_params", "timestamp",
]

MODEL_CATALOG_FILE = "model_catalog.json"
DATA_CATALOG_FILE = "data_catalog.json"


# ── Model Catalog ─────────────────────────────────────────────────────

@dataclass
class ModelEntry:
    name: str
    config: dict = field(default_factory=dict)
    enriched: bool = False
    metrics: dict = field(default_factory=dict)
    status: str = "VALID"              # VALID, SUPERSEDED, DELETED
    status_note: str = ""              # why superseded/deleted

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
    def sft_accuracy(self) -> Optional[float]:
        return self.config.get("sft_accuracy")

    def split_accuracy(self, eval_key: str = "sft_eval", split: str = None) -> Optional[float]:
        ev = self.metrics.get(eval_key, {})
        if not ev:
            return None
        if split:
            s = ev.get("splits", {}).get(split, {})
            return s.get("full_accuracy") if s else None
        return ev.get("summary", {}).get("overall_accuracy")

    @property
    def arch(self) -> str:
        L = self.config.get("n_layer", "?")
        H = self.config.get("n_head", "?")
        D = self.config.get("n_embd", "?")
        return f"{L}L/{H}H/{D}d"

    def summary(self) -> dict:
        out = {"name": self.name, "enriched": self.enriched, "status": self.status}
        for k in SUMMARY_KEYS:
            if k in self.config:
                out[k] = self.config[k]
        return out


class ModelCatalog:
    def __init__(self, repo_id: str = MODEL_REPO):
        self.repo_id = repo_id
        self.entries: list[ModelEntry] = []
        self._catalog_data: dict = {}  # name -> {status, status_note}

    def fetch(self, verbose: bool = True):
        """
        Pull catalog from HF, then scan for all models.
        New models (not in catalog) get status=VALID.
        Existing models keep their persisted status.
        """
        api = HfApi()
        all_files = api.list_repo_files(self.repo_id)

        # 1. Load persisted catalog if it exists
        self._catalog_data = {}
        if MODEL_CATALOG_FILE in all_files:
            try:
                local = hf_hub_download(
                    self.repo_id, MODEL_CATALOG_FILE,
                    local_dir="/tmp/hf_cache/arithmetic-sorl",
                    force_download=True,
                )
                for entry in json.load(open(local)):
                    self._catalog_data[entry["name"]] = {
                        "status": entry.get("status", "VALID"),
                        "status_note": entry.get("status_note", ""),
                    }
                if verbose:
                    print(f"Loaded catalog: {len(self._catalog_data)} entries")
            except Exception as e:
                if verbose:
                    print(f"Warning: could not load catalog: {e}")

        # 2. Scan all train_config.json
        config_files = [f for f in all_files if f.endswith("train_config.json")]
        metrics_files = set(f for f in all_files if f.endswith("metrics.json"))
        if verbose:
            print(f"Found {len(config_files)} models in {self.repo_id}")

        self.entries = []
        for cf in sorted(config_files):
            subfolder = str(Path(cf).parent)
            enriched = not subfolder.startswith("non_enriched/")

            # Get status from persisted catalog (default VALID for new models)
            cat_info = self._catalog_data.get(subfolder, {})
            status = cat_info.get("status", "VALID")
            status_note = cat_info.get("status_note", "")

            try:
                local = hf_hub_download(
                    self.repo_id, cf,
                    local_dir="/tmp/hf_cache/arithmetic-sorl",
                )
                config = json.load(open(local))

                metrics = {}
                mf = f"{subfolder}/metrics.json"
                if mf in metrics_files:
                    try:
                        ml = hf_hub_download(
                            self.repo_id, mf,
                            local_dir="/tmp/hf_cache/arithmetic-sorl",
                        )
                        metrics = json.load(open(ml))
                    except Exception:
                        pass

                self.entries.append(ModelEntry(
                    name=subfolder, config=config, enriched=enriched,
                    metrics=metrics, status=status, status_note=status_note,
                ))
                if verbose:
                    flag = f" [{status}]" if status != "VALID" else ""
                    print(f"  {subfolder}{flag}")
            except Exception as e:
                if verbose:
                    print(f"  {subfolder} — FAILED: {e}")

        if verbose:
            n_valid = sum(1 for e in self.entries if e.status == "VALID")
            print(f"Indexed {len(self.entries)} models ({n_valid} VALID)")
        return self

    def valid(self) -> list[ModelEntry]:
        """Return only VALID models."""
        return [e for e in self.entries if e.status == "VALID"]

    def set_status(self, name: str, status: str, note: str = ""):
        """Set status for a model. status: VALID, SUPERSEDED, DELETED."""
        assert status in ("VALID", "SUPERSEDED", "DELETED"), f"Invalid status: {status}"
        for e in self.entries:
            if e.name == name:
                e.status = status
                e.status_note = note
                return
        raise ValueError(f"Model not found: {name}")

    def push(self, commit_message: str = "Update model catalog"):
        """Upload model_catalog.json to HF."""
        catalog = []
        for e in sorted(self.entries, key=lambda x: x.name):
            catalog.append({
                "name": e.name,
                "status": e.status,
                "status_note": e.status_note,
                "mode": e.mode,
                "trainer_version": e.trainer,
                "ops": e.ops,
                "dataset_size": e.dataset_size,
                "abs_vocab": e.abs_vocab,
                "K": e.K,
                "arch": e.arch,
                "lr": e.config.get("lr"),
                "eval_method": e.config.get("eval_method", "unknown"),
                "final_accuracy": e.accuracy,
                "sft_accuracy": e.sft_accuracy,
            })

        api = HfApi()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / MODEL_CATALOG_FILE
            with open(path, "w") as f:
                json.dump(catalog, f, indent=2)
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=MODEL_CATALOG_FILE,
                repo_id=self.repo_id,
                commit_message=commit_message,
            )
        n_valid = sum(1 for c in catalog if c["status"] == "VALID")
        print(f"Pushed model catalog: {len(catalog)} models ({n_valid} VALID)")

    def filter(self, **kwargs) -> list[ModelEntry]:
        results = []
        for entry in self.entries:
            match = True
            for k, v in kwargs.items():
                if k == "enriched":
                    if entry.enriched != v:
                        match = False
                elif k == "status":
                    if entry.status != v:
                        match = False
                elif entry.config.get(k) != v:
                    match = False
            if match:
                results.append(entry)
        return results

    def print_table(self, entries: list[ModelEntry] | None = None, hard_splits: bool = True):
        """Print box-drawing table of all (or filtered) models."""
        entries = entries or self.entries
        if not entries:
            print("No models to display.")
            return

        def _fmt_acc(v):
            return f"{v:.0%}" if v is not None else "?"

        def _split_acc(e, eval_key, split):
            return _fmt_acc(e.split_accuracy(eval_key, split))

        cols = [
            ("Name",     36, lambda e: e.name.removeprefix("non_enriched/")),
            ("St",        3, lambda e: e.status[:3]),
            ("Mode",      4, lambda e: e.mode[:4]),
            ("Data",      5, lambda e: f"{e.dataset_size // 1000}K" if e.dataset_size else "?"),
            ("V",         3, lambda e: str(e.abs_vocab)),
            ("K",         2, lambda e: str(e.K)),
            ("Arch",      12, lambda e: e.arch),
            ("Acc",       5, lambda e: _fmt_acc(e.accuracy)),
            ("SFT",       5, lambda e: _fmt_acc(e.sft_accuracy)),
        ]

        if hard_splits:
            def _hard(split):
                def getter(e):
                    ek = "sorl_eval" if e.mode == "sorl" else "sft_eval"
                    return _split_acc(e, ek, split)
                return getter
            cols += [
                ("C4",    4, _hard("add_C4")),
                ("C5",    4, _hard("add_C5")),
                ("C6",    4, _hard("add_C6")),
            ]

        # Build table
        header = "│".join(f" {h:<{w}} " for h, w, _ in cols)
        sep = "┼".join("─" * (w + 2) for _, w, _ in cols)

        print(f"┌{'┬'.join('─' * (w + 2) for _, w, _ in cols)}┐")
        print(f"│{header}│")
        print(f"├{sep}┤")

        for entry in sorted(entries, key=lambda e: (e.status, e.enriched, e.mode, e.ops, e.dataset_size)):
            row = "│".join(f" {g(entry):<{w}} " for _, w, g in cols)
            print(f"│{row}│")

        print(f"└{'┴'.join('─' * (w + 2) for _, w, _ in cols)}┘")
        n_valid = sum(1 for e in entries if e.status == "VALID")
        print(f"  {len(entries)} models ({n_valid} VALID)")

    def save(self, path: str):
        """Save catalog to local JSON (avoids re-fetching)."""
        data = [{"name": e.name, "config": e.config, "enriched": e.enriched,
                 "metrics": e.metrics, "status": e.status,
                 "status_note": e.status_note} for e in self.entries]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved catalog ({len(self.entries)} models) to {path}")

    @classmethod
    def load(cls, path: str, repo_id: str = MODEL_REPO) -> "ModelCatalog":
        """Load catalog from local JSON."""
        cat = cls(repo_id=repo_id)
        data = json.load(open(path))
        cat.entries = [ModelEntry(
            name=d["name"], config=d["config"],
            enriched=d.get("enriched", False),
            metrics=d.get("metrics", {}),
            status=d.get("status", "VALID"),
            status_note=d.get("status_note", ""),
        ) for d in data]
        print(f"Loaded catalog ({len(cat.entries)} models) from {path}")
        return cat

    def __len__(self):
        return len(self.entries)

    def __repr__(self):
        n_valid = sum(1 for e in self.entries if e.status == "VALID")
        return f"ModelCatalog({len(self.entries)} total, {n_valid} VALID)"


# ── Data Catalog ──────────────────────────────────────────────────────

@dataclass
class DataEntry:
    path: str                          # path within HF repo (e.g. "eval_sets/eval_add_sub_6d_N100_seed42.json")
    description: str = ""
    version: str = "v1"
    n_examples: Optional[int] = None   # total examples across all splits
    n_splits: Optional[int] = None
    seed: Optional[int] = None
    status: str = "ACTIVE"             # ACTIVE, DEPRECATED
    notes: str = ""


class DataCatalog:
    def __init__(self, repo_id: str = DATASET_REPO):
        self.repo_id = repo_id
        self.entries: list[DataEntry] = []

    def fetch(self, verbose: bool = True):
        """Pull data_catalog.json from HF dataset repo."""
        api = HfApi()
        all_files = api.list_repo_files(self.repo_id, repo_type="dataset")

        self.entries = []
        if DATA_CATALOG_FILE in all_files:
            try:
                local = hf_hub_download(
                    self.repo_id, DATA_CATALOG_FILE, repo_type="dataset",
                    local_dir="/tmp/hf_cache/arithmetic-sorl-data",
                    force_download=True,
                )
                for entry in json.load(open(local)):
                    self.entries.append(DataEntry(**entry))
                if verbose:
                    print(f"Loaded data catalog: {len(self.entries)} entries")
            except Exception as e:
                if verbose:
                    print(f"Warning: could not load data catalog: {e}")

        # Report files on HF not in catalog
        if verbose:
            cataloged = {e.path for e in self.entries}
            for f in sorted(all_files):
                if f in (".gitattributes", "README.md", DATA_CATALOG_FILE):
                    continue
                tag = "" if f in cataloged else " [NOT IN CATALOG]"
                print(f"  {f}{tag}")

        return self

    def register(self, path: str, description: str, version: str = "v1",
                 n_examples: int = None, n_splits: int = None,
                 seed: int = None, status: str = "ACTIVE", notes: str = ""):
        """
        Register a new dataset entry. Never overwrites existing entries.
        Raises ValueError if path already exists in catalog.
        To update an existing entry, use update() instead.
        """
        for e in self.entries:
            if e.path == path:
                raise ValueError(
                    f"Dataset already in catalog: {path} (version={e.version}). "
                    f"Use update() to modify, or register with a different path/version."
                )
        entry = DataEntry(
            path=path, description=description, version=version,
            n_examples=n_examples, n_splits=n_splits, seed=seed,
            status=status, notes=notes,
        )
        self.entries.append(entry)
        return entry

    def update(self, path: str, **kwargs):
        """Update fields of an existing catalog entry."""
        for e in self.entries:
            if e.path == path:
                for k, v in kwargs.items():
                    if hasattr(e, k):
                        setattr(e, k, v)
                    else:
                        raise ValueError(f"DataEntry has no field: {k}")
                return e
        raise ValueError(f"Dataset not found in catalog: {path}")

    def get(self, path: str) -> Optional[DataEntry]:
        """Look up a dataset by path. Returns None if not found."""
        for e in self.entries:
            if e.path == path:
                return e
        return None

    def list(self, status: str = None) -> list[DataEntry]:
        """List datasets, optionally filtered by status."""
        if status:
            return [e for e in self.entries if e.status == status]
        return list(self.entries)

    def push(self, commit_message: str = "Update data catalog"):
        """Upload data_catalog.json to HF dataset repo."""
        catalog = []
        for e in sorted(self.entries, key=lambda x: x.path):
            catalog.append({
                "path": e.path,
                "description": e.description,
                "version": e.version,
                "n_examples": e.n_examples,
                "n_splits": e.n_splits,
                "seed": e.seed,
                "status": e.status,
                "notes": e.notes,
            })

        api = HfApi()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / DATA_CATALOG_FILE
            with open(path, "w") as f:
                json.dump(catalog, f, indent=2)
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=DATA_CATALOG_FILE,
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message=commit_message,
            )
        print(f"Pushed data catalog: {len(catalog)} entries")

    def active(self) -> list[DataEntry]:
        return [e for e in self.entries if e.status == "ACTIVE"]

    def print_table(self):
        entries = sorted(self.entries, key=lambda e: e.path)
        if not entries:
            print("No data entries.")
            return
        print(f"{'Path':<50s} {'Ver':>4s} {'N':>6s} {'Splits':>6s} {'Status':>10s}")
        print("─" * 80)
        for e in entries:
            n = str(e.n_examples) if e.n_examples else "?"
            s = str(e.n_splits) if e.n_splits else "?"
            print(f"  {e.path:<48s} {e.version:>4s} {n:>6s} {s:>6s} {e.status:>10s}")

    def __len__(self):
        return len(self.entries)

    def __repr__(self):
        n_active = sum(1 for e in self.entries if e.status == "ACTIVE")
        return f"DataCatalog({len(self.entries)} total, {n_active} ACTIVE)"
