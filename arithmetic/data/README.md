# arithmetic/data/

Data generation, storage, and evaluation sets for the arithmetic interpretability study.

| File | Purpose |
|------|---------|
| `addition.py` | Example generation (add/sub), Quirke subtask labels, enrichment, eval set loading |
| `multiplication.py` | Multiplication example generation (unused in current study) |
| `hub.py` | HuggingFace storage interface — `save_model`, `load_model`, `save_dataset`, `load_dataset` |
| `eval_sets/` | Cached eval set JSONs downloaded from HF (`eval_add_sub_6d_N100_seed42.json`, etc.) |

## Key constants (addition.py)

- `CANONICAL_EVAL_SET` — the one true eval set used by all experiments (N=100/split, seed=42)
- `QWEN3_TOKEN_MAP` / `QWEN3_INV_MAP` — digit/operator ↔ Qwen3 token ID mapping

## Key functions (addition.py)

- `get_eval_set()` — download + cache canonical eval from HF, returns `{split: [ArithmeticExample]}`
- `generate_batch(N, ops, ...)` — online batch generation with Quirke enrichment
- `ArithmeticExample` — dataclass with `.tokens`, `.labels`, `.split_tags`

## Key functions (hub.py)

- `save_model(model, config, metrics, subfolder)` — upload model + config + metrics to HF
- `load_model(subfolder, device)` — download and instantiate model from HF
- `save_dataset` / `load_dataset` — HF dataset repo I/O
