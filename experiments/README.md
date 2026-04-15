# Experiments

Reproducible scripts for all dashboard figures and analyses.
Each subfolder is a self-contained experiment that reads models/data from
HuggingFace and writes outputs (figures, JSON, markdown) to its own directory.

## Usage

```bash
# Run everything
bash experiments/regenerate_all.sh

# Run one experiment
python experiments/01_model_comparison/run.py
python experiments/02_vocab_scaling/run.py --device cuda:0
```

## Structure

Each experiment folder contains:
- `run.py` — self-contained script, no notebook dependency
- `*.png` — generated figures
- `results.json` — structured output data
- `summary.md` — markdown summary for dashboard embedding

## Experiments

| # | Name | Dashboard Section | Inputs |
|---|------|-------------------|--------|
| 01 | Model Comparison | Models tab | HF model catalog |
| 02 | Vocab Scaling | Interpretability: Section 1 | HF models (varying abs_vocab) |
| 03 | Token Subtask Heatmap | Interpretability: Section 2 | K=1 abs30 model + canonical eval |
| 04 | Addition Hierarchy | Interpretability: Section 2 | Same as 03, focused on add splits |
| 05 | Token Vignettes | Interpretability: Section 3 | K=1 abs30 model + canonical eval |
| 06 | Token Swap | Interpretability: Section 4 | K=1 abs30 model + canonical eval |
| 07 | Causal Ablation | Interpretability: Section 5 | K=1 abs30 model + canonical eval |

## Data Sources

All experiments read from:
- Models: `thoughtworks/arithmetic-sorl` (HuggingFace)
- Eval sets: `thoughtworks/arithmetic-sorl-data` (HuggingFace, canonical N=100)
- No local data generation — everything comes from HF
