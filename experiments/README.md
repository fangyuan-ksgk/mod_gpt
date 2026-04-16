# Experiments

Reproducible scripts for all analyses and dashboard figures.
Each subfolder is a self-contained experiment that reads models/data from
HuggingFace and writes outputs (figures, JSON, markdown) to its own directory.

## Usage

```bash
# Run everything
bash experiments/regenerate_all.sh

# Run one experiment
python experiments/01_model_comparison/run.py
python experiments/03_token_subtask_heatmap/run.py --model MODEL --device cuda:0
bash experiments/08_mechanistic_verification/run.sh MODEL cuda:1
```

## Structure

Each experiment folder contains:
- `run.py` — self-contained script, no notebook dependency
- `*.png` — generated figures
- `results.json` — structured output data
- `summary.md` — markdown summary for dashboard embedding

## Experiments

| # | Name | Status | Default Model | Key Output |
|---|------|--------|---------------|------------|
| 01 | Model Comparison | Results generated | HF catalog | Comparison tables, data efficiency plot |
| 02 | Vocab Scaling | Results generated | HF models (varying abs_vocab) | Accuracy vs vocab size plot |
| 03 | Token Subtask Heatmap | Results generated | `abs30_K1_10K` | P(subtask\|token) heatmap |
| 04 | Addition Hierarchy | Script ready, not run | `abs30_K1_100K` | Carry computation spectrum (SA→SC→SS→UC→US) |
| 05 | Token Vignettes | Script ready, not run | `abs30_K1_100K` | Deep-dive profiles for specific tokens |
| 06 | Token Swap | Script ready, not run | `abs30_K1_100K` | Surgical token transplant effects |
| 07 | Causal Ablation | Script ready, not run | `abs30_K1_100K` | Knockout/shuffle/random interventions |
| 08 | Mechanistic Verification | Results generated | `abs30_K1_100K_2L1H128d` | Findings (1)-(5) from novel.md |

## Key Models for Analysis

| Model | Architecture | Accuracy | Use |
|-------|-------------|----------|-----|
| `add_sub_sorl_v1_abs30_K1_10K` | 2L/3H/510d (standard) | 97.4% | Token distributions (saturated, clean specialization) |
| `add_sub_sorl_v1_abs30_K1_100K_2L1H128d` | 2L/1H/128d (undersized) | 77.0% | Causal interventions (SoRL matters, not saturated) |

## Data Sources

All experiments read from:
- Models: `thoughtworks/arithmetic-sorl` (HuggingFace)
- Eval sets: `thoughtworks/arithmetic-sorl-data` (HuggingFace, canonical N=100)
- No local data generation — everything comes from HF

## Experiment Notes

- Experiments 04-07 were written for the old (pre-fixed-data) models. They should work on
  clean models but haven't been run yet. Token IDs will differ — use `--tokens` args.
- Experiment 08 tests all 5 novel findings from `arithmetic/novel.md` in a single script.
  Results are architecture-dependent: run on both standard and undersized models.
- Finding (4) cross-operation unification is the only one confirmed causally so far.
