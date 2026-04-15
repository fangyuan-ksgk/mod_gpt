# Model Comparison

**78 VALID models** from `thoughtworks/arithmetic-sorl`

## SoRL K=1 vs Baseline (Standard Architecture: 2L/3H/510d)

| Model | Mode | Data | V | Overall | C3 | C4 | C5 | C6 |
|-------|------|------|---|---------|----|----|----|----|
| add_sub_baseline_10K | baseline | 10K | 0 | 72% | 78% | 88% | 76% | 92% |
| add_sub_baseline_25K | baseline | 25K | 0 | 92% | 96% | 96% | 94% | 100% |
| add_sub_baseline_50K | baseline | 50K | 0 | 100% | 100% | 100% | 100% | 100% |
| add_sub_baseline_100K | baseline | 100K | 0 | 100% | 100% | 100% | 100% | 100% |
| add_sub_sorl_v1_abs10_10K | sorl | 10K | 10 | 1% | 0% | 0% | 0% | 2% |
| add_sub_sorl_v1_abs10_K1_10K | sorl | 10K | 10 | 95% | 100% | 100% | 94% | 100% |
| add_sub_sorl_v1_abs30_10K | sorl | 10K | 30 | 0% | 0% | 0% | 0% | 0% |
| add_sub_sorl_v1_abs30_K1_10K | sorl | 10K | 30 | 96% | 100% | 100% | 100% | 100% |
| add_sub_sorl_v1_abs10_25K | sorl | 25K | 10 | 97% | 100% | 100% | 96% | 100% |
| add_sub_sorl_v1_abs10_K1_25K | sorl | 25K | 10 | 98% | 100% | 100% | 100% | 100% |
| add_sub_sorl_v1_abs30_K1_25K | sorl | 25K | 30 | 100% | 100% | 100% | 100% | 100% |
| add_sub_sorl_v1_abs10_50K | sorl | 50K | 10 | 100% | 100% | 100% | 100% | 100% |
| add_sub_sorl_v1_abs10_K1_50K | sorl | 50K | 10 | 100% | 100% | 100% | 100% | 100% |
| add_sub_sorl_v1_abs30_50K | sorl | 50K | 30 | 99% | 100% | 100% | 100% | 100% |
| add_sub_sorl_v1_abs30_K1_50K | sorl | 50K | 30 | 100% | 100% | 100% | 100% | 100% |
| add_sub_sorl_v1_abs30_K3_50K | sorl | 50K | 30 | 100% | 100% | 100% | 100% | 100% |
| add_sub_sorl_v1_abs50_50K | sorl | 50K | 50 | 100% | 100% | 100% | 100% | 100% |
| add_sub_sorl_v1_abs50_K1_50K | sorl | 50K | 50 | 100% | 100% | 100% | 100% | 100% |
| add_sub_sorl_v1_abs5_50K | sorl | 50K | 5 | 100% | 100% | 100% | 100% | 100% |
| add_sub_sorl_v1_abs5_K1_50K | sorl | 50K | 5 | 100% | 100% | 100% | 100% | 100% |
| add_sub_sorl_v1_abs100_100K | sorl | 100K | 100 | 100% | 100% | 100% | 100% | 100% |
| add_sub_sorl_v1_abs100_K1_100K | sorl | 100K | 100 | 100% | 100% | 100% | 100% | 100% |
| add_sub_sorl_v1_abs10_100K | sorl | 100K | 10 | 100% | 100% | 100% | 100% | 100% |
| add_sub_sorl_v1_abs10_K1_100K | sorl | 100K | 10 | 100% | 100% | 100% | 100% | 100% |
| add_sub_sorl_v1_abs16_100K | sorl | 100K | 16 | 100% | 100% | 100% | 100% | 100% |
| add_sub_sorl_v1_abs16_K1_100K | sorl | 100K | 16 | 99% | 100% | 100% | 100% | 100% |
| add_sub_sorl_v1_abs20_100K | sorl | 100K | 20 | 100% | 100% | 100% | 100% | 100% |
| add_sub_sorl_v1_abs20_K1_100K | sorl | 100K | 20 | 99% | 100% | 100% | 100% | 100% |
| add_sub_sorl_v1_abs2_100K | sorl | 100K | 2 | 100% | 100% | 100% | 100% | 100% |
| add_sub_sorl_v1_abs2_K1_100K | sorl | 100K | 2 | 100% | 100% | 100% | 100% | 100% |
| add_sub_sorl_v1_abs30_100K | sorl | 100K | 30 | 100% | 100% | 100% | 100% | 100% |
| add_sub_sorl_v1_abs30_K1_100K | sorl | 100K | 30 | 100% | 100% | 100% | 100% | 100% |
| add_sub_sorl_v1_abs30_K3_100K | sorl | 100K | 30 | 100% | 100% | 100% | 100% | 100% |
| add_sub_sorl_v1_abs50_100K | sorl | 100K | 50 | 100% | 100% | 100% | 100% | 100% |
| add_sub_sorl_v1_abs50_K1_100K | sorl | 100K | 50 | 100% | 100% | 100% | 100% | 100% |
| add_sub_sorl_v1_abs5_100K | sorl | 100K | 5 | 100% | 100% | 100% | 100% | 100% |
| add_sub_sorl_v1_abs5_K1_100K | sorl | 100K | 5 | 100% | 100% | 100% | 100% | 100% |
| add_sub_sorl_v1_abs70_100K | sorl | 100K | 70 | 100% | 100% | 100% | 100% | 100% |
| add_sub_sorl_v1_abs70_K1_100K | sorl | 100K | 70 | 100% | 100% | 100% | 100% | 100% |
| add_sub_sorl_v6_abs30_10K | sorl_v6 | 10K | 30 | 85% | 0% | 0% | 0% | 0% |
| add_sub_sorl_v6_abs30_K1_10K | sorl_v6 | 10K | 30 | 71% | 0% | 0% | 0% | 0% |
| add_sub_sorl_v6_abs30_25K | sorl_v6 | 25K | 30 | 75% | 0% | 0% | 0% | 0% |
| add_sub_sorl_v6_abs30_K1_25K | sorl_v6 | 25K | 30 | 82% | 0% | 0% | 0% | 0% |
| add_sub_sorl_v6_abs30_K1_50K | sorl_v6 | 50K | 30 | 98% | 0% | 0% | 0% | 0% |

## Cross-Architecture Comparison (100K data, K=1 abs30)

| Arch | Baseline | SoRL | Delta | C4 Base | C4 SoRL | C6 Base | C6 SoRL |
|------|----------|------|-------|---------|---------|---------|---------|
| 1L/2H/256d | 33% | 44% | +11% | 30% | 54% | 30% | 48% |
| 1L/3H/510d | 61% | 72% | +12% | 76% | 70% | 68% | 72% |
| 2L/1H/128d | 50% | 85% | +35% | 38% | 94% | 32% | 94% |
| 2L/3H/510d | 100% | 100% | +0% | 100% | 100% | 100% | 100% |
