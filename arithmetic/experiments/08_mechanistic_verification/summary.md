# Mechanistic Verification of Novel Findings

**Model:** `add_sub_sorl_v1_abs30_K1_100K_2L1H128d` (K=1)
**Eval set:** canonical N=100 from HuggingFace
**Token records:** 13000

## (1) Sum-9 Detector

Background rate of sum=9 in eval data: **23.0%**

Tokens with elevated sum-9 purity:

| Token | N | Sum-9 Count | Sum-9 Purity |
|-------|---|-------------|-------------|
| t4 | 1275 | 555 | 43.5% |
| t7 | 926 | 403 | 43.5% |
| t2 | 1604 | 690 | 43.0% |
| t9 | 468 | 132 | 28.2% |
| t1 | 3047 | 835 | 27.4% |
| t22 | 30 | 8 | 26.7% |
| t18 | 114 | 23 | 20.2% |
| t5 | 415 | 77 | 18.6% |
| t21 | 122 | 17 | 13.9% |
| t23 | 48 | 4 | 8.3% |

**Verdict:** WEAK — top token has 44% sum-9 purity vs 23% background

## (2) MSB Digit-Sum Encoding

Mean token-to-digit-sum purity by answer position:

| Position | Tokens | Mean Purity | Max Purity |
|----------|--------|-------------|------------|
| d0 | 10 | 45.4% | 100.0% |
| d1 | 20 | 31.0% | 63.6% |
| d2 | 21 | 28.9% | 51.0% |
| d3 | 21 | 33.3% | 60.0% |
| d4 | 19 | 35.2% | 57.1% |
| d5 | 0 | 0.0% | 0.0% |
| d6 | 0 | 0.0% | 0.0% |

**Verdict:** WEAK — MSB mean purity 35.2% vs other positions 34.6%

## (3) Position-Locked Specialization

- Same-subtask cross-position swaps: **98.5%** survived (n=6750)
- Diff-subtask cross-position swaps: **97.2%** survived (n=12067)

**Verdict:** WEAK — same-subtask swaps break accuracy (98%), confirming position matters beyond subtask identity. Gap vs diff-subtask: 1%

## (4) Cross-Operation Unification

- Add→Sub transplant: **93.5%** full accuracy, **99.1%** digit accuracy
- Sub→Add transplant: **98.0%** full accuracy, **99.7%** digit accuracy
- Random baseline: **75.5%** full accuracy, **96.1%** digit accuracy

**Verdict:** CONFIRMED — cross-operation transplant (98%) beats random (76%)

## (5) LSB Shortcut — Per-Position Ablation

| Position | N | Survived | Survive Rate |
|----------|---|----------|-------------|
| d0 | 2002 | 1870 | 93.4% |
| d1 | 2002 | 1875 | 93.7% |
| d2 | 2002 | 1887 | 94.3% |
| d3 | 2002 | 1881 | 94.0% |
| d4 | 2002 | 1899 | 94.9% |
| d5 | 0 | 0 | 0.0% |
| d6 | 0 | 0 | 0.0% |

**Verdict:** WEAK — d0 survive rate (93%) <= inner positions (94%)
