# Analysis Scripts

Post-training analysis of SoRL abstraction tokens. Each script loads models from HF,
runs analysis, saves figures/results to `arithmetic/analysis/results/`.

## Scripts

### Representation Structure
- `representation_structure.py` — embedding geometry, MI co-occurrence, substitution matrix, CKA

### Token-Subtask Mapping
- `token_subtask_correlation.py` — P(token | subtask), P(subtask | token), precision/recall

### Polysemanticity
- `polysemanticity.py` — 1-to-1 vs many-to-many token↔subtask mapping, per-complexity analysis

### Probing
- `probing.py` — linear probes for resolved cascade state (SV/MV), future digits, cascade length

### Representation Analysis
- `logit_lens.py` — logit lens + future lens at abstraction positions vs baseline
- `circuit_discovery.py` — EAP with SoRL tokens as circuit anchors vs baseline

### Auto-Interpretability
- `autointerp.py` — Juang et al. pipeline adapted for SoRL tokens, 5 scoring metrics

### Token Interventions
- `token_interventions.py` — systematic knockout/swap/shuffle/replace across splits and positions, cascade-targeted interventions, cross-complexity transfer tests. Builds on primitives in `interp_utils/interventions.py`.

### SAE ↔ SoRL Token Matching
- `sae_token_matching.py` — Hungarian matching between SAE features and SoRL tokens, causal validation via matched ablations, cross-model patching

### Vocab Utilization
- `vocab_utilization.py` — per-model vocab usage stats, Zipf fit, frequency hierarchy
