# Sweep Audit Checklist

Run these checks BEFORE launching any sweep. Each rule is mechanically verifiable.

## 1. Hyperparameter Consistency

**Rule: Baseline and SoRL at the same (arch, dataset_size) MUST use identical optimizer settings.**

This means same LR, weight_decay, beta2, warmup_ratio. The ONLY differences should be SoRL-specific (K, abs_vocab, alpha_*).

**LR policy** (set by `ArithmeticConfig.auto_scale_lr()`, override with `--lr`):

| Architecture | n_embd | LR | Source |
|---|---|---|---|
| 2L/3H/510d (standard) | 510 | 8e-5 | Fangyuan's default |
| 1L/3H/510d | 510 | 8e-5 | Same hidden dim |
| 1L/2H/256d | 256 | 2e-5 | Scaled down |
| 2L/1H/128d | 128 | 2e-5 | Scaled down |

**Verification script:**
```python
# Run this on a sweep file before launching:
python -c "
import re
from collections import defaultdict
groups = defaultdict(set)
with open('sweep_file.txt') as f:
    for line in f:
        if not line.strip() or line.startswith('#'): continue
        ds = re.search(r'--dataset_size (\d+)', line)
        embd = re.search(r'--n_embd (\d+)', line)
        lr = re.search(r'--lr (\S+)', line)
        mode = 'sorl' if '--mode sorl' in line else 'baseline'
        arch = embd.group(1) if embd else '510'
        size = ds.group(1) if ds else '?'
        lr_val = lr.group(1) if lr else 'auto'
        groups[f'{arch}d_{size}'].add((mode, lr_val))
for k, v in sorted(groups.items()):
    modes = {m for m, _ in v}
    lrs = {l for _, l in v}
    if len(lrs) > 1 and len(modes) > 1:
        print(f'MISMATCH: {k} has LRs {lrs} across {modes}')
    else:
        print(f'OK: {k} LR={lrs}')
"
```

## 2. Convergence Checks

**Rule: Before launching a full sweep at N epochs, verify convergence on a pilot run.**

- Check wandb eval/accuracy curve: is it plateaued or still climbing at the final epoch?
- If still climbing: increase epochs or LR
- Minimum epochs by data size:
  - 10K-25K: 10 epochs for SoRL, 20 for baseline
  - 50K+: 10 epochs for SoRL, 20 for baseline
  - Undersized models: always 20 epochs (slower convergence)

**Verification:** After the first few jobs complete, check:
```python
# Pull last 3 eval points from wandb, check if still increasing
last3 = [acc for _, acc in eval_history[-3:]]
if last3[-1] > last3[0] + 0.01:
    print("STILL INCREASING — consider more epochs")
```

## 3. Sweep Design Validation

**Rule: No redundant jobs. Every job must differ in at least one meaningful config axis.**

Parse the sweep file and check:
- No duplicate (mode, dataset_size, abs_vocab, K, arch) tuples
- Priority tags are correctly sticky (parse and print priority per job)
- No dataset_size > 100K for standard model unless explicitly justified (saturates by 50K)
- No dataset_size > 250K for undersized models (they plateau earlier)

**Rule: Priority tags must be verified.**
```bash
# Print priority assignments before launching:
python -c "
PMAP = {'HIGH': 0, 'NORMAL': 1, 'LOW': 2}
cur = 1
with open('sweep_file.txt') as f:
    for line in f:
        line = line.strip()
        if line.startswith('#PRIORITY:'):
            cur = PMAP.get(line.split(':')[1].strip().upper(), 1)
        elif line and not line.startswith('#'):
            name = line.split('ckpt/sweep/')[-1] if 'ckpt/sweep/' in line else '?'
            print(f'  {[\"HIGH\",\"NORMAL\",\"LOW\"][cur]:6s} {name}')
"
```

## 4. Model Catalog Hygiene

**Rule: Every model on HF must have a known status.**

Statuses:
- **VALID**: correct config, converged, results trustworthy
- **SUPERSEDED**: replaced by a rerun with better config (keep for reference, don't use in paper)
- **DELETED**: removed from HF

**Rule: Before launching reruns, verify which existing models will be overwritten.**

```bash
# Cross-reference sweep output_dirs against HF models
```

**Rule: After a sweep completes, update the catalog with actual LR, epochs, final accuracy, and status.**

## 5. Pre-launch Sanity Checks

**Rule: Smoke test MUST pass before launching.**
- Run 1 baseline + 1 SoRL job at 500 samples / 1 epoch
- Verify: training completes, eval runs, wandb config logged (check lr, seed fields), HF upload succeeds

**Rule: Print resolved configs before launching.**
```python
# For each unique (arch, mode) in the sweep:
from arithmetic.train import ArithmeticConfig
cfg = ArithmeticConfig(n_embd=510, mode="sorl")
cfg.auto_scale_lr()
print(f"510d sorl: lr={cfg.lr}, warmup={cfg.warmup_ratio}, beta2={cfg.beta2}")
```

**Rule: Git commit hash must be logged in every run's manifest.**
- Already done via `git_commit` field in train_config.json
- Verify with: `grep git_commit ckpt/sweep/*/config.json`

## 6. Post-Sweep Validation

**Rule: After sweep completes, run the LR consistency check (Section 1) against actual wandb runs, not just the sweep file.**

**Rule: Flag any model where final eval accuracy < epoch eval accuracy (possible eval bug or overfitting).**

**Rule: All models used in paper figures must be VALID status in the catalog.**
