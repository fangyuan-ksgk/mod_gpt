# Session Memory — Arithmetic Interpretability Study

Claude: read this file at the start of every session to pick up where we left off.
Also read `CLAUDE.md` (technical reference) and `TODO.md` (task list) in this directory.

## Current State (2026-04-09)

- **Branch:** `amir/arithmetic`
- **GPUs:** 3× RTX PRO 6000 Blackwell (96GB each) — all idle, no jobs running
- **No models are currently training.** Last training was pre-enrichment.

## What Just Happened

1. **Evaluator built** (`arithmetic/evaluate.py`) — `ArithmeticEvaluator` class with:
   - SFT and SoRL (recursion) eval modes
   - Per-complexity (S0-S6, M0-M6) and per-subtask (SA/SC/SS/UC/US, MD/MB/ME/UB/UD) breakdowns
   - Box-drawing tables, bar charts, heatmaps, model comparison
   - JSON save/load for result persistence

2. **Data enrichment done** — subtraction borrow cascades (40% equal digits) and addition carry cascades (40% sum-to-9). Dataset uploaded to `thoughtworks/arithmetic-sorl-data`.

3. **Old GAT references removed** from arithmetic/ — we use Qwen3 exclusively now.

4. **Hub docstrings fixed** — `thoughtworks/arithmetic-sorl` (not amirali1985).

5. **Pod persistence fixed** — repo lives at `/workspace/codes/mod_gpt`, symlinked from `~/codes/mod_gpt`.

## What Needs to Happen Next

**Immediate (start training):**
- All existing models were trained WITHOUT enriched subtraction data
- Need to retrain baseline + SoRL across the sweep grid (see TODO.md)
- Use `arithmetic/scripts/gpu_queue.py` for scheduling across 3 GPUs

**Then:**
- Run evaluator on all new models
- Update `log/arithmetic.md` with results
- 25K vs 50K data efficiency on hard cases (S3-S6)

## Key Decisions Made

- **v1 trainer only** for from-scratch arithmetic (v6 doesn't work from scratch)
- **Recursion eval only** (model.generate() gives 0% — doesn't match training)
- **Pure SFT baseline** (not SoRLTrainer with alpha=0)
- **Qwen3 tokenizer** with `QWEN3_TOKEN_MAP` for digit/operator mapping
- **Architecture:** 2L/3H/510d (Quirke's addition model size)

## Session Startup Checklist

1. Check GPU status: `nvidia-smi`
2. Check for running jobs: `ps aux | grep python`
3. Verify symlinks: `ls -la ~/codes/mod_gpt` should point to `/workspace/codes/mod_gpt`
4. **Start GPU idle monitor cron:**
   ```bash
   sudo service cron start
   echo '*/15 * * * * /workspace/scripts/gpu_idle_shutdown.sh' | crontab -
   ```
   This auto-commits, pushes, and stops the pod if all GPUs idle >3h.
   Script reads `$RUNPOD_POD_ID` from env — works across pod recreates.
5. Read this file, `CLAUDE.md`, and `TODO.md`
6. **Check for auto-shutdown:** Search this file for `## Auto-Shutdown Log`. If present, the pod was previously stopped automatically due to GPU idle timeout. **Tell the user immediately** — report the timestamp and idle duration.
7. Check `TODO.md` for what to work on
