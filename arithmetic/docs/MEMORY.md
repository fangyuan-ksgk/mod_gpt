# Session Memory — Arithmetic Interpretability Study

Claude: read this file at the start of every session to pick up where we left off.
Also read `CLAUDE.md` (technical reference) and `TODO.md` (task list) in this directory.

## Current State (2026-04-09)

- **Branch:** `amir/arithmetic`
- **GPUs:** 3× RTX PRO 6000 Blackwell (96GB each)
- **TRAINING IN PROGRESS:** 30-job enriched sweep via `gpu_queue.py`
  - Log: `/workspace/sorl_logs/sweep_enriched.log`
  - Jobs file: `arithmetic/scripts/sweep_enriched.txt`
  - 12 baselines (add + add_sub × 6 data sizes) + 18 SoRL (K={1,4} × 9 vocab sizes)
  - Each job uploads to HF and deletes local checkpoint automatically

## What Just Happened

1. **HF repo cleaned** — deleted all v6 models and 3L/4H baselines. 12 v1 SoRL models remain under `non_enriched/`.

2. **Re-evaluated all 12 models** with `ArithmeticEvaluator` (per-split hard cases). Updated metrics on HF.

3. **Model catalog** (`arithmetic/catalog.py`) — scans HF repo, indexes train_config.json, filter/table/save/load.

4. **Hard eval splits added** to `arithmetic/datasets/addition.py`:
   - Forced deep cascades: S5, S6 (addition), M4, M5 (subtraction, M6 impossible for 6-digit)
   - Hot carry chains: C3-C6 (varied answer digits, not just 0's)
   - Hot borrow chains: B3-B5 (varied answer digits, not just 9's)
   - New metrics: `carry_chain_depth()`, `borrow_chain_depth()` beyond Quirke's U-chains

5. **hub.py fixed** — safetensors loading, vocab size mismatch handling.

6. **GPU idle monitor** — cron job at `/workspace/scripts/gpu_idle_shutdown.sh`, auto-commits+pushes+stops pod after 3h idle.

## What Needs to Happen Next

**In progress:**
- Enriched sweep running (30 jobs). Check log: `tail -f /workspace/sorl_logs/sweep_enriched.log`
- Models auto-upload to HF under `enriched/` (baselines) or root (SoRL)

**When sweep finishes:**
- Run `ArithmeticEvaluator` on all new models (reeval_hf_models.py)
- Update catalog: `ModelCatalog().fetch().print_table()`
- Update `log/arithmetic.md` with enriched results
- Data efficiency sweep: best vocab at {10K, 25K, 50K, 100K, 250K}

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
