# arithmetic/job_manager/

Job queue, sweep orchestration, and model catalog management.

| File | Purpose |
|------|---------|
| `gpu_queue.py` | Multi-GPU job queue — reads sweep files, schedules jobs, handles kill/submit/modify via Redis |
| `job_state.py` | Redis-backed job state DB — CLI for kill/status/submit/clear |
| `catalog.py` | `ModelCatalog` / `DataCatalog` — persistent indexes of VALID/SUPERSEDED models on HF |
| `model_catalog.json` | Local copy of the HF model catalog (auto-updated by training jobs) |
| `run_sweep.py` | Higher-level sweep runner |
| `llm_reviewer.py` | GPT-4/Gemini code review integration |
| `auto_triage.py` | Automatic failure triage for queue jobs |
| `upload_status.py` | Monitor HF upload status |

## Usage

```bash
# Launch queue from sweep file (GPU slots 1 and 2)
export WANDB_API_KEY=... && export HF_TOKEN=...
source venv/bin/activate
nohup python -m arithmetic.job_manager.gpu_queue arithmetic/scripts/sweep.txt 1 2 > /tmp/queue.log 2>&1 &

# Check status
python -m arithmetic.job_manager.job_state

# Submit a single job to running queue
python -m arithmetic.job_manager.job_state submit "python -m arithmetic.train ..."

# Kill a job
python -m arithmetic.job_manager.job_state kill <job_name>
```

## Hash-skip

The queue computes a config hash for each job and checks HF for VALID models
with a matching hash. Already-completed jobs are skipped automatically.
SUPERSEDED models are ignored by the hash check (use `--force` to bypass all skipping).

## catalog.py

`ModelCatalog` wraps the HF model catalog JSON. Use it to query VALID models
for analysis and dashboard generation — not for queue management directly.
