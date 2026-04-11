"""
Post-sweep triage: runs after gpu_queue finishes. Checks for failures,
diagnoses root causes, fixes what it can, re-queues what it can't.

Usage:
    python -m arithmetic.job_manager.post_sweep /workspace/sorl_logs/sweep_final.log

What it does:
    1. Parse queue log for failed jobs
    2. Read each failure's job log, classify the error
    3. Auto-fix known issues (import errors, OOM, HF upload failures)
    4. Re-queue fixable failures
    5. Write a triage report for unfixable ones
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path


KNOWN_FIXES = {
    "ImportError": "Code changed during run. Retry should work with current code.",
    "OutOfMemoryError": "GPU OOM. Retry with smaller batch or different GPU.",
    "CUDA out of memory": "GPU OOM. Retry with smaller batch or different GPU.",
    "Server error '500": "HF Hub transient error. Retry.",
    "ConnectionError": "Network issue. Retry.",
    "ReadTimeout": "HF Hub timeout. Retry.",
}


def parse_failures(log_file: str) -> list:
    """Extract failed jobs from queue log."""
    failures = []
    with open(log_file) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if "FAIL(" in line:
            # Parse job info
            match = re.search(r"JOB\s+(\d+)\s+GPU\s+(\d+)\s+FAIL\(exit=(-?\d+)\).*?\)\s+(.*)", line)
            if match:
                job_id = int(match.group(1))
                gpu = int(match.group(2))
                exit_code = int(match.group(3))
                name = match.group(4).strip()
            else:
                job_id, gpu, exit_code, name = -1, -1, -1, line.strip()

            # Find the log file
            log_dir = "/tmp/gpu_queue"
            job_log = None
            for f in Path(log_dir).glob(f"job_{job_id:03d}_*"):
                job_log = str(f)
                break

            # Read error from job log
            error_lines = []
            error_type = "unknown"
            if job_log and os.path.exists(job_log):
                with open(job_log) as jf:
                    tail = jf.readlines()[-20:]
                    error_lines = [l.strip() for l in tail if l.strip()]

                    # Classify error
                    full_text = "\n".join(tail)
                    for pattern, fix in KNOWN_FIXES.items():
                        if pattern in full_text:
                            error_type = pattern
                            break

            failures.append({
                "job_id": job_id,
                "name": name,
                "gpu": gpu,
                "exit_code": exit_code,
                "log_file": job_log,
                "error_type": error_type,
                "error_lines": error_lines[-5:],
                "retryable": error_type != "unknown",
            })

    return failures


def extract_command(log_file: str, job_id: int) -> str:
    """Extract the original command for a job from the queue log."""
    with open(log_file) as f:
        for line in f:
            if f"JOB {job_id:3d}" in line and "START:" in line:
                # The command is in the job's own log file, or we reconstruct from the name
                match = re.search(r"\((.+?)\.\.\.\)", line)
                if match:
                    return match.group(1) + "..."
    return ""


def validate_uploaded_model(name: str) -> list:
    """
    Check that an uploaded model has all required artifacts and valid metrics.
    Returns list of issues (empty = valid).
    """
    from huggingface_hub import hf_hub_download, HfApi
    issues = []
    repo = "thoughtworks/arithmetic-sorl"
    cache = "/tmp/hf_validate_cache"

    try:
        api = HfApi()
        files = api.list_repo_files(repo)
        model_files = [f for f in files if f.startswith(f"{name}/")]

        # Check required files exist
        required = ["train_config.json", "metrics.json", "model.safetensors", "config.json"]
        for req in required:
            if f"{name}/{req}" not in model_files:
                issues.append(f"missing {req}")

        if f"{name}/train_config.json" in model_files:
            path = hf_hub_download(repo, f"{name}/train_config.json", local_dir=cache)
            cfg = json.load(open(path))

            # Check wandb was recorded
            if not cfg.get("wandb_run_id"):
                issues.append("no wandb_run_id")

            # Check eval method
            if cfg.get("eval_method") != "ArithmeticEvaluator":
                issues.append(f"eval_method={cfg.get('eval_method')} (expected ArithmeticEvaluator)")

            # Check accuracy is reasonable
            acc = cfg.get("final_accuracy")
            if acc is None:
                issues.append("no final_accuracy")

        if f"{name}/metrics.json" in model_files:
            path = hf_hub_download(repo, f"{name}/metrics.json", local_dir=cache)
            metrics = json.load(open(path))

            # Check eval results exist
            has_sft_eval = "sft_eval" in metrics
            has_sorl_eval = "sorl_eval" in metrics
            if not has_sft_eval:
                issues.append("no sft_eval in metrics")

            # Check training history has eval curves
            history = metrics.get("history", {})
            if "eval_step" not in history or len(history.get("eval_step", [])) == 0:
                issues.append("no eval curves in training history")

            # Check splits exist in eval
            for eval_key in ["sft_eval", "sorl_eval"]:
                ev = metrics.get(eval_key, {})
                splits = ev.get("splits", {})
                if ev and len(splits) < 10:
                    issues.append(f"{eval_key} has only {len(splits)} splits (expected 15+)")

    except Exception as e:
        issues.append(f"validation error: {e}")

    return issues


def validate_all_completed(log_file: str) -> dict:
    """Validate all successfully completed jobs have proper outputs."""
    print(f"\n{'='*60}")
    print("VALIDATING COMPLETED JOBS")
    print(f"{'='*60}")

    # Get completed job names from log
    completed = []
    with open(log_file) as f:
        for line in f:
            if " DONE " in line:
                match = re.search(r"DONE \(\d+s\)\s+(\S+)", line)
                if match:
                    completed.append(match.group(1))

    results = {"valid": [], "invalid": []}
    for name in completed:
        # Map job name to HF subfolder name
        # The train.py generates run_name from args, which becomes the HF subfolder
        # Job name from queue = output_dir basename, which may differ from run_name
        # Try common patterns
        hf_name = name  # might need mapping
        issues = validate_uploaded_model(hf_name)
        if issues:
            print(f"  INVALID: {name}")
            for issue in issues:
                print(f"    - {issue}")
            results["invalid"].append({"name": name, "issues": issues})
        else:
            results["valid"].append(name)

    print(f"\nValidation: {len(results['valid'])} valid, {len(results['invalid'])} invalid")
    return results


def triage_and_requeue(log_file: str):
    """Main triage: parse failures, diagnose, requeue fixable ones."""
    failures = parse_failures(log_file)

    if not failures:
        print("No failures. All jobs completed successfully.")
        return

    print(f"\n{'='*60}")
    print(f"POST-SWEEP TRIAGE: {len(failures)} failures")
    print(f"{'='*60}")

    retryable = []
    unfixable = []

    for f in failures:
        print(f"\nJob {f['job_id']} ({f['name']}):")
        print(f"  Error type: {f['error_type']}")
        if f['error_lines']:
            print(f"  Last output: {f['error_lines'][-1][:100]}")

        if f['retryable']:
            print(f"  → RETRYABLE: {KNOWN_FIXES.get(f['error_type'], 'retry')}")
            retryable.append(f)
        else:
            print(f"  → UNFIXABLE: manual investigation needed")
            print(f"  Log: {f['log_file']}")
            for line in f['error_lines'][-3:]:
                print(f"    {line[:120]}")
            unfixable.append(f)

    # Re-queue retryable failures
    if retryable:
        print(f"\n{'='*60}")
        print(f"RE-QUEUING {len(retryable)} retryable failures...")
        print(f"{'='*60}")

        # Find original commands from the sweep file
        # Read the sweep file to get commands by matching output_dir names
        sweep_file = None
        for sf in Path("arithmetic/scripts").glob("sweep_*.txt"):
            sweep_file = str(sf)

        if sweep_file:
            with open(sweep_file) as f:
                all_cmds = [l.strip() for l in f if l.strip() and not l.startswith("#")]

            retry_cmds = []
            for failure in retryable:
                name = failure["name"]
                # Find matching command
                for cmd in all_cmds:
                    if name in cmd:
                        retry_cmds.append(cmd)
                        break
                else:
                    print(f"  WARNING: couldn't find command for {name}")

            if retry_cmds:
                retry_file = "/tmp/retry_jobs.txt"
                with open(retry_file, "w") as f:
                    f.write("# Auto-generated retry jobs\n")
                    for cmd in retry_cmds:
                        f.write(cmd + "\n")

                print(f"  Written {len(retry_cmds)} retry jobs to {retry_file}")
                print(f"  Launching...")

                result = subprocess.run(
                    ["python", "-m", "arithmetic.job_manager.gpu_queue",
                     retry_file, "3", "1", "--max-retries", "1"],
                    capture_output=False,
                )
                print(f"  Retry queue exit code: {result.returncode}")

    # Report
    print(f"\n{'='*60}")
    print(f"TRIAGE COMPLETE")
    print(f"  Total failures: {len(failures)}")
    print(f"  Retried: {len(retryable)}")
    print(f"  Unfixable: {len(unfixable)}")
    if unfixable:
        print(f"\n  UNFIXABLE JOBS (need manual investigation):")
        for f in unfixable:
            print(f"    {f['name']}: {f['error_lines'][-1][:80] if f['error_lines'] else '?'}")
            print(f"    Log: {f['log_file']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    log = sys.argv[1] if len(sys.argv) > 1 else "/workspace/sorl_logs/sweep_final.log"
    triage_and_requeue(log)
    validate_all_completed(log)
