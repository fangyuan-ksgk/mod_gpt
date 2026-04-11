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
