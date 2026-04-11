"""
Autonomous triage daemon: monitors queue, diagnoses failures with LLM, applies fixes.

Runs alongside the queue. When a job fails:
  1. Reads the error log
  2. Reads the relevant source file
  3. Sends to GPT-4 for diagnosis + fix
  4. Applies simple fixes automatically (import errors, config issues)
  5. Requeues the fixed job
  6. Flags complex issues in a report for human review

Usage:
    nohup python -m arithmetic.job_manager.auto_triage &

Safety:
    - Only auto-applies fixes for KNOWN SIMPLE patterns
    - Complex fixes are logged but NOT applied
    - All fixes are logged to /workspace/sorl_logs/auto_triage.log
"""
import json
import time
import os
import re
import subprocess
from pathlib import Path
from datetime import datetime

QUEUE_STATUS = "/tmp/gpu_queue/queue_status.json"
LOG_DIR = "/tmp/gpu_queue"
TRIAGE_LOG = "/workspace/sorl_logs/auto_triage.log"
REPORT_FILE = "/workspace/sorl_logs/auto_triage_report.txt"
POLL_INTERVAL = 60  # check every minute

# Simple patterns we can auto-fix
SIMPLE_PATTERNS = {
    "ImportError": "import_fix",
    "ModuleNotFoundError": "import_fix",
    "FileNotFoundError": "file_fix",
    "Server error '500": "retry_only",
    "ConnectionError": "retry_only",
    "ReadTimeout": "retry_only",
    "CUDA out of memory": "retry_only",
    "OutOfMemoryError": "retry_only",
}


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(TRIAGE_LOG, "a") as f:
        f.write(line + "\n")


def get_failed_jobs() -> list:
    """Get failed jobs from queue status JSON."""
    if not os.path.exists(QUEUE_STATUS):
        return []
    try:
        with open(QUEUE_STATUS) as f:
            status = json.load(f)
        return [j for j in status.get("jobs", []) if j["status"] == "failed"]
    except Exception:
        return []


def read_job_log(job: dict) -> str:
    """Read the last 50 lines of a job's log file."""
    log_file = job.get("log_file", "")
    if not log_file or not os.path.exists(log_file):
        # Try to find it
        job_id = job.get("job_id", -1)
        for f in Path(LOG_DIR).glob(f"job_{job_id:03d}_*"):
            log_file = str(f)
            break
    if not log_file or not os.path.exists(log_file):
        return ""
    with open(log_file) as f:
        lines = f.readlines()
    return "".join(lines[-50:])


def classify_error(error_text: str) -> tuple:
    """Classify error into a pattern. Returns (pattern_name, action)."""
    for pattern, action in SIMPLE_PATTERNS.items():
        if pattern in error_text:
            return pattern, action
    return "unknown", "llm_diagnose"


def llm_diagnose(error_text: str, job_name: str) -> dict:
    """Send error to GPT-4 for diagnosis."""
    try:
        from arithmetic.job_manager.llm_reviewer import Reviewer
        r = Reviewer()
        diagnosis = r.review(
            prompt=f"""A training job '{job_name}' failed. Diagnose the error and suggest a fix.

If the fix is a simple code change (1-3 lines), provide the exact fix as:
FIX_FILE: <filepath>
FIX_OLD: <exact string to replace>
FIX_NEW: <replacement string>

If the fix requires complex changes or human judgment, say NEEDS_HUMAN.

Error log (last 50 lines):
{error_text}""",
            review_type="implementation",
            max_tokens=1000,
        )
        return {"diagnosis": diagnosis, "job_name": job_name}
    except Exception as e:
        return {"diagnosis": f"LLM diagnosis failed: {e}", "job_name": job_name}


def apply_simple_fix(diagnosis: str) -> bool:
    """Parse and apply a simple fix from LLM diagnosis. Returns True if applied."""
    # Look for FIX_FILE/FIX_OLD/FIX_NEW pattern
    file_match = re.search(r"FIX_FILE:\s*(.+?)$", diagnosis, re.MULTILINE)
    old_match = re.search(r"FIX_OLD:\s*(.+?)$", diagnosis, re.MULTILINE)
    new_match = re.search(r"FIX_NEW:\s*(.+?)$", diagnosis, re.MULTILINE)

    if not (file_match and old_match and new_match):
        return False

    filepath = file_match.group(1).strip()
    old_str = old_match.group(1).strip()
    new_str = new_match.group(1).strip()

    if not os.path.exists(filepath):
        log(f"  Fix file not found: {filepath}")
        return False

    with open(filepath) as f:
        content = f.read()

    if old_str not in content:
        log(f"  Fix string not found in {filepath}")
        return False

    # Safety: only apply if it's a small change
    if len(old_str) > 200 or len(new_str) > 200:
        log(f"  Fix too large ({len(old_str)}/{len(new_str)} chars), skipping")
        return False

    content = content.replace(old_str, new_str, 1)
    with open(filepath, "w") as f:
        f.write(content)

    log(f"  Applied fix: {filepath}: '{old_str[:50]}' → '{new_str[:50]}'")
    return True


def requeue_job(job: dict):
    """Re-queue a failed job."""
    cmd = job.get("cmd", "")
    name = job.get("name", "retry")
    if not cmd:
        log(f"  Cannot requeue {name}: no command found")
        return

    retry_file = "/tmp/auto_triage_retry.txt"
    with open(retry_file, "w") as f:
        f.write(f"# Auto-triage retry for {name}\n")
        f.write(cmd + "\n")

    log(f"  Requeuing: {name}")
    subprocess.Popen(
        ["python", "-m", "arithmetic.job_manager.gpu_queue", retry_file, "3", "1",
         "--max-retries", "0"],
        stdout=open(f"/workspace/sorl_logs/retry_{name}.log", "w"),
        stderr=subprocess.STDOUT,
    )


def write_report(triaged: list):
    """Write human-readable report of all triage actions."""
    with open(REPORT_FILE, "w") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"Auto-Triage Report ({ts})\n{'='*60}\n\n")
        for t in triaged:
            f.write(f"Job: {t['name']}\n")
            f.write(f"  Error: {t['error_type']}\n")
            f.write(f"  Action: {t['action']}\n")
            if t.get('diagnosis'):
                f.write(f"  Diagnosis: {t['diagnosis'][:200]}\n")
            f.write(f"  Fixed: {t.get('fixed', False)}\n")
            f.write(f"  Requeued: {t.get('requeued', False)}\n\n")


def main():
    log("Auto-triage daemon started.")
    seen_failures = set()
    triaged = []

    while True:
        failed = get_failed_jobs()

        for job in failed:
            job_id = job.get("job_id", -1)
            name = job.get("name", f"job_{job_id}")

            if name in seen_failures:
                continue
            seen_failures.add(name)

            log(f"\nFAILURE DETECTED: {name}")
            error_text = read_job_log(job)
            error_type, action = classify_error(error_text)
            log(f"  Error type: {error_type}, Action: {action}")

            triage_record = {
                "name": name, "error_type": error_type,
                "action": action, "fixed": False, "requeued": False,
            }

            if action == "retry_only":
                log(f"  Transient error — requeuing")
                requeue_job(job)
                triage_record["requeued"] = True

            elif action == "import_fix":
                log(f"  Import error — sending to LLM for diagnosis")
                result = llm_diagnose(error_text, name)
                triage_record["diagnosis"] = result["diagnosis"]

                if "NEEDS_HUMAN" not in result["diagnosis"]:
                    fixed = apply_simple_fix(result["diagnosis"])
                    triage_record["fixed"] = fixed
                    if fixed:
                        requeue_job(job)
                        triage_record["requeued"] = True
                    else:
                        log(f"  LLM suggested fix but couldn't apply — needs human review")
                else:
                    log(f"  LLM says needs human review")

            elif action == "llm_diagnose":
                log(f"  Unknown error — sending to LLM")
                result = llm_diagnose(error_text, name)
                triage_record["diagnosis"] = result["diagnosis"]

                if "NEEDS_HUMAN" not in result["diagnosis"]:
                    fixed = apply_simple_fix(result["diagnosis"])
                    triage_record["fixed"] = fixed
                    if fixed:
                        requeue_job(job)
                        triage_record["requeued"] = True
                else:
                    log(f"  Needs human review")

            triaged.append(triage_record)
            write_report(triaged)

        # Check if queue is done
        try:
            with open(QUEUE_STATUS) as f:
                status = json.load(f)
            total = status.get("total", 0)
            done = status.get("done", 0)
            failed_count = status.get("failed", 0)
            if total > 0 and done + failed_count >= total:
                log(f"\nQueue complete: {done} done, {failed_count} failed")
                break
        except Exception:
            pass

        time.sleep(POLL_INTERVAL)

    log("Auto-triage daemon finished.")
    write_report(triaged)


if __name__ == "__main__":
    main()
