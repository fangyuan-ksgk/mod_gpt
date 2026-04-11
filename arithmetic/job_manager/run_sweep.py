"""
Run a sweep with automatic post-sweep triage.

Usage:
    python -m arithmetic.job_manager.run_sweep arithmetic/scripts/sweep_final.txt [n_gpus] [max_per_gpu]

This is the ONE command to run a sweep end-to-end:
    1. Run all jobs via gpu_queue
    2. Run post_sweep triage (diagnose failures, retry fixable ones)
    3. Write results to log/arithmetic.md
"""
import subprocess
import sys

REPO_DIR = "/workspace/codes/mod_gpt"


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m arithmetic.job_manager.run_sweep jobs.txt [n_gpus] [max_per_gpu]")
        sys.exit(1)

    jobs_file = sys.argv[1]
    n_gpus = sys.argv[2] if len(sys.argv) > 2 else "3"
    max_per_gpu = sys.argv[3] if len(sys.argv) > 3 else "1"

    log_file = "/workspace/sorl_logs/sweep_final.log"

    # Step 1: Run the queue
    print("=" * 60)
    print("STEP 1: Running job queue")
    print("=" * 60)
    subprocess.run(
        ["python", "-m", "arithmetic.job_manager.gpu_queue",
         jobs_file, n_gpus, max_per_gpu,
         "--stale-timeout", "1800", "--max-retries", "1"],
        stdout=open(log_file, "w"),
        stderr=subprocess.STDOUT,
    )

    # Step 2: Triage failures
    print("\n" + "=" * 60)
    print("STEP 2: Post-sweep triage")
    print("=" * 60)
    subprocess.run(
        ["python", "-m", "arithmetic.job_manager.post_sweep", log_file],
    )

    # Step 3: Write results
    print("\n" + "=" * 60)
    print("STEP 3: Writing results")
    print("=" * 60)
    try:
        from arithmetic.catalog import ModelCatalog
        cat = ModelCatalog()
        cat.fetch(verbose=False)
        cat.write_results_md(f"{REPO_DIR}/log/arithmetic.md")
        cat.print_table()

        subprocess.run(["git", "-C", REPO_DIR, "add", "log/arithmetic.md"], check=False)
        subprocess.run(["git", "-C", REPO_DIR, "commit", "-m",
                        "auto-update: arithmetic.md with sweep results\n\n"
                        "Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"],
                       check=False)
        subprocess.run(["git", "-C", REPO_DIR, "push", "origin", "HEAD"], check=False)
    except Exception as e:
        print(f"Results writing failed: {e}")

    print("\n" + "=" * 60)
    print("SWEEP COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
