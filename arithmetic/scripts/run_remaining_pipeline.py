"""
Run all experiments that were missed by the original auto_zipf_sweep.py.
Launch after the 10-epoch baselines finish.

Order:
  1. Check baseline convergence → queue 30-epoch if needed
  2. Low-data SoRL (5 jobs)
  3. Low-data vocab sweep (24 jobs)
  4. Undersized models (30 jobs)
  5. Write results to log/arithmetic.md
"""
import subprocess
import time
import os
import json

REPO_DIR = "/workspace/codes/mod_gpt"
LOG_DIR = "/workspace/sorl_logs"


def wait_for_log(log_file, n_jobs):
    """Wait for a job queue log to show all jobs done."""
    while True:
        if os.path.exists(log_file):
            with open(log_file) as f:
                text = f.read()
            done = text.count(" DONE ")
            fail = text.count(" FAIL(")
            if done + fail >= n_jobs:
                print(f"  {log_file}: {done} done, {fail} failed")
                return
        time.sleep(30)


def check_baseline_convergence():
    """Check if 10-epoch baselines are still improving."""
    from huggingface_hub import hf_hub_download

    KEY_BASELINES = [
        ("add_sub_baseline_25K", 25000),
        ("add_sub_baseline_50K", 50000),
        ("add_sub_baseline_100K", 100000),
    ]

    still_improving = []
    for name, ds_size in KEY_BASELINES:
        try:
            path = hf_hub_download(
                "thoughtworks/arithmetic-sorl", f"{name}/metrics.json",
                local_dir="/tmp/hf_cache/arithmetic-sorl",
            )
            m = json.load(open(path))
            h = m.get("history", {})
            evals = h.get("eval_accuracy", [])
            if len(evals) < 6:
                print(f"  {name}: only {len(evals)} eval points, skipping")
                continue

            recent = sum(evals[-3:]) / 3
            earlier = sum(evals[-6:-3]) / 3
            delta = recent - earlier
            print(f"  {name}: last3={recent:.3f} prev3={earlier:.3f} delta={delta:+.3f}")

            if delta > 0.01:
                still_improving.append((name, ds_size))
        except Exception as e:
            print(f"  {name}: {e}")

    if not still_improving:
        print("Baselines converged. No extended runs needed.")
        return

    print(f"\n{len(still_improving)} baselines still improving — queuing 30-epoch runs")
    jobs_file = f"{REPO_DIR}/arithmetic/scripts/sweep_baselines_30ep.txt"
    with open(jobs_file, "w") as f:
        f.write("# Auto-generated: baselines still improving at 10 epochs\n")
        for name, ds_size in still_improving:
            ds_k = ds_size // 1000
            f.write(
                f"python -m arithmetic.train --mode baseline --ops add_sub "
                f"--dataset_size {ds_size} --num_epochs 30 --push_to_hub --no_wandb "
                f"--output_dir ckpt/sweep/add_sub_baseline_{ds_k}K_30ep\n"
            )

    log = f"{LOG_DIR}/sweep_baselines_30ep.log"
    subprocess.run(
        ["python", "-m", "arithmetic.scripts.gpu_queue", jobs_file, "3", "2"],
        stdout=open(log, "w"), stderr=subprocess.STDOUT,
    )
    print(f"30-epoch baselines done. Log: {log}")


def run_queue(name, jobs_file, n_gpus=3, max_per_gpu=1):
    log = f"{LOG_DIR}/sweep_{name}.log"
    print(f"\n=== {name} ===")
    subprocess.run(
        ["python", "-m", "arithmetic.scripts.gpu_queue", jobs_file, str(n_gpus), str(max_per_gpu)],
        stdout=open(log, "w"), stderr=subprocess.STDOUT,
    )
    print(f"Done. Log: {log}")


def write_results():
    from arithmetic.catalog import ModelCatalog
    print("\nWriting results to log/arithmetic.md...")
    cat = ModelCatalog()
    cat.fetch(verbose=False)
    cat.write_results_md(f"{REPO_DIR}/log/arithmetic.md")
    cat.save(f"{REPO_DIR}/log/catalog.json")
    cat.print_table()

    subprocess.run(["git", "-C", REPO_DIR, "add", "log/arithmetic.md", "log/catalog.json"], check=False)
    subprocess.run(["git", "-C", REPO_DIR, "commit", "-m",
                    "auto-update: arithmetic.md with all sweep results\n\n"
                    "Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"],
                   check=False)
    subprocess.run(["git", "-C", REPO_DIR, "push", "origin", "HEAD"], check=False)


def main():
    # Wait for 10-epoch baselines to finish
    print("Waiting for 10-epoch baselines...")
    wait_for_log(f"{LOG_DIR}/sweep_baselines_10ep.log", 12)

    # Check convergence
    print("\nChecking baseline convergence...")
    check_baseline_convergence()

    # Low-data SoRL
    run_queue("low_data_sorl", f"{REPO_DIR}/arithmetic/scripts/sweep_low_data_sorl.txt")

    # Low-data vocab sweep
    run_queue("low_data_vocab", f"{REPO_DIR}/arithmetic/scripts/sweep_low_data_vocab.txt")

    # Undersized models
    run_queue("undersize", f"{REPO_DIR}/arithmetic/scripts/sweep_undersize.txt")

    # Write results
    write_results()

    print("\n=== ALL EXPERIMENTS COMPLETE ===")


if __name__ == "__main__":
    main()
