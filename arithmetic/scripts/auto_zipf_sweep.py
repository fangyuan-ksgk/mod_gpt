"""
Auto-queue Fangyuan's zipf diversity sweep after the main enriched sweep finishes.

Watches the main sweep log for completion, picks the 2 best vocab sizes
from SoRL results, then queues:
  zipf={2.0, 5.0, 10.0} × K={1, 4} × best_2_vocabs = 12 runs

Run this AFTER launching the main sweep:
  nohup python -m arithmetic.scripts.auto_zipf_sweep &
"""
import time
import os
import subprocess
import json
from pathlib import Path

SWEEP_LOG = "/workspace/sorl_logs/sweep_enriched.log"
ZIPF_LOG = "/workspace/sorl_logs/sweep_zipf.log"
ZIPF_JOBS = "/workspace/codes/mod_gpt/arithmetic/scripts/sweep_zipf.txt"
TOTAL_MAIN_JOBS = 30


def sweep_finished():
    """Check if all main sweep jobs are done."""
    if not os.path.exists(SWEEP_LOG):
        return False
    with open(SWEEP_LOG) as f:
        text = f.read()
    done_count = text.count(" DONE ")
    fail_count = text.count(" FAIL(")
    return (done_count + fail_count) >= TOTAL_MAIN_JOBS


def find_best_vocabs():
    """Fetch catalog and find 2 best vocab sizes from enriched SoRL add_sub models."""
    # Import here so we don't fail if catalog isn't available at import time
    from arithmetic.catalog import ModelCatalog

    cat = ModelCatalog()
    cat.fetch(verbose=False)

    # Filter to enriched SoRL add_sub models at 500K
    sorl_models = [e for e in cat.entries
                   if e.mode == "sorl"
                   and e.ops == "add_sub"
                   and e.dataset_size == 500000
                   and e.enriched]

    if not sorl_models:
        print("WARNING: No enriched SoRL models found. Using defaults: vocab=10, 20")
        return [10, 20]

    # Sort by accuracy (sorl accuracy if available, else final_accuracy)
    def get_acc(entry):
        acc = entry.config.get("final_accuracy", 0)
        return acc if acc is not None else 0

    # Group by vocab, take best accuracy per vocab
    vocab_best = {}
    for e in sorl_models:
        v = e.abs_vocab
        acc = get_acc(e)
        if v not in vocab_best or acc > vocab_best[v]:
            vocab_best[v] = acc

    # Sort vocabs by best accuracy, take top 4
    sorted_vocabs = sorted(vocab_best.items(), key=lambda x: x[1], reverse=True)
    best_4 = [v for v, acc in sorted_vocabs[:4]]

    print(f"Vocab performance (enriched SoRL add_sub 500K):")
    for v, acc in sorted_vocabs:
        marker = " <-- SELECTED" if v in best_4 else ""
        print(f"  vocab={v}: {acc:.1%}{marker}")

    return best_4


def generate_zipf_jobs(best_vocabs):
    """Generate zipf sweep job commands."""
    jobs = []
    zipf_values = [2.0, 5.0, 10.0]
    K_values = [1, 4]

    for zipf in zipf_values:
        for K in K_values:
            for vocab in best_vocabs:
                name = f"as_sorl_abs{vocab}_K{K}_zipf{zipf}"
                cmd = (
                    f"python -m arithmetic.train --mode sorl --ops add_sub "
                    f"--dataset_size 500000 --abs_vocab {vocab} --K {K} "
                    f"--alpha_soft_zipf {zipf} "
                    f"--num_epochs 10 --push_to_hub --no_wandb "
                    f"--output_dir ckpt/sweep/{name}"
                )
                jobs.append(cmd)

    return jobs


def check_baseline_convergence():
    """
    Check if 10-epoch baselines are still improving. If the last 3 eval points
    show >1% improvement, queue 30-epoch runs for those data sizes.
    """
    from huggingface_hub import hf_hub_download
    import json

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
                print(f"  {name}: only {len(evals)} eval points, skipping convergence check")
                continue

            # Compare last 3 eval points vs 3 before that
            recent = sum(evals[-3:]) / 3
            earlier = sum(evals[-6:-3]) / 3
            delta = recent - earlier
            print(f"  {name}: last3={recent:.3f} prev3={earlier:.3f} delta={delta:+.3f}")

            if delta > 0.01:  # >1% improvement in second half
                still_improving.append((name, ds_size))
        except Exception as e:
            print(f"  {name}: couldn't check — {e}")

    if not still_improving:
        print("\nBaselines converged at 10 epochs. No extended runs needed.")
        return

    print(f"\n{len(still_improving)} baselines still improving — queuing 30-epoch runs:")
    jobs = []
    for name, ds_size in still_improving:
        ds_k = ds_size // 1000
        cmd = (
            f"python -m arithmetic.train --mode baseline --ops add_sub "
            f"--dataset_size {ds_size} --num_epochs 30 --push_to_hub --no_wandb "
            f"--output_dir ckpt/sweep/add_sub_baseline_{ds_k}K_30ep"
        )
        jobs.append(cmd)
        print(f"  {cmd[:80]}...")

    # Write and run
    jobs_file = "/workspace/codes/mod_gpt/arithmetic/scripts/sweep_baselines_30ep.txt"
    with open(jobs_file, "w") as f:
        f.write("# Auto-generated: baselines still improving at 10 epochs\n")
        for j in jobs:
            f.write(j + "\n")

    log_file = "/workspace/sorl_logs/sweep_baselines_30ep.log"
    subprocess.run(
        ["python", "-m", "arithmetic.job_manager.gpu_queue", jobs_file, "3", "2"],
        stdout=open(log_file, "w"),
        stderr=subprocess.STDOUT,
    )
    print(f"30-epoch baselines done. Log: {log_file}")


def main():
    print(f"Waiting for main sweep to finish ({TOTAL_MAIN_JOBS} jobs)...")
    print(f"Watching: {SWEEP_LOG}")

    while not sweep_finished():
        time.sleep(60)  # check every minute

    print("\nMain sweep finished!")
    time.sleep(5)  # let last writes flush

    # First: re-run baselines at 10 epochs (overwrites 5-epoch versions)
    print("\n=== Running 10-epoch baselines (apples-to-apples with SoRL) ===")
    BASELINE_JOBS = "/workspace/codes/mod_gpt/arithmetic/scripts/sweep_baselines_10ep.txt"
    BASELINE_LOG = "/workspace/sorl_logs/sweep_baselines_10ep.log"
    subprocess.run(
        ["python", "-m", "arithmetic.job_manager.gpu_queue",
         BASELINE_JOBS, "3", "2"],  # 2 per GPU — baselines are light
        stdout=open(BASELINE_LOG, "w"),
        stderr=subprocess.STDOUT,
    )
    print(f"10-epoch baselines done. Log: {BASELINE_LOG}")

    # Check if baselines are still improving at epoch 10 — if so, queue longer runs
    check_baseline_convergence()

    # SoRL at low data sizes — the data efficiency comparison
    print("\n=== Running SoRL at low data sizes (K=1 vocab=10) ===")
    LOW_DATA_JOBS = "/workspace/codes/mod_gpt/arithmetic/scripts/sweep_low_data_sorl.txt"
    LOW_DATA_LOG = "/workspace/sorl_logs/sweep_low_data_sorl.log"
    subprocess.run(
        ["python", "-m", "arithmetic.job_manager.gpu_queue",
         LOW_DATA_JOBS, "3", "1"],
        stdout=open(LOW_DATA_LOG, "w"),
        stderr=subprocess.STDOUT,
    )
    print(f"Low-data SoRL done. Log: {LOW_DATA_LOG}")

    # Low-data vocab sweep — tests H1: vocab size × K interaction
    print("\n=== Running low-data vocab sweep (K={1,4} × vocab={5,10,30,50} × data={25K,50K,100K}) ===")
    LOW_VOCAB_JOBS = "/workspace/codes/mod_gpt/arithmetic/scripts/sweep_low_data_vocab.txt"
    LOW_VOCAB_LOG = "/workspace/sorl_logs/sweep_low_data_vocab.log"
    subprocess.run(
        ["python", "-m", "arithmetic.job_manager.gpu_queue",
         LOW_VOCAB_JOBS, "3", "1"],
        stdout=open(LOW_VOCAB_LOG, "w"),
        stderr=subprocess.STDOUT,
    )
    print(f"Low-data vocab sweep done. Log: {LOW_VOCAB_LOG}")

    # Undersized model experiments: does SoRL converge when baseline fails?
    print("\n=== Running undersized model experiments ===")
    UNDERSIZE_JOBS = "/workspace/codes/mod_gpt/arithmetic/scripts/sweep_undersize.txt"
    UNDERSIZE_LOG = "/workspace/sorl_logs/sweep_undersize.log"
    subprocess.run(
        ["python", "-m", "arithmetic.job_manager.gpu_queue",
         UNDERSIZE_JOBS, "3", "1"],
        stdout=open(UNDERSIZE_LOG, "w"),
        stderr=subprocess.STDOUT,
    )
    print(f"Undersized experiments done. Log: {UNDERSIZE_LOG}")

    # Find best vocabs
    best_vocabs = find_best_vocabs()
    print(f"\nBest 2 vocab sizes: {best_vocabs}")

    # Generate jobs
    jobs = generate_zipf_jobs(best_vocabs)
    print(f"\nGenerated {len(jobs)} zipf sweep jobs:")
    for j in jobs:
        print(f"  {j[:80]}...")

    # Write jobs file
    with open(ZIPF_JOBS, "w") as f:
        f.write("# Auto-generated zipf diversity sweep (Fangyuan's guidance)\n")
        f.write(f"# Best vocabs from main sweep: {best_vocabs}\n")
        f.write(f"# zipf={{2.0, 5.0, 10.0}} x K={{1, 4}} x vocab={best_vocabs}\n\n")
        for j in jobs:
            f.write(j + "\n")

    # Launch via gpu_queue
    print(f"\nLaunching zipf sweep ({len(jobs)} jobs, 3 GPUs)...")
    subprocess.run(
        ["python", "-m", "arithmetic.job_manager.gpu_queue",
         ZIPF_JOBS, "3", "1"],
        stdout=open(ZIPF_LOG, "w"),
        stderr=subprocess.STDOUT,
    )

    print(f"\nZipf sweep complete! Log: {ZIPF_LOG}")

    # Write final results to arithmetic.md
    write_results()


def write_results():
    """Fetch full catalog and write results to log/arithmetic.md."""
    from arithmetic.catalog import ModelCatalog

    print("\nWriting results to log/arithmetic.md...")
    cat = ModelCatalog()
    cat.fetch(verbose=False)
    cat.write_results_md("/workspace/codes/mod_gpt/log/arithmetic.md")
    cat.save("/workspace/codes/mod_gpt/log/catalog.json")
    cat.print_table()

    # Also commit results
    import subprocess
    subprocess.run(["git", "-C", "/workspace/codes/mod_gpt", "add",
                    "log/arithmetic.md", "log/catalog.json"], check=False)
    subprocess.run(["git", "-C", "/workspace/codes/mod_gpt", "commit", "-m",
                    "auto-update: arithmetic.md with sweep results\n\n"
                    "Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"],
                   check=False)
    subprocess.run(["git", "-C", "/workspace/codes/mod_gpt", "push", "origin", "HEAD"],
                   check=False)
    print("Results committed and pushed.")


if __name__ == "__main__":
    main()
