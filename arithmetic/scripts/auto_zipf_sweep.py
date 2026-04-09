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

    # Sort vocabs by best accuracy, take top 2
    sorted_vocabs = sorted(vocab_best.items(), key=lambda x: x[1], reverse=True)
    best_2 = [v for v, acc in sorted_vocabs[:2]]

    print(f"Vocab performance (enriched SoRL add_sub 500K):")
    for v, acc in sorted_vocabs:
        marker = " <-- SELECTED" if v in best_2 else ""
        print(f"  vocab={v}: {acc:.1%}{marker}")

    return best_2


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


def main():
    print(f"Waiting for main sweep to finish ({TOTAL_MAIN_JOBS} jobs)...")
    print(f"Watching: {SWEEP_LOG}")

    while not sweep_finished():
        time.sleep(60)  # check every minute

    print("\nMain sweep finished!")
    time.sleep(5)  # let last writes flush

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
        ["python", "-m", "arithmetic.scripts.gpu_queue",
         ZIPF_JOBS, "3", "1"],
        stdout=open(ZIPF_LOG, "w"),
        stderr=subprocess.STDOUT,
    )

    print(f"\nZipf sweep complete! Log: {ZIPF_LOG}")


if __name__ == "__main__":
    main()
