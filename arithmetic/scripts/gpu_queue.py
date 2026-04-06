#!/usr/bin/env python3
"""
GPU job queue: submits shell commands to the least-loaded GPU.

Usage:
    # From a job file (one command per line, {GPU} is replaced with GPU id):
    python gpu_queue.py jobs.txt

    # Programmatic:
    from gpu_queue import GPUQueue
    q = GPUQueue(n_gpus=3, max_per_gpu=1)
    q.submit("CUDA_VISIBLE_DEVICES={GPU} python train.py --config a")
    q.submit("CUDA_VISIBLE_DEVICES={GPU} python train.py --config b")
    q.wait()

Features:
    - Picks GPU with fewest running jobs (ties broken by lowest GPU id)
    - Waits if all GPUs are at max_per_gpu capacity
    - Logs start/finish/fail for each job
    - Returns summary of all jobs when done
"""
import subprocess
import threading
import time
import sys
import os
from dataclasses import dataclass, field
from typing import List, Optional
from collections import defaultdict


@dataclass
class Job:
    job_id: int
    cmd: str
    gpu: int = -1
    status: str = "pending"  # pending, running, done, failed
    proc: Optional[subprocess.Popen] = field(default=None, repr=False)
    start_time: float = 0
    end_time: float = 0
    exit_code: int = -1
    log_file: str = ""


class GPUQueue:
    def __init__(self, n_gpus: int = 3, max_per_gpu: int = 1,
                 poll_interval: float = 5.0, log_dir: str = "/tmp/gpu_queue"):
        self.n_gpus = n_gpus
        self.max_per_gpu = max_per_gpu
        self.poll_interval = poll_interval
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.jobs: List[Job] = []
        self.gpu_running: defaultdict = defaultdict(int)  # gpu_id -> count
        self.lock = threading.Lock()
        self._job_counter = 0
        self._threads: List[threading.Thread] = []

    def submit(self, cmd: str) -> int:
        """Submit a command. {GPU} in cmd will be replaced with the assigned GPU id."""
        with self.lock:
            job_id = self._job_counter
            self._job_counter += 1
            job = Job(job_id=job_id, cmd=cmd)
            self.jobs.append(job)

        t = threading.Thread(target=self._run_job, args=(job,), daemon=True)
        self._threads.append(t)
        t.start()
        return job_id

    def _pick_gpu(self) -> Optional[int]:
        """Return GPU with fewest running jobs, or None if all at capacity."""
        with self.lock:
            best_gpu = None
            best_count = self.max_per_gpu + 1
            for gpu in range(self.n_gpus):
                count = self.gpu_running[gpu]
                if count < self.max_per_gpu and count < best_count:
                    best_count = count
                    best_gpu = gpu
            return best_gpu

    def _run_job(self, job: Job):
        # Wait for a free GPU
        while True:
            gpu = self._pick_gpu()
            if gpu is not None:
                break
            time.sleep(self.poll_interval)

        # Claim the GPU
        with self.lock:
            self.gpu_running[gpu] += 1
            job.gpu = gpu
            job.status = "running"
            job.start_time = time.time()
            job.log_file = os.path.join(self.log_dir, f"job_{job.job_id:03d}_gpu{gpu}.log")

        cmd = job.cmd.replace("{GPU}", str(gpu))
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] JOB {job.job_id:3d} GPU {gpu} START: {cmd[:80]}...")

        try:
            with open(job.log_file, "w") as logf:
                proc = subprocess.Popen(
                    cmd, shell=True, stdout=logf, stderr=subprocess.STDOUT,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                )
                job.proc = proc
                proc.wait()
                job.exit_code = proc.returncode
        except Exception as e:
            job.exit_code = -1
            with open(job.log_file, "a") as logf:
                logf.write(f"\nEXCEPTION: {e}\n")

        # Release GPU
        with self.lock:
            self.gpu_running[gpu] -= 1
            job.end_time = time.time()
            job.status = "done" if job.exit_code == 0 else "failed"

        elapsed = job.end_time - job.start_time
        ts = time.strftime("%H:%M:%S")
        tag = "DONE" if job.exit_code == 0 else f"FAIL(exit={job.exit_code})"
        print(f"[{ts}] JOB {job.job_id:3d} GPU {gpu} {tag} ({elapsed:.0f}s)")

    def wait(self):
        """Block until all submitted jobs finish."""
        for t in self._threads:
            t.join()

    def summary(self) -> str:
        lines = []
        n_done = sum(1 for j in self.jobs if j.status == "done")
        n_fail = sum(1 for j in self.jobs if j.status == "failed")
        lines.append(f"Total: {len(self.jobs)} jobs | Done: {n_done} | Failed: {n_fail}")
        lines.append("")
        for j in self.jobs:
            elapsed = j.end_time - j.start_time if j.end_time else 0
            status = f"{'DONE' if j.status == 'done' else 'FAIL':4s}"
            lines.append(f"  {j.job_id:3d} GPU {j.gpu} {status} {elapsed:6.0f}s  {j.cmd[:70]}")
        return "\n".join(lines)


def main():
    """Read jobs from a file (one command per line) and run them."""
    if len(sys.argv) < 2:
        print("Usage: python gpu_queue.py jobs.txt [n_gpus] [max_per_gpu]")
        print("")
        print("jobs.txt: one shell command per line. {GPU} replaced with GPU id.")
        print("Lines starting with # are skipped.")
        sys.exit(1)

    jobs_file = sys.argv[1]
    n_gpus = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    max_per_gpu = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    with open(jobs_file) as f:
        commands = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    print(f"GPU Queue: {len(commands)} jobs, {n_gpus} GPUs, max {max_per_gpu} per GPU")
    print("")

    q = GPUQueue(n_gpus=n_gpus, max_per_gpu=max_per_gpu)
    for cmd in commands:
        q.submit(cmd)
    q.wait()

    print("")
    print(q.summary())


if __name__ == "__main__":
    main()
