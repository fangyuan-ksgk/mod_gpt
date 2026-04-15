#!/usr/bin/env python3
"""
GPU job queue with monitoring, callbacks, failure detection, and auto-retry.

Usage:
    # From a job file:
    python gpu_queue.py jobs.txt [n_gpus] [max_per_gpu]

    # Programmatic:
    from gpu_queue import GPUQueue, JobConfig
    q = GPUQueue(n_gpus=3, max_per_gpu=1)
    q.submit("python train.py --config a", name="baseline_25K")
    q.submit("python train.py --config b", name="sorl_25K",
             on_complete=lambda j: print(f"done: {j.name}"),
             on_fail=lambda j: print(f"FAILED: {j.name}"))
    q.wait()
    q.print_summary()

Features:
    - Real-time status polling (configurable interval)
    - Per-job callbacks: on_complete, on_fail
    - Stale job detection (no output for N seconds → killed + requeued)
    - Failure auto-retry (configurable max_retries)
    - JSON status file updated every poll (machine-readable progress)
    - Heartbeat file per job (last output timestamp)
"""
import subprocess
import threading
import time
import sys
import os
import json
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from collections import defaultdict
from pathlib import Path


@dataclass
class Job:
    job_id: int
    cmd: str
    name: str = ""
    gpu: int = -1
    status: str = "pending"  # pending, running, done, failed, stale, retrying
    priority: int = 1        # 0=high, 1=normal, 2=low (lower number = higher priority)
    proc: Optional[subprocess.Popen] = field(default=None, repr=False)
    start_time: float = 0
    end_time: float = 0
    exit_code: int = -1
    log_file: str = ""
    last_output_time: float = 0
    retries: int = 0
    max_retries: int = 1
    on_complete: Optional[Callable] = field(default=None, repr=False)
    on_fail: Optional[Callable] = field(default=None, repr=False)
    stale_timeout: float = 1800  # 30 min no output → stale
    gpu_slot_released: bool = False  # prevents double-decrement

    @property
    def elapsed(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return time.time() - self.start_time
        return 0

    @property
    def idle_time(self) -> float:
        """Seconds since last output."""
        if self.last_output_time:
            return time.time() - self.last_output_time
        elif self.start_time:
            return time.time() - self.start_time
        return 0

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id, "name": self.name, "cmd": self.cmd[:100],
            "gpu": self.gpu, "status": self.status,
            "elapsed": round(self.elapsed), "idle_time": round(self.idle_time),
            "exit_code": self.exit_code, "retries": self.retries,
            "log_file": self.log_file,
        }


class GPUQueue:
    def __init__(self, n_gpus: int = 3, max_per_gpu: int = 1,
                 poll_interval: float = 5.0, log_dir: str = "/tmp/gpu_queue",
                 status_file: str = None, stale_timeout: float = 1800,
                 max_retries: int = 1):
        self.n_gpus = n_gpus
        self.max_per_gpu = max_per_gpu
        self.poll_interval = poll_interval
        self.log_dir = log_dir
        self.status_file = status_file or os.path.join(log_dir, "queue_status.json")
        self.stale_timeout = stale_timeout
        self.max_retries = max_retries
        os.makedirs(log_dir, exist_ok=True)

        self.jobs: List[Job] = []
        self.gpu_running: defaultdict = defaultdict(int)
        self.lock = threading.Lock()
        self._job_counter = 0
        self._threads: List[threading.Thread] = []
        self._monitor_thread = None
        self._stop_monitor = threading.Event()

        # Redis state DB integration
        self.db = None
        try:
            from arithmetic.job_manager.job_state import JobStateDB
            self.db = JobStateDB()
            self.db.clear()  # fresh state for this queue run
        except Exception:
            pass  # Redis not available, run without it

    def submit(self, cmd: str, name: str = "", priority: int = 1,
               on_complete: Callable = None, on_fail: Callable = None) -> int:
        """Submit a command. priority: 0=high, 1=normal, 2=low. Returns job ID."""
        with self.lock:
            job_id = self._job_counter
            self._job_counter += 1
            job = Job(
                job_id=job_id, cmd=cmd, name=name or f"job_{job_id}",
                priority=priority,
                max_retries=self.max_retries, stale_timeout=self.stale_timeout,
                on_complete=on_complete, on_fail=on_fail,
            )
            self.jobs.append(job)
            if self.db:
                self.db.create_job(name or f"job_{job_id}", cmd=cmd)

        t = threading.Thread(target=self._run_job, args=(job,), daemon=True)
        self._threads.append(t)
        t.start()

        # Start monitor if not running
        if self._monitor_thread is None:
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()

        return job_id

    def _pick_gpu(self) -> Optional[int]:
        with self.lock:
            best_gpu = None
            best_count = self.max_per_gpu + 1
            for gpu in range(self.n_gpus):
                count = self.gpu_running[gpu]
                if count < self.max_per_gpu and count < best_count:
                    best_count = count
                    best_gpu = gpu
            return best_gpu

    def _release_gpu(self, job: Job):
        """Release GPU slot exactly once (prevents double-decrement)."""
        with self.lock:
            if not job.gpu_slot_released and job.gpu >= 0:
                self.gpu_running[job.gpu] -= 1
                job.gpu_slot_released = True
            job.end_time = time.time()

    def _run_job(self, job: Job):
        # Check for pending kill before even waiting for GPU
        if self.db and self.db.pending_kill(job.name):
            job.status = "failed"
            job.exit_code = -9
            if self.db:
                self.db.fail_job(job.name, error="killed before dispatch")
            return

        # Wait for a free GPU, yielding to higher-priority pending jobs
        while True:
            gpu = self._pick_gpu()
            if gpu is not None:
                # Yield if a higher-priority job is also waiting
                with self.lock:
                    higher = any(j.status == "pending" and j.priority < job.priority
                                 for j in self.jobs)
                if higher:
                    time.sleep(self.poll_interval)
                    continue
                break
            # Re-check kill while waiting
            if self.db and self.db.pending_kill(job.name):
                job.status = "failed"
                job.exit_code = -9
                if self.db:
                    self.db.fail_job(job.name, error="killed while waiting")
                return
            time.sleep(self.poll_interval)

        # Claim the GPU
        with self.lock:
            self.gpu_running[gpu] += 1
            job.gpu = gpu
            job.gpu_slot_released = False
            job.status = "running"
            job.start_time = time.time()
            job.last_output_time = time.time()
            job.log_file = os.path.join(self.log_dir, f"job_{job.job_id:03d}_{job.name[:30]}_gpu{gpu}.log")

        cmd = job.cmd.replace("{GPU}", str(gpu))
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] JOB {job.job_id:3d} GPU {gpu} START: {job.name} ({cmd[:60]}...)")
        if self.db:
            self.db.start_job(job.name, gpu=gpu)

        try:
            with open(job.log_file, "w") as logf:
                proc = subprocess.Popen(
                    cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                )
                job.proc = proc

                # Stream output to log file and track last output time
                with proc.stdout:
                    for line in iter(proc.stdout.readline, b""):
                        logf.write(line.decode("utf-8", errors="replace"))
                        logf.flush()
                        job.last_output_time = time.time()
                        if self.db:
                            self.db.heartbeat(job.name)

                        # Check for kill command from Redis
                        if self.db and self.db.pending_kill(job.name):
                            print(f"  Kill command received for {job.name}")
                            proc.kill()
                            proc.wait(timeout=10)
                            job.exit_code = -9
                            break

                if job.exit_code != -9:
                    proc.wait()
                    job.exit_code = proc.returncode
        except Exception as e:
            job.exit_code = -1
            try:
                with open(job.log_file, "a") as logf:
                    logf.write(f"\nEXCEPTION: {e}\n")
            except Exception:
                pass
        finally:
            # Always release GPU slot, even on unexpected exceptions
            self._release_gpu(job)

        elapsed = job.end_time - job.start_time
        ts = time.strftime("%H:%M:%S")

        if job.exit_code == 0:
            with self.lock:
                job.status = "done"
            print(f"[{ts}] JOB {job.job_id:3d} GPU {gpu} DONE ({elapsed:.0f}s) {job.name}")
            if self.db:
                self.db.complete_job(job.name)
            if job.on_complete:
                try:
                    job.on_complete(job)
                except Exception as e:
                    print(f"  on_complete callback error: {e}")
        else:
            with self.lock:
                job.status = "failed"
            print(f"[{ts}] JOB {job.job_id:3d} GPU {gpu} FAIL(exit={job.exit_code}) ({elapsed:.0f}s) {job.name}")
            if self.db:
                error_msg = ""
                try:
                    with open(job.log_file) as f:
                        error_msg = f.readlines()[-1].strip()[:200]
                except Exception:
                    pass
                self.db.fail_job(job.name, error=error_msg)

            # Check last few lines of log for error message
            try:
                with open(job.log_file) as f:
                    lines = f.readlines()
                    error_lines = [l.strip() for l in lines[-5:] if l.strip()]
                    if error_lines:
                        print(f"  Last output: {error_lines[-1][:100]}")
            except Exception:
                pass

            # Auto-retry
            if job.retries < job.max_retries:
                with self.lock:
                    job.retries += 1
                    job.status = "retrying"
                print(f"  Retrying ({job.retries}/{job.max_retries})...")
                with self.lock:
                    job.exit_code = -1
                    job.start_time = 0
                    job.end_time = 0
                t = threading.Thread(target=self._run_job, args=(job,), daemon=True)
                self._threads.append(t)
                t.start()
            else:
                if job.on_fail:
                    try:
                        job.on_fail(job)
                    except Exception as e:
                        print(f"  on_fail callback error: {e}")

        self._write_status()

    def _monitor_loop(self):
        """Periodic monitoring: detect stale jobs, check kill commands, write status."""
        while not self._stop_monitor.is_set():
            time.sleep(30)
            self._check_stale_jobs()
            self._check_kill_commands()
            self._write_status()

    def _dynamic_submit(self, cmd_str: str, name: str = ""):
        """Submit a new job to the running queue dynamically."""
        if not name:
            # Extract name from output_dir
            for part in cmd_str.split():
                if part.startswith("ckpt/sweep/"):
                    name = part.split("/")[-1]
                    break
            if not name:
                name = f"dynamic_{self._job_counter}"

        job_id = self.submit(cmd_str, name=name)
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] DYNAMIC SUBMIT: {name} (job {job_id})")

    def _check_kill_commands(self):
        """Check Redis for kill/modify/submit commands targeting QUEUE or ALL."""
        if not self.db:
            return

        # Check for QUEUE-level commands
        cmds = self.db.read_commands("QUEUE")
        for cmd in cmds:
            if cmd.get("command") == "kill":
                ts = time.strftime("%H:%M:%S")
                print(f"[{ts}] QUEUE KILL received — killing all running jobs")
                with self.lock:
                    for job in self.jobs:
                        if job.status == "running" and job.proc:
                            try:
                                job.proc.kill()
                                job.proc.wait(timeout=10)
                            except Exception:
                                pass
                            job.status = "killed"
                            self._release_gpu(job)
                            if self.db:
                                self.db.fail_job(job.name, error="killed by QUEUE command")
                self._stop_monitor.set()  # stop the queue
            elif cmd.get("command") == "modify":
                self._handle_modify(cmd)
            elif cmd.get("command") == "submit":
                payload = cmd.get("payload", {})
                self._dynamic_submit(payload.get("cmd", ""), name=payload.get("name", ""))

        # Check per-job modify commands for ALL targets
        all_cmds = self.db.read_commands("ALL")
        for cmd in all_cmds:
            if cmd.get("command") == "modify":
                self._handle_modify(cmd, target_all=True)

    def _handle_modify(self, cmd, target_all=False):
        """Modify pending job commands by replacing flag values."""
        payload = cmd.get("payload", {})
        flag = payload.get("flag", "")
        value = payload.get("value", "")
        target = cmd.get("target", "")
        if not flag:
            return

        ts = time.strftime("%H:%M:%S")
        with self.lock:
            for job in self.jobs:
                if job.status != "pending":
                    continue
                if not target_all and target not in ("QUEUE", "ALL") and job.name != target:
                    continue
                # Check if this flag exists in the command
                if flag in job.cmd:
                    import re
                    # Replace --flag <old_value> with --flag <new_value>
                    pattern = rf"({re.escape(flag)})\s+\S+"
                    new_cmd = re.sub(pattern, rf"\1 {value}", job.cmd)
                    if new_cmd != job.cmd:
                        print(f"[{ts}] MODIFY {job.name}: {flag} → {value}")
                        job.cmd = new_cmd

    def _check_stale_jobs(self):
        """Kill and requeue jobs that have gone silent."""
        with self.lock:
            for job in self.jobs:
                if job.status != "running":
                    continue
                if job.idle_time > job.stale_timeout:
                    ts = time.strftime("%H:%M:%S")
                    print(f"[{ts}] JOB {job.job_id:3d} STALE ({job.idle_time:.0f}s no output) — killing")
                    if job.proc:
                        try:
                            job.proc.kill()
                            job.proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            print(f"  WARNING: Job {job.name} did not exit after kill")
                        except Exception:
                            pass
                    job.status = "stale"
                    # Use _release_gpu to prevent double-decrement
                    if not job.gpu_slot_released and job.gpu >= 0:
                        self.gpu_running[job.gpu] -= 1
                        job.gpu_slot_released = True
                    job.end_time = time.time()

                    # Retry if possible
                    if job.retries < job.max_retries:
                        job.retries += 1
                        job.status = "retrying"
                        print(f"  Retrying ({job.retries}/{job.max_retries})...")
                        job.exit_code = -1
                        job.start_time = 0
                        job.end_time = 0
                        t = threading.Thread(target=self._run_job, args=(job,), daemon=True)
                        self._threads.append(t)
                        t.start()

    def _write_status(self):
        """Write machine-readable status JSON and upload to HuggingFace."""
        status = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total": len(self.jobs),
            "pending": sum(1 for j in self.jobs if j.status == "pending"),
            "running": sum(1 for j in self.jobs if j.status == "running"),
            "done": sum(1 for j in self.jobs if j.status == "done"),
            "failed": sum(1 for j in self.jobs if j.status == "failed"),
            "stale": sum(1 for j in self.jobs if j.status == "stale"),
            "retrying": sum(1 for j in self.jobs if j.status == "retrying"),
            "jobs": [j.to_dict() for j in self.jobs],
        }
        try:
            with open(self.status_file, "w") as f:
                json.dump(status, f, indent=2)
        except Exception:
            pass

        # Upload status to HuggingFace model repo
        try:
            from huggingface_hub import HfApi
            HfApi().upload_file(
                path_or_fileobj=self.status_file,
                path_in_repo="queue_status.json",
                repo_id="thoughtworks/arithmetic-sorl",
                repo_type="model",
            )
        except Exception:
            pass  # don't crash the queue on upload failure

    def wait(self):
        """Block until all submitted jobs finish."""
        for t in self._threads:
            t.join()
        self._stop_monitor.set()
        self._write_status()

    def get_status(self) -> dict:
        """Get current queue status."""
        return {
            "total": len(self.jobs),
            "done": sum(1 for j in self.jobs if j.status == "done"),
            "failed": sum(1 for j in self.jobs if j.status in ("failed", "stale")),
            "running": sum(1 for j in self.jobs if j.status == "running"),
            "pending": sum(1 for j in self.jobs if j.status == "pending"),
        }

    def print_summary(self):
        """Print human-readable summary."""
        s = self.get_status()
        print(f"\n{'='*60}")
        print(f"Queue: {s['total']} total | {s['done']} done | {s['failed']} failed | {s['running']} running | {s['pending']} pending")
        print(f"{'='*60}")
        for j in self.jobs:
            status = f"{'DONE' if j.status == 'done' else j.status.upper():8s}"
            elapsed = f"{j.elapsed:6.0f}s" if j.elapsed else "     -"
            retry = f" (retry {j.retries})" if j.retries else ""
            print(f"  {j.job_id:3d} GPU {j.gpu:1d} {status} {elapsed}  {j.name}{retry}")

        failed = [j for j in self.jobs if j.status in ("failed", "stale")]
        if failed:
            print(f"\n  FAILURES ({len(failed)}):")
            for j in failed:
                print(f"    {j.name}: exit={j.exit_code}, log={j.log_file}")
                try:
                    with open(j.log_file) as f:
                        lines = f.readlines()
                        last = [l.strip() for l in lines[-3:] if l.strip()]
                        for l in last:
                            print(f"      {l[:100]}")
                except Exception:
                    pass


def main():
    """Read jobs from a file and run them."""
    if len(sys.argv) < 2:
        print("Usage: python gpu_queue.py jobs.txt [n_gpus] [max_per_gpu] [--stale-timeout N] [--max-retries N]")
        sys.exit(1)

    jobs_file = sys.argv[1]
    n_gpus = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    max_per_gpu = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    # Parse optional flags
    stale_timeout = 1800
    max_retries = 1
    for i, arg in enumerate(sys.argv):
        if arg == "--stale-timeout" and i + 1 < len(sys.argv):
            stale_timeout = float(sys.argv[i + 1])
        if arg == "--max-retries" and i + 1 < len(sys.argv):
            max_retries = int(sys.argv[i + 1])

    # Parse jobs file with optional #PRIORITY: tags
    PRIORITY_MAP = {"HIGH": 0, "NORMAL": 1, "LOW": 2}
    jobs = []  # list of (cmd, priority)
    current_priority = 1  # default: NORMAL
    with open(jobs_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#PRIORITY:"):
                tag = line.split(":", 1)[1].strip().upper()
                current_priority = PRIORITY_MAP.get(tag, 1)
                continue
            if line.startswith("#"):
                continue
            jobs.append((line, current_priority))

    # Extract job names from output_dir
    def extract_name(cmd):
        for part in cmd.split():
            if part.startswith("ckpt/sweep/"):
                return part.split("/")[-1]
        return cmd.split("--mode")[-1].strip().split()[0] if "--mode" in cmd else "job"

    n_high = sum(1 for _, p in jobs if p == 0)
    n_low = sum(1 for _, p in jobs if p == 2)
    print(f"GPU Queue: {len(jobs)} jobs, {n_gpus} GPUs, max {max_per_gpu}/GPU")
    print(f"  priority: {n_high} high, {len(jobs) - n_high - n_low} normal, {n_low} low")
    print(f"  stale_timeout={stale_timeout}s, max_retries={max_retries}")
    print()

    q = GPUQueue(
        n_gpus=n_gpus, max_per_gpu=max_per_gpu,
        stale_timeout=stale_timeout, max_retries=max_retries,
    )
    for cmd, priority in jobs:
        q.submit(cmd, name=extract_name(cmd), priority=priority)
    q.wait()
    q.print_summary()


if __name__ == "__main__":
    main()
