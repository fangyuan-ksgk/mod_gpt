"""
Job state DB backed by Redis. Provides communication between queue, jobs, and orchestrator.

Features:
  - Atomic updates (no file locking issues)
  - Pub/sub for real-time notifications (job done, job failed, command sent)
  - TTL on heartbeats for automatic stale detection
  - CLI for quick status checks and sending commands

Channels:
  - job:events — publishes all job state changes (start, done, fail, heartbeat)
  - job:commands — publishes commands (kill, pause, retry)

Key schema:
  - job:{name}        — hash with job state
  - job:{name}:hb     — heartbeat key with TTL (auto-expires when stale)
  - jobs:index        — set of all job names
  - commands:{name}   — list of pending commands for a job

Usage:
    from arithmetic.scripts.job_state import JobStateDB

    db = JobStateDB()

    # Queue creates jobs
    db.create_job("baseline_25K", cmd="python train.py ...", gpu=0)
    db.start_job("baseline_25K", gpu=0)

    # Job reports progress
    db.heartbeat("baseline_25K", step=500, loss=0.03, progress=0.5)

    # Orchestrator sends kill
    db.send_command("baseline_25K", "kill")

    # Job checks for commands
    cmds = db.read_commands("baseline_25K")  # returns and consumes

    # Anyone queries state
    db.print_status()
    stale = db.stale_jobs(timeout=1800)

    # Subscribe to events (blocking)
    for event in db.subscribe():
        print(event)  # {"job": "baseline_25K", "event": "done", ...}
"""
import json
import time
import redis
import sys
from typing import Optional


class JobStateDB:
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0,
                 heartbeat_ttl: int = 1800):
        self.r = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.heartbeat_ttl = heartbeat_ttl
        self.r.ping()  # fail fast if redis not available

    # ── Job lifecycle ─────────────────────────────────────────────

    def create_job(self, name: str, cmd: str = "", gpu: int = -1, **extra):
        key = f"job:{name}"
        mapping = {
            "status": "pending",
            "cmd": cmd,
            "gpu": gpu,
            "created_at": time.time(),
            "started_at": "",
            "completed_at": "",
            "progress": 0.0,
            "step": 0,
            "loss": "",
            "error": "",
            "retries": 0,
            "metrics": "{}",
            **{k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in extra.items()},
        }
        with self.r.pipeline() as pipe:
            pipe.hset(key, mapping=mapping)
            pipe.sadd("jobs:index", name)
            pipe.execute()
        self._publish("created", name)

    def start_job(self, name: str, gpu: int = -1):
        key = f"job:{name}"
        now = time.time()
        self.r.hset(key, mapping={
            "status": "running",
            "gpu": gpu,
            "started_at": now,
        })
        # Set heartbeat with TTL
        self.r.setex(f"job:{name}:hb", self.heartbeat_ttl, now)
        self._publish("started", name, gpu=gpu)

    def heartbeat(self, name: str, step: int = None, loss: float = None,
                  progress: float = None, **extra):
        """Called by the job to report liveness + progress."""
        key = f"job:{name}"
        update = {}
        if step is not None:
            update["step"] = step
        if loss is not None:
            update["loss"] = loss
        if progress is not None:
            update["progress"] = progress
        for k, v in extra.items():
            update[k] = json.dumps(v) if isinstance(v, (dict, list)) else str(v)
        if update:
            self.r.hset(key, mapping=update)
        # Refresh heartbeat TTL
        self.r.setex(f"job:{name}:hb", self.heartbeat_ttl, time.time())

    def complete_job(self, name: str, metrics: dict = None):
        key = f"job:{name}"
        self.r.hset(key, mapping={
            "status": "done",
            "completed_at": time.time(),
            "progress": 1.0,
            "metrics": json.dumps(metrics or {}),
        })
        self.r.delete(f"job:{name}:hb")  # no longer needs heartbeat
        self._publish("done", name, metrics=metrics)

    def fail_job(self, name: str, error: str = ""):
        key = f"job:{name}"
        self.r.hset(key, mapping={
            "status": "failed",
            "completed_at": time.time(),
            "error": error,
        })
        self.r.delete(f"job:{name}:hb")
        self._publish("failed", name, error=error)

    def update_job(self, name: str, **fields):
        key = f"job:{name}"
        mapped = {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in fields.items()}
        self.r.hset(key, mapping=mapped)

    # ── Commands (orchestrator → jobs/queue) ──────────────────────

    def send_command(self, target: str, command: str, **payload):
        """
        target: job name, "ALL", or "QUEUE"
        command: "kill", "pause", "resume", "retry", "priority"
        """
        msg = json.dumps({
            "target": target,
            "command": command,
            "payload": payload,
            "sent_at": time.time(),
        })
        if target == "ALL":
            # Push to all running jobs
            for name in self.r.smembers("jobs:index"):
                self.r.rpush(f"commands:{name}", msg)
        else:
            self.r.rpush(f"commands:{target}", msg)
        # Also publish for subscribers
        self.r.publish("job:commands", msg)

    def read_commands(self, name: str) -> list:
        """Pop all pending commands for a job (atomic consume)."""
        commands = []
        while True:
            msg = self.r.lpop(f"commands:{name}")
            if msg is None:
                break
            commands.append(json.loads(msg))
        return commands

    def pending_kill(self, name: str) -> bool:
        """Non-destructive peek for kill command."""
        msgs = self.r.lrange(f"commands:{name}", 0, -1)
        return any(json.loads(m).get("command") == "kill" for m in msgs)

    # ── Pub/Sub ───────────────────────────────────────────────────

    def _publish(self, event: str, name: str, **extra):
        msg = json.dumps({"job": name, "event": event, "time": time.time(), **extra})
        self.r.publish("job:events", msg)

    def subscribe(self):
        """Yields events as they happen. Blocking iterator."""
        pubsub = self.r.pubsub()
        pubsub.subscribe("job:events", "job:commands")
        try:
            for message in pubsub.listen():
                if message["type"] == "message":
                    yield json.loads(message["data"])
        finally:
            pubsub.close()

    # ── Queries ───────────────────────────────────────────────────

    def get_job(self, name: str) -> Optional[dict]:
        data = self.r.hgetall(f"job:{name}")
        if not data:
            return None
        # Parse numeric fields
        for k in ("created_at", "started_at", "completed_at", "progress", "loss"):
            if k in data and data[k]:
                try:
                    data[k] = float(data[k])
                except (ValueError, TypeError):
                    pass
        for k in ("gpu", "step", "retries"):
            if k in data and data[k]:
                try:
                    data[k] = int(data[k])
                except (ValueError, TypeError):
                    pass
        if "metrics" in data:
            try:
                data["metrics"] = json.loads(data["metrics"])
            except (json.JSONDecodeError, TypeError):
                pass
        return data

    def get_all_jobs(self) -> dict:
        names = self.r.smembers("jobs:index")
        return {name: self.get_job(name) for name in sorted(names) if self.get_job(name)}

    def stale_jobs(self, timeout: float = None) -> list:
        """Find running jobs whose heartbeat key has expired (or is close)."""
        timeout = timeout or self.heartbeat_ttl
        stale = []
        for name in self.r.smembers("jobs:index"):
            job = self.get_job(name)
            if not job or job.get("status") != "running":
                continue
            # If heartbeat key expired, job is stale
            if not self.r.exists(f"job:{name}:hb"):
                stale.append((name, timeout))  # at least timeout seconds stale
            else:
                ttl = self.r.ttl(f"job:{name}:hb")
                idle = self.heartbeat_ttl - ttl
                if idle > timeout:
                    stale.append((name, idle))
        return stale

    def summary(self) -> dict:
        jobs = self.get_all_jobs()
        counts = {"total": len(jobs)}
        for status in ["pending", "running", "done", "failed"]:
            counts[status] = sum(1 for j in jobs.values() if j.get("status") == status)
        return counts

    def print_status(self):
        jobs = self.get_all_jobs()
        s = self.summary()
        print(f"Jobs: {s['total']} total | {s['done']} done | {s['failed']} failed | {s['running']} running | {s['pending']} pending")
        if not jobs:
            print("  (no jobs)")
            return
        print()
        for name, job in sorted(jobs.items()):
            status = job.get("status", "?").upper()
            gpu = job.get("gpu", -1)
            progress = job.get("progress", 0)
            if isinstance(progress, str):
                progress = float(progress) if progress else 0
            step = job.get("step", 0)
            loss = job.get("loss", "")

            elapsed = ""
            started = job.get("started_at")
            if started and isinstance(started, (int, float)) and started > 0:
                end = job.get("completed_at")
                if isinstance(end, (int, float)) and end > 0:
                    elapsed = f"{end - started:.0f}s"
                else:
                    elapsed = f"{time.time() - started:.0f}s"

            loss_str = f"loss={float(loss):.4f}" if loss and loss != "" else ""
            print(f"  {name:40s} {status:8s} GPU {gpu}  {float(progress):5.0%}  step={step:>6}  {elapsed:>8s}  {loss_str}")

            error = job.get("error", "")
            if error:
                print(f"    ERROR: {error[:100]}")

    def clear(self):
        """Remove all job state. Use carefully."""
        for name in self.r.smembers("jobs:index"):
            self.r.delete(f"job:{name}", f"job:{name}:hb", f"commands:{name}")
        self.r.delete("jobs:index")


if __name__ == "__main__":
    db = JobStateDB()

    if len(sys.argv) < 2:
        db.print_status()
    elif sys.argv[1] == "kill":
        target = sys.argv[2] if len(sys.argv) > 2 else "ALL"
        db.send_command(target, "kill")
        print(f"Kill signal sent to {target}")
    elif sys.argv[1] == "submit":
        # Submit a new job to the running queue:
        #   job_state.py submit "python -m arithmetic.train ..." [--name my_job]
        if len(sys.argv) < 3:
            print("Usage: job_state.py submit \"<full command>\" [--name <name>]")
            sys.exit(1)
        cmd_str = sys.argv[2]
        name = ""
        for i, arg in enumerate(sys.argv):
            if arg == "--name" and i + 1 < len(sys.argv):
                name = sys.argv[i + 1]
        db.send_command("QUEUE", "submit", cmd=cmd_str, name=name)
        print(f"Submit command sent: {name or '(auto-name)'}")
    elif sys.argv[1] == "modify":
        # Modify a pending job's command: job_state.py modify <name> <new_cmd_fragment>
        # e.g.: job_state.py modify add_sub_baseline_10K --num_epochs 20
        # Replaces matching flag values in the stored command
        if len(sys.argv) < 4:
            print("Usage: job_state.py modify <name|ALL_BASELINE> <flag> <value>")
            sys.exit(1)
        target = sys.argv[2]
        flag = sys.argv[3]
        value = sys.argv[4] if len(sys.argv) > 4 else ""
        db.send_command(target, "modify", flag=flag, value=value)
        print(f"Modify command sent: {target} {flag} {value}")
    elif sys.argv[1] == "stale":
        timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 1800
        stale = db.stale_jobs(timeout)
        for name, idle in stale:
            print(f"  STALE: {name} ({idle:.0f}s idle)")
        if not stale:
            print("No stale jobs")
    elif sys.argv[1] == "summary":
        print(json.dumps(db.summary(), indent=2))
    elif sys.argv[1] == "clear":
        db.clear()
        print("Cleared all job state")
    elif sys.argv[1] == "listen":
        print("Listening for events (Ctrl+C to stop)...")
        for event in db.subscribe():
            print(f"  [{event.get('event', '?')}] {event.get('job', '?')}: {json.dumps({k:v for k,v in event.items() if k not in ('event','job','time')})}")
    else:
        print("Usage: python job_state.py [kill [name|ALL]] [stale [timeout]] [summary] [clear] [listen]")
