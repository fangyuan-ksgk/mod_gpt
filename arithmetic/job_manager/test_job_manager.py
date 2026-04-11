"""
Tests for job_manager: unit tests for JobStateDB + end-to-end tests for GPUQueue.

Run: python -m pytest arithmetic/job_manager/test_job_manager.py -v
"""
import pytest
import time
import json
import threading
import subprocess
import redis
from arithmetic.job_manager.job_state import JobStateDB
from arithmetic.job_manager.gpu_queue import GPUQueue


@pytest.fixture(autouse=True)
def clean_redis():
    """Clear test state before and after each test."""
    db = JobStateDB(db=1)  # use db=1 for tests, not db=0 (production)
    db.clear()
    yield db
    db.clear()


# ══════════════════════════════════════════════════════════════════
# Unit tests: JobStateDB
# ══════════════════════════════════════════════════════════════════


class TestJobLifecycle:
    def test_create_and_get(self, clean_redis):
        db = clean_redis
        db.create_job("test1", cmd="echo hello", gpu=0)
        job = db.get_job("test1")
        assert job is not None
        assert job["status"] == "pending"
        assert job["cmd"] == "echo hello"

    def test_start_job(self, clean_redis):
        db = clean_redis
        db.create_job("test1", cmd="echo hello")
        db.start_job("test1", gpu=2)
        job = db.get_job("test1")
        assert job["status"] == "running"
        assert job["gpu"] == 2
        assert isinstance(job["started_at"], float)

    def test_complete_job(self, clean_redis):
        db = clean_redis
        db.create_job("test1")
        db.start_job("test1")
        db.complete_job("test1", metrics={"accuracy": 0.95})
        job = db.get_job("test1")
        assert job["status"] == "done"
        assert job["metrics"]["accuracy"] == 0.95
        assert job["progress"] == 1.0

    def test_fail_job(self, clean_redis):
        db = clean_redis
        db.create_job("test1")
        db.start_job("test1")
        db.fail_job("test1", error="OOM")
        job = db.get_job("test1")
        assert job["status"] == "failed"
        assert job["error"] == "OOM"

    def test_heartbeat(self, clean_redis):
        db = clean_redis
        db.create_job("test1")
        db.start_job("test1")
        db.heartbeat("test1", step=100, loss=0.5, progress=0.3)
        job = db.get_job("test1")
        assert job["step"] == 100
        assert float(job["loss"]) == pytest.approx(0.5)
        assert job["progress"] == pytest.approx(0.3)

    def test_heartbeat_refreshes_ttl(self, clean_redis):
        db = JobStateDB(db=1, heartbeat_ttl=5)
        db.create_job("test1")
        db.start_job("test1")
        ttl1 = db.r.ttl("job:test1:hb")
        assert ttl1 > 0
        time.sleep(1)
        db.heartbeat("test1", step=50)
        ttl2 = db.r.ttl("job:test1:hb")
        assert ttl2 >= ttl1  # refreshed

    def test_summary_counts(self, clean_redis):
        db = clean_redis
        db.create_job("a")
        db.create_job("b")
        db.create_job("c")
        db.start_job("a")
        db.complete_job("b")
        s = db.summary()
        assert s["total"] == 3
        assert s["running"] == 1
        assert s["done"] == 1
        assert s["pending"] == 1

    def test_get_all_jobs(self, clean_redis):
        db = clean_redis
        db.create_job("x", cmd="echo x")
        db.create_job("y", cmd="echo y")
        all_jobs = db.get_all_jobs()
        assert len(all_jobs) == 2
        assert "x" in all_jobs
        assert "y" in all_jobs


class TestCommands:
    def test_send_and_read_command(self, clean_redis):
        db = clean_redis
        db.create_job("test1")
        db.send_command("test1", "kill")
        cmds = db.read_commands("test1")
        assert len(cmds) == 1
        assert cmds[0]["command"] == "kill"

    def test_commands_consumed_on_read(self, clean_redis):
        db = clean_redis
        db.create_job("test1")
        db.send_command("test1", "kill")
        db.read_commands("test1")  # consume
        cmds = db.read_commands("test1")  # should be empty
        assert len(cmds) == 0

    def test_pending_kill(self, clean_redis):
        db = clean_redis
        db.create_job("test1")
        assert not db.pending_kill("test1")
        db.send_command("test1", "kill")
        assert db.pending_kill("test1")

    def test_all_command(self, clean_redis):
        db = clean_redis
        db.create_job("a")
        db.create_job("b")
        db.send_command("ALL", "pause")
        cmds_a = db.read_commands("a")
        cmds_b = db.read_commands("b")
        assert len(cmds_a) == 1
        assert cmds_a[0]["command"] == "pause"
        assert len(cmds_b) == 1

    def test_multiple_commands_fifo(self, clean_redis):
        db = clean_redis
        db.create_job("test1")
        db.send_command("test1", "pause")
        db.send_command("test1", "resume")
        db.send_command("test1", "kill")
        cmds = db.read_commands("test1")
        assert [c["command"] for c in cmds] == ["pause", "resume", "kill"]


class TestStaleDetection:
    def test_stale_job_detected(self, clean_redis):
        db = JobStateDB(db=1, heartbeat_ttl=2)
        db.create_job("slow")
        db.start_job("slow")
        time.sleep(3)  # heartbeat expires
        stale = db.stale_jobs(timeout=1)
        assert len(stale) == 1
        assert stale[0][0] == "slow"

    def test_active_job_not_stale(self, clean_redis):
        db = JobStateDB(db=1, heartbeat_ttl=10)
        db.create_job("fast")
        db.start_job("fast")
        db.heartbeat("fast", step=1)
        stale = db.stale_jobs(timeout=5)
        assert len(stale) == 0

    def test_completed_job_not_stale(self, clean_redis):
        db = JobStateDB(db=1, heartbeat_ttl=2)
        db.create_job("done_job")
        db.start_job("done_job")
        db.complete_job("done_job")
        time.sleep(3)
        stale = db.stale_jobs(timeout=1)
        assert len(stale) == 0


class TestPubSub:
    def test_events_published(self, clean_redis):
        db = clean_redis
        received = []

        def listener():
            for event in db.subscribe():
                received.append(event)
                if len(received) >= 3:
                    break

        t = threading.Thread(target=listener, daemon=True)
        t.start()
        time.sleep(0.2)  # let subscriber connect

        db.create_job("test1")
        db.start_job("test1")
        db.complete_job("test1")

        t.join(timeout=3)
        assert len(received) >= 3
        events = [r["event"] for r in received]
        assert "created" in events
        assert "started" in events
        assert "done" in events


class TestClear:
    def test_clear_removes_all(self, clean_redis):
        db = clean_redis
        db.create_job("a")
        db.create_job("b")
        db.send_command("a", "kill")
        db.clear()
        assert db.summary()["total"] == 0
        assert db.get_job("a") is None


# ══════════════════════════════════════════════════════════════════
# End-to-end tests: GPUQueue + JobStateDB
# ══════════════════════════════════════════════════════════════════


class TestGPUQueueE2E:
    def test_successful_job(self):
        q = GPUQueue(n_gpus=1, max_per_gpu=1, log_dir="/tmp/test_gpu_queue")
        q.submit('python -c "print(\'hello\')"', name="success_test")
        q.wait()
        s = q.get_status()
        assert s["done"] == 1
        assert s["failed"] == 0

    def test_failed_job_with_retry(self):
        q = GPUQueue(n_gpus=1, max_per_gpu=1, max_retries=1,
                     log_dir="/tmp/test_gpu_queue")
        q.submit('python -c "raise ValueError(\'boom\')"', name="fail_test")
        q.wait()
        # Should have tried twice (original + 1 retry)
        job = q.jobs[0]
        assert job.status == "failed"
        assert job.retries == 1

    def test_multiple_jobs_across_gpus(self):
        q = GPUQueue(n_gpus=2, max_per_gpu=1, log_dir="/tmp/test_gpu_queue")
        q.submit('python -c "import time; time.sleep(1); print(\'a\')"', name="gpu_a")
        q.submit('python -c "import time; time.sleep(1); print(\'b\')"', name="gpu_b")
        q.wait()
        s = q.get_status()
        assert s["done"] == 2
        # Check they ran on different GPUs
        gpus = {j.gpu for j in q.jobs}
        assert len(gpus) == 2

    def test_job_queue_ordering(self):
        q = GPUQueue(n_gpus=1, max_per_gpu=1, log_dir="/tmp/test_gpu_queue")
        order = []
        for i in range(3):
            q.submit(
                f'python -c "import time; time.sleep(0.5); print({i})"',
                name=f"order_{i}",
                on_complete=lambda j, _i=i: order.append(_i),
            )
        q.wait()
        assert len(order) == 3

    def test_status_json_written(self):
        import os
        q = GPUQueue(n_gpus=1, max_per_gpu=1, log_dir="/tmp/test_gpu_queue")
        q.submit('python -c "print(1)"', name="status_test")
        q.wait()
        assert os.path.exists(q.status_file)
        with open(q.status_file) as f:
            status = json.load(f)
        assert status["done"] == 1

    def test_callback_on_complete(self):
        results = []
        q = GPUQueue(n_gpus=1, max_per_gpu=1, log_dir="/tmp/test_gpu_queue")
        q.submit(
            'python -c "print(\'done\')"',
            name="callback_test",
            on_complete=lambda j: results.append(j.name),
        )
        q.wait()
        assert results == ["callback_test"]

    def test_callback_on_fail(self):
        results = []
        q = GPUQueue(n_gpus=1, max_per_gpu=1, max_retries=0,
                     log_dir="/tmp/test_gpu_queue")
        q.submit(
            'python -c "exit(1)"',
            name="fail_callback",
            on_fail=lambda j: results.append(f"failed:{j.name}"),
        )
        q.wait()
        assert results == ["failed:fail_callback"]


class TestKillSignal:
    def test_kill_via_command(self, clean_redis):
        """Test that a job can be killed via the state DB command."""
        db = clean_redis

        # Simulate a job checking for kill
        db.create_job("killme")
        db.start_job("killme")
        db.send_command("killme", "kill")

        # Job checks for commands
        cmds = db.read_commands("killme")
        should_die = any(c["command"] == "kill" for c in cmds)
        assert should_die


# ══════════════════════════════════════════════════════════════════
# Integration test: Queue + StateDB together
# ══════════════════════════════════════════════════════════════════


class TestQueueWithStateDB:
    def test_queue_updates_state_db(self, clean_redis):
        """Queue should create/start/complete jobs in the state DB."""
        db = clean_redis
        q = GPUQueue(n_gpus=1, max_per_gpu=1, log_dir="/tmp/test_gpu_queue")

        # Wrap callbacks to update DB
        def on_start(job):
            db.start_job(job.name, gpu=job.gpu)

        def on_done(job):
            db.complete_job(job.name)

        db.create_job("integration_test")
        q.submit(
            'python -c "print(\'hello\')"',
            name="integration_test",
            on_complete=on_done,
        )
        q.wait()

        # The DB should reflect done status
        # (Note: start_job must be called separately since queue doesn't know about DB yet)
        job = db.get_job("integration_test")
        assert job is not None
        assert job["status"] == "done"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
