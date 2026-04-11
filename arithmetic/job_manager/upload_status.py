"""
Uploads queue status to HF repo periodically so the dashboard can show live job info.

Run alongside the queue:
    nohup python -m arithmetic.job_manager.upload_status &

Uploads /tmp/gpu_queue/queue_status.json to thoughtworks/arithmetic-sorl as queue_status.json
every 2 minutes.
"""
import json
import time
import os
from huggingface_hub import HfApi

QUEUE_STATUS = "/tmp/gpu_queue/queue_status.json"
REPO = "thoughtworks/arithmetic-sorl"
INTERVAL = 120  # 2 minutes


def main():
    api = HfApi()
    print("Status uploader started.")

    while True:
        if os.path.exists(QUEUE_STATUS):
            try:
                api.upload_file(
                    path_or_fileobj=QUEUE_STATUS,
                    path_in_repo="queue_status.json",
                    repo_id=REPO,
                    commit_message="queue status update",
                )
            except Exception:
                pass  # transient HF errors, skip

            # Check if queue is done
            try:
                with open(QUEUE_STATUS) as f:
                    status = json.load(f)
                total = status.get("total", 0)
                done = status.get("done", 0)
                failed = status.get("failed", 0)
                if total > 0 and done + failed >= total:
                    print("Queue complete. Final upload done.")
                    break
            except Exception:
                pass

        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
