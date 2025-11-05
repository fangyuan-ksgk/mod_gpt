# Distributed run (debug on Nvidia Pod)
# -------------------------------------

# Distributed Training Debug Script with Rank-Specific Logging
# -----------------------------------------------------------------------------
import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import time
import datetime

# --- IMPORTANT: Redirect output to rank-specific log files ---
# This will capture the actual error from the child process that torchrun hides.
try:
    rank = int(os.environ.get("RANK", -1))
    log_dir = "/workspace/mod_gpt"  # Assumes this path is writable
    if not os.path.exists(log_dir) and rank == 0:
        os.makedirs(log_dir, exist_ok=True)
    
    # Wait for rank 0 to create directory
    time.sleep(0.1) 
    
    log_file = os.path.join(log_dir, f"rank_{rank}.log")
    print(f"[Launcher] Redirecting stdout/stderr for rank {rank} to: {log_file}")

    # Redirect stdout and stderr
    f = open(log_file, 'w')
    sys.stdout = f
    sys.stderr = f
except Exception as e:
    # Print to original stdout if redirection fails
    print(f"FATAL: Could not redirect output for rank {rank}. Error: {e}", file=sys.__stdout__)
    sys.exit(1)
# -------------------------------------------------------------------

def print_all(msg, rank=None):
    """Print from all ranks with rank prefix to the log file."""
    if rank is None:
        rank = int(os.environ.get("RANK", -1))
    timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')
    print(f"[{timestamp}] [Rank {rank}] {msg}", flush=True)

import datetime

def test_3_distributed_init():
    """Test 3: Initialize distributed process group using NCCL"""
    print_all("\n" + "="*60)
    print_all("TEST 3: Distributed Initialization (NCCL)")
    print_all("="*60)
    
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    try:
        # This is the most critical step. It tells this process which GPU to use.
        # All subsequent CUDA operations in this process will default to this device.
        torch.cuda.set_device(local_rank)
        print_all(f"Step 1: Set active CUDA device to local_rank:{local_rank}")

        # This is the simplest possible init call. PyTorch and NCCL should
        # automatically detect the correct GPU from the line above without
        # needing the `device_id` argument.
        print_all(f"Step 2: Initializing process group (backend=nccl)...")
        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(seconds=120)
        )

        print_all(f"Step 3: Init complete. Waiting at barrier...")
        dist.barrier()
        
        print_all(f"Step 4: Barrier passed! World size: {world_size}")
        print_all("✅ PASSED: Distributed initialization successful!")
        return True
        
    except Exception as e:
        print_all(f"❌ FAILED in test_3: {type(e).__name__}: {e}")
        import traceback
        print_all(traceback.format_exc())
        return False

def main():
    """Run all tests sequentially"""
    print_all("\n" + "="*60)
    print_all("DISTRIBUTED TRAINING DEBUG SCRIPT")
    print_all("="*60)
    tests = [
        test_3_distributed_init,
    ]
    for i, test in enumerate(tests, 1):
        passed = test()
        if not passed:
            print_all(f"\n❌ Test {i} ({test.__name__}) failed. Stopping here.")
            break
    if dist.is_initialized():
        dist.destroy_process_group()
    print_all("\nDebug session complete!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print_all(f"\n\n❌ UNCAUGHT FATAL ERROR: {e}")
        import traceback
        print_all(traceback.format_exc())
        if dist.is_initialized():
            dist.destroy_process_group()
        sys.exit(1)