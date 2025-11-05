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

def test_1_environment():
    """Test 1: Check environment variables"""
    print_all("\n" + "="*60)
    print_all("TEST 1: Environment Variables")
    print_all("="*60)
    required_vars = ["RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]
    for var in required_vars:
        val = os.environ.get(var, "NOT SET")
        print_all(f"  {var}: {val}")
    if any(os.environ.get(var) is None for var in required_vars):
        print_all("❌ FAILED: Missing required environment variables!")
        return False
    print_all("✅ PASSED: All environment variables present")
    return True

def test_2_cuda():
    """Test 2: Check CUDA availability"""
    print_all("\n" + "="*60)
    print_all("TEST 2: CUDA Availability")
    print_all("="*60)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    print_all(f"  PyTorch version: {torch.__version__}")
    print_all(f"  CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print_all("❌ FAILED: CUDA not available!")
        return False
    print_all(f"  CUDA version: {torch.version.cuda}")
    print_all(f"  Number of GPUs: {torch.cuda.device_count()}")
    try:
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        print_all(f"  Device name: {torch.cuda.get_device_name(device)}")
        print_all("✅ PASSED: CUDA initialized successfully")
        return True
    except Exception as e:
        print_all(f"❌ FAILED: CUDA initialization error: {e}")
        return False

def test_3_distributed_init():
    """Test 3: Initialize distributed process group"""
    print_all("\n" + "="*60)
    print_all("TEST 3: Distributed Initialization")
    print_all("="*60)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    try:
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        print_all(f"Initializing process group (backend=nccl) with device_id={local_rank}...")
        dist.init_process_group(
            backend="nccl", 
            device_id=local_rank,
            timeout=datetime.timedelta(seconds=120)
        )
        print_all(f"Init complete. Testing barrier...")
        dist.barrier()
        print_all(f"Barrier passed! Initialized! World size: {world_size}")
        print_all("✅ PASSED: Distributed initialized successfully")
        return True
    except Exception as e:
        print_all(f"❌ FAILED in test_3: {type(e).__name__}: {e}")
        import traceback
        print_all(traceback.format_exc())
        return False

def test_4_communication():
    """Test 4: Test distributed communication"""
    print_all("\n" + "="*60)
    print_all("TEST 4: Distributed Communication")
    print_all("="*60)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    try:
        tensor = torch.tensor([rank], dtype=torch.float32, device="cuda")
        dist.broadcast(tensor, src=0)
        print_all(f"After broadcast from rank 0: {tensor.item()}")
        tensor = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        expected = float(world_size)
        print_all(f"After all_reduce (sum): {tensor.item()} (expected: {expected})")
        if abs(tensor.item() - expected) < 1e-5:
            print_all("✅ PASSED: Communication working correctly")
            return True
        else:
            print_all(f"❌ FAILED: all_reduce gave {tensor.item()}, expected {expected}")
            return False
    except Exception as e:
        print_all(f"❌ FAILED: Communication error: {e}")
        import traceback
        print_all(traceback.format_exc())
        return False

def test_5_simple_model():
    """Test 5: Create and sync a simple model"""
    print_all("\n" + "="*60)
    print_all("TEST 5: Model Creation and Sync")
    print_all("="*60)
    rank = int(os.environ["RANK"])
    try:
        model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10)).cuda()
        for param in model.parameters():
            dist.broadcast(param.detach(), 0)
        checksum = sum(p.sum().item() for p in model.parameters())
        print_all(f"Parameter checksum: {checksum:.4f}")
        print_all("✅ PASSED: Model created and synced")
        return True
    except Exception as e:
        print_all(f"❌ FAILED: Model creation error: {e}")
        import traceback
        print_all(traceback.format_exc())
        return False

def test_6_forward_backward():
    """Test 6: Forward and backward pass"""
    print_all("\n" + "="*60)
    print_all("TEST 6: Forward and Backward Pass")
    print_all("="*60)
    rank = int(os.environ["RANK"])
    try:
        model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10)).cuda()
        for param in model.parameters():
            dist.broadcast(param.detach(), 0)
        x = torch.randn(4, 128, device="cuda")
        target = torch.randint(0, 10, (4,), device="cuda")
        output = model(x)
        loss = nn.functional.cross_entropy(output, target)
        print_all(f"Loss: {loss.item():.4f}")
        loss.backward()
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        print_all("✅ PASSED: Forward/backward working correctly")
        return True
    except Exception as e:
        print_all(f"❌ FAILED: Forward/backward error: {e}")
        import traceback
        print_all(traceback.format_exc())
        return False

def test_7_compile():
    """Test 7: Test torch.compile (optional)"""
    print_all("\n" + "="*60)
    print_all("TEST 7: torch.compile (Optional)")
    print_all("="*60)
    rank = int(os.environ["RANK"])
    try:
        model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10)).cuda()
        compiled_model = torch.compile(model, dynamic=True)
        x = torch.randn(4, 128, device="cuda")
        output = compiled_model(x)
        print_all(f"Compiled output shape: {output.shape}")
        print_all("✅ PASSED: torch.compile working")
        return True
    except Exception as e:
        print_all(f"⚠️  WARNING: torch.compile failed (may not be supported): {e}")
        return True

def main():
    """Run all tests sequentially"""
    print_all("\n" + "="*60)
    print_all("DISTRIBUTED TRAINING DEBUG SCRIPT")
    print_all("="*60)
    tests = [
        test_1_environment,
        test_2_cuda,
        test_3_distributed_init,
        test_4_communication,
        test_5_simple_model,
        test_6_forward_backward,
        test_7_compile,
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