# Distributed run (debug on Nvidia Pod)
# -------------------------------------

# Distributed Training Debug Script
# Tests each component step-by-step with verbose logging
# -----------------------------------------------------------------------------

import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import time

def print_all(msg, rank=None):
    """Print from all ranks with rank prefix"""
    if rank is None:
        rank = int(os.environ.get("RANK", -1))
    print(f"[Rank {rank}] {msg}", flush=True)

def test_1_environment():
    """Test 1: Check environment variables"""
    print("\n" + "="*60)
    print("TEST 1: Environment Variables")
    print("="*60)
    
    required_vars = ["RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]
    for var in required_vars:
        val = os.environ.get(var, "NOT SET")
        print(f"  {var}: {val}")
    
    if any(os.environ.get(var) is None for var in required_vars):
        print("❌ FAILED: Missing required environment variables!")
        print("   Run with: torchrun --standalone --nproc_per_node=N script.py")
        return False
    
    print("✅ PASSED: All environment variables present")
    return True

def test_2_cuda():
    """Test 2: Check CUDA availability"""
    print("\n" + "="*60)
    print("TEST 2: CUDA Availability")
    print("="*60)
    
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("❌ FAILED: CUDA not available!")
        return False
    
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  Number of GPUs: {torch.cuda.device_count()}")
    print(f"  Local rank: {local_rank}")
    
    try:
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        print(f"  Device name: {torch.cuda.get_device_name(device)}")
        print(f"  Device properties: {torch.cuda.get_device_properties(device)}")
        print("✅ PASSED: CUDA initialized successfully")
        return True
    except Exception as e:
        print(f"❌ FAILED: CUDA initialization error: {e}")
        return False

def test_3_distributed_init():
    """Test 3: Initialize distributed process group"""
    print("\n" + "="*60)
    print("TEST 3: Distributed Initialization")
    print("="*60)
    
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    try:
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        
        print_all(f"Initializing process group (backend=nccl)...", rank)
        dist.init_process_group(backend="nccl", device_id=device)
        
        print_all(f"Barrier test...", rank)
        dist.barrier()
        
        print_all(f"Initialized! World size: {world_size}", rank)
        print("✅ PASSED: Distributed initialized successfully")
        return True
    except Exception as e:
        print(f"❌ FAILED: Distributed initialization error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_4_communication():
    """Test 4: Test distributed communication"""
    print("\n" + "="*60)
    print("TEST 4: Distributed Communication")
    print("="*60)
    
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    try:
        # Test broadcast
        print_all("Testing broadcast...", rank)
        tensor = torch.tensor([rank], dtype=torch.float32, device="cuda")
        dist.broadcast(tensor, src=0)
        print_all(f"After broadcast from rank 0: {tensor.item()}", rank)
        
        # Test all_reduce
        print_all("Testing all_reduce...", rank)
        tensor = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        expected = float(world_size)
        print_all(f"After all_reduce (sum): {tensor.item()} (expected: {expected})", rank)
        
        if abs(tensor.item() - expected) < 1e-5:
            print("✅ PASSED: Communication working correctly")
            return True
        else:
            print(f"❌ FAILED: all_reduce gave {tensor.item()}, expected {expected}")
            return False
            
    except Exception as e:
        print(f"❌ FAILED: Communication error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_5_simple_model():
    """Test 5: Create and sync a simple model"""
    print("\n" + "="*60)
    print("TEST 5: Model Creation and Sync")
    print("="*60)
    
    rank = int(os.environ["RANK"])
    
    try:
        # Simple 2-layer MLP
        model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        ).cuda()
        
        print_all(f"Model created, {sum(p.numel() for p in model.parameters())} params", rank)
        
        # Broadcast parameters from rank 0
        for param in model.parameters():
            dist.broadcast(param.detach(), 0)
        
        # Verify all ranks have same parameters
        checksum = sum(p.sum().item() for p in model.parameters())
        print_all(f"Parameter checksum: {checksum:.4f}", rank)
        
        print("✅ PASSED: Model created and synced")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: Model creation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_6_forward_backward():
    """Test 6: Forward and backward pass"""
    print("\n" + "="*60)
    print("TEST 6: Forward and Backward Pass")
    print("="*60)
    
    rank = int(os.environ["RANK"])
    
    try:
        model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        ).cuda()
        
        # Sync model
        for param in model.parameters():
            dist.broadcast(param.detach(), 0)
        
        # Create random input (different per rank)
        x = torch.randn(4, 128, device="cuda")
        target = torch.randint(0, 10, (4,), device="cuda")
        
        print_all("Forward pass...", rank)
        output = model(x)
        loss = nn.functional.cross_entropy(output, target)
        print_all(f"Loss: {loss.item():.4f}", rank)
        
        print_all("Backward pass...", rank)
        loss.backward()
        
        # Check gradients
        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        print_all(f"Gradient norm: {grad_norm:.4f}", rank)
        
        print_all("All-reduce gradients...", rank)
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        
        reduced_grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        print_all(f"Reduced gradient norm: {reduced_grad_norm:.4f}", rank)
        
        print("✅ PASSED: Forward/backward working correctly")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: Forward/backward error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_7_compile():
    """Test 7: Test torch.compile (optional)"""
    print("\n" + "="*60)
    print("TEST 7: torch.compile (Optional)")
    print("="*60)
    
    rank = int(os.environ["RANK"])
    
    try:
        model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        ).cuda()
        
        print_all("Compiling model...", rank)
        compiled_model = torch.compile(model, dynamic=True)
        
        # Test compiled model
        x = torch.randn(4, 128, device="cuda")
        output = compiled_model(x)
        print_all(f"Compiled output shape: {output.shape}", rank)
        
        print("✅ PASSED: torch.compile working")
        return True
        
    except Exception as e:
        print(f"⚠️  WARNING: torch.compile failed (may not be supported): {e}")
        return True  # Don't fail on this

def main():
    """Run all tests sequentially"""
    print("\n" + "="*60)
    print("DISTRIBUTED TRAINING DEBUG SCRIPT")
    print("="*60)
    
    tests = [
        test_1_environment,
        test_2_cuda,
        test_3_distributed_init,
        test_4_communication,
        test_5_simple_model,
        test_6_forward_backward,
        test_7_compile,
    ]
    
    results = []
    for i, test in enumerate(tests, 1):
        try:
            passed = test()
            results.append((test.__name__, passed))
            if not passed and i < 7:  # Don't stop on compile failure
                print(f"\n❌ Test {i} failed. Stopping here.")
                break
        except Exception as e:
            print(f"\n❌ Test {i} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test.__name__, False))
            break
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
    
    # Cleanup
    if dist.is_initialized():
        print("\nCleaning up...")
        dist.destroy_process_group()
    
    print("\nDebug session complete!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        print(f"\n\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        if dist.is_initialized():
            dist.destroy_process_group()
        sys.exit(1)