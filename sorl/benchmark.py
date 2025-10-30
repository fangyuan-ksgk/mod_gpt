# --- Benchmark Speed & Memory Cost --- 

import torch
import time
from sorl.gat_sim import GAT, GATConfig, recursion
import numpy as np

# ============================================================================
# Utility Functions
# ============================================================================

class GPUMemoryMonitor:
    """Track GPU memory usage for a code block"""
    def __enter__(self):
        torch.cuda.reset_peak_memory_stats()
        self.start_mem = torch.cuda.memory_allocated()
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        torch.cuda.synchronize()
        self.end_time = time.time()
        self.end_mem = torch.cuda.memory_allocated()
        self.peak_mem = torch.cuda.max_memory_allocated()
        
        self.time_elapsed = self.end_time - self.start_time
        self.memory_used = (self.end_mem - self.start_mem) / 1024**3  # GB
        self.peak_memory = (self.peak_mem - self.start_mem) / 1024**3  # GB

def print_stats(name, times, memories, peak_memories=None):
    """Print statistics for a benchmark"""
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"Time:   {np.mean(times)*1000:.2f} ± {np.std(times)*1000:.2f} ms")
    print(f"Memory: {np.mean(memories):.3f} ± {np.std(memories):.3f} GB")
    if peak_memories:
        print(f"Peak:   {np.mean(peak_memories):.3f} ± {np.std(peak_memories):.3f} GB")

# ============================================================================
# Benchmark Functions
# ============================================================================
def benchmark_forward_no_grad(model, tokens, memory_span, num_runs=10, warmup=3):
    """Benchmark: Forward pass without gradient tracking"""
    print("\n[1/5] Benchmarking: Forward (no grad)")
    
    times, memories, peaks = [], [], []
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(tokens, memory_span=memory_span)
    
    torch.cuda.empty_cache()  # Clean before benchmark
    
    # Benchmark
    with torch.no_grad():
        for i in range(num_runs):
            torch.cuda.empty_cache()
            with GPUMemoryMonitor() as mon:
                loss, logits = model(tokens, memory_span=memory_span)
                # Explicitly delete to free memory
                del loss, logits
            
            times.append(mon.time_elapsed)
            memories.append(mon.memory_used)
            peaks.append(mon.peak_memory)
            print(f"  Run {i+1}/{num_runs}: {mon.time_elapsed*1000:.1f}ms", end='\r')
    
    torch.cuda.empty_cache()  # Clean after benchmark
    print_stats("Forward (no grad)", times, memories, peaks)
    return times, memories, peaks


def benchmark_forward_with_grad(model, tokens, memory_span, num_runs=10, warmup=3):
    """Benchmark: Forward pass with gradient tracking"""
    print("\n[2/5] Benchmarking: Forward (with grad)")
    
    times, memories, peaks = [], [], []
    
    # Warmup
    for _ in range(warmup):
        model.zero_grad()
        loss, logits = model(tokens, memory_span=memory_span)
        # Clean up computation graph
        del loss, logits
    
    torch.cuda.empty_cache()  # Clean before benchmark
    
    # Benchmark
    for i in range(num_runs):
        torch.cuda.empty_cache()
        model.zero_grad()
        
        with GPUMemoryMonitor() as mon:
            loss, logits = model(tokens, memory_span=memory_span)
            loss_mean = loss.mean()
            # Store value but release graph
            loss_value = loss_mean.item()
        
        # CRITICAL: Delete tensors to free computation graph
        del loss, logits, loss_mean
        
        times.append(mon.time_elapsed)
        memories.append(mon.memory_used)
        peaks.append(mon.peak_memory)
        print(f"  Run {i+1}/{num_runs}: {mon.time_elapsed*1000:.1f}ms", end='\r')
    
    torch.cuda.empty_cache()  # Clean after benchmark
    print_stats("Forward (with grad)", times, memories, peaks)
    return times, memories, peaks


def benchmark_backward(model, tokens, memory_span, num_runs=10, warmup=3):
    """Benchmark: Backward pass"""
    print("\n[3/5] Benchmarking: Backward")
    
    times, memories, peaks = [], [], []
    
    # Warmup
    for _ in range(warmup):
        model.zero_grad()
        loss, _ = model(tokens, memory_span=memory_span)
        loss.mean().backward()
        model.zero_grad()  # Clear gradients
    
    torch.cuda.empty_cache()  # Clean before benchmark
    
    # Benchmark
    for i in range(num_runs):
        # Aggressive cleanup
        torch.cuda.empty_cache()
        model.zero_grad()
        
        # Forward first (not timed)
        loss, _ = model(tokens, memory_span=memory_span)
        loss_scalar = loss.mean()
        
        # Benchmark backward only
        with GPUMemoryMonitor() as mon:
            loss_scalar.backward()
        
        # Clean up immediately
        model.zero_grad()
        del loss, loss_scalar
        
        times.append(mon.time_elapsed)
        memories.append(mon.memory_used)
        peaks.append(mon.peak_memory)
        print(f"  Run {i+1}/{num_runs}: {mon.time_elapsed*1000:.1f}ms", end='\r')
    
    torch.cuda.empty_cache()  # Clean after benchmark
    print_stats("Backward", times, memories, peaks)
    return times, memories, peaks


def benchmark_recursion(model, tokens, memory_span, max_iterations=5, num_runs=5, warmup=2):
    """Benchmark: Recursion/Search"""
    print("\n[4/5] Benchmarking: Recursion (Search)")
    
    times, memories, peaks = [], [], []
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            tokens_copy = tokens.clone()
            idx, loss = recursion(model, tokens_copy, max_iterations=max_iterations, 
                     memory_span=memory_span, temperature=0.0)
            del idx, loss, tokens_copy
    
    torch.cuda.empty_cache()  # Clean before benchmark
    
    # Benchmark
    with torch.no_grad():
        for i in range(num_runs):
            torch.cuda.empty_cache()
            tokens_copy = tokens.clone()
            
            with GPUMemoryMonitor() as mon:
                idx, loss = recursion(model, tokens_copy, max_iterations=max_iterations,
                                     memory_span=memory_span, temperature=0.0)
            
            # Clean up
            del idx, loss, tokens_copy
            
            times.append(mon.time_elapsed)
            memories.append(mon.memory_used)
            peaks.append(mon.peak_memory)
            print(f"  Run {i+1}/{num_runs}: {mon.time_elapsed*1000:.1f}ms", end='\r')
    
    torch.cuda.empty_cache()  # Clean after benchmark
    print_stats(f"Recursion ({max_iterations} iters)", times, memories, peaks)
    return times, memories, peaks


def benchmark_full_iteration(model, tokens, memory_span, num_runs=10, warmup=3):
    """Benchmark: Full training iteration (forward + backward)"""
    print("\n[5/5] Benchmarking: Full Iteration (Forward + Backward)")
    
    times, memories, peaks = [], [], []
    
    # Warmup
    for _ in range(warmup):
        model.zero_grad()
        loss, _ = model(tokens, memory_span=memory_span)
        loss.mean().backward()
        model.zero_grad()
    
    torch.cuda.empty_cache()  # Clean before benchmark
    
    # Benchmark
    for i in range(num_runs):
        torch.cuda.empty_cache()
        model.zero_grad()
        
        with GPUMemoryMonitor() as mon:
            loss, _ = model(tokens, memory_span=memory_span)
            loss.mean().backward()
        
        # Clean up
        model.zero_grad()
        del loss
        
        times.append(mon.time_elapsed)
        memories.append(mon.memory_used)
        peaks.append(mon.peak_memory)
        print(f"  Run {i+1}/{num_runs}: {mon.time_elapsed*1000:.1f}ms", end='\r')
    
    torch.cuda.empty_cache()  # Clean after benchmark
    print_stats("Full Iteration", times, memories, peaks)
    return times, memories, peaks


# ============================================================================
# Main Benchmark Suite
# ============================================================================

def run_benchmark_suite(model, tokens, memory_span=1024, num_runs=10):
    """Run complete benchmark suite"""
    
    print(f"\n{'#'*60}")
    print(f"# GAT Benchmark Suite")
    print(f"{'#'*60}")
    # print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Sequence Length: {tokens.shape[1]}")
    print(f"Batch Size: {tokens.shape[0]}")
    print(f"Memory Span: {memory_span}")
    print(f"Model Config: {model.n_embd}d, {len(model.transformer.h)} layers, {model.transformer.h[0].attn.n_head} heads")
    
    results = {}
    
    # 1. Forward (no grad)
    results['forward_no_grad'] = benchmark_forward_no_grad(
        model, tokens, memory_span, num_runs=num_runs
    )
    
    # 2. Forward (with grad)
    results['forward_with_grad'] = benchmark_forward_with_grad(
        model, tokens, memory_span, num_runs=num_runs
    )
    
    # 3. Backward
    results['backward'] = benchmark_backward(
        model, tokens, memory_span, num_runs=num_runs
    )
    
    # 4. Recursion
    results['recursion'] = benchmark_recursion(
        model, tokens, memory_span, max_iterations=5, num_runs=max(5, num_runs//2)
    )
    
    # 5. Full iteration
    results['full_iteration'] = benchmark_full_iteration(
        model, tokens, memory_span, num_runs=num_runs
    )
    
    # Summary
    print(f"\n{'#'*60}")
    print(f"# Summary")
    print(f"{'#'*60}")
    for name, (times, mems, peaks) in results.items():
        print(f"{name:20s}: {np.mean(times)*1000:7.2f}ms  |  {np.mean(peaks):5.3f}GB peak")
    
    return results

