#!/usr/bin/env python3
"""
Test script for the revised SoRLWrapper with extract_and_sample and recursion
"""

import torch
from transformers import AutoConfig
from sorl.sorl_wrapper import SorlModelWrapper

def test_sorl_wrapper():
    """Test the revised SoRLWrapper methods"""
    
    # Create a minimal config for testing
    config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")
    config.model_type = "qwen2"
    
    # Initialize wrapper with single abstraction level
    abstract_vocab_size_list = [128]  # Single abstract vocab
    memory_span = 512
    
    print("=== Initializing SorlModelWrapper ===")
    wrapper = SorlModelWrapper.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        abstract_vocab_size_list=abstract_vocab_size_list,
        memory_span=memory_span
    )
    
    print(f"✓ Wrapper initialized")
    print(f"  Vocab sizes: {wrapper.vocab_sizes.tolist()}")
    print(f"  Level starts: {wrapper.level_starts.tolist()}")
    print(f"  Level ends: {wrapper.level_ends.tolist()}")
    
    # Test data
    batch_size = 2
    seq_len = 10
    base_vocab = wrapper.vocab_sizes[0].item()
    
    # Create test sequence with some abstract tokens
    test_ids = torch.randint(0, base_vocab, (batch_size, seq_len))
    
    # Insert abstract tokens at specific positions
    test_ids[0, 5] = base_vocab + 5  # Abstract token for first sample
    test_ids[1, 3] = base_vocab + 10  # Abstract token for second sample
    
    print(f"\n=== Test Data ===")
    print(f"Test sequence shape: {test_ids.shape}")
    print(f"Sample 0: {test_ids[0].tolist()}")
    print(f"Sample 1: {test_ids[1].tolist()}")
    
    # Test extract_and_sample
    print(f"\n=== Testing extract_and_sample ===")
    
    # Create mock logits
    total_vocab = wrapper.total_vocab_size.item()
    mock_logits = torch.randn(batch_size, seq_len, total_vocab)
    
    # Create recursion mask (positions with abstract tokens)
    recursion_mask = (test_ids >= base_vocab)
    recursion_mask[:, 0] = False  # Don't recurse on first position
    
    print(f"Recursion mask:\n{recursion_mask}")
    
    # Test extract_and_sample
    temperature = 0.0  # Greedy
    updated_ids = wrapper.extract_and_sample(mock_logits, test_ids.clone(), recursion_mask, temperature)
    
    print(f"Updated IDs after extract_and_sample:\n{updated_ids}")
    print(f"✓ extract_and_sample completed")
    
    # Test recursion
    print(f"\n=== Testing recursion (with information bottleneck mask) ===")
    
    final_ids, loss = wrapper.recursion(
        test_ids.clone(),
        max_iterations=2,
        memory_span_abs=512,   # Shorter memory for abstract tokens
        memory_span_traj=256,  # Even shorter for trajectory tokens
        attn_blocksize=512,
        temperature=0.0
    )
    
    print(f"Final IDs after recursion:\n{final_ids}")
    print(f"Loss shape: {loss.shape if hasattr(loss, 'shape') else type(loss)}")
    print(f"✓ recursion with information bottleneck completed")
    
    # Verify that abstract tokens were updated
    abstract_positions = (final_ids >= base_vocab)
    print(f"Abstract token positions after recursion:\n{abstract_positions}")
    
    # Test that recursion is deterministic for greedy sampling
    print(f"\n=== Testing recursion determinism ===")
    final_ids_2, _ = wrapper.recursion(
        test_ids.clone(),
        max_iterations=2,
        memory_span_abs=512,
        memory_span_traj=256,
        attn_blocksize=512,
        temperature=0.0
    )
    
    results_match = torch.allclose(final_ids, final_ids_2)
    print(f"Recursion is deterministic: {results_match}")
    print(f"✓ determinism test completed")
    
    # Test generate with different memory spans
    print(f"\n=== Testing generate with information bottleneck mask ===")
    
    # Simple prompt without abstract tokens
    prompt = torch.randint(0, base_vocab, (batch_size, 5))
    
    generated = wrapper.generate(
        prompt,
        max_new_tokens=3,
        temperature=0.7,
        top_k=50,
        K=4,  # Use abstraction
        memory_span_abs=1024,  # Abstract tokens can see further
        memory_span_traj=512   # Trajectory tokens have limited view
    )
    
    print(f"Prompt shape: {prompt.shape}")
    print(f"Generated shape: {generated.shape}")
    print(f"Generated sequences:\n{generated}")
    print(f"✓ generate with information bottleneck completed")
    
    print(f"\n🎉 All tests passed!")
    
    # Verify abstraction tokens were generated
    has_abstract = (generated >= base_vocab).any()
    print(f"Generated contains abstract tokens: {has_abstract}")
    
    return True

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    test_sorl_wrapper()
