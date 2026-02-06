import torch
import time
from contextlib import contextmanager

# Force CPU device
torch.set_default_device('cpu')

@contextmanager
def timer(name):
    start = time.time()
    yield
    end = time.time()
    print(f"{name}: {end - start:.4f}s")

def benchmark_sorl_components(model, input_ids, attention_mask, pad_token_id, n=2, K=4, max_iterations=2):
    """Benchmark each SoRL component separately"""
    
    print("=== SoRL Component Benchmark ===")
    print(f"Input shape: {input_ids.shape}")
    print(f"Device: {input_ids.device}")
    print(f"Batch size: {input_ids.shape[0]}, Seq len: {input_ids.shape[1]}")
    print()
    
    # 1. Insert mask inference
    with timer("1. Insert mask inference"):
        from sorl.sorl_trainer import infer_insert_mask
        insert_mask = infer_insert_mask(input_ids, K, model.vocab_sizes[0], attention_mask)
    
    # 2. Token insertion
    with timer("2. Token insertion"):
        from sorl.sorl_trainer import insert_tokens_with_padding
        expanded_data, expanded_mask = insert_tokens_with_padding(
            input_ids, attention_mask, insert_mask, model.vocab_sizes[0], pad_token_id
        )
    
    # 3. Batch expansion
    with timer("3. Batch expansion"):
        repeat_data = expanded_data.repeat_interleave(n, dim=0)
        repeat_mask = expanded_mask.repeat_interleave(n, dim=0)
    
    # 4. Block mask creation (single time)
    with timer("4. Block mask creation (single)"):
        block_mask = model._create_sorl_block_mask(repeat_data, 1792, 1792)
    
    # 5. Single forward pass
    with timer("5. Single forward pass"):
        outputs = model.forward(input_ids=repeat_data, attention_mask=repeat_mask)
    
    # 6. Recursion (multiple forward passes)
    with timer("6. Full recursion"):
        search_data, search_ppt = model.recursion(
            repeat_data, 
            repeat_mask,
            max_iterations=max_iterations,
            memory_span_abs=1792,
            memory_span_traj=1792,
            temperature=0.0
        )
    
    # 7. Best sequence selection
    with timer("7. Best sequence selection"):
        from sorl.sorl_trainer import select_best_sequences
        best_data, best_ppt, best_ppt_advantage = select_best_sequences(
            search_data, search_ppt, n, expanded_data.shape[0]
        )
    
    # 8. Loss computation
    with timer("8. Loss computation"):
        from sorl.sorl_trainer import SoRLLoss
        abs_vocab_size = model.total_vocab_size - model.vocab_sizes[0]
        loss_fn = SoRLLoss(abs_vocab_size=abs_vocab_size)
        
        # Base trajectory loss
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        base_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, memory_span_abs=1792, memory_span_traj=1792)
        base_traj_loss = base_outputs.loss
        
        # Auxiliary losses
        info_gain_loss, abs_loss, zipf_bigram_loss = loss_fn(
            best_data, model, base_traj_loss.detach(), expanded_mask, 1792, 1792
        )
    
    # 9. Block mask creation overhead (repeated calls)
    print("\n=== Block Mask Creation Overhead ===")
    with timer("Block mask creation (repeated 3x)"):
        for i in range(3):
            model._create_sorl_block_mask(repeat_data, 1792, 1792)
    
    # 10. Forward pass overhead (repeated calls)
    print("\n=== Forward Pass Overhead ===")
    with timer("Forward pass (repeated 3x)"):
        for i in range(3):
            model.forward(input_ids=repeat_data, attention_mask=repeat_mask, block_mask=block_mask)
    
    print("\n=== Summary ===")
    print(f"Total search time (recursion): ~6s")
    print(f"Total loss time: ~3s")
    print(f"Expected total: ~9s")
    print("\nKey bottlenecks to investigate:")
    print("- Block mask creation (should be <0.1s)")
    print("- Forward passes (should be <1s each)")
    print("- Recursion iterations")

def benchmark_attention_mask_creation(input_ids, attention_mask, pad_token_id):
    """Benchmark attention mask creation overhead"""
    print("\n=== Attention Mask Creation Benchmark ===")
    
    from sorl.sorl_trainer import infer_insert_mask, insert_tokens_with_padding
    
    with timer("Insert mask inference"):
        insert_mask = infer_insert_mask(input_ids, 4, 151936, attention_mask)
    
    with timer("Token insertion"):
        expanded_tokens, expanded_mask = insert_tokens_with_padding(
            input_ids, attention_mask, insert_mask, 151936, pad_token_id
        )
    
    print(f"Original shape: {input_ids.shape}")
    print(f"Expanded shape: {expanded_tokens.shape}")
    print(f"Mask shape: {expanded_mask.shape}")