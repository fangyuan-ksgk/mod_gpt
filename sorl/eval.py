import torch 
# from sorl.neo_utils import generate

def compute_vocab_utilization_rate(data: torch.Tensor, model):
    
    si, ei = model.vocab_sizes.cumsum(dim=0)
    n_unique_abs_tokens = data[(data >= si) & (data < ei)].unique().size(0)
    n_total_abs_tokens = (ei - si).item()
    abs_util_rate = n_unique_abs_tokens / n_total_abs_tokens
    # print(f" Total abs tokens: {n_total_abs_tokens} | Unique abs tokens: {n_unique_abs_tokens} | Abs util rate: {abs_util_rate * 100:.2f}%")
    return abs_util_rate

# def eval_cp(model, tokens, K, max_iterations, memory_span, seq_len, loader, temperature=0.0, num_samples=100):
#     n_correct = 0
#     num_samples = 10
#     abs_idx = []
#     for _ in range(num_samples): 
#         tokens, loss_mask = loader.get_batch(1)
#         idx = tokens[:, :1 + seq_len].clone()
#         for i in range(seq_len): 
#             idx = generate(model, idx, K=K, max_iterations=max_iterations, memory_span=memory_span, attn_blocksize=1792, temperature=temperature)
#             idx_without_abstraction = idx[idx < model.vocab_sizes[0]]
#             abs_idx.append(idx[idx >= model.vocab_sizes[0]])
#         correct_cp = torch.allclose(idx_without_abstraction[1 : 1 + seq_len], idx_without_abstraction[1 + seq_len : 2 + 2*seq_len])
#         n_correct += correct_cp
#     acc = n_correct / num_samples
#     print(f"=== Copy & Paste Evaluation ===\n Copy correct: {acc * 100:.2f}%")

#     data = torch.cat(abs_idx, dim=0)
#     abs_util_rate = compute_vocab_utilization_rate(data, model)
#     return acc, abs_util_rate