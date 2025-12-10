# Catastrophic Forgetting probes 
# -------------------------------
import torch
from sorl.neo_utils import sorl_rollout_v2, select_best_per_doc_v2, sorl_evaluate_v2, sorl_rollout_v3, avg_ppt_per_sample
from sorl.gat_act import BOS_TOKEN_ID

def compute_abs_stats(tokens, model, n, K, max_iterations, memory_span, attn_blocksize, temperature, truncate_seq_len):
        
    search_data, search_ppt, abs_logits = sorl_rollout_v3(tokens, model, n=n, K=K, 
                                                            max_iterations=max_iterations,
                                                            memory_span=memory_span,
                                                            attn_blocksize=attn_blocksize,
                                                            temperature=temperature,
                                                            truncate_seq_len=truncate_seq_len)
    search_ppt = search_ppt.reshape(search_data.shape[0], -1)

    # Get valid positions
    bos_pos_mask = torch.logical_and(
        search_data[:, :-1] != BOS_TOKEN_ID, 
        search_data[:, 1:] != BOS_TOKEN_ID
    ).float()

    traj_mask = (search_data[:, 1:] < model.vocab_sizes[0]).float()

    # --- greedy rollout's advantage ---
    valid_traj_mask = bos_pos_mask * traj_mask
    raw_ppt_adv = (search_ppt[1:].mean(dim=0) - search_ppt[0]) / (search_ppt[1:].mean(dim=0) + 1e-8)
    search_adv = (raw_ppt_adv * valid_traj_mask[0]).sum() / valid_traj_mask[0].sum().clamp(min=1)

    # --- losses ---
    greedy_ppt = search_ppt[0]
    abs_mask = 1 - traj_mask[0]

    valid_traj = valid_traj_mask[0]
    traj_ppt = search_ppt[0] * valid_traj
    doc_idx = (search_data == BOS_TOKEN_ID).cumsum(dim=1)
    doc_idx = doc_idx - doc_idx.min(dim=1, keepdim=True).values # idx starts from 0  
    doc_ppt = avg_ppt_per_sample(traj_ppt.unsqueeze(0), doc_idx[:1, 1:]).squeeze(0)

    greedy_abs_logits = abs_logits[0, abs_mask.bool(), :]
    greedy_abs_tokens = search_data[0, 1:][abs_mask.bool()]

    return doc_ppt, greedy_abs_logits, greedy_abs_tokens


def train_forget_vec(train_idx, loader, model, abs_stats, abs_stats_post, optimizer,
                     max_iterations, memory_span, attn_blocksize, temperature, K, r_min, reward_mode, loss_fn, alpha_abs, alpha_soft_zipf, alpha_topo,
                     ckpt_path="sorl_tinystories.pt"): 

    # ---- Load Checkpoint Model ----
    model.load_state_dict(torch.load(ckpt_path))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    temperatures_eval = torch.tensor([0.0, 10.0], device=model.device)

    i = 0
    batch_size = 16
    while i < len(loader.stories): 
        batch_indices = torch.arange(i, min(i + batch_size, len(loader.stories)))
        loader.get_specific(batch_indices) 
        i += batch_size
        tokens, doc_ids = loader.get_specific(batch_indices) 
        traj_loss, abs_logits, abs_tokens = compute_abs_stats(tokens, model, n=2, K=K, max_iterations=max_iterations, memory_span=memory_span, attn_blocksize=attn_blocksize, temperature=temperatures_eval,
                                                                                            truncate_seq_len=False)
        abs_stats_post.update(abs_logits, traj_loss, abs_tokens, doc_ids)

    # --- Training on 1 extra data point ---
    batch_size = 8
    train_indices = torch.tensor([train_idx] * batch_size)
    num_steps = 4

    for step in range(num_steps): 
        optimizer.zero_grad()
        tokens, doc_ids = loader.get_specific(train_indices)

        with torch.no_grad(): 
            # --- breakdown of SoRL search (select one per-document) ---
            search_data, search_ppt = sorl_rollout_v2(tokens, model, n=2, K=K, 
                                        max_iterations=max_iterations,
                                        memory_span=memory_span,
                                        attn_blocksize=attn_blocksize,
                                        temperature=temperature,
                                        truncate_seq_len=False)
            search_ppt = search_ppt.reshape(search_data.shape[0], -1)

            levels = (search_data >= model.vocab_sizes[0]).long()
            best_data, best_ppt, best_ppt_advantage, utility_reward = select_best_per_doc_v2(search_data, search_ppt, levels, r_min=r_min, reward_mode=reward_mode)

        # --- compute loss --- 
        traj_loss, abs_loss, zipf_bigram_loss, topo_loss = loss_fn(best_data, model, memory_span, attn_blocksize, utility_reward[1:])
        loss = traj_loss + alpha_abs * abs_loss + alpha_soft_zipf * zipf_bigram_loss + alpha_topo * topo_loss
        print(f"Step {step} | traj_loss: {traj_loss.item():.2f} | abs_loss: {abs_loss.item():.2f} | search adv: {best_ppt_advantage.item() * 100:.2f}% | bigram-zipf kl: {zipf_bigram_loss.item():.2f} | topo similarity: {-topo_loss.item():.2f} | bigram rep rate: {abs_stats.bigram_rep_rate:.2f}")
        
        # --- optimize --- 
        loss.backward() 
        optimizer.step()

    # ---- Full dataset statistics (after train) ---
    i = 0 
    batch_size = 16
    while i < len(loader.stories): 
        batch_indices = torch.arange(i, min(i + batch_size, len(loader.stories)))
        loader.get_specific(batch_indices) 
        i += batch_size
        tokens, doc_ids = loader.get_specific(batch_indices) 
        traj_loss, abs_logits, abs_tokens = compute_abs_stats(tokens, model, n=2, K=K, max_iterations=max_iterations, memory_span=memory_span, attn_blocksize=attn_blocksize, temperature=temperatures_eval,
                                                                                            truncate_seq_len=False)
        abs_stats_post.update(abs_logits, traj_loss, abs_tokens, doc_ids)

    # ----- Fill-in forget matrix ----
    forget_vec = abs_stats_post.traj_perplexity - abs_stats.traj_perplexity

    return forget_vec