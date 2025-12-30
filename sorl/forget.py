# Catastrophic Forgetting probes 
# -------------------------------
import torch
from data.tinystory_local import fig_to_pil
from sorl.neo_utils import sorl_rollout_v2, select_best_per_doc_v2, sorl_evaluate_v2, sorl_rollout_v3, avg_ppt_per_sample
from sorl.gat_act import BOS_TOKEN_ID
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np

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
    greedy_adv = (raw_ppt_adv * valid_traj_mask[0]).sum() / valid_traj_mask[0].sum().clamp(min=1)

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

    # --- base traj loss ---
    base_traj_ppt, _  = model.forward(tokens, memory_span, attn_blocksize)
    levels = (search_data >= model.vocab_sizes[0]).long()
    best_data, best_ppt, _ = select_best_per_doc(search_data, search_ppt, levels, model)
    best_traj_ppt = best_ppt[best_data[0, 1:] < model.vocab_sizes[0]]
    info_gain = (base_traj_ppt[..., :best_traj_ppt.shape[-1]] - best_traj_ppt) # traj ppt might be truncated
    
    
    rel_info_gain = info_gain.mean() / base_traj_ppt[..., :best_traj_ppt.shape[-1]].mean() # focus on hard case

    rel_info_gain_v2 = (info_gain / base_traj_ppt[..., :best_traj_ppt.shape[-1]].clamp(min=1e-8)).mean() # focus on simple case

    traj_mask = best_data[0, 1:] < model.vocab_sizes[0]
    traj_doc_idx = doc_idx[0, 1:][traj_mask]
    doc_info_gain = avg_ppt_per_sample(info_gain.unsqueeze(0), traj_doc_idx.unsqueeze(0)).squeeze(0)
    doc_base_traj_ppt = avg_ppt_per_sample(base_traj_ppt.unsqueeze(0), traj_doc_idx.unsqueeze(0)).squeeze(0)
    doc_rel_info_gain = doc_info_gain / doc_base_traj_ppt

    return doc_ppt, greedy_abs_logits, greedy_abs_tokens, doc_rel_info_gain, rel_info_gain, greedy_adv, base_traj_ppt.mean() 


from sorl.dream_utils import sorl_rollout_v3 as dream_rollout
def compute_abs_stats_v2(tokens, model, n, K, max_iterations, memory_span_abs, memory_span_traj, attn_blocksize, temperature, truncate_seq_len):
        
    search_data, search_ppt, abs_logits = dream_rollout(tokens, model, n=n, K=K, 
                                                            max_iterations=max_iterations,
                                                            memory_span_abs=memory_span_abs,
                                                            memory_span_traj=memory_span_traj,
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
    greedy_adv = (raw_ppt_adv * valid_traj_mask[0]).sum() / valid_traj_mask[0].sum().clamp(min=1)

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

    # --- base traj loss ---
    base_traj_ppt, _  = model.forward(tokens, memory_span_abs, memory_span_traj, attn_blocksize)
    levels = (search_data >= model.vocab_sizes[0]).long()
    best_data, best_ppt, _ = select_best_per_doc(search_data, search_ppt, levels, model)
    best_traj_ppt = best_ppt[best_data[0, 1:] < model.vocab_sizes[0]]
    info_gain = (base_traj_ppt[..., :best_traj_ppt.shape[-1]] - best_traj_ppt) # traj ppt might be truncated
    
    
    rel_info_gain = info_gain.mean() / base_traj_ppt[..., :best_traj_ppt.shape[-1]].mean() # focus on hard case

    rel_info_gain_v2 = (info_gain / base_traj_ppt[..., :best_traj_ppt.shape[-1]].clamp(min=1e-8)).mean() # focus on simple case

    traj_mask = best_data[0, 1:] < model.vocab_sizes[0]
    traj_doc_idx = doc_idx[0, 1:][traj_mask]
    doc_info_gain = avg_ppt_per_sample(info_gain.unsqueeze(0), traj_doc_idx.unsqueeze(0)).squeeze(0)
    doc_base_traj_ppt = avg_ppt_per_sample(base_traj_ppt.unsqueeze(0), traj_doc_idx.unsqueeze(0)).squeeze(0)
    doc_rel_info_gain = doc_info_gain / doc_base_traj_ppt

    return doc_ppt, greedy_abs_logits, greedy_abs_tokens, doc_rel_info_gain, rel_info_gain, greedy_adv, base_traj_ppt.mean() 


def compute_abs_stats_v3(tokens, model, base_model, n, K, max_iterations, memory_span, attn_blocksize, temperature, truncate_seq_len):

    search_data, search_ppt, abs_logits = sorl_rollout_v3(tokens, model, n=n, K=K, 
                                                                max_iterations=max_iterations,
                                                            memory_span=memory_span,
                                                            attn_blocksize=attn_blocksize,
                                                            temperature=temperature,
                                                            truncate_seq_len=truncate_seq_len)
    search_ppt = search_ppt.reshape(search_data.shape[0], -1)

    # --- greedy & random rollout ppt ---
    cond_traj_mask = (search_data[0, 1:] < model.vocab_sizes[0])
    greedy_traj_ppt = search_ppt[0][cond_traj_mask]
    random_traj_ppt = search_ppt[1:].mean(dim=0)[cond_traj_mask]

    # --- base traj loss ---
    base_traj_ppt = base_model.forward(tokens[:, :-1].long(), tokens[:, 1:].long(), attn_blocksize)
    base_traj_mask = (tokens[:, 1:] != BOS_TOKEN_ID).float()
    base_traj_loss = (base_traj_ppt * base_traj_mask[0]).sum() / base_traj_mask.sum().clamp(min=1)

    # --- search (per-doc-best rollout) info gain ---
    levels = (search_data >= model.vocab_sizes[0]).long()
    best_data, best_ppt, _ = select_best_per_doc(search_data, search_ppt, levels, model)
    best_traj_ppt = best_ppt[cond_traj_mask]

    # --- information gain, search advantage (relative & absolute) ---
    greedy_adv = ((random_traj_ppt - greedy_traj_ppt) / (random_traj_ppt + 1e-8)).mean()
    greedy_abs_adv = (random_traj_ppt - greedy_traj_ppt).mean()
    search_adv = ((random_traj_ppt - best_traj_ppt) / (random_traj_ppt + 1e-8)).mean()
    search_abs_adv = (random_traj_ppt - best_traj_ppt).mean()
    greedy_info_gain = (base_traj_ppt - greedy_traj_ppt).mean()
    search_info_gain = (base_traj_ppt - best_traj_ppt).mean()

    greedy_traj_loss = greedy_traj_ppt.mean()
    search_traj_loss = best_traj_ppt.mean()

    return base_traj_loss, greedy_traj_loss, search_traj_loss, greedy_adv, greedy_abs_adv, search_adv, search_abs_adv, greedy_info_gain, search_info_gain
    

from sorl.dream_utils import sorl_rollout_v3 as dream_rollout
from sorl.dream_utils import select_best_per_doc
def compute_abs_stats_v5(tokens, model, n, K, max_iterations, memory_span_abs, memory_span_traj, attn_blocksize, temperature, truncate_seq_len, pad_token):

    search_data, search_traj_ppt, search_abs_ppt, abs_logits = dream_rollout(tokens, model, n=n, K=K, 
                                                        max_iterations=max_iterations,
                                                        memory_span_abs=memory_span_abs,
                                                        memory_span_traj=memory_span_traj,
                                                        attn_blocksize=attn_blocksize,
                                                        temperature=temperature,
                                                        truncate_seq_len=truncate_seq_len)
    search_traj_ppt = search_traj_ppt.reshape(search_data.shape[0], -1)
    search_abs_ppt = search_abs_ppt.reshape(search_data.shape[0], -1)

    # --- greedy & random rollout ppt ---
    cond_traj_mask = (search_data[0, 1:] < model.vocab_sizes[0])
    greedy_traj_ppt = search_traj_ppt[0][cond_traj_mask]
    random_traj_ppt = search_traj_ppt[1:].mean(dim=0)[cond_traj_mask]

    # --- base traj loss ---
    base_traj_ppt = model.forward(tokens, memory_span_abs, memory_span_traj, attn_blocksize)[0]
    base_traj_mask = (tokens[:, 1:] != BOS_TOKEN_ID).float()
    base_traj_loss = (base_traj_ppt * base_traj_mask[0]).sum() / base_traj_mask.sum().clamp(min=1)

    # --- search (per-doc-best rollout) info gain ---
    levels = (search_data >= model.vocab_sizes[0]).long()
    best_data, best_traj_ppt, best_abs_ppt, _ = select_best_per_doc(search_data, search_traj_ppt, search_abs_ppt, levels, model)
    best_traj_ppt = best_traj_ppt[cond_traj_mask]

    # --- information gain, search advantage (relative & absolute) ---
    greedy_adv = ((random_traj_ppt - greedy_traj_ppt) / (random_traj_ppt + 1e-8))
    greedy_abs_adv = (random_traj_ppt - greedy_traj_ppt)
    search_adv = ((random_traj_ppt - best_traj_ppt) / (random_traj_ppt + 1e-8))
    search_abs_adv = (random_traj_ppt - best_traj_ppt)
    greedy_info_gain = (base_traj_ppt - greedy_traj_ppt)
    search_info_gain = (base_traj_ppt - best_traj_ppt)

    greedy_traj_loss = greedy_traj_ppt.mean()
    search_traj_loss = best_traj_ppt.mean()

    # --- information gain, search advantage (relative & absolute) ---
    greedy_adv = ((random_traj_ppt - greedy_traj_ppt) / (random_traj_ppt + 1e-8)).mean()
    greedy_abs_adv = (random_traj_ppt - greedy_traj_ppt).mean()
    search_adv = ((random_traj_ppt - best_traj_ppt) / (random_traj_ppt + 1e-8)).mean()
    search_abs_adv = (random_traj_ppt - best_traj_ppt).mean()
    greedy_info_gain = (base_traj_ppt - greedy_traj_ppt).mean()
    search_info_gain = (base_traj_ppt - best_traj_ppt).mean()

    return {
            "base_traj_loss": base_traj_loss, 
            "greedy_traj_loss": greedy_traj_loss, 
            "search_traj_loss": search_traj_loss, 
            "greedy_adv": greedy_adv, 
            "search_adv": search_adv, 
            "greedy_abs_adv": greedy_abs_adv, 
            "search_abs_adv": search_abs_adv, 
            "greedy_info_gain": greedy_info_gain,  
            "search_info_gain": search_info_gain
            }   
      

def visualize_info_gain(tokens, model, base_model, K, max_iterations, memory_span, attn_blocksize, 
                        n=5, temperature=None, figsize=(14, 8)):
    """Visualize per-token perplexity comparison between base and greedy (abstraction) model."""
    
    if temperature is None:
        temperature = torch.tensor([0.0] + [5.0] * (n - 1), device=tokens.device)
    
    with torch.no_grad():
        # Greedy rollout perplexity
        search_data, search_ppt, _ = sorl_rollout_v3(
            tokens, model, n=n, K=K, max_iterations=max_iterations,
            memory_span=memory_span, attn_blocksize=attn_blocksize,
            temperature=temperature, truncate_seq_len=False
        )
        search_ppt = search_ppt.reshape(search_data.shape[0], -1)
        cond_traj_mask = (search_data[0, 1:] < model.vocab_sizes[0])
        greedy_ppt = search_ppt[0][cond_traj_mask].cpu()
        
        # Base model perplexity
        base_ppt = base_model.forward(
            tokens[:, :-1].long(), tokens[:, 1:].long(), attn_blocksize
        ).cpu().squeeze()
    
    # Align lengths
    min_len = min(len(base_ppt), len(greedy_ppt))
    base_ppt, greedy_ppt = base_ppt[:min_len], greedy_ppt[:min_len]
    info_gain = (base_ppt - greedy_ppt).numpy()
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    axes[0].plot(base_ppt.numpy(), alpha=0.7, label='Base Model', linewidth=0.8)
    axes[0].plot(greedy_ppt.numpy(), alpha=0.7, label='Greedy (with abstraction)', linewidth=0.8)
    axes[0].set_ylabel('Per-token PPT')
    axes[0].legend()
    axes[0].set_title('Per-Token Perplexity: Base vs Greedy')
    
    colors = ['green' if g > 0 else 'red' for g in info_gain]
    axes[1].bar(range(len(info_gain)), info_gain, color=colors, alpha=0.6, width=1.0)
    axes[1].axhline(0, color='black', linewidth=0.5)
    axes[1].set_ylabel('Info Gain')
    axes[1].set_title(f'Per-Token Info Gain (mean={info_gain.mean():.4f})')
    
    cumsum = info_gain.cumsum()
    axes[2].plot(cumsum, color='blue', linewidth=1.0)
    axes[2].fill_between(range(len(cumsum)), cumsum, alpha=0.3)
    axes[2].axhline(0, color='black', linewidth=0.5)
    axes[2].set_xlabel('Token Position')
    axes[2].set_ylabel('Cumulative Info Gain')
    axes[2].set_title('Cumulative Information Gain Over Sequence')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Mean base PPT: {base_ppt.mean():.4f} | Mean greedy PPT: {greedy_ppt.mean():.4f} | Mean info gain: {info_gain.mean():.4f}")
    
    return {"base_ppt": base_ppt, "greedy_ppt": greedy_ppt, "info_gain": info_gain}

    
def train_forget_vec(train_idx, loader, model, abs_stats, abs_stats_post, optimizer, num_steps,
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
        abs_stats.update(abs_logits, traj_loss, abs_tokens, doc_ids)

    # --- Training on 1 extra data point ---
    batch_size = 1
    train_indices = torch.tensor([train_idx] * batch_size)

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


def plot_forget_trend(train_idx, forget_vec, ham_dist_vec, logit_sim_vec):
    mask = torch.arange(len(forget_vec)) != train_idx
    y_forget = forget_vec[mask].cpu().numpy()
    x_hamming_dist = ham_dist_vec[mask].cpu().numpy()  # DISTANCE now
    x_logit_sim = logit_sim_vec[mask].cpu().numpy()

    # 1. Pearson Correlation
    corr_hamming, p_hamming = scipy.stats.pearsonr(x_hamming_dist, y_forget)
    corr_logit, p_logit = scipy.stats.pearsonr(x_logit_sim, y_forget)

    print(f"Forgetting vs Hamming DISTANCE: r={corr_hamming:.3f} (p={p_hamming:.4f})")
    print(f"Forgetting vs Logit SIMILARITY: r={corr_logit:.3f} (p={p_logit:.4f})")

    # 2. Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Structure (Hamming Distance)
    axes[0].scatter(x_hamming_dist, y_forget, alpha=0.6, color='purple')
    axes[0].set_xlabel('Abstract DISTANCE (Hamming)')  # Corrected label
    axes[0].set_ylabel('Forgetting (Δ Perplexity)')
    axes[0].set_title(f'Structure vs Forgetting (r={corr_hamming:.2f})')
    z = np.polyfit(x_hamming_dist, y_forget, 1)
    axes[0].plot(x_hamming_dist, np.poly1d(z)(x_hamming_dist), 'k--', alpha=0.5)

    # Plot 2: Semantics (Logit Sim)
    axes[1].scatter(x_logit_sim, y_forget, alpha=0.6, color='teal')
    axes[1].set_xlabel('Semantic SIMILARITY (Logit Cosine)')
    axes[1].set_ylabel('Forgetting (Δ Perplexity)')
    axes[1].set_title(f'Semantics vs Forgetting (r={corr_logit:.2f})')
    z = np.polyfit(x_logit_sim, y_forget, 1)
    axes[1].plot(x_logit_sim, np.poly1d(z)(x_logit_sim), 'k--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    # 3. New Interpretation
    if corr_hamming < -0.3:
        print(">> Evidence: Distant abstract structures are PROTECTED (Less Forgetting).")
        print(">> Samples with SHARED structure (Low Distance) interfere more.")
    elif corr_hamming > 0.3:
        print(">> Evidence: Distant abstract structures INTERFERE more (Unexpected).")


def collect_forget_data(forget_mat, abs_stats): 
    ham_dist_vec = abs_stats.compute_cross_doc_hamming()[0]
    # logit_sim_vec = abs_stats.compute_cross_doc_logit_sim()[0]

    ham_corrs = []
    correlation_data = []  # Store (x, y) pairs for plotting

    for train_idx in range(forget_mat.shape[0]): 
        forget_vec = forget_mat[train_idx]
        mask = torch.arange(len(forget_vec)) != train_idx
        y_forget = forget_vec[mask].cpu().numpy()
        x_hamming_dist = ham_dist_vec[mask].cpu().numpy()

        # Store for visualization
        correlation_data.append((x_hamming_dist, y_forget))
        
        # Pearson Correlation
        corr_hamming, _ = scipy.stats.pearsonr(x_hamming_dist, y_forget)
        ham_corrs.append(corr_hamming)

    neg_corr_count = (np.array(ham_corrs) <= 0).sum()
    print(f"{neg_corr_count/len(ham_corrs)*100:.1f}% exhibit negative correlation")
    return correlation_data, ham_corrs

def plot_normalized_correlation_lines(correlation_data, correlations, 
                                          xlabel="X", ylabel="Y", title=""):

    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_lines_neg, y_lines_pos = [], []
    neg_count = 0
    
    # Use percentile-based x-axis for common scale
    x_common = np.linspace(0, 1, 50)  # 0 = min distance, 1 = max distance
    
    for (x, y), corr in zip(correlation_data, correlations):
        # Min-max normalize X (distance) to [0, 1]
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
        # Min-max normalize Y (forgetting) to [0, 1] — all positive
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8)
        
        slope, intercept = np.polyfit(x_norm, y_norm, 1)
        y_line = slope * x_common + intercept
        
        if corr <= 0:
            y_lines_neg.append(y_line)
            ax.plot(x_common, y_line, color='steelblue', alpha=0.25, linewidth=1.5)
            neg_count += 1
        else:
            y_lines_pos.append(y_line)
            ax.plot(x_common, y_line, color='coral', alpha=0.12, linewidth=0.8)
    
    # Mean trends
    y_neg_mean = np.array(y_lines_neg).mean(axis=0)
    ax.plot(x_common, y_neg_mean, color='darkblue', linewidth=4)
    
    # Shade zones
    ax.fill_betweenx([0, 1], 0.5, 1.0, alpha=0.04, color='steelblue')  # far = less forget zone
    ax.fill_betweenx([0, 1], 0, 0.5, alpha=0.04, color='coral')  # near = more forget zone
    
    # ===== LEGEND with interpretation =====
    from matplotlib.lines import Line2D
    neg_pct = 100 * neg_count / len(correlations)
    pos_pct = 100 - neg_pct
    legend_elements = [
        Line2D([0], [0], color='steelblue', lw=3, 
               label=f'Far → Less Forgetting ({neg_pct:.0f}%)'),
        Line2D([0], [0], color='coral', lw=3, 
               label=f'Far → More Forgetting ({pos_pct:.0f}%)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=16, 
              framealpha=0.95, title='Trend Direction')
    
    # ===== INSET DONUT =====
    ax_inset = fig.add_axes([0.12, 0.12, 0.36, 0.36])
    wedges, _ = ax_inset.pie([neg_pct/100, pos_pct/100], colors=['steelblue', 'coral'],
                              startangle=90, wedgeprops=dict(width=0.45, edgecolor='white'))
    ax_inset.text(0, 0, f'{neg_pct:.0f}%', ha='center', va='center',
                  fontsize=18, fontweight='bold', color='steelblue')
    ax_inset.set_aspect('equal')
    
    # Labels — meaningful scale
    ax.set_xlabel(f'{xlabel}\n(0 = closest, 1 = farthest)', fontsize=16)
    ax.set_ylabel(f'{ylabel}\n(0 = least, 1 = most)', fontsize=16)
    ax.set_title(title, fontsize=20, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0.1, 0.95)
    
    # Corner annotations
    ax.text(0.02, 0.94, 'Near & High Forget', fontsize=16, color='gray', 
            ha='left', va='top', style='italic')
    ax.text(0.98, 0.10, 'Far & Low Forget', fontsize=16, color='gray', 
            ha='right', va='bottom', style='italic')
    
    plt.tight_layout()
    plt.show()


def plot_rel_info_gain_vs_step(record):
    
    steps = range(len(record['rel_search_info_gain']))

    fig, ax1 = plt.subplots(figsize=(8,5))

    color1 = 'tab:blue'
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Rel Search Info Gain', color=color1)
    line1, = ax1.plot(steps, record['rel_search_info_gain'], label='Rel Search Info Gain', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    # Draw rel search info gain = 0 line as a dotted line
    ax1.axhline(0, color=color1, linestyle=':', linewidth=1, label='Info Gain = 0')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color2 = 'tab:orange'
    ax2.set_ylabel('Search Adv', color=color2)  
    line2, = ax2.plot(steps, record['greedy_adv'], label='Search Adv', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title('SoRL Search Info Gain & Search Adv vs. Step')

    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    plt.legend(lines, labels, loc='best')

    return fig_to_pil(fig)