# Mutual Information Loss 
import torch 
from torch import nn 
import torch.nn.functional as F
from sorl.gat_sim import BOS_TOKEN_ID
from sorl.topo import compute_topo_loss, pairwise_hamming_dist

class MutualInformationLoss(nn.Module):
    def __init__(self, vocab_size, decay=0.8, target_vocab_util=0.8):
        super().__init__()
        self.decay = decay       
        target_val = torch.log(vocab_size * target_vocab_util) 
        self.register_buffer('target_marginal_entropy', target_val)
        
        self.register_buffer('running_marginal', torch.ones(vocab_size, device=vocab_size.device) / vocab_size)

    def forward(self, logits):
        """
        logits: [Batch, Vocab]
        """
        probs = F.softmax(logits, dim=-1)
        
        # --- TERM 1: MARGINAL ENTROPY (Maximize H(A)) ---
        batch_marginal = probs.mean(dim=[0, 1])
        
        with torch.no_grad():
             new_avg = self.decay * self.running_marginal + (1 - self.decay) * batch_marginal
             self.running_marginal.copy_(new_avg)

        mixed_marginal = self.decay * self.running_marginal + (1 - self.decay) * batch_marginal
        marginal_entropy = -torch.sum(mixed_marginal * torch.log(mixed_marginal + 1e-10))
        marginal_entropy = torch.clamp(marginal_entropy - self.target_marginal_entropy, max=0.0) + self.target_marginal_entropy

        # --- TERM 2: CONDITIONAL ENTROPY (Minimize H(A|S)) ---
        per_sample_entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        conditional_entropy = per_sample_entropy.mean()

        # --- Return marginal & conditional entropy ---
        return marginal_entropy, conditional_entropy

class SoRLLoss(nn.Module): 
    """
    SoRL loss: p(s | a), p(a | s), H(p(a)), H(p(a | s)) 
    """

    def __init__(self, abs_vocab_size, decay=0.8, target_vocab_util=0.8):
        super().__init__()
        self.decay = decay
        self.mutual_info_loss = MutualInformationLoss(abs_vocab_size, decay, target_vocab_util)

    def forward(self, data, model, memory_span: int, attn_blocksize: int, utility_reward: torch.Tensor):

        ppt, logits = model.forward(data, memory_span, attn_blocksize)
        ppt = ppt * utility_reward
        ppt = ppt.reshape(data.shape[0], -1)
        logits = logits.reshape(data.shape[0], -1, logits.size(-1))

        levels = (data >= model.vocab_sizes[0]).long()[:, 1:]
        bos_pos_mask = torch.logical_and(
            data[:, :-1] != BOS_TOKEN_ID, 
            data[:, 1:] != BOS_TOKEN_ID
        ).float()

        traj_mask = (levels[0] == 0).float()
        abs_mask = 1 - traj_mask

        valid_traj_mask = bos_pos_mask * traj_mask
        valid_abs_mask = bos_pos_mask * abs_mask

        traj_loss = (ppt * valid_traj_mask).sum() / valid_traj_mask.sum().clamp(min=1)
        abs_loss = (ppt * valid_abs_mask).sum() / valid_abs_mask.sum().clamp(min=1)

        abs_positions = abs_mask.bool()
        abs_logits = logits[:, :-1][:, abs_positions, model.vocab_sizes[0]:]

        marginal_entropy, _ = self.mutual_info_loss(abs_logits)

        # --- Return: p(s | a), p(a | s), H(p(a)), H(p(a | s)) ---
        return traj_loss, abs_loss, marginal_entropy

# Zipfian distribution matching loss >> Entropy regularization ? 

# --- Zipfian distribution ---- 
def get_zipf_prior(total_vocab_size, target_vocab_util: float = 1.0, alpha: float = 1.0):
    vocab_size = int(total_vocab_size * target_vocab_util)
    ranks = torch.arange(1, vocab_size + 1, dtype=torch.float32, device=total_vocab_size.device)
    freqs = 1.0 / (ranks ** alpha)
    probs = torch.zeros(total_vocab_size, dtype=torch.float32, device=total_vocab_size.device)
    probs[:vocab_size] = (freqs / freqs.sum())
    return probs

# --- Sinkhorn-Knopp iteration ---- 
def sinkhorn_prob_transform(prob, prior, n_iters=3): 
    B = prob.shape[0]
    target_cols = prior * B
    M = prob
    for _ in range(n_iters):
        col_sums = M.sum(dim=0, keepdim=True) + 1e-8
        M = M * (target_cols.unsqueeze(0) / col_sums)
        row_sums = M.sum(dim=1, keepdim=True) + 1e-8
        M = M / row_sums
    return M

class ZipfianLoss(nn.Module):
    def __init__(self, vocab_size, decay=0.8, target_vocab_util=0.8):
        super().__init__()
        self.decay = decay   
        
        zipf_prior = get_zipf_prior(vocab_size, target_vocab_util)
        self.register_buffer('zipf_prior', zipf_prior)
        self.register_buffer('running_marginal', torch.ones(vocab_size) / vocab_size)

    def forward(self, logits):
        """
        logits: [Batch, SeqLen, Vocab]
        """
        probs = F.softmax(logits, dim=-1) # [B, L, V]
        probs_flat = probs.flatten(0, 1)  # [N, V]
        
        # --- 1. EMA Update & Mixed Marginal ---
        batch_marginal = probs_flat.mean(dim=0)
        
        if self.training:
            with torch.no_grad():
                new_avg = self.decay * self.running_marginal + (1 - self.decay) * batch_marginal
                self.running_marginal.copy_(new_avg)
                
        mixed_marginal = self.decay * self.running_marginal + (1 - self.decay) * batch_marginal

        sorted_marginal, _ = torch.sort(mixed_marginal, descending=True)

        p_safe = torch.clamp(sorted_marginal, min=1e-10)
        term1 = p_safe * torch.log(p_safe)
        term2 = p_safe * torch.log(self.zipf_prior + 1e-10)
        kl_div = torch.sum(term1 - term2)

        # ce = -torch.sum(sorted_marginal * torch.log(self.zipf_prior + 1e-10))

        # --- Method B: Soft Zipfian (Sinkhorn on Mixed Probs) ---
        group_marginals = self.decay * self.running_marginal.unsqueeze(0) + (1 - self.decay) * probs_flat
        soft_sinkhorn = sinkhorn_prob_transform(group_marginals, self.zipf_prior, n_iters=3)
        
        p_safe = torch.clamp(group_marginals, min=1e-10)
        term1 = p_safe * torch.log(p_safe)
        term2 = p_safe * torch.log(soft_sinkhorn.detach() + 1e-10)
        soft_kl_div = torch.sum(term1 - term2) / group_marginals.shape[0]

        # ce_soft = -torch.sum(group_marginals * torch.log(soft_sinkhorn.detach() + 1e-10)) / group_marginals.shape[0]

        return kl_div, soft_kl_div

class SoRLLoss_v2(nn.Module): 
    """
    SoRL loss: p(s | a), p(a | s), KL(p(a), zipf_prior), KL(p(a), soft_zipf_prior)
    """

    def __init__(self, abs_vocab_size, decay=0.8, target_vocab_util=0.8):
        super().__init__()
        self.decay = decay
        self.zipf_loss = ZipfianLoss(abs_vocab_size, decay, target_vocab_util)

    def forward(self, data, model, memory_span: int, attn_blocksize: int):

        ppt, logits = model.forward(data, memory_span, attn_blocksize)
        ppt = ppt.reshape(data.shape[0], -1)
        logits = logits.reshape(data.shape[0], -1, logits.size(-1))

        levels = (data >= model.vocab_sizes[0]).long()[:, 1:]
        bos_pos_mask = torch.logical_and(
            data[:, :-1] != BOS_TOKEN_ID, 
            data[:, 1:] != BOS_TOKEN_ID
        ).float()

        traj_mask = (levels[0] == 0).float()
        abs_mask = 1 - traj_mask

        valid_traj_mask = bos_pos_mask * traj_mask
        valid_abs_mask = bos_pos_mask * abs_mask

        traj_loss = (ppt * valid_traj_mask).sum() / valid_traj_mask.sum().clamp(min=1)
        abs_loss = (ppt * valid_abs_mask).sum() / valid_abs_mask.sum().clamp(min=1)

        abs_positions = abs_mask.bool()
        abs_logits = logits[:, :-1][:, abs_positions, model.vocab_sizes[0]:]

        zipf_kl, soft_zipf_kl = self.zipf_loss(abs_logits)

        # --- Return: p(s | a), p(a | s), KL(p(a), zipf_prior), KL(p(a), soft_zipf_prior) ---
        return traj_loss, abs_loss, soft_zipf_kl


# ----- BiGram Zipfian Regularization (to avoid repetition) ---- 

# --- Hollow Sinkhorn ---- 
def hollow_sinkhorn_transform(prob_2gram, zipf_prior, n_iters=3, diag_penalty=1e-5):
    B, V, _ = prob_2gram.shape
    device = prob_2gram.device
    mask = 1.0 - torch.eye(V, device=device).unsqueeze(0) # [1, V, V] 0 on diag, 1 elsewhere
    M = prob_2gram * (mask + diag_penalty * (1 - mask))
    
    r = zipf_prior.unsqueeze(0) # [1, V]
    # r needs to be [1, V, 1] for row scaling and [1, 1, V] for col scaling to broadcast correctly with [B, V, V]
    r_row = r.unsqueeze(2) # [1, V, 1]
    r_col = r.unsqueeze(1) # [1, 1, V]
    
    for _ in range(n_iters):
        # Scale rows -> sum to r
        row_sum = M.sum(dim=2, keepdim=True) + 1e-10
        M = M * (r_row / row_sum)
        
        # Scale cols -> sum to r (assuming symmetric marginals for Sinkhorn)
        col_sum = M.sum(dim=1, keepdim=True) + 1e-10
        M = M * (r_col / col_sum)
    return M

class Zipfian2gramLoss(nn.Module):
    def __init__(self, vocab_size, decay=0.8, target_vocab_util=0.8):
        super().__init__()
        self.decay = decay   
        
        zipf_prior = get_zipf_prior(vocab_size, target_vocab_util)
        self.register_buffer('zipf_prior', zipf_prior)
        # Running marginal is a transition matrix [V, V]
        self.register_buffer('running_marginal', torch.ones(vocab_size, vocab_size, device=vocab_size.device) / vocab_size)

    def forward(self, logits):
        """
        logits: [Batch, SeqLen, Vocab]
        """
        probs = F.softmax(logits, dim=-1) # [B, L, V]
        
        # Compute joint probs P(a_t, a_{t+1}) = P(a_t) * P(a_{t+1})
        # Use Outer Product! Summing probs is wrong.
        probs_2gram = (probs[:, :-1].unsqueeze(3) * probs[:, 1:].unsqueeze(2)).mean(dim=1) # [B, V, V]
        
        # --- 1. EMA Update & Mixed Marginal ---
        batch_marginal = probs_2gram.mean(dim=0) # [V, V]
        
        if self.training:
            with torch.no_grad():
                new_avg = self.decay * self.running_marginal + (1 - self.decay) * batch_marginal
                self.running_marginal.copy_(new_avg)
        
        # --- Soft Zipfian (Hollow Sinkhorn) ---
        # group_marginals: [B, V, V]
        # Mix global history [1, V, V] with local batch [B, V, V]
        group_marginals = self.decay * self.running_marginal.unsqueeze(0) + (1 - self.decay) * probs_2gram
        
        # Target: Hollow Matrix with Zipfian Marginals
        soft_sinkhorn = hollow_sinkhorn_transform(group_marginals, self.zipf_prior, n_iters=3)
        
        # Loss
        p_safe = torch.clamp(group_marginals, min=1e-10)
        term1 = p_safe * torch.log(p_safe)
        term2 = p_safe * torch.log(soft_sinkhorn.detach() + 1e-10)
        soft_kl_div = torch.sum(term1 - term2) / group_marginals.shape[0] # Divide by B

        return soft_kl_div

class SoRLLoss_v3(nn.Module): 
    """
    SoRL loss: p(s | a), p(a | s), KL(p(a_t, a_t+1), soft_zipf_prior)
    """

    def __init__(self, abs_vocab_size, decay=0.8, target_vocab_util=0.8):
        super().__init__()
        self.decay = decay
        self.zipf_loss = Zipfian2gramLoss(abs_vocab_size, decay, target_vocab_util)

    def forward(self, data, model, memory_span: int, attn_blocksize: int):

        ppt, logits = model.forward(data, memory_span, attn_blocksize)
        ppt = ppt.reshape(data.shape[0], -1)
        logits = logits.reshape(data.shape[0], -1, logits.size(-1))

        levels = (data >= model.vocab_sizes[0]).long()[:, 1:]
        bos_pos_mask = torch.logical_and(
            data[:, :-1] != BOS_TOKEN_ID, 
            data[:, 1:] != BOS_TOKEN_ID
        ).float()

        traj_mask = (levels[0] == 0).float()
        abs_mask = 1 - traj_mask

        valid_traj_mask = bos_pos_mask * traj_mask
        valid_abs_mask = bos_pos_mask * abs_mask

        traj_loss = (ppt * valid_traj_mask).sum() / valid_traj_mask.sum().clamp(min=1)
        abs_loss = (ppt * valid_abs_mask).sum() / valid_abs_mask.sum().clamp(min=1)

        abs_positions = abs_mask.bool()
        abs_logits = logits[:, :-1][:, abs_positions, model.vocab_sizes[0]:]

        soft_zipf_kl = self.zipf_loss(abs_logits)

        # --- Return: p(s | a), p(a | s), KL(p(a_t, a_t+1), soft_zipf_prior) ---
        return traj_loss, abs_loss, soft_zipf_kl

class SoRLLoss_v4(nn.Module): 
    """
    SoRL loss: p(s | a), p(a | s), KL(p(a_t, a_t+1), soft_zipf_prior)
    """

    def __init__(self, abs_vocab_size, decay=0.8, target_vocab_util=0.8):
        super().__init__()
        self.decay = decay
        self.zipf_loss = Zipfian2gramLoss(abs_vocab_size, decay, target_vocab_util)

    def forward(self, data, model, memory_span: int, attn_blocksize: int, 
                      utility_reward: torch.Tensor):

        ppt, logits = model.forward(data, memory_span, attn_blocksize)
        ppt = ppt.reshape(data.shape[0], -1)
        logits = logits.reshape(data.shape[0], -1, logits.size(-1))

        levels = (data >= model.vocab_sizes[0]).long()[:, 1:]
        bos_pos_mask = torch.logical_and(
            data[:, :-1] != BOS_TOKEN_ID, 
            data[:, 1:] != BOS_TOKEN_ID
        ).float()

        traj_mask = (levels[0] == 0).float()
        abs_mask = 1 - traj_mask

        valid_traj_mask = bos_pos_mask * traj_mask
        valid_abs_mask = bos_pos_mask * abs_mask

        traj_loss = (ppt * valid_traj_mask).sum() / valid_traj_mask.sum().clamp(min=1)
        abs_loss = (ppt * utility_reward * valid_abs_mask).sum() / valid_abs_mask.sum().clamp(min=1)

        abs_positions = abs_mask.bool()
        abs_logits = logits[:, :-1][:, abs_positions, model.vocab_sizes[0]:]

        soft_zipf_kl = self.zipf_loss(abs_logits)

        # --- Return: p(s | a), p(a | s), KL(p(a_t, a_t+1), soft_zipf_prior) ---
        return traj_loss, abs_loss, soft_zipf_kl


class SoRLLoss_v5(nn.Module): 
    """
    SoRL loss: p(s | a), p(a | s), KL(p(a_t, a_t+1), soft_zipf_prior), corr(d(a), d(r))
    """

    def __init__(self, abs_vocab_size, decay=0.8, target_vocab_util=0.8):
        super().__init__()
        self.decay = decay
        self.zipf_loss = Zipfian2gramLoss(abs_vocab_size, decay, target_vocab_util)

    def forward(self, data, model, memory_span: int, attn_blocksize: int, 
                      utility_reward: torch.Tensor):
 
        # topo distance on hidden states instead of logits 

        ppt, logits, hidden_states = model.forward_with_hidden_states(data, memory_span, attn_blocksize)
        ppt = ppt.reshape(data.shape[0], -1)
        ppt = ppt * utility_reward
        logits = logits.reshape(data.shape[0], -1, logits.size(-1))
        hidden_states = hidden_states.reshape(data.shape[0], -1, hidden_states.size(-1))

        levels = (data >= model.vocab_sizes[0]).long()[:, 1:]
        bos_pos_mask = torch.logical_and(
            data[:, :-1] != BOS_TOKEN_ID, 
            data[:, 1:] != BOS_TOKEN_ID
        ).float()

        traj_mask = (levels[0] == 0).float()
        abs_mask = 1 - traj_mask

        valid_traj_mask = bos_pos_mask * traj_mask
        valid_abs_mask = bos_pos_mask * abs_mask

        traj_loss = (ppt * valid_traj_mask).sum() / valid_traj_mask.sum().clamp(min=1)
        abs_loss = (ppt * valid_abs_mask).sum() / valid_abs_mask.sum().clamp(min=1)

        abs_positions = abs_mask.bool()
        abs_logits = logits[:, :-1][:, abs_positions, model.vocab_sizes[0]:]

        soft_zipf_kl = self.zipf_loss(abs_logits)

        # --- topo similarity loss --- 
        levels = (data >= model.vocab_sizes[0])
        abs_data = data[levels].reshape((data == BOS_TOKEN_ID).sum(), -1)
        abs_dist = pairwise_hamming_dist(abs_data)

        eos_mask = torch.cat([(data[0, 1:] == BOS_TOKEN_ID), torch.tensor([True], device=data.device)])
        eos_reps = hidden_states[:, eos_mask, :]
        # eos_dist = torch.cdist(eos_reps, eos_reps, p=1)
        # eos_logits = logits[:, eos_mask, :]
        # eos_dist = torch.cdist(eos_logits, eos_logits, p=1)

        eos_rep_norm = F.normalize(eos_reps, p=2, dim=-1)
        cosine_sim = torch.bmm(eos_rep_norm, eos_rep_norm.transpose(1, 2))
        eos_dist = 1 - cosine_sim

        topo_loss = compute_topo_loss(abs_dist, eos_dist, mode=1)

        # --- Return: p(s | a), p(a | s), KL(p(a_t, a_t+1), soft_zipf_prior) ---
        return traj_loss, abs_loss, soft_zipf_kl, topo_loss


class SoRLLoss_v6(nn.Module): 
    """
    SoRL loss: p(s | a) - p(s), p(a | s), KL(p(a_t, a_t+1), soft_zipf_prior), corr(d(a), d(r))
    """

    def __init__(self, abs_vocab_size, decay=0.8, target_vocab_util=0.8):
        super().__init__()
        self.decay = decay
        self.zipf_loss = Zipfian2gramLoss(abs_vocab_size, decay, target_vocab_util)

    def forward(self, data, model, base_traj_ppt, memory_span: int, attn_blocksize: int, 
                      utility_reward: torch.Tensor):
 
        ppt, logits, hidden_states = model.forward_with_hidden_states(data, memory_span, attn_blocksize)
        ppt = ppt.reshape(data.shape[0], -1)
        logits = logits.reshape(data.shape[0], -1, logits.size(-1))
        hidden_states = hidden_states.reshape(data.shape[0], -1, hidden_states.size(-1))

        traj_mask = (data[:, 1:] < model.vocab_sizes[0])
        traj_ppt = ppt[traj_mask]
        info_loss = (traj_ppt - base_traj_ppt.detach()[..., :traj_ppt.shape[-1]]).mean() # traj ppt might be truncated
        
        abs_loss = (ppt * utility_reward)[~traj_mask].mean() # scale on policy gradient only

        # --- KL(p(a_t, a_t+1), soft_zipf_prior) --- 
        abs_logits = logits[:, :-1][:, ~traj_mask.squeeze(0), model.vocab_sizes[0]:]
        soft_zipf_kl = self.zipf_loss(abs_logits)

        # --- topo similarity loss --- 
        levels = (data >= model.vocab_sizes[0])
        abs_data = data[levels].reshape((data == BOS_TOKEN_ID).sum(), -1)
        abs_dist = pairwise_hamming_dist(abs_data)

        eos_mask = torch.cat([(data[0, 1:] == BOS_TOKEN_ID), torch.tensor([True], device=data.device)])
        eos_reps = hidden_states[:, eos_mask, :]

        eos_rep_norm = F.normalize(eos_reps, p=2, dim=-1)
        cosine_sim = torch.bmm(eos_rep_norm, eos_rep_norm.transpose(1, 2))
        eos_dist = 1 - cosine_sim

        topo_loss = compute_topo_loss(abs_dist, eos_dist, mode=1)

        # --- Return: p(s | a), p(a | s), KL(p(a_t, a_t+1), soft_zipf_prior) ---
        return info_loss, abs_loss, soft_zipf_kl, topo_loss 