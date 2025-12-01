# Mutual Information Loss 
import torch 
from torch import nn 
import torch.nn.functional as F
from sorl.gat_sim import BOS_TOKEN_ID

class MutualInformationLoss(nn.Module):
    def __init__(self, vocab_size, decay=0.99, target_vocab_util=0.8):
        super().__init__()
        self.decay = decay        
        target_val = torch.log(torch.tensor(vocab_size * target_vocab_util, device=vocab_size.device))
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

    def __init__(self, abs_vocab_size, decay=0.99, target_vocab_util=0.8):
        super().__init__()
        self.decay = decay
        self.mutual_info_loss = MutualInformationLoss(abs_vocab_size, decay, target_vocab_util)

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

        marginal_entropy, conditional_entropy = self.mutual_info_loss(abs_logits)

        # --- Return: p(s | a), p(a | s), H(p(a)), H(p(a | s)) ---
        return traj_loss, abs_loss, marginal_entropy, conditional_entropy