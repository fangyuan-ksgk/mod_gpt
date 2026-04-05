import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Default config (paper: Token Assorted [26]) ----
DEFAULT_L      = 16     # chunk size (compression rate)
DEFAULT_C_SIZE = 1024   # codebook size
DEFAULT_D_BOT  = 256    # bottleneck dim
DEFAULT_BETA   = 0.25   # commitment loss weight
DEFAULT_M_SET  = [0, 72, 128, 160, 192, 224, 256]  # replacement schedule


# ---------------------------------------------------------------------------
# VQ-VAE components
# ---------------------------------------------------------------------------

class ChunkEncoder(nn.Module):
    """(B, L, D) -> (B, D_bot): flatten L token embeddings, project to bottleneck."""
    def __init__(self, D, L, D_bot):
        super().__init__()
        self.proj = nn.Linear(D * L, D_bot)

    def forward(self, x):
        B = x.shape[0]
        return self.proj(x.reshape(B, -1))


class VQModule(nn.Module):
    """
    Codebook in D_bot space with EMA update + dead-code reinitialization.

    Standard trick (VQ-VAE-2 / VQGAN / Jukebox) to prevent codebook collapse:
    - Codebook entries updated via EMA toward their assigned encoder outputs.
    - Only commitment loss (encoder → codebook) flows back to the encoder.
    - Entries whose EMA usage < dead_threshold are reinitialized from batch.
    """
    def __init__(self, C_SIZE, D_bot, beta=DEFAULT_BETA, decay=0.99, dead_threshold=0.01):
        super().__init__()
        self.C_SIZE         = C_SIZE
        self.beta           = beta
        self.decay          = decay
        self.dead_threshold = dead_threshold

        self.codebook = nn.Embedding(C_SIZE, D_bot)
        self.register_buffer('cluster_size', torch.zeros(C_SIZE))
        self.register_buffer('cluster_sum',  torch.zeros(C_SIZE, D_bot))

    def forward(self, z):
        # z: (B, D_bot)
        z_sq  = (z ** 2).sum(-1, keepdim=True)
        e_sq  = (self.codebook.weight ** 2).sum(-1)
        dists = z_sq + e_sq - 2 * (z @ self.codebook.weight.T)

        ids = dists.argmin(-1)          # (B,)
        e_k = self.codebook(ids)        # (B, D_bot)

        # Commitment loss: pull encoder toward (stop-grad) codebook entry.
        # No codebook gradient here — codebook updated via EMA below.
        commit_loss = F.mse_loss(e_k.detach(), z) * self.beta

        if self.training:
            with torch.no_grad():
                z_sg    = z.detach()
                one_hot = F.one_hot(ids, self.C_SIZE).float()  # (B, C)
                counts  = one_hot.sum(0)                        # (C,)
                sums    = one_hot.T @ z_sg                      # (C, D_bot)

                self.cluster_size.mul_(self.decay).add_(counts * (1 - self.decay))
                self.cluster_sum.mul_(self.decay).add_(sums   * (1 - self.decay))

                # Laplace-smoothed codebook update
                n        = self.cluster_size.sum()
                smoothed = (self.cluster_size + 1e-5) / (n + self.C_SIZE * 1e-5) * n
                self.codebook.weight.data.copy_(self.cluster_sum / smoothed.unsqueeze(1))

                # Dead-code reinit: replace unused entries with random batch encodings
                dead   = self.cluster_size < self.dead_threshold
                n_dead = int(dead.sum().item())
                if n_dead > 0:
                    src = torch.randint(0, z_sg.shape[0], (n_dead,), device=z_sg.device)
                    self.codebook.weight.data[dead] = z_sg[src]
                    self.cluster_size[dead]         = self.dead_threshold
                    self.cluster_sum[dead]          = z_sg[src] * self.dead_threshold

        z_st = z + (e_k - z).detach()  # straight-through estimator
        return ids, z_st, commit_loss

    @torch.no_grad()
    def assign(self, z):
        z_sq = (z ** 2).sum(-1, keepdim=True)
        e_sq = (self.codebook.weight ** 2).sum(-1)
        return (z_sq + e_sq - 2 * (z @ self.codebook.weight.T)).argmin(-1)


class ChunkDecoder(nn.Module):
    """(B, D_bot) -> (B, L, D): reconstruct L token embeddings."""
    def __init__(self, D, L, D_bot):
        super().__init__()
        self.proj = nn.Linear(D_bot, D * L)
        self.D, self.L = D, L

    def forward(self, z):
        return self.proj(z).reshape(z.shape[0], self.L, self.D)


class TokenAssortedVQVAE(nn.Module):
    """
    VQ-VAE labeler for Token Assorted baseline.

    Training (forward):   returns recon_loss + commitment_loss
    Inference (encode):   maps (B, L, D) chunks -> discrete code ids (B,)
    """
    def __init__(self, D, L=DEFAULT_L, D_bot=DEFAULT_D_BOT,
                 C_SIZE=DEFAULT_C_SIZE, beta=DEFAULT_BETA, decay=0.99, dead_threshold=0.01):
        super().__init__()
        self.encoder = ChunkEncoder(D, L, D_bot)
        self.vq      = VQModule(C_SIZE, D_bot, beta, decay=decay, dead_threshold=dead_threshold)
        self.decoder = ChunkDecoder(D, L, D_bot)

    def forward(self, x):
        """x: (B, L, D) -> ids, x_hat, recon_loss, commit_loss, total_loss"""
        z                       = self.encoder(x)
        ids, z_q, commit_loss   = self.vq(z)
        x_hat                   = self.decoder(z_q)
        recon_loss              = F.mse_loss(x_hat, x)
        return ids, x_hat, recon_loss, commit_loss, recon_loss + commit_loss

    @torch.no_grad()
    def encode(self, x):
        """x: (B, L, D) -> ids (B,)"""
        x = x.to(self.encoder.proj.weight.device)
        z = self.encoder(x)
        return self.vq.assign(z)

    @torch.no_grad()
    def vocab_utilization(self, x):
        """Returns the fraction of the codebook used by batch x."""
        ids = self.encode(x)
        unique_ids = torch.unique(ids)
        return len(unique_ids) / self.vq.C_SIZE


# -------------------------------------------------------------------
# Mixed Sequence Data Prep (PyTorch Dataset & Collate)
# -------------------------------------------------------------------

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class MixedSequenceDataset(Dataset):
    """
    On-the-fly generation of mixed sequences using the frozen VQ-VAE.
    Produces input_ids, attention_mask, labels (for standard causal LM training).
    Expects dataset items to be parsed via `parse_sample` (which returns prompt, full_text).
    """
    def __init__(self, dataset, tokenizer, emb_table, vqvae, latent_offset, abs_begin_id, abs_end_id, L=16):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.emb_table = emb_table
        self.vqvae = vqvae
        self.latent_offset = latent_offset
        self.abs_begin_id = abs_begin_id
        self.abs_end_id = abs_end_id
        self.L = L

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ex = self.dataset.dataset[idx]
        
        # Get standardized prompt and full text from the specific dataset class
        prompt, full_text = self.dataset.parse_sample(ex)
        
        # Determine prompt_len (used for loss masking)
        q_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_len = len(q_ids)

        # Tokenize full text to get raw token IDs
        full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
        
        # Identify the "reasoning" part (everything after prompt)
        cot_ids = full_ids[prompt_len:]
        
        # Extract chunks (n_chunks, L, D_MODEL)
        n_chunks = len(cot_ids) // self.L
        if n_chunks == 0:
            cot_chunks = torch.zeros(0, self.L, self.emb_table.shape[1], device=self.emb_table.device)
        else:
            ids_trunc = torch.tensor(cot_ids[:n_chunks * self.L]).reshape(n_chunks, self.L)
            cot_chunks = self.emb_table[ids_trunc]
            
        # Determine how many chunks to replace stochastically
        m = sample_replacement_length(len(cot_ids), L=self.L)
        n_replace = m // self.L
        
        # Build mixed sequence
        mixed = list(q_ids)
        if n_replace > 0 and n_chunks > 0:
            if self.abs_begin_id is not None:
                mixed.append(self.abs_begin_id)
                
            with torch.no_grad():
                # ensure cot_chunks is on the right device for the vqvae model
                lat_ids = self.vqvae.encode(cot_chunks)
                
            for k in range(min(n_replace, n_chunks)):
                mixed.append(int(lat_ids[k].item()) + self.latent_offset)
                
            if self.abs_end_id is not None:
                mixed.append(self.abs_end_id)
                
        # Append the rest of the uncompressed text
        mixed += cot_ids[m:]

        return {
            "input_ids": torch.tensor(mixed, dtype=torch.long),
            "prompt_len": prompt_len
        }

def mixed_sequence_collate_fn(batch, pad_token_id):
    """Pads sequences and builds standard causal LM labels (-100 for prompt & padding)."""
    input_ids = [item["input_ids"] for item in batch]
    prompt_lens = [item["prompt_len"] for item in batch]

    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_mask = (padded_input_ids != pad_token_id).long()

    labels = padded_input_ids.clone()
    labels[labels == pad_token_id] = -100
    
    # Mask out the prompt so loss is only computed on CoT + Answer
    for i, p_len in enumerate(prompt_lens):
        labels[i, :p_len] = -100

    return {
        "input_ids": padded_input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# ---------------------------------------------------------------------------
# VQCodebook: hidden-state-space quantization for abs-proj pre-training
# ---------------------------------------------------------------------------

class VQCodebook(nn.Module):
    """
    VQ codebook operating directly in hidden-state space (D).

    Used to pre-train abstract embedding rows in the v6 trainer via
    topology-preserving quantization of frozen LLM hidden states.
    Motivation: 5x better mNN overlap vs diagonal/orthogonal init (vq-vae.ipynb).

    Unlike VQModule (D_bot bottleneck for chunk compression), this module
    works in the full hidden-state dimension and is intended for use with
    _pretrain_abs_projection_vq() in SoRLTrainer.
    """

    def __init__(self, V: int, D: int, beta: float = 0.25):
        super().__init__()
        self.V    = V
        self.beta = beta
        self.codebook = nn.Embedding(V, D)

    def forward(self, h: torch.Tensor):
        """h: (N, D) — returns (ids, h_st, vq_loss)"""
        h_sq  = (h ** 2).sum(-1, keepdim=True)
        e_sq  = (self.codebook.weight ** 2).sum(-1)
        dots  = h @ self.codebook.weight.T
        dists = h_sq + e_sq - 2 * dots

        ids  = dists.argmin(-1)
        e_k  = self.codebook(ids)

        loss = F.mse_loss(e_k.detach(), h) * self.beta + \
               F.mse_loss(e_k, h.detach())

        h_st = h + (e_k - h).detach()
        return ids, h_st, loss

    @torch.no_grad()
    def assign(self, h: torch.Tensor) -> torch.Tensor:
        h_sq  = (h ** 2).sum(-1, keepdim=True)
        e_sq  = (self.codebook.weight ** 2).sum(-1)
        dots  = h @ self.codebook.weight.T
        return (h_sq + e_sq - 2 * dots).argmin(-1)

    @torch.no_grad()
    def vocab_utilization(self, h: torch.Tensor) -> float:
        return self.assign(h).unique().numel() / self.V


# ---------------------------------------------------------------------------
# Special token registration helpers
# ---------------------------------------------------------------------------

ABS_BEGIN_TOKEN = "<abs_begin>"
ABS_END_TOKEN   = "<abs_end>"

def add_abs_special_tokens(tokenizer):
    """
    Add <abs_begin> / <abs_end> to the tokenizer.
    Call this before resizing the LM embedding table.
    Returns (abs_begin_id, abs_end_id).
    """
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [ABS_BEGIN_TOKEN, ABS_END_TOKEN]}
    )
    abs_begin_id = tokenizer.convert_tokens_to_ids(ABS_BEGIN_TOKEN)
    abs_end_id   = tokenizer.convert_tokens_to_ids(ABS_END_TOKEN)
    return abs_begin_id, abs_end_id


# ---------------------------------------------------------------------------
# Randomized latent code replacement
# ---------------------------------------------------------------------------

def sample_replacement_length(cot_len, M_set=None, L=DEFAULT_L):
    """
    Sample m in {0, L, 2L, ..., m_max}, capped at cot_len.
    Following paper's schedule: M = {0, 72, 128, 160, 192, 224, 256}.
    """
    if M_set is None:
        M_set = DEFAULT_M_SET
    m_max   = random.choice(M_set)
    m_max   = min(m_max, (cot_len // L) * L)
    choices = list(range(0, m_max + 1, L))
    return random.choice(choices)


@torch.no_grad()
def assign_latent_ids(cot_embeddings, labeler, L=DEFAULT_L):
    """
    cot_embeddings: (T, D) float32 -- CoT token embeddings from a frozen LLM
    Returns: latent_ids (n_chunks,), where n_chunks = T // L.
    Trailing T % L tokens are left as text (not quantized).
    """
    T        = cot_embeddings.shape[0]
    n_chunks = T // L
    if n_chunks == 0:
        return torch.tensor([], dtype=torch.long)
    chunks = cot_embeddings[:n_chunks * L].reshape(n_chunks, L, -1)
    return labeler.encode(chunks)


def build_mixed_sequence(
    question_ids,
    cot_ids,
    cot_embeddings,
    labeler,
    answer_ids=None,
    latent_offset=0,
    abs_begin_id=None,
    abs_end_id=None,
    M_set=None,
    L=DEFAULT_L,
):
    """
    Replace the first m CoT tokens with discrete latent codes, wrapped in
    <abs_begin> ... <abs_end> bracket tokens (paper convention).

    Args:
        question_ids:   list[int]   question token ids
        cot_ids:        list[int]   full CoT token ids (text)
        cot_embeddings: (T, D)      frozen-LLM embeddings for cot_ids
        labeler:        TokenAssortedVQVAE (trained)
        answer_ids:     list[int]   optional answer token ids
        latent_offset:  int         offset so latent codes don't collide with text vocab
                                    (set to tokenizer.vocab_size + 2 after adding special tokens)
        abs_begin_id:   int         token id for <abs_begin>  (from add_abs_special_tokens)
        abs_end_id:     int         token id for <abs_end>    (from add_abs_special_tokens)
        M_set:          replacement schedule; defaults to paper's schedule
        L:              chunk size

    Returns:
        mixed_ids (list[int]), m_used (int)

    Sequence layout (with brackets):
        [question] <abs_begin> latent_0 .. latent_{m/L-1} <abs_end> [cot_text_m ..] [answer]

    Sequence layout (without brackets, abs_begin_id=None):
        [question] latent_0 .. latent_{m/L-1} [cot_text_m ..] [answer]
    """
    m         = sample_replacement_length(len(cot_ids), M_set=M_set, L=L)
    n_replace = m // L

    latent_ids_all = assign_latent_ids(cot_embeddings, labeler, L)

    mixed = list(question_ids)

    if n_replace > 0:
        if abs_begin_id is not None:
            mixed.append(abs_begin_id)
        for k in range(n_replace):
            mixed.append(int(latent_ids_all[k].item()) + latent_offset)
        if abs_end_id is not None:
            mixed.append(abs_end_id)

    mixed += list(cot_ids[m:])
    if answer_ids:
        mixed += list(answer_ids)

    return mixed, m
