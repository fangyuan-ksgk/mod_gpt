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
    """Codebook of size C_SIZE in D_bot space. STE + commitment loss."""
    def __init__(self, C_SIZE, D_bot, beta=DEFAULT_BETA, gamma=0.0):
        super().__init__()
        self.codebook = nn.Embedding(C_SIZE, D_bot)
        self.beta  = beta
        self.gamma = gamma  # encoder spread regularization weight

    def forward(self, z):
        z_sq  = (z ** 2).sum(-1, keepdim=True)
        e_sq  = (self.codebook.weight ** 2).sum(-1)
        dots  = z @ self.codebook.weight.T
        dists = z_sq + e_sq - 2 * dots

        ids = dists.argmin(-1)
        e_k = self.codebook(ids)

        commit_loss = F.mse_loss(e_k.detach(), z) * self.beta + \
                      F.mse_loss(e_k, z.detach())

        # Spread regularization: maximize batch variance of encoder output z
        # Prevents encoder collapse where all inputs map to the same point.
        spread_loss = (-self.gamma * z.var(dim=0).mean()) if (self.gamma > 0.0 and z.shape[0] > 1) else z.new_tensor(0.0)

        z_st = z + (e_k - z).detach()  # straight-through estimator
        return ids, z_st, commit_loss + spread_loss

    @torch.no_grad()
    def assign(self, z):
        z_sq  = (z ** 2).sum(-1, keepdim=True)
        e_sq  = (self.codebook.weight ** 2).sum(-1)
        dots  = z @ self.codebook.weight.T
        return (z_sq + e_sq - 2 * dots).argmin(-1)


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
                 C_SIZE=DEFAULT_C_SIZE, beta=DEFAULT_BETA, gamma=0.0):
        super().__init__()
        self.encoder = ChunkEncoder(D, L, D_bot)
        self.vq      = VQModule(C_SIZE, D_bot, beta, gamma=gamma)
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
        """Labeler inference: (B, L, D) -> discrete code ids (B,)"""
        return self.vq.assign(self.encoder(x))

    @torch.no_grad()
    def vocab_utilization(self, x_all):
        """Fraction of codebook entries used across all chunks in x_all."""
        ids = self.encode(x_all)
        return ids.unique().numel() / self.vq.codebook.num_embeddings


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
