from sorl.gat import GAT
from sorl.utils import infer_level, infer_timestamp, infer_rythmic_insertion_mask, insert_tokens, infer_spike_insertion_mask, infer_valid_masks, group_argmax, group_mean, compute_switch_ratio, combine_rollout, group_min
from copy import deepcopy
import torch
import torch.nn.functional as F
from typing import Optional, Union 
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
# from sorl.model_sorl import SorlModelWrapper


def compute_per_token_loss(model, idx: torch.Tensor, pad_token_id: Optional[int] = None) -> torch.Tensor:
    """
    Computes per-token loss (negative log-likelihood) using a standard forward pass.
    This replaces the custom `model(idx, target)` call from the original GAT model.
    """
    targets = idx[:, 1:].contiguous()
    inputs = idx[:, :-1].contiguous()
    
    outputs = model.forward(input_ids=inputs, output_hidden_states=False)
    logits = outputs.logits
    
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = targets.view(-1)
    
    pad_token_id = pad_token_id if pad_token_id is not None else model.level_mask_tokens[0].item()
    losses = F.cross_entropy(logits_flat, targets_flat, reduction='none', ignore_index=pad_token_id)
    
    ppt = losses.view(targets.shape)
    return ppt


@dataclass 
class SORLConfig: 
    n: int # number of candidates to rollout 
    temperature: float 
    K: int # Rythmic stride for level-1 abstraction
    
    causal_rollout: bool = False # whether to use causal rollout
    budget: Optional[int] = None # max number of abstract tokens allowed

    # parameter used if not causal rollout
    l: Optional[int] = None # level of abstraction to search & learn
    steps: Optional[int] = None # steps for chunk-wise denoise
    max_t_search: Optional[int] = None # max number of tokens to search for abstraction
    start_ts: Optional[torch.Tensor] = None # start timestamp for adding placeholders | one for each sample
    end_ts: Optional[torch.Tensor] = None # end timestamp for adding placeholders | one for each sample
    abstract_budget: int = 5 # max number of spiky abstraction allowed
    use_rhythmic_placeholders: bool = True # whether to use rhythmic placeholders
    use_spike_placeholders: bool = True # whether to use spike placeholders
    temperature_flip: bool = False # Whether to alternate temperature between 0.0 and the specified value
    use_special_placeholders: bool = False # whether to use special placeholders
    special_token_id: int = None # special token id for special placeholders

    # incremental abstraction search 
    curriculum_ratio: float = 0.6 # ratio of curriculum iterations (after this total_step * ratio, abstraction is full-length)

    # memory fading 
    max_seq_len: int = 1024 # max sequence length for training data, useful for memory fading
    use_fade_memory: bool = False # whether to use memory fading
    min_keep: int = 1024 # default to same value as max_length
    use_compression_mask: bool = True # whether to use compression mask
    compression_curriculum_ratio: float = 0.25 # New: Ratio of training to warmup drop_ratio

    # dataset specific
    train_dataset_path: Optional[str] = None
    val_dataset_path: Optional[str] = None
    train_batch_size: int = 128
    val_batch_size: int = 128
    train_iterations: int = 1000
    val_iterations: int = 10
    max_length: int = 1024

    # optimization specific 
    learning_rate: float = 1e-3
    log_interval: int = 100

    # GAPT
    default_phase: Optional[int] = None # default phase for GAPT | when used, GAPT is disabled
    delta: float = 0.01
    tau: float = 0.1
    p_m: int = 10
    p_c: int = 10



# Placeholder addition function (for parallel search & training)
# -----------------------------------------------------------------------------------------------------
def add_rhythmic_placeholders(model, idx: torch.Tensor, l: int, K: int, t_search: Optional[Union[int, torch.Tensor]] = None, start_ts: Optional[torch.Tensor] = None, end_ts: Optional[torch.Tensor] = None): 
    
    tokens = deepcopy(idx)
    levels = infer_level(tokens, model.vocab_sizes, model.level_mask_tokens[0])
    timestamps = infer_timestamp(levels, K, l)
    insert_masks = infer_rythmic_insertion_mask(levels, timestamps, K, l)

    valid_masks = infer_valid_masks(timestamps, start_ts, t_search, end_ts)
    insert_masks *= valid_masks

    tokens = insert_tokens(tokens, insert_masks.int(), model.level_mask_tokens[l], model.level_mask_tokens[0])

    return tokens
    
def add_spike_placeholders(model, data: torch.Tensor, l: int, K: int, abstract_budget: int, t_search: Optional[Union[int, torch.Tensor]] = None, start_ts: Optional[torch.Tensor] = None, end_ts: Optional[torch.Tensor] = None): 

    tokens = deepcopy(data)
    idx = tokens[:, :-1].contiguous()
    target = tokens[:, 1:].contiguous()
    with torch.no_grad():
        ppt = compute_per_token_loss(model, tokens)

    levels = infer_level(tokens, model.vocab_sizes, model.level_mask_tokens[0])
    timestamps = infer_timestamp(levels, K, l)
    insert_masks = infer_spike_insertion_mask(levels, ppt, l, abstract_budget)

    valid_masks = infer_valid_masks(timestamps, start_ts, t_search, end_ts)
    insert_masks *= valid_masks

    tokens = insert_tokens(tokens, insert_masks, model.level_mask_tokens[l], model.level_mask_tokens[0])

    return tokens

def add_special_placeholders(model, idx: torch.Tensor, l: int, t_search: int, special_token_id: int): 
    tokens = deepcopy(idx)
    insert_masks = (tokens == special_token_id).int()
    insert_masks[insert_masks.cumsum(dim=1) > t_search] = 0
    tokens = insert_tokens(tokens, insert_masks, model.level_mask_tokens[l], model.level_mask_tokens[0])

    return tokens

def pad_abstract_tokens(tokens: torch.Tensor, 
                        model,
                        l: int,
                        config: SORLConfig,
                        t_search: Optional[Union[int, torch.Tensor]] = None,
                        start_ts: Optional[torch.Tensor] = None, 
                        end_ts: Optional[torch.Tensor] = None):

    if config.use_spike_placeholders:
        assert config.abstract_budget is not None, "abstract_budgets must be provided for spike placeholders"

    if config.use_spike_placeholders:
        batch_data = add_spike_placeholders(model, tokens, l, config.K, config.abstract_budget, t_search, start_ts, end_ts)

    if config.use_rhythmic_placeholders:
        batch_data = add_rhythmic_placeholders(model, tokens, l, config.K, t_search, start_ts, end_ts)

    if config.use_special_placeholders: 
        batch_data = add_special_placeholders(model, tokens, l, t_search, config.special_token_id)

    return batch_data

def repad_abstract_tokens(tokens: torch.Tensor, model, l: int, K: int, start_ts: torch.Tensor): 
    levels = infer_level(tokens, model.vocab_sizes, model.level_mask_tokens[0])
    timestamps = infer_timestamp(levels, K, l)
    repad_mask = (timestamps >= start_ts.unsqueeze(1)) & (levels == l)
    tokens[repad_mask] = model.level_mask_tokens[l]
    return tokens

def prep_denoise(tokens: torch.Tensor, model):
    levels = infer_level(tokens, model.vocab_sizes, model.level_mask_tokens[0])
    denoise_mask = torch.isin(tokens, model.level_mask_tokens[1:])
    denoise_levels = levels[denoise_mask]
    return denoise_mask, denoise_levels

def chunk_denoise(data: torch.Tensor, model, l: int, K: int, steps: int, 
                   max_t_search: Optional[int] = None, temperature: float = 0.0):

    tokens = deepcopy(data)
    levels = infer_level(tokens, model.vocab_sizes, model.level_mask_tokens[0])
    timestamps = infer_timestamp(levels, K, l)
    max_ts = timestamps.max(dim=1).values
    if max_t_search is not None:
        max_ts = torch.minimum(max_ts, torch.tensor(max_t_search))

    search_ts = torch.ceil(max_ts / steps)

    for step in range(steps): 

        start_ts = (search_ts * step).int()

        # What we actually need here is "repad" -- not new tricks required, just replace abstract tokens with "mask" from certain timestamp onwards
        tokens = repad_abstract_tokens(tokens, model, l, K, start_ts)

        denoise_mask, denoise_levels = prep_denoise(tokens, model)

        if denoise_levels.numel() > 0:
            with torch.no_grad():  
                tokens = model.denoise(tokens, denoise_mask, denoise_levels, temperature=temperature)

        if (start_ts >= max_ts).all(): 
            break 

    return tokens

def heuristic_rollout(data: torch.Tensor, model, l: int, n: int, config: SORLConfig):
    """Heuristic-based decision on when to generate abstraction"""

    data_idx = torch.arange(data.shape[0], device=data.device)

    repeat_data = data.repeat_interleave(n, dim=0)
    repeat_data_idx = data_idx.repeat_interleave(n, dim=0)

    repeat_data = pad_abstract_tokens(repeat_data, model, l, config, t_search=config.max_t_search, start_ts=config.start_ts, end_ts=config.end_ts)

    repeat_data = chunk_denoise(repeat_data, model, l, config.K, steps=config.steps, max_t_search=config.max_t_search, temperature=config.temperature)

    return repeat_data, repeat_data_idx


def sorl_search(data: torch.Tensor, loss_mask: Optional[torch.Tensor],model, config: SORLConfig): 

    # greedy-involved rollout
    assert config.n > 1, "n must be greater than 1"
    with torch.no_grad(): 
        greedy_config = deepcopy(config)
        greedy_config.temperature = 0.0
        search_config = deepcopy(config)
        search_config.temperature = config.temperature

        greedy_data, greedy_data_idx = heuristic_rollout(data, model, l=config.l, n=1, config=greedy_config)
        search_data, search_data_idx = heuristic_rollout(data, model, l=config.l, n=config.n-1, config=search_config)

    combined_data, combined_data_idx = combine_rollout(greedy_data, greedy_data_idx, search_data, search_data_idx, model.level_mask_tokens[0])

    with torch.no_grad():
        ppt = compute_per_token_loss(model, combined_data)

    # select best for each sample idx
    # (TBD). picking the sequence with lowest perplexity is a brute-force selection gadget
    #        a more careful scoring / reward is needed, info-gain is a good candidate

    padded_loss_mask = torch.nn.functional.pad(loss_mask, (0, combined_data.size(1) - loss_mask.size(1)), value=0)
    broadcasted_mask = padded_loss_mask[combined_data_idx][:, 1:].bool()
    masked_ppt = (ppt * broadcasted_mask).sum(dim=1) / broadcasted_mask.sum(dim=1).clamp(min=1)

    reward = - masked_ppt
    # reward = masked_ppt
    idx_max = group_argmax(reward, combined_data_idx) # this is a bug... we want lowest perplexity sample ...
    best_data = combined_data[idx_max]

    switch_ratio = compute_switch_ratio(idx_max, data.size(0))

    return best_data, switch_ratio

def _get_per_level_tensors(data: torch.Tensor, model, tensor_to_distribute: torch.Tensor) -> dict:
    """
    Separates a tensor (like ppt or -ppt for log_probs) into a dictionary keyed by level.
    """
    levels = infer_level(data, model.vocab_sizes, model.level_mask_tokens[0])
    per_level_tensors = {}
    
    # Ensure all levels are present in the dict, even if with None
    for l in range(len(model.vocab_sizes)):
        level_mask = (levels[:, 1:] == l)
        if level_mask.any():
            per_level_tensors[l] = tensor_to_distribute[level_mask]
        else:
            per_level_tensors[l] = None
            
    return per_level_tensors

# loss computation (level-0 is trajectory loss, level >= 1 is abstract loss)
# ------------------------------------------------------------------------------------------------
def compute_loss(data: torch.Tensor, model, ppt: torch.Tensor, loss_mask: torch.Tensor): 
    # loss mask is for trajectory token only
    levels = infer_level(data, model.vocab_sizes, model.level_mask_tokens[0])
    level_loss = {l: torch.tensor(0.) for l in range(1, len(model.full_vocab_size_list))}

    ppt_l0 = ppt[(levels[:, 1:] <= 0)].reshape(ppt.shape[0], -1)
    ppt_l0 = (ppt_l0 * loss_mask[:, 1:]).sum(dim=1) / loss_mask[:, 1:].sum(dim=1)
    ppt_l0 = ppt_l0.mean()

    level_loss.update(group_mean(ppt, levels[:, 1:]))
    level_loss[0] = ppt_l0

    return level_loss[0], sum(level_loss[l] for l in level_loss if l > 0)

# Sub-optimal way of evaluating search improvement || We'd like to have "evaluate" function that properly does token-by-token generation
# ---------------------------------------------------------------------------------------------------------------------------------------
def compute_vocab_utilization_rate(data: torch.Tensor, model):
    si, ei = model.vocab_sizes.cumsum(dim=0)
    return data[(data >= si) & (data < ei)].unique().size(0) / (ei - si).item()

def evaluate(data: torch.Tensor, loss_mask: torch.Tensor, config: SORLConfig, model, 
                       search_temperature: float = 100., search_n: int = 10, ref_model = None): 

    assert config.n > 1, "n must be greater than 1"
    with torch.no_grad(): 
        greedy_config = deepcopy(config)
        greedy_config.temperature = 0.0
        search_config = deepcopy(config)
        search_config.temperature = search_temperature

        greedy_data, greedy_data_idx = heuristic_rollout(data, model, l=config.l, n=1, config=greedy_config)
        search_data, search_data_idx = heuristic_rollout(data, model, l=config.l, n=search_n, config=search_config)

    combined_data, combined_data_idx = combine_rollout(greedy_data, greedy_data_idx, search_data, search_data_idx, model.level_mask_tokens[0])

    ref_model = ref_model if ref_model is not None else model # tentative, ready for EMA
    with torch.no_grad():
        ppt = compute_per_token_loss(model, combined_data)
        ppt_no_abs = compute_per_token_loss(ref_model, data)

    # Broadcast and mask the loss
    padded_loss_mask = torch.nn.functional.pad(loss_mask, (0, combined_data.size(1) - loss_mask.size(1)), value=0)
    broadcasted_mask = padded_loss_mask[combined_data_idx][:, 1:].bool()
    mean_losses = (ppt * broadcasted_mask).sum(dim=1) / broadcasted_mask.sum(dim=1).clamp(min=1)

    # Advantage 1: Greedy vs. High-Temp Search
    greedy_losses = mean_losses[:data.shape[0]]
    search_losses = mean_losses[data.shape[0]:]
    avg_search_losses = torch.tensor([group_mean(search_losses, search_data_idx).get(i, 0) for i in range(data.shape[0])], device=model.device)
    greedy_advantage = (avg_search_losses - greedy_losses) / avg_search_losses.clamp(min=1e-8)

    # Advantage 2: Best vs. Average
    best_losses = group_min(mean_losses, combined_data_idx)
    avg_total_losses = torch.tensor([group_mean(mean_losses, combined_data_idx).get(i, 0) for i in range(data.shape[0])], device=model.device)
    best_advantage = (avg_total_losses - best_losses) / avg_total_losses.clamp(min=1e-8)

    # Advantage 3: Greedy vs. Abstraction-Free
    abstract_free_losses = (ppt_no_abs * loss_mask[:, 1:]).mean(axis=1)
    greedy_info_gain = (abstract_free_losses - greedy_losses) / abstract_free_losses.clamp(min=1e-8)

    return 100 * greedy_advantage.mean(), 100 * best_advantage.mean(), 100 * greedy_info_gain.mean(), greedy_losses.mean(), abstract_free_losses.mean()


class SearchScheduler: 
    def __init__(self, sorl_config: SORLConfig): 
        self.sorl_config = sorl_config
        self.K = sorl_config.K
        self.max_ts = min(sorl_config.max_length // self.K, sorl_config.max_t_search)

        if self.max_ts == 0: 
            print("- No abstraction allowed")
        
        # t_search curriculum
        t_search_curriculum_iters = int(sorl_config.train_iterations * sorl_config.curriculum_ratio)
        self.t_delta = self.max_ts / t_search_curriculum_iters if t_search_curriculum_iters > 0 else 0
        self.t_search = 0 

        # New: Independent drop_ratio curriculum
        self.use_compression_mask = sorl_config.use_compression_mask
        self.drop_ratio = 0.0
        compression_curriculum_iters = int(sorl_config.train_iterations * sorl_config.compression_curriculum_ratio)
        self.drop_ratio_delta = 1.0 / compression_curriculum_iters if compression_curriculum_iters > 0 else 0

        # memory fading experiment || gradually decrease 't_keep' at the later half of training
        self.use_fade_memory = sorl_config.use_fade_memory
        self.max_memory_span = sorl_config.max_seq_len
        self.min_memory_span = sorl_config.min_keep
        self.use_compression_mask = sorl_config.use_compression_mask
        self.memory_span = self.max_memory_span

    def step(self): 
        self.t_search = min(self.t_search + self.t_delta, self.max_ts)
        
        if self.use_fade_memory:
            new_memory_span = max(self.max_memory_span - (self.t_search * self.K), self.min_memory_span)
            # Apply smoothing to avoid abrupt changes
            self.memory_span = max(int(0.7 * self.memory_span + 0.3 * new_memory_span), self.min_memory_span)
        
        if self.use_compression_mask:
            self.drop_ratio = min(self.drop_ratio + self.drop_ratio_delta, 1.0)
        
        return int(self.t_search), self.drop_ratio


# Phase change regulator
class GatedPhaseTransition:
    """
    Gated Phase Transition (GAPT) : https://arxiv.org/pdf/2505.08727
    """
    def __init__(self, delta: float, tau: float, p_m: int, p_c: int):

        # Hyperparameters
        self.delta = delta
        self.tau = tau
        self.p_m = p_m
        self.p_c = p_c

        # State
        self.phi = 1  # 1 for memorization, 2 for compression
        self.s_m = 0
        self.s_c = 0
        self.E_min = float('inf')
        self.abs_loss_min = float('inf')

    def step(self, ssl_loss: float, abs_loss: float):

        L_ce = ssl_loss

        delta_E = self.E_min - L_ce
        self.E_min = min(self.E_min, L_ce)

        if self.phi == 1:  # Memorization phase
            if delta_E > self.delta:
                self.s_m = 0
            else:
                self.s_m += 1
            
            if self.s_m >= self.p_m:
                self.phi = 2
                self.s_c = 0
                self.E_min = float('inf')  # Reset for the new phase
                self.abs_loss_min = float('inf')

        elif self.phi == 2:  # Compression phase
            # Condition 1: SSL loss spikes
            if self.E_min != float('inf') and L_ce > self.E_min * (1 + self.tau):
                self.phi = 1
                self.s_m = 0
            else:
                # Condition 2: Abstraction loss plateaus
                delta_M = self.abs_loss_min - abs_loss
                self.abs_loss_min = min(self.abs_loss_min, abs_loss)

                if delta_M > self.delta:
                    self.s_c = 0
                else:
                    self.s_c += 1
                
                if self.s_c >= self.p_c:
                    self.phi = 1
                    self.s_m = 0
        
        return self.phi 