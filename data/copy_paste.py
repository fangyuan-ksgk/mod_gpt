import torch
import numpy as np
from dataclasses import dataclass
from sorl.gat_sim import BOS_TOKEN_ID

import torch
import numpy as np

class CopyPasteDataLoader:
    """Generate sequences: [abc...][PLACEHOLDER][abc...]"""
    
    def __init__(self, vocab_size=128, max_token=10, seq_len=256, K=8, device='cpu'):

        self.vocab_size = min(vocab_size, max_token)
        self.seq_len = seq_len
        self.device = device
        self.PLACEHOLDER_TOKEN = vocab_size
    
    def get_batch(self, batch_size):
        """Generate flattened copy-paste sequences."""
        repeat_seqs = torch.randint(0, self.vocab_size, (batch_size, self.seq_len), device=self.device)
        
        bos = torch.full((batch_size, 1), BOS_TOKEN_ID, device=self.device)
        placeholder = torch.full((batch_size, 1), self.PLACEHOLDER_TOKEN, device=self.device)
        
        samples = torch.cat([bos, repeat_seqs, placeholder, repeat_seqs], dim=1)  # [batch_size, 2*seq_len+2]
        
        data = samples.flatten().unsqueeze(0)  # [1, batch_size*(2*seq_len+2)]
        
        mask = torch.ones_like(samples, dtype=torch.float)
        mask[:, self.seq_len + 1] = 0.0  # Placeholder is at position seq_len+1
        loss_mask = mask.flatten().unsqueeze(0)
        
        return data, loss_mask