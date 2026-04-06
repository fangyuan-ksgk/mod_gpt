#!/bin/bash
# ──────────────────────────────────────────────────────────────
# Train SAE on activations from a trained addition model
# Uses EleutherAI's sparsify (eai-sparsify) via SparseCoder
# ──────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/../.."

MODEL_DIR=${1:?Usage: run_sae.sh <model_dir> [sae_save_dir]}
SAE_DIR=${2:-${MODEL_DIR}/sae}

python -c "
import torch, json, sys
sys.path.insert(0, '.')
from arithmetic.reference.addition_6digit import AdditionGAT
from arithmetic.interp_utils.sae_trainer import SAETrainer, SAETrainerConfig, collect_activations

# Load trained model
config = json.load(open('${MODEL_DIR}/config.json'))
wrapper = AdditionGAT(
    n_digits=config['n_digits'],
    n_layer=config['n_layer'],
    n_head=config['n_head'],
    n_embd=config['n_embd'],
    device='cuda',
    compile_model=not config.get('no_compile', False),
)
wrapper.model.load_state_dict(torch.load('${MODEL_DIR}/model.pt'))

# Collect activations
print('Collecting activations...')
acts = collect_activations(wrapper, n_digits=config['n_digits'],
                           n_batches=200, batch_size=64)
print(f'Collected {acts.shape[0]} activation vectors, dim={acts.shape[1]}')

# Train SAE
sae_cfg = SAETrainerConfig(
    d_in=config['n_embd'],
    k=32,
    expansion_factor=16,
    lr=5e-4,
    auxk_alpha=1/32,
)
trainer = SAETrainer(sae_cfg)

n_epochs = 10
batch_size = 256
n_samples = acts.shape[0]
indices = torch.randperm(n_samples)

for epoch in range(n_epochs):
    indices = torch.randperm(n_samples)
    epoch_loss = 0
    n_steps = 0
    for i in range(0, n_samples - batch_size, batch_size):
        batch = acts[indices[i:i+batch_size]]
        info = trainer.step(batch)
        epoch_loss += info['loss']
        n_steps += 1
    avg = epoch_loss / n_steps
    print(f'epoch {epoch+1:2d} | loss: {avg:.4f} | dead: {info[\"num_dead\"]}')

trainer.save('${SAE_DIR}')
print(f'SAE saved to ${SAE_DIR}')
"
