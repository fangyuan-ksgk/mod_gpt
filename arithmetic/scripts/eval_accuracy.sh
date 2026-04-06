#!/bin/bash
# ────────────────────────────────────────────────────────────���─
# Evaluate a trained model on structured addition test sets
# Reports per-subtask accuracy: BA, MC1, MS9, UC1, US9
# ──────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/../.."

MODEL_DIR=${1:?Usage: eval_accuracy.sh <model_dir>}

python -c "
import torch, json, sys
sys.path.insert(0, '.')
from arithmetic.reference.addition_6digit import AdditionGAT

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

# Quick accuracy
acc = wrapper.eval_accuracy(n_examples=256)
print(f'Overall accuracy (256 random): {acc:.3f}')
print()

# Detailed by subtask
results = wrapper.eval_by_subtask()
for cat, res in results.items():
    subtask_str = ' | '.join(f'{t}: {v:.3f}' for t, v in sorted(res['per_subtask'].items()))
    print(f'  {cat:15s} | full: {res[\"full_acc\"]:.3f} | {subtask_str}')
"
