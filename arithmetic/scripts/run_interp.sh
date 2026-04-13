#!/bin/bash
# Run interpretability analysis on SoRL models.
# Produces token-subtask heatmaps and per-token profiles.
#
# Usage:
#   bash arithmetic/scripts/run_interp.sh                    # all default models
#   bash arithmetic/scripts/run_interp.sh abs30_100K K=4     # specific model
#
# Output: arithmetic/interp_results/<model_name>/
#   - token_profiles.txt    (per-token subtask/carry/position breakdown)
#   - token_heatmap.png     (P(token|subtask) heatmap)
#   - analysis.json         (raw data for dashboard)

set -e

MODELS="${1:-add_sub_sorl_v1_abs10_100K add_sub_sorl_v1_abs30_100K add_sub_sorl_v1_abs50_100K add_sub_sorl_v1_abs100_100K add_sub_sorl_v1_abs10_K1_100K add_sub_sorl_v1_abs30_K1_100K}"
DEVICE="${2:-cuda:0}"
N_PER_SPLIT="${3:-100}"

echo "═══════════════════════════════════════════════════"
echo "  SoRL Token Interpretability Analysis"
echo "  Models: $(echo $MODELS | wc -w)"
echo "  N/split: $N_PER_SPLIT"
echo "═══════════════════════════════════════════════════"

for MODEL in $MODELS; do
    echo ""
    echo "━━━ $MODEL ━━━"

    python3 -c "
import torch, sys, json, os
from collections import defaultdict, Counter
sys.path.insert(0, '.')

from arithmetic.hub import load_model
from arithmetic.datasets.addition import get_eval_set
from arithmetic.train import QWEN3_TOKEN_MAP, QWEN3_INV_MAP
from sorl.sorl_trainer import infer_insert_mask, insert_tokens_with_padding, expand_prompt_len
from transformers import AutoTokenizer

PROMPT_LEN = 14
ANSWER_LEN = 7
device = '${DEVICE}'
model_name = '${MODEL}'
n_per_split = ${N_PER_SPLIT}
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')

model, config, metrics = load_model(model_name, device=device)
base_v = model.vocab_sizes[0].item()
pad_id = tokenizer.pad_token_id
K = config.get('K', 4)
abs_vocab = config.get('abs_vocab', 10)

out_dir = f'arithmetic/interp_results/{model_name}'
os.makedirs(out_dir, exist_ok=True)

categories = get_eval_set(6, 'add_sub', N=n_per_split)

token_data = defaultdict(lambda: {
    'subtask': Counter(), 'answer_digit_value': Counter(),
    'answer_position': Counter(), 'operation': Counter(),
    'carry_state': Counter(), 'input_sum_mod10': Counter(),
    'count': 0,
})

model.eval()
n = 0
for split_name, examples in categories.items():
    for ex in examples:
        qwen_ids = torch.tensor([QWEN3_TOKEN_MAP[t] for t in ex.tokens], dtype=torch.long, device=device)
        seq = qwen_ids.unsqueeze(0)
        attn = torch.ones_like(seq)
        pl = torch.tensor([PROMPT_LEN], dtype=torch.long, device=device)

        with torch.no_grad():
            im = infer_insert_mask(seq, K, attn)
            ep = expand_prompt_len(pl, im)
            ed, ea = insert_tokens_with_padding(seq, attn, im, model.vocab_sizes[0], pad_id)
            data, ppt, logits = model.recursion(
                ed, ea, max_iterations=2,
                memory_span_abs=1792, memory_span_traj=1792,
                temperature=0.0, prompt_len=ep,
            )

        expanded = data[0]
        is_abs = expanded >= base_v
        is_traj = ~is_abs
        traj_indices = is_traj.nonzero(as_tuple=True)[0]
        abs_indices = is_abs.nonzero(as_tuple=True)[0]

        orig_tokens = [QWEN3_INV_MAP.get(t.item(), -1) for t in qwen_ids]
        op = 'add' if orig_tokens[6] == 10 else 'sub'

        for abs_idx in abs_indices:
            tok_id = (expanded[abs_idx] - base_v).item()
            traj_before = (traj_indices < abs_idx).sum().item() - 1

            if traj_before >= PROMPT_LEN and traj_before < PROMPT_LEN + ANSWER_LEN:
                answer_pos = traj_before - PROMPT_LEN
                label = ex.labels[answer_pos] if answer_pos < len(ex.labels) else '?'
                answer_val = orig_tokens[traj_before] if traj_before < len(orig_tokens) else -1
                d1 = orig_tokens[answer_pos] if answer_pos < 6 else -1
                d2 = orig_tokens[7 + answer_pos] if answer_pos < 6 else -1

                td = token_data[tok_id]
                td['subtask'][label] += 1
                td['answer_digit_value'][answer_val] += 1
                td['answer_position'][answer_pos] += 1
                td['operation'][op] += 1
                if d1 >= 0 and d2 >= 0:
                    s = d1 + d2
                    td['input_sum_mod10'][s % 10] += 1
                    td['carry_state']['carry' if s >= 10 else 'no_carry'] += 1
                td['count'] += 1
        n += 1

# Save profiles
with open(f'{out_dir}/token_profiles.txt', 'w') as f:
    f.write(f'Model: {model_name}\n')
    f.write(f'K={K}, abs_vocab={abs_vocab}, n_examples={n}\n\n')

    for pos in sorted(set(td['answer_position'].most_common(1)[0][0] for td in token_data.values() if td['answer_position'])):
        toks = [(tid, td) for tid, td in token_data.items()
                if td['answer_position'] and td['answer_position'].most_common(1)[0][0] == pos]
        toks.sort(key=lambda x: -x[1]['count'])

        f.write(f'POSITION d{pos} ({len(toks)} tokens)\n')
        f.write('-' * 80 + '\n')
        for tok_id, td in toks:
            cnt = td['count']
            if cnt < 10: continue
            subs = ', '.join(f'{l}={c*100//cnt}%' for l, c in td['subtask'].most_common(3))
            ops = ', '.join(f'{o}={c*100//cnt}%' for o, c in td['operation'].most_common())
            carries = ', '.join(f'{s}={c*100//cnt}%' for s, c in td['carry_state'].most_common())
            sums = ', '.join(f'{v}={c*100//cnt}%' for v, c in td['input_sum_mod10'].most_common(3))
            f.write(f'  t{tok_id:>2d} (n={cnt:>4d}) sub=[{subs}] op=[{ops}] carry=[{carries}] sum%10=[{sums}]\n')
        f.write('\n')

# Save JSON for dashboard
json_data = {}
for tok_id, td in token_data.items():
    json_data[str(tok_id)] = {
        'count': td['count'],
        'subtask': dict(td['subtask']),
        'operation': dict(td['operation']),
        'carry_state': dict(td['carry_state']),
        'answer_position': {str(k): v for k, v in td['answer_position'].items()},
        'answer_digit_value': {str(k): v for k, v in td['answer_digit_value'].items()},
        'input_sum_mod10': {str(k): v for k, v in td['input_sum_mod10'].items()},
    }
with open(f'{out_dir}/analysis.json', 'w') as f:
    json.dump({'model': model_name, 'K': K, 'abs_vocab': abs_vocab,
               'n_examples': n, 'tokens': json_data}, f, indent=2)

# Plot heatmap
try:
    import matplotlib.pyplot as plt
    import numpy as np

    labels = sorted(set(l for td in token_data.values() for l in td['subtask']))
    tids = sorted(token_data.keys())

    matrix = np.zeros((len(tids), len(labels)))
    for i, tid in enumerate(tids):
        cnt = token_data[tid]['count']
        if cnt == 0: continue
        for j, label in enumerate(labels):
            matrix[i, j] = token_data[tid]['subtask'].get(label, 0) / cnt

    fig, ax = plt.subplots(figsize=(max(8, len(labels)*0.7), max(4, len(tids)*0.4)))
    im = ax.imshow(matrix, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticks(range(len(tids)))
    ax.set_yticklabels([f't{t} (n={token_data[t][\"count\"]})' for t in tids], fontsize=8)
    for i in range(len(tids)):
        for j in range(len(labels)):
            v = matrix[i, j]
            if v >= 0.05:
                ax.text(j, i, f'{v:.0%}', ha='center', va='center', fontsize=6,
                        color='white' if v > 0.5 else 'black')
    plt.colorbar(im, label='P(subtask | token)')
    ax.set_title(f'{model_name} (K={K}, abs={abs_vocab})')
    ax.set_xlabel('Quirke Subtask')
    ax.set_ylabel('Abstraction Token')
    plt.tight_layout()
    fig.savefig(f'{out_dir}/token_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved to {out_dir}/')
except Exception as e:
    print(f'  Plot error: {e}')
    print(f'  Text results saved to {out_dir}/')

print(f'  {len(token_data)} unique tokens, {n} examples')
del model
torch.cuda.empty_cache()
"
done

echo ""
echo "Done! Results in arithmetic/interp_results/"
