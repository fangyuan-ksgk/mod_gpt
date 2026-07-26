#!/usr/bin/env bash
# Stage 2 for CodeNet: matched training budget + causal knockout.
#
# The first CodeNet run trained for 125 optimizer steps against the arithmetic
# study's 3125. That is a confound I introduced, not a property of the domain: a
# null at that budget cannot be told apart from undertraining. This re-runs at
# 20K examples (~625 steps, 5x) and changes NOTHING else -- same loss weights,
# same layer, same codebook size. Whatever it produces is what gets reported.
#
# It also runs the codes-on/codes-off knockout that, on arithmetic, was the
# single measurement that explained the otherwise-confusing split verdict.
set -uo pipefail

PY=/lambda/nfs/Amir-steering/codes/dlr/bin/python3
MODEL=Qwen/Qwen3-0.6B
LOGS=amir_interp_rebuttal/logs
mkdir -p "$LOGS" amir_interp_rebuttal/results

# Wait for the stage-1 analysis to release the GPU. Matches the python process
# only -- never this script -- to avoid the self-matching pgrep deadlock that
# stalled the first chain.
while pgrep -f "amir_interp_rebuttal.analyze --study codenet" >/dev/null; do sleep 30; done
echo "[chain2] stage-1 analysis released the GPU at $(date -u +%H:%M:%S)"

# ── knockout on the 125-step model (cheap, and it contextualises stage 1) ──
echo "[chain2] knockout: codenet_v9 (125 steps)"
$PY -u -W ignore -c "
import json
from amir_interp_rebuttal.load_local import load_local_steered
from amir_interp_rebuttal.codenet_dataset import CodeNetDataset
from amir_interp_rebuttal.runner import batched_generate
w, tok, a = load_local_steered('ckpt/codenet_v9', device='cuda')
ds = CodeNetDataset(split='test', tokenizer=tok, max_length=256, size=800)
idxs = list(range(len(ds))); out = {}
for name, sc in [('codes_ON', float(a['scale'])), ('codes_OFF', 0.0)]:
    recs = batched_generate(w, tok, ds, 'cuda', idxs, eval_batch_size=32,
                            max_new_tokens=32, record_codes=False, decode_scale=sc)
    out[name] = sum(r['correct'] for r in recs)/len(recs)
    print(f'{name:>10} decode_scale={sc} acc={out[name]:.1%}')
print(f'DELTA = {100*(out[\"codes_ON\"]-out[\"codes_OFF\"]):.1f} pp')
json.dump(out, open('amir_interp_rebuttal/results/codenet_125step_knockout.json','w'), indent=2)
" > "$LOGS/codenet_knockout_125.log" 2>&1
echo "[chain2] knockout done"

# ── matched-budget retrain ────────────────────────────────────────────
echo "[chain2] retraining codenet at CODENET_SIZE=20000"
CODENET_SIZE=20000 $PY -u train_steer_pt.py \
    --mode v9 --dataset codenet --model_name "$MODEL" \
    --L 8 --C_SIZE 30 --scale 0.1 --inject_layers 14 \
    --alpha_info 10.0 --alpha_abs 0.1 --alpha_zipf 1.0 \
    --max_length 256 --batch_size 8 --gradient_accumulation_steps 4 \
    --num_epochs 1 --lr 1e-5 --steer_lr 1e-3 \
    --num_rollouts 4 --search_temp 1.0 \
    --eval_samples 200 --eval_batch_size 16 --eval_every 200 --max_new_tokens 32 \
    --eval_decode_scale 0.1 \
    --log_every 25 --output_dir ckpt/codenet_v9_20k > "$LOGS/codenet_train_20k.log" 2>&1
echo "[chain2] retrain done"

if [ -f ckpt/codenet_v9_20k/final.pt ]; then
  echo "[chain2] analysing codenet_v9_20k"
  $PY -u -W ignore -m amir_interp_rebuttal.analyze \
      --study codenet --ckpt ckpt/codenet_v9_20k --model_name "$MODEL" \
      --eval_n 800 --max_new_tokens 32 --max_swap_examples 200 \
      > "$LOGS/codenet_analyze_20k.log" 2>&1

  echo "[chain2] knockout: codenet_v9_20k"
  $PY -u -W ignore -c "
import json
from amir_interp_rebuttal.load_local import load_local_steered
from amir_interp_rebuttal.codenet_dataset import CodeNetDataset
from amir_interp_rebuttal.runner import batched_generate
w, tok, a = load_local_steered('ckpt/codenet_v9_20k', device='cuda')
ds = CodeNetDataset(split='test', tokenizer=tok, max_length=256, size=800)
idxs = list(range(len(ds))); out = {}
for name, sc in [('codes_ON', float(a['scale'])), ('codes_OFF', 0.0)]:
    recs = batched_generate(w, tok, ds, 'cuda', idxs, eval_batch_size=32,
                            max_new_tokens=32, record_codes=False, decode_scale=sc)
    out[name] = sum(r['correct'] for r in recs)/len(recs)
    print(f'{name:>10} decode_scale={sc} acc={out[name]:.1%}')
print(f'DELTA = {100*(out[\"codes_ON\"]-out[\"codes_OFF\"]):.1f} pp')
json.dump(out, open('amir_interp_rebuttal/results/codenet_20k_knockout.json','w'), indent=2)
" > "$LOGS/codenet_knockout_20k.log" 2>&1
else
  echo "[chain2] ERROR: no checkpoint at ckpt/codenet_v9_20k/final.pt" >&2
fi

echo "[chain2] ALL COMPLETE $(date -u +%H:%M:%S)"
