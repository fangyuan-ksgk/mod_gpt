#!/usr/bin/env bash
# Gate a freshly-trained arithmetic checkpoint, then -- only if the gate opens --
# run the targeted repair sweep on it.
#
# The whole repair programme has been blocked on one thing: repair is gated on
# causal load. arith_v9_paperhp answers arithmetic well (86.35%) but its codes
# are a read-out (+0.15pp knockout), and you cannot repair a computation through
# a channel it ignores. The gated CodeNet checkpoint has the causal load
# (-39.3%) but only 17% accuracy, so its errors are unfixable for the opposite
# reason. Neither can show repair working.
#
# These runs train arithmetic at scale 0.5 / 1.0 -- the knob that opened the
# CodeNet gate and the one dimension the earlier escalation sweep never varied.
# If the gate opens we finally have the missing quadrant: a competent model with
# load-bearing codes, which is the only place targeted repair could show a
# positive.
#
# Usage: bash gate_then_repair.sh ckpt/arith_s0.5_i10_z1_u8
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/lambda/nfs/Amir-steering/codes/dlr/bin/python3
CK="${1:?usage: gate_then_repair.sh <ckpt_dir>}"
TAG=$(basename "$CK")
LOGS=amir_interp_rebuttal/logs
RES=amir_interp_rebuttal/results

[ -f "$CK/final.pt" ] || { echo "no final.pt in $CK"; exit 1; }

echo "=== gating $TAG (4 arms, batch 32, arithmetic prompt is fixed-length) ==="
$PY -u -W ignore -c "
import json, torch
from amir_interp_rebuttal.load_local import load_local_steered
from amir_interp_rebuttal.arith_dataset import ArithmeticDataset
from amir_interp_rebuttal.runner import batched_generate
w,tok,a = load_local_steered('$CK', device='cuda')
ds = ArithmeticDataset(split='test', tokenizer=tok, max_length=64)
idxs=list(range(min(2600,len(ds)))); sc=float(a['scale']); out={'scale':sc,'L':int(a['L'])}
def run(tag, **kw):
    r=batched_generate(w,tok,ds,'cuda',idxs,eval_batch_size=32,max_new_tokens=8,
                       record_codes=False,decode_scale=sc,**kw)
    out[tag]=sum(x['correct'] for x in r)/len(r); print(tag, round(out[tag],4), flush=True)
sv=w.steering_emb.weight.data.clone()
run('codes_ON')
g=torch.Generator(device='cpu').manual_seed(0)
w.steering_emb.weight.data.copy_(sv[torch.randperm(sv.shape[0],generator=g)]); run('codes_RANDOM')
w.steering_emb.weight.data.zero_(); run('codes_OFF_full')
w.steering_emb.weight.data.copy_(sv)
out['delta_pp']=100*(out['codes_ON']-out['codes_OFF_full'])
out['delta_rel_pct']=(100*(out['codes_ON']-out['codes_OFF_full'])/out['codes_ON']) if out['codes_ON'] else 0.0
out['gate_open']=bool(out['delta_pp']>=3.0 or out['delta_rel_pct']>=15.0)
print('DELTA_PP',round(out['delta_pp'],2),'DELTA_REL',round(out['delta_rel_pct'],1),
      'GATE', 'OPEN' if out['gate_open'] else 'closed', flush=True)
json.dump(out,open('$RES/${TAG}_knockout4.json','w'),indent=2)
" 2>&1 | tee "$LOGS/${TAG}_knockout.log"

OPEN=$($PY -c "import json;print(json.load(open('$RES/${TAG}_knockout4.json'))['gate_open'])")
if [ "$OPEN" != "True" ]; then
  echo "gate CLOSED for $TAG -- not running repair. Repair on a read-out channel"
  echo "is uninformative, and running it anyway would produce a null that means"
  echo "nothing. Recorded in $RES/${TAG}_knockout4.json."
  exit 0
fi

echo "=== gate OPEN -- building error taxonomy on $TAG ==="
$PY -u -m amir_interp_rebuttal.error_repair --study arithmetic --ckpt "$CK" \
    --dump --eval_n 2600 --max_new_tokens 8 2>&1 | tee "$LOGS/${TAG}_taxonomy.log"

echo "=== targeted repair on single-digit carry errors ==="
$PY -u -m amir_interp_rebuttal.error_repair --study arithmetic --ckpt "$CK" \
    --repair --from_taxonomy --eval_n 2600 --max_new_tokens 8 \
    --error_class single_digit --mode targeted --max_examples 40 \
    2>&1 | tee "$LOGS/${TAG}_repair_single_digit.log"
