# mod_gpt
Modified GPT model pre-training for GPU poor

```bash
git clone https://github.com/fangyuan-ksgk/mod_gpt.git && cd mod_gpt
pip install torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn==2.8.3 --no-build-isolation
pip install -r requirements.txt
python data/cached_fineweb10B.py 8 # downloads only the first 800M training tokens to save time

python data/tinystory.py # prepare tinystory dataset
```

For baseline (GPT-2), run 
```bash
torchrun --standalone --nproc_per_node=1 train_base.py
```

For SoRL, run 
```bash
torchrun --standalone --nproc_per_node=1 train_sorl_v5.py
```
