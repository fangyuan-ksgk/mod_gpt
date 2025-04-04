# mod_gpt
Modified GPT model pre-training for GPU poor

```bash
git clone https://github.com/fangyuan-ksgk/mod_gpt.git && cd mod_gpt
pip install -r requirements.txt
pip install --pre torch==2.7.0.dev20250110+cu126 --index-url https://download.pytorch.org/whl/nightly/cu126 --upgrade
python data/cached_fineweb10B.py 8 # downloads only the first 800M training tokens to save time
bash run.sh
```
