
python eval_sorl.py \
    --hf_repo_id "Ksgk-fy/sorl" \
    --hf_filename "ts-k4-v128.pt" \
    --model_size "small" \
    --abstract_vocab_size 128 \
    --num_rollouts 5 \
    --K 4 \
    --max_iterations 2 \
    --min_temperature 0.0 \
    --max_temperature 5.0 \
    --save_path "eval_sorl.csv" \
    --split "validation" \
    --num_stories 1000 \
    --max_len 32 \
    --batch_size 8