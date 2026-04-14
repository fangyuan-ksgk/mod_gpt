#!/usr/bin/env python3
"""Re-evaluate all HF models with updated eval (C1/C2 splits + digit accuracy)."""
import torch, sys, json, os
sys.path.insert(0, ".")

from arithmetic.hub import load_model
from arithmetic.evaluate import ArithmeticEvaluator
from transformers import AutoTokenizer
from huggingface_hub import HfApi, hf_hub_download

device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
api = HfApi()

files = api.list_repo_files("thoughtworks/arithmetic-sorl")
folders = sorted(set(f.split("/")[0] for f in files if "/" in f and "metrics" in f and "interp" not in f))

for i, name in enumerate(folders):
    print(f"\n[{i+1}/{len(folders)}] {name}")
    try:
        model, cfg, old_metrics = load_model(name, device=device)
        evaluator = ArithmeticEvaluator(model, tokenizer, device=device)
        
        K = cfg.get("K") if cfg.get("mode") in ("sorl", "sorl_v6") else None
        
        # Run updated eval
        sft_results = evaluator.run(ops="add_sub", K=None, n_per_split=50)
        sorl_results = evaluator.run(ops="add_sub", K=K, n_per_split=50) if K is not None else None
        
        # Merge into existing metrics
        old_metrics["sft_eval"] = sft_results
        if sorl_results:
            old_metrics["sorl_eval"] = sorl_results
        
        # Upload updated metrics
        metrics_path = f"/tmp/reeval_{name}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(old_metrics, f, indent=2)
        api.upload_file(
            path_or_fileobj=metrics_path,
            path_in_repo=f"{name}/metrics.json",
            repo_id="thoughtworks/arithmetic-sorl",
            repo_type="model",
        )
        print(f"  Updated: sft={sft_results['summary']['overall_accuracy']:.1%} "
              f"digit={sft_results['summary'].get('digit_accuracy', 0):.1%}"
              + (f" sorl={sorl_results['summary']['overall_accuracy']:.1%}" if sorl_results else ""))
        
        del model; torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ERROR: {e}")
