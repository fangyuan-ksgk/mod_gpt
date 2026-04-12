"""
Re-evaluate all SoRL models on HF using ArithmeticEvaluator (per-split, hard cases).
Updates metrics.json and train_config.json with corrected final_accuracy.
"""
import json
import torch
import tempfile
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

from arithmetic.catalog import ModelCatalog
from arithmetic.hub import load_model, MODEL_REPO
from arithmetic.evaluate import ArithmeticEvaluator


TOKENIZER = None

def get_tokenizer():
    global TOKENIZER
    if TOKENIZER is None:
        TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    return TOKENIZER


def reeval_model(name: str, device: str = "cuda", n_per_split: int = 250):
    """Load model from HF, eval with ArithmeticEvaluator, return results."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"{'='*60}")

    model, config, old_metrics = load_model(name, device=device)
    tokenizer = get_tokenizer()
    evaluator = ArithmeticEvaluator(model, tokenizer, device=device, n_digits=config.get("n_digits", 6))

    K = config.get("K", 4)
    ops = config.get("ops", "add_sub")

    # Run both SFT and SoRL (recursion) eval
    results_sft = evaluator.run(ops=ops, K=None, n_per_split=n_per_split)
    results_sorl = evaluator.run(ops=ops, K=K, n_per_split=n_per_split)

    print(f"\n  SFT eval:")
    evaluator.print_table(results_sft)
    print(f"\n  SoRL eval (K={K}):")
    evaluator.print_table(results_sorl)

    return {
        "sft_eval": results_sft,
        "sorl_eval": results_sorl,
        "sft_overall_accuracy": results_sft["summary"]["overall_accuracy"],
        "sorl_overall_accuracy": results_sorl["summary"]["overall_accuracy"],
        "eval_K": K,
        "eval_method": "ArithmeticEvaluator",
        "n_per_split": n_per_split,
    }


def update_hf_metrics(name: str, new_metrics: dict, repo_id: str = MODEL_REPO):
    """Upload updated metrics and config to HF."""
    api = HfApi()

    local_dir = f"/tmp/hf_cache/{repo_id.split('/')[-1]}"
    config_path = hf_hub_download(repo_id, f"{name}/train_config.json", local_dir=local_dir)
    config = json.load(open(config_path))

    # Update config with corrected accuracy
    config["final_accuracy"] = new_metrics["sorl_overall_accuracy"]
    config["sft_accuracy"] = new_metrics["sft_overall_accuracy"]
    config["eval_method"] = "ArithmeticEvaluator"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with open(tmp / "train_config.json", "w") as f:
            json.dump(config, f, indent=2)
        with open(tmp / "metrics.json", "w") as f:
            json.dump(new_metrics, f, indent=2)

        api.upload_folder(
            folder_path=str(tmp),
            repo_id=repo_id,
            path_in_repo=name,
            commit_message=f"Re-eval {name}: sorl={new_metrics['sorl_overall_accuracy']:.1%}, sft={new_metrics['sft_overall_accuracy']:.1%}",
        )
    print(f"  Updated on HF: {repo_id}/{name}")


def main():
    cat = ModelCatalog()
    cat.fetch(verbose=False)
    print(f"Found {len(cat.entries)} models to re-evaluate")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}

    for entry in cat.entries:
        try:
            new_metrics = reeval_model(entry.name, device=device)
            update_hf_metrics(entry.name, new_metrics)
            results[entry.name] = {
                "sorl": new_metrics["sorl_overall_accuracy"],
                "sft": new_metrics["sft_overall_accuracy"],
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[entry.name] = {"error": str(e)}

        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, m in sorted(results.items()):
        if "error" in m:
            print(f"  {name}: ERROR — {m['error']}")
        else:
            print(f"  {name}: sorl={m['sorl']:.1%}, sft={m['sft']:.1%}")


if __name__ == "__main__":
    main()
