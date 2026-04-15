"""
Re-evaluate all VALID SoRL models on HF using canonical eval set (N=100 from HF).
Updates metrics.json and train_config.json with consistent numbers.

Usage:
    python -m arithmetic.scripts.reeval_hf_models [--device cuda:0] [--dry-run]
"""
import json
import torch
import tempfile
import argparse
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


def reeval_model(name: str, device: str = "cuda"):
    """Load model from HF, eval with canonical N=100 eval set, return results."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"{'='*60}")

    model, config, old_metrics = load_model(name, device=device)
    tokenizer = get_tokenizer()
    evaluator = ArithmeticEvaluator(
        model, tokenizer, device=device,
        n_digits=config.get("n_digits", 6),
    )

    K = config.get("K", 4)
    is_sorl = config.get("mode", "baseline") != "baseline"

    # SFT eval (no recursion)
    results_sft = evaluator.run(K=None)
    print(f"\n  SFT eval:")
    evaluator.print_table(results_sft)

    # SoRL eval (with recursion) — only for SoRL models
    results_sorl = None
    if is_sorl:
        results_sorl = evaluator.run(K=K)
        print(f"\n  SoRL eval (K={K}):")
        evaluator.print_table(results_sorl)

    del model
    torch.cuda.empty_cache()

    out = {
        "sft_eval": results_sft,
        "sft_overall_accuracy": results_sft["summary"]["overall_accuracy"],
        "sft_digit_accuracy": results_sft["summary"]["digit_accuracy"],
        "eval_method": "ArithmeticEvaluator_canonical_N100",
    }
    if results_sorl is not None:
        out["sorl_eval"] = results_sorl
        out["sorl_overall_accuracy"] = results_sorl["summary"]["overall_accuracy"]
        out["sorl_digit_accuracy"] = results_sorl["summary"]["digit_accuracy"]
        out["eval_K"] = K

    return out


def update_hf_metrics(name: str, new_metrics: dict, repo_id: str = MODEL_REPO):
    """Upload updated metrics and config to HF."""
    api = HfApi()

    local_dir = f"/tmp/hf_cache/{repo_id.split('/')[-1]}"
    config_path = hf_hub_download(
        repo_id, f"{name}/train_config.json", local_dir=local_dir)
    config = json.load(open(config_path))

    # Update config with corrected accuracy
    sorl_acc = new_metrics.get("sorl_overall_accuracy")
    sft_acc = new_metrics["sft_overall_accuracy"]
    config["final_accuracy"] = sorl_acc if sorl_acc is not None else sft_acc
    config["sft_accuracy"] = sft_acc
    config["eval_method"] = "ArithmeticEvaluator_canonical_N100"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with open(tmp / "train_config.json", "w") as f:
            json.dump(config, f, indent=2)
        with open(tmp / "metrics.json", "w") as f:
            json.dump(new_metrics, f, indent=2)

        label = f"sorl={sorl_acc:.1%}" if sorl_acc is not None else ""
        api.upload_folder(
            folder_path=str(tmp),
            repo_id=repo_id,
            path_in_repo=name,
            commit_message=f"Re-eval {name} (canonical N=100): sft={sft_acc:.1%} {label}",
        )
    print(f"  Updated on HF: {repo_id}/{name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dry-run", action="store_true",
                   help="Eval but don't upload to HF")
    args = p.parse_args()

    cat = ModelCatalog()
    cat.fetch(verbose=False)
    valid = [e for e in cat.entries if e.status == "VALID"]
    print(f"Found {len(valid)} VALID models to re-evaluate")

    results = {}
    for entry in valid:
        try:
            new_metrics = reeval_model(entry.name, device=args.device)
            if not args.dry_run:
                update_hf_metrics(entry.name, new_metrics)
            results[entry.name] = {
                "sft": new_metrics["sft_overall_accuracy"],
                "sorl": new_metrics.get("sorl_overall_accuracy"),
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
        elif m["sorl"] is not None:
            print(f"  {name}: sft={m['sft']:.1%}, sorl={m['sorl']:.1%}")
        else:
            print(f"  {name}: sft={m['sft']:.1%} (baseline)")


if __name__ == "__main__":
    main()
