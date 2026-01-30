import lm_eval
import argparse
import json
from lm_eval.utils import handle_non_serializable

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a HF model using lm-eval")
    
    # Model configuration
    parser.add_argument("--pretrained", type=str, default="Ksgk-fy/gpt2-xl-fineweb10B", help="Path to the pretrained model (local or HF Hub)")

    return parser.parse_args()

def main():
    args = parse_args()
    
    results = lm_eval.simple_evaluate(
    model="hf",
    model_args={
        "pretrained": args.pretrained,
        "trust_remote_code": True,
        "tokenizer": "gpt2",
    },
    device="cuda:0",
    tasks=["hellaswag", "arc_easy", "piqa", 'winogrande'], # List your tasks here
    num_fewshot=0,
    batch_size=64,
    log_samples=True
)

    output_name = args.pretrained.split('/')[-1]
    with open(f"results_{output_name}.json", "w") as f:
        json.dump(results, f, default=handle_non_serializable, indent=2)

if __name__ == "__main__":
    main()