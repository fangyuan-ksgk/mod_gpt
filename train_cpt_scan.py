import torch
import argparse
import pandas as pd
import os
import random
import requests
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType
from src.gapt_trainer import GaptTrainer, GaptConfig
from tqdm.auto import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Train CPT on SCAN with GAPT")
    
    # Model & Data
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B", help="Model checkpoint")
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--use_lora", action="store_true", default=False, help="Use LoRA")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lora_target", type=str, default="q_proj,v_proj", help="LoRA target modules (comma-sep)")
    
    # GAPT Configuration
    parser.add_argument("--mbe_comp_mode", type=str, default="naive", choices=["naive", "spike", "max"], help="MBE compression mode")
    parser.add_argument("--patch_size", type=int, default=4, help="Patch size for MBE")
    parser.add_argument("--mbe_weight", type=float, default=1.0, help="Weight for MBE loss")
    parser.add_argument("--entropy_patience", type=int, default=500, help="Steps to wait before switching to compression")
    parser.add_argument("--mbe_patience", type=int, default=500, help="Steps to wait before switching back to memorization")
    parser.add_argument("--tau_plateau", type=float, default=0.01, help="Plateau threshold")
    parser.add_argument("--tau_spike", type=float, default=0.1, help="Spike threshold")
    parser.add_argument("--static_phase", action="store_true", help="If set, phase never changes")
    parser.add_argument("--initial_phase", type=int, default=1, choices=[1, 2], help="Initial phase (1=Mem, 2=Comp)")
    
    # Training Hyperparams
    parser.add_argument("--output_dir", type=str, default="./output_scan", help="Output directory")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Per device batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Num epochs")
    parser.add_argument("--weight_decay", type=float, default=0.00, help="Weight decay")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every N steps")
    parser.add_argument("--eval_steps", type=int, default=50, help="Eval every N steps")
    parser.add_argument("--report_to", type=str, default="none", help="wandb or none")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--acc_eval_steps", type=int, default=200, help="Steps between accuracy evaluation")
    parser.add_argument("--acc_eval_samples", type=int, default=100, help="Number of samples for accuracy evaluation")

    return parser.parse_args()

def aggregate_log_history(log_history):
    """
    Merge train and eval logs, forward-fill so every row has latest values.
    """
    exclude = {'learning_rate', 'grad_norm', 'train_runtime', 'train_samples_per_second',
               'train_steps_per_second', 'total_flos', 'train_loss',
               'eval_runtime', 'eval_samples_per_second', 'eval_steps_per_second'}
    
    rows = {}
    for entry in log_history:
        step = entry.get('step', 0)
        if step not in rows:
            rows[step] = {'step': step}
        for k, v in entry.items():
            if isinstance(v, (int, float, str)) and k not in exclude:
                if k not in rows[step] or pd.isna(rows[step].get(k)):
                    rows[step][k] = v
    
    df = pd.DataFrame(list(rows.values()))
    df = df.sort_values('step').reset_index(drop=True)
    
    eval_cols = [c for c in df.columns if c.startswith('eval_')]
    df[eval_cols] = df[eval_cols].ffill()
    
    train_cols = ['loss', 'ce_loss', 'mbe_loss', 'gapt_loss', 'gapt_phi']
    train_cols = [c for c in train_cols if c in df.columns]
    df[train_cols] = df[train_cols].ffill()
    
    if 'loss' in df.columns:
        df = df.dropna(subset=['loss']).reset_index(drop=True)
    
    return df

def load_scan_length_manual():
    """Manually download and parse SCAN length split."""
    base_url = "https://raw.githubusercontent.com/brendenlake/SCAN/master/length_split"
    
    def parse_file(url):
        print(f"Downloading {url}...")
        response = requests.get(url)
        response.raise_for_status()
        
        commands = []
        actions = []
        
        for line in response.text.strip().split('\n'):
            # Format: IN: jump left OUT: JUMP LTURN
            if line.startswith("IN:"):
                parts = line.split(" OUT: ")
                cmd = parts[0].replace("IN: ", "").strip()
                act = parts[1].strip()
                commands.append(cmd)
                actions.append(act)
                
        return Dataset.from_dict({"commands": commands, "actions": actions})

    train_ds = parse_file(f"{base_url}/tasks_train_length.txt")
    test_ds = parse_file(f"{base_url}/tasks_test_length.txt")
    
    return DatasetDict({"train": train_ds, "test": test_ds})

# ----- Accuracy Evaluation Logic -----

def check_scan_match(target, pred):
    target = target.strip()
    pred = pred.strip()
    
    # 1. Exact match
    if pred == target:
        return True
    
    # 2. Prefix match (if model keeps generating)
    if pred.startswith(target):
        return True

    # 3. Missing first char fix
    if target.startswith("I_") and pred.startswith("_"):
        corrected_pred = "I" + pred
        if corrected_pred == target or corrected_pred.startswith(target):
            return True
            
    # 4. First char split fix (e.g. "I" + " _TURN...")
    if target.startswith("I_") and pred.startswith("I _"):
        corrected_pred = pred.replace("I _", "I_", 1)
        if corrected_pred == target or corrected_pred.startswith(target):
            return True
    
    return False

def compute_scan_accuracy(model, tokenizer, dataset, max_samples=200):
    model.eval()
    correct = 0
    total = 0
    
    # Select samples
    num_samples = min(max_samples, len(dataset))
    samples = dataset.select(range(num_samples))
    
    # Note: running sequential generation in loop for simplicity
    for i, example in enumerate(samples):
        # Extract prompt
        full_text = example["text"]
        input_part = full_text.split("\nOutput:")[0] + "\nOutput:"
        target_action = full_text.split("\nOutput: ")[1].strip()
        
        inputs = tokenizer(input_part, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_len:]
        prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        is_correct = check_scan_match(target_action, prediction)
        if is_correct: 
            correct += 1
        
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy

class ScanAccuracyCallback(TrainerCallback):
    def __init__(self, tokenizer, test_id, test_ood, eval_steps=200, max_samples=100, log_to_console=True):
        self.tokenizer = tokenizer
        self.test_id = test_id
        self.test_ood = test_ood
        self.eval_steps = eval_steps
        self.max_samples = max_samples
        self.log_to_console = log_to_console
        
    def on_step_end(self, args, state, control, model=None, **kwargs):
        # Only run on main process to avoid duplicates and redundant computation
        if args.local_rank != -1 and args.local_rank != 0:
            return

        if state.global_step > 0 and state.global_step % self.eval_steps == 0:
            if self.log_to_console:
                print(f"\n[Step {state.global_step}] Running Accuracy Evaluation...")
            
            acc_id = compute_scan_accuracy(model, self.tokenizer, self.test_id, self.max_samples)
            acc_ood = compute_scan_accuracy(model, self.tokenizer, self.test_ood, self.max_samples)
            
            if self.log_to_console:
                print(f"ID Accuracy:  {acc_id:.2%}")
                print(f"OOD Accuracy: {acc_ood:.2%}")
            
            # Log to trainer state so it appears in logs/csv
            state.log_history.append({
                "step": state.global_step,
                "eval_id_acc": acc_id,
                "eval_ood_acc": acc_ood,
            })

def main():
    args = parse_args()
    print(f"Args: {args}")
    
    # Set all random seeds for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Device - detect distributed mode
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_distributed = local_rank != -1
    
    if is_distributed:
        device = f"cuda:{local_rank}"
        print(f"[Rank {local_rank}] Using device: {device}")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

    # Load Model (no device_map for distributed training)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if is_distributed:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # Apply LoRA if requested
    if args.use_lora:
        target_modules = [m.strip() for m in args.lora_target.split(",")]
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        model.gradient_checkpointing_enable()

    # Load Data (Manual)
    if local_rank <= 0:
        print("Loading SCAN dataset...")
    
    dataset = load_scan_length_manual()

    # Format
    def format_scan(example):
        return {"text": f"Input: {example['commands']}\nOutput: {example['actions']}"}
    
    dataset = dataset.map(format_scan)
    
    # Create Splits
    # Train: Short sequences
    # Test ID: Held-out short sequences
    # Test OOD: Long sequences (original test split)
    full_train = dataset["train"]
    # Ensure shuffle is deterministic across ranks
    train_test_split = full_train.train_test_split(test_size=0.1, seed=args.seed)
    
    train_data = train_test_split["train"]
    test_id = train_test_split["test"]
    test_ood = dataset["test"]

    # Subsample OOD for speed during eval (optional, keeping it full or large enough)
    if len(test_ood) > 500:
        test_ood = test_ood.shuffle(seed=args.seed).select(range(500))

    if local_rank <= 0:
        print(f"Train: {len(train_data)}")
        print(f"Test ID: {len(test_id)}")
        print(f"Test OOD: {len(test_ood)}")

    # Tokenize with Masking
    def tokenize_with_answer_mask(examples):
        texts = examples["text"]
        tokenized = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )
        
        labels = []
        for i, text in enumerate(texts):
            input_ids = tokenized["input_ids"][i]
            
            # Find start of Output
            split_marker = "\nOutput: "
            answer_start = text.find(split_marker)
            
            if answer_start != -1:
                # Tokenize header to find mask length
                # Note: tokenizer is fast enough
                header_text = text[:answer_start + len(split_marker)]
                mask_len = len(tokenizer(header_text, add_special_tokens=False)["input_ids"])
                
                # Mask question (-100), keep answer
                label = [-100] * mask_len + input_ids[mask_len:]
                # Pad/Truncate
                label = label[:args.max_length] + [-100] * max(0, args.max_length - len(label))
            else:
                label = input_ids.copy()
                
            labels.append(label)
        
        tokenized["labels"] = labels
        return tokenized

    # Map with batched=True
    tokenized_train = train_data.map(tokenize_with_answer_mask, batched=True, remove_columns=train_data.column_names)
    tokenized_id = test_id.map(tokenize_with_answer_mask, batched=True, remove_columns=test_id.column_names)
    tokenized_ood = test_ood.map(tokenize_with_answer_mask, batched=True, remove_columns=test_ood.column_names)
    
    tokenized_train.set_format("torch")
    tokenized_id.set_format("torch")
    tokenized_ood.set_format("torch")

    # GAPT Config
    gapt_config = GaptConfig(
        patch_size=args.patch_size,
        mode=args.mbe_comp_mode,
        mbe_weight=args.mbe_weight,
        entropy_patience=args.entropy_patience,
        mbe_patience=args.mbe_patience,
        tau_plateau_m=args.tau_plateau,
        tau_plateau_a=args.tau_plateau,
        tau_spike=args.tau_spike,
        static_phase=args.static_phase,
        initial_phase=args.initial_phase
    )

    # Initialize callback
    acc_callback = ScanAccuracyCallback(
        tokenizer=tokenizer,
        test_id=test_id,
        test_ood=test_ood,
        eval_steps=args.acc_eval_steps,
        max_samples=args.acc_eval_samples
    )

    trainer = GaptTrainer(
        gapt_config=gapt_config,
        model=model,
        train_dataset=tokenized_train,
        eval_dataset={"id": tokenized_id, "ood": tokenized_ood},
        args=TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            logging_steps=args.logging_steps,
            logging_first_step=True,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            weight_decay=args.weight_decay,
            save_strategy="no",
            eval_on_start=True,
            report_to=args.report_to,
            bf16=True,
            dataloader_pin_memory=False,
            seed=args.seed,
            log_level="error",
            gradient_checkpointing=args.use_lora,
            ddp_find_unused_parameters=False if is_distributed else None, # Optimization for DDP
        ),
        callbacks=[acc_callback]
    )

    # Train
    trainer.train()

    # Save Log History
    if local_rank <= 0:
        log_df = aggregate_log_history(trainer.state.log_history)
        os.makedirs(args.output_dir, exist_ok=True)
        csv_path = os.path.join(args.output_dir, "training_log.csv")
        log_df.to_csv(csv_path, index=False)
        print(f"Training logs saved to {csv_path}")

if __name__ == "__main__":
    main()
