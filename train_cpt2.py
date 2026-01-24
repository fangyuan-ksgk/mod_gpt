import torch
import argparse
import pandas as pd
import os
import re
import random
import numpy as np
from datasets import load_dataset
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType
from src.gapt_trainer import GaptTrainer, GaptConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train on GSM8K, Eval on GSM-Symbolic with GAPT")
    
    # Model & Data
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B", help="Model checkpoint")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Limit training samples (None=all)")
    parser.add_argument("--max_eval_samples", type=int, default=500, help="Max eval samples per split")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--use_lora", action="store_true", default=False, help="Use LoRA")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lora_target", type=str, default="q_proj,v_proj", help="LoRA target modules (comma-sep)")
    
    # GAPT Configuration
    parser.add_argument("--mbe_comp_mode", type=str, default="spike", choices=["naive", "spike", "max"], help="MBE compression mode")
    parser.add_argument("--patch_size", type=int, default=8, help="Patch size for MBE")
    parser.add_argument("--mbe_weight", type=float, default=1.0, help="Weight for MBE loss")
    parser.add_argument("--entropy_patience", type=int, default=200, help="Steps to wait before switching to compression")
    parser.add_argument("--mbe_patience", type=int, default=100, help="Steps to wait before switching back to memorization")
    parser.add_argument("--tau_plateau", type=float, default=0.01, help="Plateau threshold")
    parser.add_argument("--tau_spike", type=float, default=0.1, help="Spike threshold")
    parser.add_argument("--static_phase", action="store_true", help="If set, phase never changes")
    parser.add_argument("--initial_phase", type=int, default=1, choices=[1, 2], help="Initial phase (1=Mem, 2=Comp)")
    
    # Training Hyperparams
    parser.add_argument("--output_dir", type=str, default="./output_gsm_symbolic", help="Output directory")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Per device batch size")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=3, help="Num epochs")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--logging_steps", type=int, default=20, help="Log every N steps")
    parser.add_argument("--eval_steps", type=int, default=100, help="Eval every N steps")
    parser.add_argument("--acc_eval_steps", type=int, default=200, help="Accuracy eval every N steps")
    parser.add_argument("--acc_eval_samples", type=int, default=50, help="Samples for accuracy eval")
    parser.add_argument("--report_to", type=str, default="none", help="wandb or none")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def aggregate_log_history(log_history):
    """Merge train and eval logs, forward-fill so every row has latest values."""
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


# ----- Accuracy Evaluation -----
def extract_answer(text):
    """Extract final numeric answer after ####"""
    match = re.search(r'####\s*([\d,\-\.]+)', text)
    if match:
        return match.group(1).replace(',', '')
    return None


def evaluate_accuracy(model, tokenizer, dataset, max_samples=50):
    """Generate answers and check accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    samples = dataset.select(range(min(max_samples, len(dataset))))
    
    for example in samples:
        question = example["question"]
        prompt = f"Question: {question}\nAnswer:"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred_answer = extract_answer(generated)
        true_answer = extract_answer(example["answer"])
        
        if pred_answer and true_answer:
            if pred_answer == true_answer:
                correct += 1
            total += 1
    
    return correct / total if total > 0 else 0.0


class AccuracyCallback(TrainerCallback):
    def __init__(self, test_main, test_p1, test_p2, tokenizer, eval_steps=200, max_samples=50):
        self.test_main = test_main
        self.test_p1 = test_p1
        self.test_p2 = test_p2
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps
        self.max_samples = max_samples
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.eval_steps == 0 and state.global_step > 0:
            acc_main = evaluate_accuracy(model, self.tokenizer, self.test_main, self.max_samples)
            acc_p1 = evaluate_accuracy(model, self.tokenizer, self.test_p1, self.max_samples)
            acc_p2 = evaluate_accuracy(model, self.tokenizer, self.test_p2, self.max_samples)
            print(f"Step {state.global_step} | Acc Main: {acc_main:.2%} | Acc P1: {acc_p1:.2%} | Acc P2: {acc_p2:.2%}")


def main():
    args = parse_args()
    print(f"Args: {args}")
    
    # Set all random seeds
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

    # Load Model
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

    # ----- Load Datasets -----
    print("Loading GSM8K train set...")
    gsm8k = load_dataset("gsm8k", "main")
    train_data = gsm8k["train"]
    
    if args.max_train_samples:
        train_data = train_data.shuffle(seed=args.seed).select(range(args.max_train_samples))
    
    print("Loading GSM-Symbolic evaluation sets...")
    test_main = load_dataset("apple/GSM-Symbolic", name="main")["test"]
    test_p1 = load_dataset("apple/GSM-Symbolic", name="p1")["test"]
    test_p2 = load_dataset("apple/GSM-Symbolic", name="p2")["test"]
    
    # Subsample for faster evaluation
    test_main = test_main.shuffle(seed=args.seed).select(range(min(args.max_eval_samples, len(test_main))))
    test_p1 = test_p1.shuffle(seed=args.seed).select(range(min(args.max_eval_samples, len(test_p1))))
    test_p2 = test_p2.shuffle(seed=args.seed).select(range(min(args.max_eval_samples, len(test_p2))))
    
    print(f"Train: {len(train_data)}, Test Main: {len(test_main)}, P1: {len(test_p1)}, P2: {len(test_p2)}")

    # ----- Format Functions -----
    def format_gsm8k(example):
        return {"text": f"Question: {example['question']}\nAnswer: {example['answer']}"}

    def format_symbolic(example):
        return {"text": f"Question: {example['question']}\nAnswer: {example['answer']}"}

    train_data = train_data.map(format_gsm8k)
    test_main_formatted = test_main.map(format_symbolic)
    test_p1_formatted = test_p1.map(format_symbolic)
    test_p2_formatted = test_p2.map(format_symbolic)

    # ----- Tokenization with Answer-Only Loss -----
    def tokenize_with_answer_mask(examples):
        """Compute loss on answer portion only."""
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
            
            answer_start = text.find("\nAnswer:")
            
            if answer_start != -1:
                question_tokens = tokenizer(text[:answer_start], add_special_tokens=False)["input_ids"]
                mask_until = len(question_tokens)
                label = [-100] * mask_until + input_ids[mask_until:]
                label = label[:args.max_length] + [-100] * max(0, args.max_length - len(label))
            else:
                label = input_ids.copy()
            
            labels.append(label)
        
        tokenized["labels"] = labels
        return tokenized

    tokenized_train = train_data.map(tokenize_with_answer_mask, batched=True, remove_columns=train_data.column_names)
    tokenized_main = test_main_formatted.map(tokenize_with_answer_mask, batched=True, remove_columns=test_main_formatted.column_names)
    tokenized_p1 = test_p1_formatted.map(tokenize_with_answer_mask, batched=True, remove_columns=test_p1_formatted.column_names)
    tokenized_p2 = test_p2_formatted.map(tokenize_with_answer_mask, batched=True, remove_columns=test_p2_formatted.column_names)
    
    tokenized_train.set_format("torch")
    tokenized_main.set_format("torch")
    tokenized_p1.set_format("torch")
    tokenized_p2.set_format("torch")

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

    trainer = GaptTrainer(
        gapt_config=gapt_config,
        model=model,
        train_dataset=tokenized_train,
        eval_dataset={
            "main": tokenized_main,
            "p1": tokenized_p1,
            "p2": tokenized_p2,
        },
        args=TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
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
        ),
        callbacks=[AccuracyCallback(
            test_main, test_p1, test_p2, tokenizer, 
            eval_steps=args.acc_eval_steps,
            max_samples=args.acc_eval_samples
        )],
    )

    # Train
    trainer.train()

    # Final accuracy eval
    if local_rank <= 0:
        print("\n" + "="*50)
        print("Final Accuracy Evaluation")
        print("="*50)
        acc_main = evaluate_accuracy(model, tokenizer, test_main, max_samples=100)
        acc_p1 = evaluate_accuracy(model, tokenizer, test_p1, max_samples=100)
        acc_p2 = evaluate_accuracy(model, tokenizer, test_p2, max_samples=100)
        print(f"Main: {acc_main:.2%} | P1: {acc_p1:.2%} | P2: {acc_p2:.2%}")

    # Save Log History
    if local_rank <= 0:
        log_df = aggregate_log_history(trainer.state.log_history)
        os.makedirs(args.output_dir, exist_ok=True)
        csv_path = os.path.join(args.output_dir, "training_log.csv")
        log_df.to_csv(csv_path, index=False)
        print(f"Training logs saved to {csv_path}")


if __name__ == "__main__":
    main()