import torch
import argparse
import pandas as pd
import os
import logging
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType
from src.gapt_trainer import GaptTrainer, GaptConfig
from data.symbol_multiply import load_symbol_multiply_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train CPT on Symbol Multiply with GAPT")
    
    # Model & Data
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B", help="Model checkpoint")
    parser.add_argument("--symbol_set", type=str, default="geometric", help="Symbol set to use")
    parser.add_argument("--num_train", type=int, default=1000, help="Number of training samples")
    parser.add_argument("--num_test", type=int, default=100, help="Number of test samples (ID/OOD)")
    parser.add_argument("--train_digits", type=int, nargs=2, default=[1, 2], help="Min/Max digits for train")
    parser.add_argument("--ood_digits", type=int, nargs=2, default=[3, 3], help="Min/Max digits for OOD test")
    parser.add_argument("--use_lora", action="store_true", default=False, help="Use LoRA")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lora_target", type=str, default="q_proj,v_proj", help="LoRA target modules (comma-sep)")
    
    # GAPT Configuration
    parser.add_argument("--mbe_comp_mode", type=str, default="spike", choices=["naive", "spike", "max"], help="MBE compression mode")
    parser.add_argument("--patch_size", type=int, default=8, help="Patch size for MBE")
    parser.add_argument("--mbe_weight", type=float, default=10.0, help="Weight for MBE loss")
    parser.add_argument("--entropy_patience", type=int, default=1, help="Steps to wait before switching to compression")
    parser.add_argument("--mbe_patience", type=int, default=1000, help="Steps to wait before switching back to memorization")
    parser.add_argument("--tau_plateau", type=float, default=0.01, help="Plateau threshold")
    parser.add_argument("--tau_spike", type=float, default=0.1, help="Spike threshold")
    parser.add_argument("--static_phase", action="store_true", help="If set, phase never changes")
    parser.add_argument("--initial_phase", type=int, default=1, choices=[1, 2], help="Initial phase (1=Mem, 2=Comp)")
    
    # Training Hyperparams
    parser.add_argument("--output_dir", type=str, default="./output_cpt", help="Output directory")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Per device batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Num epochs")
    parser.add_argument("--logging_steps", type=int, default=5, help="Log every N steps")
    parser.add_argument("--eval_steps", type=int, default=20, help="Eval every N steps")
    parser.add_argument("--report_to", type=str, default="none", help="wandb or none")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

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
                # Don't overwrite existing values (keeps first non-NaN)
                if k not in rows[step] or pd.isna(rows[step].get(k)):
                    rows[step][k] = v
    
    df = pd.DataFrame(list(rows.values()))
    df = df.sort_values('step').reset_index(drop=True)
    
    # Forward-fill
    eval_cols = [c for c in df.columns if c.startswith('eval_')]
    df[eval_cols] = df[eval_cols].ffill()
    
    train_cols = ['loss', 'ce_loss', 'mbe_loss', 'gapt_loss', 'gapt_phi']
    train_cols = [c for c in train_cols if c in df.columns]
    df[train_cols] = df[train_cols].ffill()
    
    # Drop pre-training rows
    if 'loss' in df.columns:
        df = df.dropna(subset=['loss']).reset_index(drop=True)
    
    return df

def main():
    args = parse_args()
    print(f"Args: {args}")
    
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
        # Enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

    # Load Data
    dataset, mapper_info = load_symbol_multiply_dataset(
        symbol_set=args.symbol_set,
        shuffle_seed=args.seed,
        num_train=args.num_train,
        train_min_digits=args.train_digits[0],
        train_max_digits=args.train_digits[1],
        num_test_id=args.num_test,
        num_test_ood=args.num_test,
        ood_min_digits=args.ood_digits[0],
        ood_max_digits=args.ood_digits[1],
    )
    print(f"Symbol mapping: {mapper_info['digit_to_symbol']}")

    # Tokenize
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=64,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_train = dataset["train"].map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)
    tokenized_test_id = dataset["test"].map(tokenize_function, batched=True, remove_columns=dataset["test"].column_names)
    tokenized_test_ood = dataset["test_ood"].map(tokenize_function, batched=True, remove_columns=dataset["test"].column_names)
    
    tokenized_train.set_format("torch")
    tokenized_test_id.set_format("torch")
    tokenized_test_ood.set_format("torch")

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
        eval_dataset={"id": tokenized_test_id, "ood": tokenized_test_ood},
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
            save_strategy="no",  # No checkpoints
            eval_on_start=True,
            report_to=args.report_to,
            bf16=True,
            dataloader_pin_memory=False,
            seed=args.seed,
            log_level="error",
            gradient_checkpointing=args.use_lora,  # Memory-efficient for LoRA
        )
    )

    # Train
    trainer.train()

    # Save Log History (only on main process to avoid race conditions)
    if local_rank <= 0:  # rank 0 or single-GPU (-1)
        log_df = aggregate_log_history(trainer.state.log_history)
        os.makedirs(args.output_dir, exist_ok=True)
        csv_path = os.path.join(args.output_dir, "training_log.csv")
        log_df.to_csv(csv_path, index=False)
        print(f"Training logs saved to {csv_path}")

if __name__ == "__main__":
    main()
