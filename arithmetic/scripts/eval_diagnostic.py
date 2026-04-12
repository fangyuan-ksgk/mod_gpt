"""
Eval diagnostic: trains v1/v6/baseline on 10K samples, then evaluates with 3 AR methods.

Methods:
  A) Fangyuan-growing: rhythmic mask, growing sequence (his generate() approach)
  B) Fixed-length AR: full 21-token seq with dummies, rebuild mask each digit
  C) Growing-rebuild: strip abstractions, rebuild from scratch each step

Usage:
  GPU 0: python -m arithmetic.scripts.eval_diagnostic --mode sorl_v1 --device cuda:0
  GPU 1: python -m arithmetic.scripts.eval_diagnostic --mode sorl_v6 --device cuda:1
  GPU 2: python -m arithmetic.scripts.eval_diagnostic --mode baseline --device cuda:2
"""
import sys, os, argparse, time, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
from transformers import Qwen3Config, AutoTokenizer
from sorl.sorl_wrapper import SorlModelWrapper
from sorl.sorl_trainer import infer_insert_mask, insert_tokens_with_padding, expand_prompt_len
from sorl.neo_utils import infer_rythmic_insert_mask, insert_tokens
from arithmetic.datasets.addition import generate_batch, get_eval_set
from arithmetic.train import (
    QWEN3_TOKEN_MAP, QWEN3_INV_MAP, TOKENIZER_NAME,
    Qwen3ArithmeticDataset, collate_fn, make_model,
    WandbSoRLTrainer, train_sft,
)
from sorl.trainer_ablate import SoRLTrainer, SoRLConfig

PROMPT_LEN = 14  # 2*6+2
ANSWER_LEN = 7   # 6+1
SEQ_LEN = 21     # PROMPT_LEN + ANSWER_LEN


# ═══════════════════════════════════════════════════════════════════
# Method A: Fangyuan-style growing sequence
# Mirrors neo_utils.generate() as closely as possible.
# ═══════════════════════════════════════════════════════════════════
@torch.no_grad()
def eval_method_A_fangyuan_growing(model, eval_examples, device, K=4, max_iterations=2):
    """
    Fangyuan's approach from neo_utils.py:284-311:
      1. Start with prompt only
      2. Insert abstractions with infer_rythmic_insert_mask
      3. Recursion fills abstractions
      4. Predict next trajectory token from logits[-1]
      5. Append to sequence (with abstractions still in it)
      6. Repeat from step 2

    DIFFERENCE from Fangyuan: Our model uses SorlModelWrapper.recursion()
    (HF-based) not gat_sim.recursion() (custom GPT). API differs but logic is same.

    KNOWN ISSUE: Our model trained with infer_insert_mask (position modulo K),
    not infer_rythmic_insert_mask (count from last abstraction). These produce
    the same result on clean sequences but DIVERGE on sequences that already
    contain abstractions — which is exactly what happens in growing generation.
    """
    model.eval()
    base_v = model.vocab_sizes[0].item()
    n_correct = 0

    for ex in eval_examples:
        qwen_ids = torch.tensor([QWEN3_TOKEN_MAP[t] for t in ex.tokens],
                                dtype=torch.long, device=device)
        target = qwen_ids[PROMPT_LEN:]

        # Start with just the prompt (Fangyuan's pattern)
        idx = qwen_ids[:PROMPT_LEN].unsqueeze(0)  # [1, 14]

        for digit in range(ANSWER_LEN):
            # Insert placeholders using Fangyuan's rhythmic mask
            # This counts positions from last BOS or abstraction token
            insert_mask = infer_rythmic_insert_mask(idx, K, base_v)
            expanded = insert_tokens(idx, insert_mask, model.vocab_sizes[0].item())

            # Recursion: fill abstraction placeholders
            attn = torch.ones_like(expanded)
            recursion_mask = (expanded >= base_v)
            recursion_mask[:, 0] = False

            for _ in range(max_iterations):
                outputs = model.model.forward(
                    input_ids=expanded, attention_mask=attn, use_cache=False,
                )
                expanded = model.extract_and_sample(
                    outputs.logits, expanded, recursion_mask, 0.0
                )

            # Predict next trajectory token (Fangyuan: logits[:, -1, :vocab[0]])
            outputs = model.model.forward(
                input_ids=expanded, attention_mask=attn, use_cache=False,
            )
            next_logits = outputs.logits[:, -1, :base_v]
            next_token = next_logits.argmax(dim=-1)  # [1]

            # Append to expanded sequence (matches Fangyuan line 310)
            idx = torch.cat([expanded, next_token.unsqueeze(1)], dim=1)

        # Extract answer: last ANSWER_LEN trajectory tokens
        is_traj = idx[0] < base_v
        traj_tokens = idx[0][is_traj]
        pred = traj_tokens[-ANSWER_LEN:]

        if (pred == target).all():
            n_correct += 1

    return n_correct / max(len(eval_examples), 1)


# ═══════════════════════════════════════════════════════════════════
# Method B: Fixed-length AR (our current approach)
# Matches training distribution exactly.
# ═══════════════════════════════════════════════════════════════════
@torch.no_grad()
def eval_method_B_fixed_length(model, eval_examples, device, K=4, max_iterations=2):
    """
    Our approach — matches training distribution:
      1. Build full 21-token sequence: prompt + dummy answer (zeros)
      2. Insert abstractions with infer_insert_mask (same as training)
      3. Recursion fills abstractions
      4. Read logits at answer position, fill prediction into dummy
      5. Repeat from step 2 for next digit

    WHY this works: Causal masking means future dummy tokens are invisible.
    Model sees the same prefix structure as training at each position.
    Verified: zeros/random/ones dummies all give identical results.

    DIFFERENCE from Fangyuan: Fixed-length instead of growing. We re-insert
    abstractions from scratch each iteration (strip and rebuild). This is
    necessary because infer_insert_mask doesn't handle existing abstractions.
    """
    model.eval()
    base_v = model.vocab_sizes[0].item()
    pad_id = 151643  # tokenizer pad_id
    n_correct = 0

    for ex in eval_examples:
        qwen_ids = torch.tensor([QWEN3_TOKEN_MAP[t] for t in ex.tokens],
                                dtype=torch.long, device=device)
        target = qwen_ids[PROMPT_LEN:]

        # Full-length sequence with dummy answer (token 0)
        seq = qwen_ids[:PROMPT_LEN].clone()
        pad_answer = torch.zeros(ANSWER_LEN, dtype=torch.long, device=device)
        seq = torch.cat([seq, pad_answer])  # [21]

        for digit_idx in range(ANSWER_LEN):
            ids = seq.unsqueeze(0)
            attn = torch.ones_like(ids)
            pl_t = torch.tensor([PROMPT_LEN], dtype=torch.long, device=device)

            # Insert abstractions from scratch (same function as training)
            im = infer_insert_mask(ids, K, attn)
            ep = expand_prompt_len(pl_t, im)
            ed, ea = insert_tokens_with_padding(ids, attn, im, model.vocab_sizes[0], pad_id)

            # Recursion fills abstractions
            data, ppt, logits = model.recursion(
                ed, ea, max_iterations=max_iterations,
                memory_span_abs=1792, memory_span_traj=1792,
                temperature=0.0, prompt_len=ep,
            )

            # Find this answer digit's position in expanded sequence
            is_traj = data[0] < base_v
            traj_indices = is_traj.nonzero(as_tuple=True)[0]
            answer_pos = traj_indices[PROMPT_LEN + digit_idx].item()
            pred_token = logits[0, answer_pos - 1, :base_v].argmax()

            seq[PROMPT_LEN + digit_idx] = pred_token

        pred = seq[PROMPT_LEN:]
        if (pred == target).all():
            n_correct += 1

    return n_correct / max(len(eval_examples), 1)


# ═══════════════════════════════════════════════════════════════════
# Method C: Growing with rebuild
# Hybrid: grows like Fangyuan, but rebuilds mask from scratch like training.
# ═══════════════════════════════════════════════════════════════════
@torch.no_grad()
def eval_method_C_growing_rebuild(model, eval_examples, device, K=4, max_iterations=2):
    """
    Hybrid approach:
      1. Start with prompt only (14 tokens)
      2. Insert abstractions with infer_insert_mask (training's function)
      3. Recursion fills abstractions
      4. Predict next token from logits[-1] over trajectory vocab
      5. STRIP all abstractions back to clean trajectory
      6. Append predicted token to clean trajectory
      7. Repeat from step 2

    This avoids the mask incompatibility of Method A (no existing abstractions
    when we re-insert) while using growing sequences like Fangyuan.

    TRADEOFF: Sequence length grows from 14 to 21 over 7 steps. The model
    trained on fixed 21-token sequences (expanded to ~41). Shorter sequences
    may produce different attention patterns.
    """
    model.eval()
    base_v = model.vocab_sizes[0].item()
    pad_id = 151643
    n_correct = 0

    for ex in eval_examples:
        qwen_ids = torch.tensor([QWEN3_TOKEN_MAP[t] for t in ex.tokens],
                                dtype=torch.long, device=device)
        target = qwen_ids[PROMPT_LEN:]

        # Clean trajectory (no abstractions), starts as just prompt
        traj = qwen_ids[:PROMPT_LEN].clone()  # [14]

        for digit in range(ANSWER_LEN):
            ids = traj.unsqueeze(0)
            attn = torch.ones_like(ids)

            # Insert abstractions from scratch (same function as training)
            im = infer_insert_mask(ids, K, attn)
            ed, ea = insert_tokens_with_padding(ids, attn, im, model.vocab_sizes[0], pad_id)

            # Recursion
            for _ in range(max_iterations):
                outputs = model.model.forward(
                    input_ids=ed, attention_mask=ea, use_cache=False,
                )
                ed = model.extract_and_sample(
                    outputs.logits, ed, (ed >= base_v), 0.0
                )

            # Predict next token from last position
            outputs = model.model.forward(
                input_ids=ed, attention_mask=ea, use_cache=False,
            )
            next_token = outputs.logits[:, -1, :base_v].argmax(dim=-1)

            # Append to clean trajectory (strip abstractions)
            traj = torch.cat([traj, next_token.squeeze().unsqueeze(0)])

        pred = traj[PROMPT_LEN:]
        if (pred == target).all():
            n_correct += 1

    return n_correct / max(len(eval_examples), 1)


# ═══════════════════════════════════════════════════════════════════
# SFT eval (growing, no abstractions)
# ═══════════════════════════════════════════════════════════════════
@torch.no_grad()
def eval_sft_growing(model, eval_examples, device):
    """SFT baseline: growing-sequence AR, no abstractions."""
    model.eval()
    base_v = model.vocab_sizes[0].item()
    n_correct = 0

    for ex in eval_examples:
        qwen_ids = torch.tensor([QWEN3_TOKEN_MAP[t] for t in ex.tokens],
                                dtype=torch.long, device=device)
        target = qwen_ids[PROMPT_LEN:]

        generated = qwen_ids[:PROMPT_LEN].clone()
        for _ in range(ANSWER_LEN):
            ids = generated.unsqueeze(0)
            attn = torch.ones_like(ids)
            out = model(ids, attention_mask=attn, memory_span_abs=1792, memory_span_traj=1792)
            next_token = out.logits[0, -1, :base_v].argmax()
            generated = torch.cat([generated, next_token.unsqueeze(0)])

        pred = generated[PROMPT_LEN:]
        if (pred == target).all():
            n_correct += 1

    return n_correct / max(len(eval_examples), 1)


# ═══════════════════════════════════════════════════════════════════
# Training helpers
# ═══════════════════════════════════════════════════════════════════

def train_v1(model, tokenizer, args, device):
    """Train SoRL v1 (info-gain loss)."""
    train_ds = Qwen3ArithmeticDataset(tokenizer, 6, args.ops, args.dataset_size)
    val_ds = Qwen3ArithmeticDataset(tokenizer, 6, args.ops, 1000)

    cfg = SoRLConfig(
        K=args.K, batch_size=args.batch_size,
        num_epochs=args.num_epochs, lr=args.lr,
        output_dir=args.output_dir,
        log_every=50, eval_every=999999, save_every=999999,
        eval_samples=0,
        alpha_info_gain=10.0, alpha_abs=0.1,
        alpha_soft_zipf=1.0, alpha_traj=0.0,
    )
    trainer = SoRLTrainer(
        model, tokenizer, train_ds, val_ds,
        compute_accuracy=None, collate_fn=collate_fn,
        config=cfg, device=device,
    )
    trainer.train()
    return trainer.history


def train_v6(model, tokenizer, args, device):
    """Train SoRL v6 (self-routing, traj-only loss)."""
    from sorl.selfroute import SoRLTrainerv6
    train_ds = Qwen3ArithmeticDataset(tokenizer, 6, args.ops, args.dataset_size)
    val_ds = Qwen3ArithmeticDataset(tokenizer, 6, args.ops, 1000)

    cfg = SoRLConfig(
        K=args.K, batch_size=args.batch_size,
        num_epochs=args.num_epochs, lr=args.lr,
        output_dir=args.output_dir,
        log_every=50, eval_every=999999, save_every=999999,
        eval_samples=0,
        alpha_traj=1.0, alpha_info_gain=0.0, alpha_abs=0.0,
        alpha_soft_zipf=0.0,
    )
    trainer = SoRLTrainerv6(
        model, tokenizer, train_ds, val_ds,
        compute_accuracy=None, collate_fn=collate_fn,
        config=cfg, device=device,
    )
    trainer.train()
    return trainer.history


def train_baseline(model, tokenizer, args, device):
    """Train SFT baseline."""
    train_ds = Qwen3ArithmeticDataset(tokenizer, 6, args.ops, args.dataset_size)
    val_ds = Qwen3ArithmeticDataset(tokenizer, 6, args.ops, 1000)

    # Inline SFT training (simpler than importing full train_sft with wandb)
    model.to(device).train()
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(loader) * args.num_epochs
    warmup_steps = int(total_steps * 0.2)
    step = 0

    for epoch in range(args.num_epochs):
        for batch in loader:
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            out = model(ids, attention_mask=attn, memory_span_abs=1792, memory_span_traj=1792)
            labels = ids.clone()
            labels[:, :PROMPT_LEN] = -100
            shift_logits = out.logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            # LR schedule
            if step < warmup_steps:
                lr = args.lr * step / max(warmup_steps, 1)
            else:
                progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
                lr = args.lr * 0.5 * (1 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1

            if step % 100 == 0:
                print(f"  step {step}/{total_steps} | loss={loss.item():.4f}")

    return {"final_loss": loss.item()}


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["sorl_v1", "sorl_v6", "baseline"], required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ops", default="add_sub")
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--abs_vocab", type=int, default=16)
    p.add_argument("--dataset_size", type=int, default=10_000)
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=8e-5)
    p.add_argument("--n_eval", type=int, default=100, help="examples per split for eval")
    p.add_argument("--output_dir", default="ckpt/eval_diagnostic")
    args = p.parse_args()

    print(f"{'═' * 60}")
    print(f"  EVAL DIAGNOSTIC: {args.mode} on {args.device}")
    print(f"  data={args.dataset_size}, epochs={args.num_epochs}, K={args.K}")
    print(f"{'═' * 60}")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # Build model
    class Args:
        pass
    model_args = Args()
    model_args.n_layer = 2
    model_args.n_head = 3
    model_args.n_embd = 510
    model_args.abs_vocab = args.abs_vocab if args.mode != "baseline" else 0
    model = make_model(model_args, tokenizer).to(args.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: 2L/3H/510d | {n_params:,} params | abs_vocab={model_args.abs_vocab}")

    # ── Train ──
    t0 = time.time()
    if args.mode == "sorl_v1":
        history = train_v1(model, tokenizer, args, args.device)
    elif args.mode == "sorl_v6":
        history = train_v6(model, tokenizer, args, args.device)
    else:
        history = train_baseline(model, tokenizer, args, args.device)

    train_time = time.time() - t0
    print(f"\n  Training done in {train_time:.0f}s")

    # ── Eval ──
    print(f"\n{'─' * 60}")
    print(f"  EVALUATION ({args.n_eval} examples per split)")
    print(f"{'─' * 60}")

    categories = get_eval_set(6, args.ops, N=args.n_eval)
    all_examples = []
    for split_name, examples in categories.items():
        all_examples.extend(examples)
    print(f"  Total eval examples: {len(all_examples)}")

    results = {}

    if args.mode == "baseline":
        # Baseline: only SFT eval (no abstractions)
        print("\n  [SFT growing-seq AR]...")
        t0 = time.time()
        acc = eval_sft_growing(model, all_examples, args.device)
        print(f"    Accuracy: {acc:.1%} ({time.time()-t0:.0f}s)")
        results["sft_growing"] = acc

        # Also try fixed-length and growing-rebuild (should be same since no abstractions involved)
        # Skip — SFT doesn't use abstractions, these methods don't apply

    else:
        K = args.K

        # Method A: Fangyuan growing
        print(f"\n  [A] Fangyuan growing (rhythmic mask, K={K})...")
        t0 = time.time()
        acc_A = eval_method_A_fangyuan_growing(model, all_examples, args.device, K=K)
        print(f"    Accuracy: {acc_A:.1%} ({time.time()-t0:.0f}s)")
        results["A_fangyuan_growing"] = acc_A

        # Method B: Fixed-length AR
        print(f"\n  [B] Fixed-length AR (training mask, K={K})...")
        t0 = time.time()
        acc_B = eval_method_B_fixed_length(model, all_examples, args.device, K=K)
        print(f"    Accuracy: {acc_B:.1%} ({time.time()-t0:.0f}s)")
        results["B_fixed_length"] = acc_B

        # Method C: Growing rebuild
        print(f"\n  [C] Growing rebuild (strip+reinsert, K={K})...")
        t0 = time.time()
        acc_C = eval_method_C_growing_rebuild(model, all_examples, args.device, K=K)
        print(f"    Accuracy: {acc_C:.1%} ({time.time()-t0:.0f}s)")
        results["C_growing_rebuild"] = acc_C

        # Also SFT eval (no abstractions — shows what model does without them)
        print(f"\n  [SFT] No abstractions (raw model)...")
        t0 = time.time()
        acc_sft = eval_sft_growing(model, all_examples, args.device)
        print(f"    Accuracy: {acc_sft:.1%} ({time.time()-t0:.0f}s)")
        results["sft_no_abs"] = acc_sft

    # ── Summary ──
    print(f"\n{'═' * 60}")
    print(f"  RESULTS: {args.mode}")
    print(f"{'═' * 60}")
    for method, acc in results.items():
        print(f"  {method:30s}: {acc:.1%}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"diagnostic_{args.mode}.json")
    with open(out_path, "w") as f:
        json.dump({
            "mode": args.mode, "device": args.device,
            "K": args.K, "abs_vocab": model_args.abs_vocab,
            "dataset_size": args.dataset_size, "num_epochs": args.num_epochs,
            "results": results, "train_time_s": train_time,
        }, f, indent=2)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
