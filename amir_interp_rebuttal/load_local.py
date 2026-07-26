"""
Load a v9 steered checkpoint from a LOCAL directory.

`sorl.analyze.load_steered_model` resolves its `run` argument through
`hf_hub_download`, so it can only read checkpoints that have been pushed to the
Hub. Our runs stay on disk, so this mirrors the same construction path against a
local `final.pt`.

Kept faithful to the original: same wrapper class selection, same LoRA
reconstruction, same three state_dict loads.
"""
from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sorl.steer import StackedAbstractionWrapperV6, StackedAbstractionWrapperV9


def load_local_steered(ckpt_dir, device=None, dtype=torch.bfloat16, verbose=True):
    """Return (wrapper, tokenizer, args) from `<ckpt_dir>/final.pt`."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    path = Path(ckpt_dir)
    ckpt_file = path if path.is_file() else path / "final.pt"
    if not ckpt_file.exists():
        raise FileNotFoundError(f"no checkpoint at {ckpt_file}")

    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    args = ckpt["args"]

    model = AutoModelForCausalLM.from_pretrained(args["model_name"], torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(args["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    D_MODEL = model.config.hidden_size

    WrapperCls = (StackedAbstractionWrapperV9 if args["mode"] == "v9"
                  else StackedAbstractionWrapperV6)
    inject_layers = [int(l) for l in str(args["inject_layers"]).split(" ")]
    wrapper = WrapperCls(
        model, C_SIZE=args["C_SIZE"], D_MODEL=D_MODEL,
        inject_layers=inject_layers, scale=args["scale"], L=args["L"],
    )

    if args.get("use_lora", False):
        from peft import LoraConfig, get_peft_model
        lora_cfg = LoraConfig(
            r=args["lora_rank"], lora_alpha=args["lora_alpha"],
            target_modules=args["lora_target_modules"].split(","),
            lora_dropout=args["lora_dropout"], bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        wrapper.model = model

    wrapper.model.load_state_dict(ckpt["model"])
    wrapper.steering_emb.load_state_dict(ckpt["steering_emb"])
    if args["mode"] == "v9" and "abs_proj" in ckpt:
        wrapper.abs_proj.load_state_dict(ckpt["abs_proj"])

    wrapper = wrapper.to(device).eval()

    if verbose:
        print(f"loaded {ckpt_file} | mode={args['mode']} C={args['C_SIZE']} "
              f"L={args['L']} scale={args['scale']} layers={args['inject_layers']}")

    return wrapper, tokenizer, args
