# Re-export shim — train.py moved to arithmetic/training/train.py
# Keeps `python -m arithmetic.train` working.
from arithmetic.training.train import *  # noqa: F401, F403
from arithmetic.training.train import (  # noqa: F401
    main, train_sft, WandbSoRLTrainer, ArithmeticConfig,
    QWEN3_TOKEN_MAP, QWEN3_INV_MAP,
)

if __name__ == "__main__":
    main()
