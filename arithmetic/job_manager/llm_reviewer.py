"""
Persistent LLM code reviewer using OpenAI Responses API (stateful).

Server-side conversation memory — no re-sending of prior context.
Each review chains via previous_response_id.

Usage:
    from arithmetic.job_manager.llm_reviewer import Reviewer

    reviewer = Reviewer()
    feedback = reviewer.review(
        files={"gpu_queue.py": open("...").read()},
        prompt="Check for race conditions",
    )

State persists to /workspace/sorl_logs/reviewer_state.json (just the response ID).
"""
import json
import os
from pathlib import Path
from openai import OpenAI

STATE_PATH = "/workspace/sorl_logs/reviewer_state.json"
MODEL = "gpt-4.1"

SYSTEM_PROMPT = """You are a senior ML engineer embedded in a research team working on SoRL (Self-Organized Reinforcement Learning) applied to arithmetic interpretability.

Project context:
- Tiny Qwen3 transformers (2L/3H/510d) trained from scratch on 6-digit addition/subtraction
- SoRL v1 adds "abstraction tokens" every K positions — discrete tokens assigned by search to maximize info-gain
- Comparing SoRL vs SFT baseline on data efficiency and model interpretability
- Eval: per-complexity splits (S0-S6 carry cascades, M0-M5 borrow cascades, C3-C6 hot chains, B3-B5 hot borrow chains)
- Infrastructure: Redis-backed job queue, HuggingFace for model storage, wandb for training logs
- Reference: Quirke et al. 2024 (https://arxiv.org/abs/2402.02619) for arithmetic subtask definitions
- Interpretability: autointerp (Juang et al. 2024), EAP circuits, SAE↔token matching, probing

You have reviewed code from this project before. Use your memory of prior reviews to give consistent, non-repetitive feedback. Be specific and actionable. Only flag real bugs and design issues, not style."""


class Reviewer:
    def __init__(self, state_path: str = STATE_PATH, model: str = MODEL):
        self.state_path = state_path
        self.model = model
        self.client = OpenAI()
        self.state = self._load_state()

    def _load_state(self) -> dict:
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                return json.load(f)
        return {"last_response_id": None, "n_reviews": 0}

    def _save_state(self):
        Path(self.state_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(self.state, f, indent=2)

    def review(self, files: dict = None, prompt: str = "",
               review_type: str = "both", max_tokens: int = 3000) -> str:
        """
        Send code for review. Returns feedback string.
        Chains with prior conversation via previous_response_id.

        review_type:
            "implementation" — bugs, concurrency, error handling, resource leaks
            "architecture" — design decisions, better approaches, cost, scalability
            "both" — both (default)
        """
        parts = []

        if review_type in ("architecture", "both"):
            parts.append(
                "ARCHITECTURE REVIEW: For each design decision in this code, "
                "state the decision explicitly, then evaluate: is there a better "
                "approach for accuracy, scalability, cost, or maintainability? "
                "What existing APIs/tools/patterns could replace custom code? "
                "What would you do differently?"
            )
        if review_type in ("implementation", "both"):
            parts.append(
                "IMPLEMENTATION REVIEW: Check for bugs, concurrency issues, "
                "error handling gaps, resource leaks, race conditions, and "
                "anything that could produce incorrect results."
            )

        if prompt:
            parts.append(prompt)
        if files:
            parts.append("")
            for name, content in files.items():
                parts.append(f"=== {name} ===\n{content}")

        user_msg = "\n\n".join(parts)

        kwargs = {
            "model": self.model,
            "input": user_msg,
            "instructions": SYSTEM_PROMPT,
            "max_output_tokens": max_tokens,
            "temperature": 0.3,
        }

        # Chain with prior conversation if we have one
        if self.state["last_response_id"]:
            kwargs["previous_response_id"] = self.state["last_response_id"]

        response = self.client.responses.create(**kwargs)

        self.state["last_response_id"] = response.id
        self.state["n_reviews"] += 1
        self._save_state()

        return response.output_text

    def reset(self):
        """Start a fresh conversation (new context)."""
        self.state = {"last_response_id": None, "n_reviews": 0}
        self._save_state()

    @property
    def n_reviews(self) -> int:
        return self.state["n_reviews"]


if __name__ == "__main__":
    import sys
    r = Reviewer()
    if len(sys.argv) > 1 and sys.argv[1] == "reset":
        r.reset()
        print("Reviewer conversation reset.")
    else:
        print(f"Reviewer: {r.n_reviews} prior reviews.")
        print(f"Response ID: {r.state['last_response_id'] or 'none (fresh)'}")
        print(f"State file: {r.state_path}")
