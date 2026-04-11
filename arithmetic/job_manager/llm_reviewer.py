"""
Persistent LLM code reviewer. Maintains conversation history across reviews
so GPT-4/5 has full context of the codebase and prior feedback.

Usage:
    from arithmetic.job_manager.llm_reviewer import Reviewer

    reviewer = Reviewer()
    feedback = reviewer.review(
        files={"gpu_queue.py": open("...").read()},
        prompt="Check for race conditions in the Redis integration",
    )
    print(feedback)

    # Next review — GPT remembers prior context
    feedback = reviewer.review(
        files={"train.py": open("...").read()},
        prompt="Check eval logic for correctness",
    )

History persists to /workspace/sorl_logs/reviewer_history.json
"""
import json
import os
from pathlib import Path
from openai import OpenAI

HISTORY_PATH = "/workspace/sorl_logs/reviewer_history.json"
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
    def __init__(self, history_path: str = HISTORY_PATH, model: str = MODEL):
        self.history_path = history_path
        self.model = model
        self.client = OpenAI()
        self.messages = self._load_history()

    def _load_history(self) -> list:
        """Load prior conversation history."""
        if os.path.exists(self.history_path):
            with open(self.history_path) as f:
                data = json.load(f)
                return data.get("messages", [{"role": "system", "content": SYSTEM_PROMPT}])
        return [{"role": "system", "content": SYSTEM_PROMPT}]

    def _save_history(self):
        """Persist conversation history."""
        Path(self.history_path).parent.mkdir(parents=True, exist_ok=True)
        # Keep last 20 exchanges to avoid context overflow
        # Always keep system prompt + last 40 messages (20 user + 20 assistant)
        trimmed = [self.messages[0]]  # system prompt
        trimmed.extend(self.messages[-40:] if len(self.messages) > 41 else self.messages[1:])
        with open(self.history_path, "w") as f:
            json.dump({"messages": trimmed, "n_reviews": self.n_reviews}, f, indent=2)
        self.messages = trimmed

    @property
    def n_reviews(self) -> int:
        return sum(1 for m in self.messages if m["role"] == "assistant")

    def review(self, files: dict = None, prompt: str = "", max_tokens: int = 3000) -> str:
        """
        Send code for review. Returns feedback string.

        Args:
            files: dict of {filename: content} to review
            prompt: specific review instructions
            max_tokens: max response length
        """
        # Build the user message
        parts = []
        if prompt:
            parts.append(prompt)
        if files:
            parts.append("")
            for name, content in files.items():
                parts.append(f"=== {name} ===\n{content}")

        user_msg = "\n\n".join(parts)
        self.messages.append({"role": "user", "content": user_msg})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=0.3,
            max_tokens=max_tokens,
        )

        feedback = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": feedback})
        self._save_history()

        return feedback

    def summarize_history(self) -> str:
        """Get a summary of what's been reviewed so far."""
        reviews = [(m["content"][:200], self.messages[i+1]["content"][:200])
                   for i, m in enumerate(self.messages[:-1])
                   if m["role"] == "user" and i+1 < len(self.messages)]
        lines = [f"Reviews so far: {len(reviews)}"]
        for i, (q, a) in enumerate(reviews):
            lines.append(f"  {i+1}. Q: {q[:80]}... → A: {a[:80]}...")
        return "\n".join(lines)


if __name__ == "__main__":
    import sys
    r = Reviewer()
    if len(sys.argv) > 1 and sys.argv[1] == "history":
        print(r.summarize_history())
    else:
        print(f"Reviewer loaded. {r.n_reviews} prior reviews in history.")
        print(f"History: {r.history_path}")
