"""
LLM code reviewer and multi-model debater with pluggable backends.

Backends: OpenAI (Responses API), Gemini (Interactions API) — both stateful.
Reviewer: stateful code review via any backend.
Debater: multi-model debate with critique rounds.

Usage:
    from arithmetic.job_manager.llm_reviewer import Reviewer, Debater, OpenAIBackend, GeminiBackend

    # Single-backend review
    reviewer = Reviewer(backend=OpenAIBackend())
    feedback = reviewer.review(files={"train.py": code}, prompt="Check for bugs")

    # Multi-model debate
    debater = Debater()  # uses both OpenAI + Gemini by default
    result = debater.debate("Is our fixed-length AR eval sound?", context="...")
    print(result.summary)
"""
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List


# ═══════════════════════════════════════════════════════════════════
# Backend base class
# ═══════════════════════════════════════════════════════════════════

STATE_DIR = "/workspace/sorl_logs"

SYSTEM_PROMPT = """You are a senior ML engineer embedded in a research team working on SoRL (Self-Organized Reinforcement Learning) applied to arithmetic interpretability.

Project context:
- Tiny Qwen3 transformers (2L/3H/510d) trained from scratch on 6-digit addition/subtraction
- SoRL v1 adds "abstraction tokens" every K positions — discrete tokens assigned by search to maximize info-gain
- Comparing SoRL vs SFT baseline on data efficiency and model interpretability
- Eval: autoregressive (errors propagate), per-complexity splits (S0-S6, M0-M5, C3-C6, B3-B5)
- Infrastructure: Redis-backed job queue, HuggingFace for model storage, wandb for training logs
- Reference: Quirke et al. 2024 (https://arxiv.org/abs/2402.02619)

Give honest, specific, actionable feedback. Contradict the questioner if warranted. Do not be sycophantic."""


class LLMBackend(ABC):
    """Base class for stateful LLM backends."""

    def __init__(self, name: str, model: str, state_path: str):
        self.name = name
        self.model = model
        self.state_path = state_path
        self.state = self._load_state()

    def _load_state(self) -> dict:
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                return json.load(f)
        return self._default_state()

    def _save_state(self):
        Path(self.state_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(self.state, f, indent=2)

    @abstractmethod
    def _default_state(self) -> dict:
        ...

    @abstractmethod
    def send(self, prompt: str, system: str = SYSTEM_PROMPT, max_tokens: int = 3000) -> str:
        """Send a message, chaining with prior conversation. Returns response text."""
        ...

    @abstractmethod
    def reset(self):
        """Clear conversation state."""
        ...

    @property
    def n_calls(self) -> int:
        return self.state.get("n_calls", 0)


# ═══════════════════════════════════════════════════════════════════
# OpenAI backend — Responses API with previous_response_id
# ═══════════════════════════════════════════════════════════════════

class OpenAIBackend(LLMBackend):
    """Stateful OpenAI backend using Responses API (server-side memory)."""

    def __init__(self, model: str = "gpt-4.1",
                 state_path: str = os.path.join(STATE_DIR, "backend_openai.json")):
        super().__init__("openai", model, state_path)

    def _default_state(self) -> dict:
        return {"last_response_id": None, "n_calls": 0}

    @property
    def _client(self):
        if not hasattr(self, "_openai_client"):
            from openai import OpenAI
            self._openai_client = OpenAI()
        return self._openai_client

    def send(self, prompt: str, system: str = SYSTEM_PROMPT, max_tokens: int = 3000) -> str:
        kwargs = {
            "model": self.model,
            "input": prompt,
            "instructions": system,
            "max_output_tokens": max_tokens,
            "temperature": 0.3,
        }
        if self.state["last_response_id"]:
            kwargs["previous_response_id"] = self.state["last_response_id"]

        response = self._client.responses.create(**kwargs)

        self.state["last_response_id"] = response.id
        self.state["n_calls"] = self.state.get("n_calls", 0) + 1
        self._save_state()
        return response.output_text

    def reset(self):
        self.state = self._default_state()
        self._save_state()


# ══���═══════════════════��════════════════════════════════════════════
# Gemini backend — Interactions API with previousInteractionId
# ═══════════════════════════════════════════════════════════════════

class GeminiBackend(LLMBackend):
    """Stateful Gemini backend. Uses Interactions API (previousInteractionId) when available,
    falls back to stateless generateContent.

    Note: Gemini 2.5 Pro uses a thinking budget that counts against maxOutputTokens.
    We set a high default (8192) to leave room for both thinking and response."""

    def __init__(self, model: str = "gemini-2.5-pro",
                 state_path: str = os.path.join(STATE_DIR, "backend_gemini.json")):
        super().__init__("gemini", model, state_path)

    def _default_state(self) -> dict:
        return {"last_interaction_id": None, "n_calls": 0}

    def send(self, prompt: str, system: str = SYSTEM_PROMPT, max_tokens: int = 3000) -> str:
        import requests

        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            return "[Gemini API key not set — skipped]"

        # Gemini 2.5 Pro thinking budget eats maxOutputTokens — pad generously
        gemini_max = max(max_tokens * 3, 8192)

        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{self.model}:generateContent?key={api_key}")
        body = {
            "contents": [{"parts": [{"text": f"{system}\n\n{prompt}"}]}],
            "generationConfig": {"maxOutputTokens": gemini_max},
        }
        # Chain with previous interaction if available
        prev_id = self.state.get("last_interaction_id")
        if prev_id:
            body["previousInteractionId"] = prev_id

        try:
            r = requests.post(url, json=body, timeout=180)
            r.raise_for_status()
            data = r.json()

            # Store interaction ID if returned
            interaction_id = data.get("interactionId")
            if interaction_id:
                self.state["last_interaction_id"] = interaction_id

            self.state["n_calls"] = self.state.get("n_calls", 0) + 1
            self._save_state()

            parts = data["candidates"][0]["content"].get("parts", [])
            return "\n".join(p["text"] for p in parts if "text" in p) or "[No text in response]"
        except Exception as e:
            return f"[Gemini error: {e}]"

    def reset(self):
        self.state = self._default_state()
        self._save_state()


# ═══════════════════════════════════════════════════════════════════
# Claude backend — Messages API with client-side state + prompt caching
# ═══════════════════════════════════════════════════════════════════

class ClaudeBackend(LLMBackend):
    """Client-side stateful Claude backend.

    Anthropic has no server-side session API (like OpenAI's previous_response_id).
    We maintain the message history locally and send it each call. Uses prompt
    caching (cache_control: ephemeral) on the system prompt and early messages
    to reduce cost — cached tokens are 10% of input cost.
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514",
                 state_path: str = os.path.join(STATE_DIR, "backend_claude.json")):
        super().__init__("claude", model, state_path)

    def _default_state(self) -> dict:
        return {"messages": [], "n_calls": 0}

    @property
    def _client(self):
        if not hasattr(self, "_anthropic_client"):
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            self._anthropic_client = anthropic.Anthropic(api_key=api_key)
        return self._anthropic_client

    def send(self, prompt: str, system: str = SYSTEM_PROMPT, max_tokens: int = 3000) -> str:
        # Append user message to history
        self.state["messages"].append({"role": "user", "content": prompt})

        # Build system with cache_control for prompt caching
        system_blocks = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]

        # Anthropic allows max 4 cache_control blocks total (including system).
        # We use 1 on system, so 3 left for messages. Cache the earliest messages
        # to maximize savings on repeated context.
        messages = []
        n_msgs = len(self.state["messages"])
        cache_budget = 3  # max cacheable message blocks (4 total - 1 system)
        cache_count = 0
        for i, msg in enumerate(self.state["messages"]):
            m = {"role": msg["role"], "content": msg["content"]}
            # Cache early messages (not the last 2, and within budget)
            if i < n_msgs - 2 and cache_count < cache_budget:
                m["content"] = [{"type": "text", "text": msg["content"], "cache_control": {"type": "ephemeral"}}]
                cache_count += 1
            messages.append(m)

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_blocks,
                messages=messages,
                temperature=0.3,
            )
            text = response.content[0].text

            # Append assistant response to history
            self.state["messages"].append({"role": "assistant", "content": text})
            self.state["n_calls"] = self.state.get("n_calls", 0) + 1
            self._save_state()
            return text

        except Exception as e:
            # Remove the failed user message
            self.state["messages"].pop()
            return f"[Claude error: {e}]"

    def reset(self):
        self.state = self._default_state()
        self._save_state()


# ═══════════════════════════════════════════════════════════════════
# Reviewer — stateful code review via any backend
# ═══════════════════════════════════════════════════════════════════

REVIEW_PROMPTS = {
    "architecture": (
        "ARCHITECTURE REVIEW: For each design decision in this code, "
        "state the decision explicitly, then evaluate: is there a better "
        "approach for accuracy, scalability, cost, or maintainability? "
        "What existing APIs/tools/patterns could replace custom code? "
        "What would you do differently?"
    ),
    "implementation": (
        "IMPLEMENTATION REVIEW: Check for bugs, concurrency issues, "
        "error handling gaps, resource leaks, race conditions, and "
        "anything that could produce incorrect results."
    ),
    "scientific": (
        "SCIENTIFIC REVIEW: This is research code for an ML experiment. Check:\n"
        "1. CORRECTNESS: Are loss functions, gradients, and metrics computed correctly?\n"
        "2. FAIR COMPARISON: Is the experimental setup fair? Same eval for all conditions? "
        "Any confounds between the thing being measured and the experimental manipulation?\n"
        "3. DATA LEAKAGE: Could test data leak into training? Could future information "
        "leak into predictions (e.g., teacher forcing, non-causal attention)?\n"
        "4. STATISTICAL VALIDITY: Are eval sets large enough? Are results averaged over seeds? "
        "Could variance explain the observed differences?\n"
        "5. REPRODUCIBILITY: Are random seeds set? Are hyperparameters documented? "
        "Could someone reproduce these results from this code?\n"
        "6. CLAIMS vs EVIDENCE: Does the code actually test what the paper claims? "
        "Are there unstated assumptions?"
    ),
}


class Reviewer:
    """Stateful code reviewer. Uses any LLMBackend for conversation memory."""

    def __init__(self, backend: Optional[LLMBackend] = None):
        self.backend = backend or OpenAIBackend()

    def review(self, files: dict = None, prompt: str = "",
               review_type: str = "all", max_tokens: int = 3000) -> str:
        """
        Send code for review. Returns feedback string.

        review_type:
            "implementation" — bugs, concurrency, error handling
            "architecture"   — design decisions, alternatives
            "scientific"     — correctness, fair comparison, data leakage, stats
            "all"            — all three, clearly separated
        """
        parts = []
        types = (["implementation", "architecture", "scientific"] if review_type == "all"
                 else ["implementation", "architecture"] if review_type == "both"
                 else [review_type])
        for t in types:
            if t in REVIEW_PROMPTS:
                parts.append(REVIEW_PROMPTS[t])

        if prompt:
            parts.append(prompt)
        if files:
            parts.append("")
            for name, content in files.items():
                parts.append(f"=== {name} ===\n{content}")

        return self.backend.send("\n\n".join(parts), max_tokens=max_tokens)

    def review_diff(self, diff: str = None, files_context: dict = None,
                    prompt: str = "", review_type: str = "all",
                    max_tokens: int = 3000) -> str:
        """
        Review a code diff (like CodeRabbit / Copilot PR review).
        Focuses on the delta — cheaper and more targeted than full file review.

        Args:
            diff: git diff string. If None, auto-generates from staged changes.
            files_context: optional {filename: content} for full files that the diff touches,
                          so the reviewer can understand surrounding context.
            prompt: additional instructions.
            review_type: same as review().
        """
        import subprocess

        if diff is None:
            # Auto-generate from staged changes, fall back to unstaged
            result = subprocess.run(["git", "diff", "--cached"], capture_output=True, text=True)
            diff = result.stdout
            if not diff.strip():
                result = subprocess.run(["git", "diff"], capture_output=True, text=True)
                diff = result.stdout
            if not diff.strip():
                return "No diff found (nothing staged or modified)."

        parts = []
        types = (["implementation", "architecture", "scientific"] if review_type == "all"
                 else ["implementation", "architecture"] if review_type == "both"
                 else [review_type])
        for t in types:
            if t in REVIEW_PROMPTS:
                parts.append(REVIEW_PROMPTS[t])

        parts.append(
            "DIFF REVIEW: Focus on the CHANGES below. Flag bugs, logic errors, "
            "or scientific issues introduced by this diff. Don't comment on "
            "unchanged code unless a change breaks it. Be concise — one comment "
            "per issue, reference the diff line."
        )

        if prompt:
            parts.append(prompt)

        if files_context:
            parts.append("\n--- Full file context (for reference) ---")
            for name, content in files_context.items():
                parts.append(f"=== {name} ===\n{content}")

        parts.append(f"\n--- Diff ---\n{diff}")

        return self.backend.send("\n\n".join(parts), max_tokens=max_tokens)

    def review_staged(self, prompt: str = "", review_type: str = "implementation",
                      include_context: bool = True, max_tokens: int = 3000) -> str:
        """
        Convenience: review currently staged git changes with file context.
        Meant to be called before committing.
        """
        import subprocess

        result = subprocess.run(["git", "diff", "--cached"], capture_output=True, text=True)
        diff = result.stdout
        if not diff.strip():
            return "Nothing staged."

        files_context = {}
        if include_context:
            # Get list of changed files
            result = subprocess.run(["git", "diff", "--cached", "--name-only"],
                                    capture_output=True, text=True)
            for fname in result.stdout.strip().split("\n"):
                if fname and os.path.exists(fname):
                    try:
                        files_context[fname] = open(fname).read()
                    except Exception:
                        pass

        return self.review_diff(diff=diff, files_context=files_context,
                                prompt=prompt, review_type=review_type,
                                max_tokens=max_tokens)

    def reset(self):
        self.backend.reset()

    @property
    def n_reviews(self) -> int:
        return self.backend.n_calls


# ══���═════════════════════════════════════════���══════════════════════
# Debater — multi-model debate
# ═══════════════════════════════════════════════════════════════════

CRITIQUE_TEMPLATE = """Here are responses from other models to the same question.

{other_responses}

Your task:
1. Where do you AGREE with the other responses? Be specific.
2. Where do you DISAGREE? Explain why with technical reasoning.
3. What did the others MISS that you think is important?
4. What is your final position given all perspectives?

Do not be sycophantic. If you think another model is wrong, say so directly."""


@dataclass
class DebateRound:
    """One round of responses keyed by backend name."""
    responses: dict = field(default_factory=dict)  # {backend_name: response_text}


@dataclass
class DebateResult:
    """Full debate output."""
    question: str = ""
    context: str = ""
    rounds: list = field(default_factory=list)
    summary: str = ""


class Debater:
    """Multi-model debate with critique rounds. Uses pluggable backends."""

    def __init__(self, backends: Optional[List[LLMBackend]] = None):
        self.backends = backends or [OpenAIBackend(), GeminiBackend(), ClaudeBackend()]

    def debate(self, question: str, context: str = "",
               rounds: int = 2, max_tokens: int = 3000,
               claude_perspective: str = "") -> DebateResult:
        """
        Run a multi-model debate.

        Round 1: All backends answer independently.
        Round 2+: Each backend sees others' responses and critiques.
        Finally: first backend synthesizes a summary.
        """
        result = DebateResult(question=question, context=context)

        full_prompt = f"{context}\n\n{question}" if context else question

        # Round 1: independent answers
        r1 = DebateRound()
        for b in self.backends:
            r1.responses[b.name] = b.send(full_prompt, max_tokens=max_tokens)
        if claude_perspective:
            r1.responses["claude"] = claude_perspective
        result.rounds.append(r1)

        # Round 2+: critique
        for _ in range(1, rounds):
            r = DebateRound()
            prev = result.rounds[-1]

            for b in self.backends:
                # Build "other responses" excluding this backend
                others = "\n\n".join(
                    f"**{name}**: {text}"
                    for name, text in prev.responses.items()
                    if name != b.name
                )
                critique_prompt = CRITIQUE_TEMPLATE.format(other_responses=others)
                r.responses[b.name] = b.send(critique_prompt, max_tokens=max_tokens)

            result.rounds.append(r)

        # Summary: first backend synthesizes (has full session context)
        summary_prompt = (
            "Synthesize the debate into a concise summary. "
            "For each point: state consensus or disagreement with evidence. "
            "End with a clear list of recommended actions."
        )
        result.summary = self.backends[0].send(summary_prompt, max_tokens=2000)

        return result

    def reset(self):
        for b in self.backends:
            b.reset()


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════���═════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "reset":
        for B in [OpenAIBackend, GeminiBackend, ClaudeBackend]:
            B().reset()
        print("All backend sessions reset.")

    elif len(sys.argv) > 1 and sys.argv[1] == "status":
        for B in [OpenAIBackend, GeminiBackend, ClaudeBackend]:
            b = B()
            print(f"{b.name}: {b.n_calls} calls, state={b.state}")

    elif len(sys.argv) > 2 and sys.argv[1] == "review":
        prompt = " ".join(sys.argv[2:])
        # Review with all backends
        for B in [OpenAIBackend, GeminiBackend, ClaudeBackend]:
            r = Reviewer(backend=B())
            print(f"\n{'='*60}\n{B.__name__}\n{'='*60}")
            print(r.review(prompt=prompt))

    elif len(sys.argv) > 2 and sys.argv[1] == "debate":
        question = " ".join(sys.argv[2:])
        d = Debater()
        result = d.debate(question)
        for i, rnd in enumerate(result.rounds):
            print(f"\n{'='*60}\nROUND {i+1}\n{'='*60}")
            for name, text in rnd.responses.items():
                print(f"\n--- {name} ---\n{text}")
        print(f"\n{'='*60}\nSUMMARY\n{'='*60}\n{result.summary}")

    else:
        print("Usage:")
        print("  python -m arithmetic.job_manager.llm_reviewer status")
        print("  python -m arithmetic.job_manager.llm_reviewer reset")
        print("  python -m arithmetic.job_manager.llm_reviewer review 'Check this code for bugs'")
        print("  python -m arithmetic.job_manager.llm_reviewer debate 'Is our eval sound?'")
