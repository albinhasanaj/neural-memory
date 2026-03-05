"""
LLM Chat with Neural Memory — interactive CLI chat with brain-inspired memory.

Connects the neural memory system to any OpenAI-compatible LLM API
(OpenAI, Anthropic, local Ollama, LM Studio, etc.) and provides an
interactive chat with automatic memory storage, retrieval, and injection.

The system:
1. Observes each user message → encodes, stores if salient, trains online
2. Activates relevant memories via spreading activation + Hopfield retrieval
3. Injects memory context into the LLM prompt
4. Observes the assistant response for additional memory encoding
5. Rewards the gate network when memories are successfully retrieved

Usage::

    # With OpenAI
    python -m pipeline.llm_chat --provider openai --model gpt-4o --api-key sk-...

    # With local Ollama
    python -m pipeline.llm_chat --provider openai --model llama3 --base-url http://localhost:11434/v1

    # With Anthropic
    python -m pipeline.llm_chat --provider anthropic --model claude-sonnet-4-20250514 --api-key sk-ant-...

    # Resume from checkpoint
    python -m pipeline.llm_chat --provider openai --model gpt-4o --checkpoint checkpoints/ultrachat_1k/final
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-28s │ %(levelname)-5s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("llm_chat")


# ────────────────────────────────────────────────────────────────────
# LLM Client
# ────────────────────────────────────────────────────────────────────


class LLMClient:
    """Synchronous client for OpenAI-compatible and Anthropic chat APIs."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o",
        base_url: str | None = None,
        api_key: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> None:
        self.provider = provider
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = httpx.Client(timeout=120.0)

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Send messages and return the assistant's response text."""
        if self.provider == "openai":
            return self._chat_openai(messages)
        elif self.provider == "anthropic":
            return self._chat_anthropic(messages)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _chat_openai(self, messages: list[dict[str, str]]) -> str:
        base = self.base_url or "https://api.openai.com/v1"
        url = f"{base}/chat/completions"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        resp = self._client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _chat_anthropic(self, messages: list[dict[str, str]]) -> str:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        # Anthropic expects system message separately
        system_text = ""
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_text += msg["content"] + "\n"
            else:
                chat_messages.append(msg)

        body: dict[str, Any] = {
            "model": self.model,
            "messages": chat_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if system_text.strip():
            body["system"] = system_text.strip()

        resp = self._client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"]

    def close(self) -> None:
        self._client.close()


# ────────────────────────────────────────────────────────────────────
# Memory-augmented chat session
# ────────────────────────────────────────────────────────────────────


class MemoryChat:
    """Interactive chat session with neural memory augmentation.

    Ties together:
    - NeuralMemoryObserver for memory processing
    - LLMClient for language model calls
    - Gate reward feedback loop
    - Checkpoint saving
    """

    SYSTEM_PROMPT = (
        "You are a helpful AI assistant with a brain-inspired memory system. "
        "You can remember facts, preferences, and context from previous conversations. "
        "When memory context is provided, use it naturally in your responses. "
        "If you recall something from memory, mention it conversationally."
    )

    def __init__(
        self,
        llm: LLMClient,
        checkpoint_path: str | None = None,
        save_transcript: str | None = None,
    ) -> None:
        self._enable_neural_modules()

        from config.settings import settings
        settings.ensure_data_dirs()

        from memory.observer import NeuralMemoryObserver
        self.observer = NeuralMemoryObserver()
        self.llm = llm
        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
        ]
        self.transcript: list[dict[str, str]] = []
        self._save_transcript_path = save_transcript
        self._turn_count = 0

        if checkpoint_path:
            self.observer._trainer.load_checkpoint(Path(checkpoint_path))
            logger.info("Loaded neural checkpoint from %s", checkpoint_path)

    @staticmethod
    def _enable_neural_modules() -> None:
        os.environ["BRAIN_USE_PATTERN_SEPARATION"] = "true"
        os.environ["BRAIN_USE_DOPAMINERGIC_GATE"] = "true"
        os.environ["BRAIN_USE_HOPFIELD_MEMORY"] = "true"
        os.environ["BRAIN_USE_VAE_CONSOLIDATION"] = "true"
        os.environ["BRAIN_USE_TRANSFORMER_WM"] = "true"
        os.environ["BRAIN_USE_LEARNED_FORGETTING"] = "true"

    def process_user_message(self, user_text: str) -> str:
        """Process a user message through the full pipeline:

        1. Observe user message (encode, gate, store, train)
        2. Activate & inject memories into prompt
        3. Send to LLM
        4. Observe assistant response
        5. Return response text
        """
        self._turn_count += 1

        # 1. Observe user turn
        obs_info = self.observer.observe(user_text, speaker="user")

        # 2. Activate & inject memories into messages
        self.messages.append({"role": "user", "content": user_text})
        injected_messages, activated = self.observer.activate_and_inject(self.messages)

        # Log memory activity
        n_activated = len(activated)
        stored = obs_info["stored"]
        salience = obs_info["salience"]
        logger.info(
            "Turn %d: salience=%.3f stored=%s activated=%d entities=%s",
            self._turn_count, salience, stored, n_activated,
            obs_info.get("entities", []),
        )

        # Log if memories were injected
        injected_count = len(injected_messages) - len(self.messages)
        if injected_count > 0:
            logger.info("  → Injected %d memory context message(s)", injected_count)

        # 3. Call LLM
        try:
            response_text = self.llm.chat(injected_messages)
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            response_text = f"[LLM Error: {e}]"

        # 4. Observe assistant response
        self.observer.observe(response_text, speaker="assistant")

        # 5. Update message history (without injection, to keep it clean)
        self.messages.append({"role": "assistant", "content": response_text})

        # Record transcript
        self.transcript.append({"role": "user", "content": user_text})
        self.transcript.append({"role": "assistant", "content": response_text})

        # 6. Log training diagnostics
        train_diag = obs_info.get("neural_training", {})
        active_losses = {
            k: v for k, v in train_diag.items()
            if isinstance(v, dict) and "loss" in v or isinstance(v, dict) and "total" in v
        }
        if active_losses:
            loss_str = ", ".join(
                f"{k}={v.get('total', v.get('loss', '?')):.4f}"
                for k, v in active_losses.items()
            )
            logger.info("  → Training: %s", loss_str)

        return response_text

    def save_transcript_to_file(self) -> None:
        """Save the conversation transcript to disk."""
        if self._save_transcript_path and self.transcript:
            path = Path(self._save_transcript_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.transcript, f, indent=2, ensure_ascii=False)
            logger.info("Transcript saved to %s", path)

    def get_memory_stats(self) -> dict:
        """Return current memory system statistics."""
        return {
            "episodic_count": self.observer.episodic_store.active_count,
            "graph_nodes": self.observer.graph.num_nodes,
            "graph_edges": self.observer.graph.num_edges,
            "wm_buffer_size": self.observer.working_memory.buffer.size,
            "hopfield_patterns": (
                self.observer._hopfield.num_patterns
                if self.observer._hopfield else 0
            ),
            "turn_count": self._turn_count,
            "trainer_step": self.observer._trainer.global_step,
        }


# ────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interactive chat with neural memory system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local Ollama
  python -m pipeline.llm_chat --provider openai --model llama3 --base-url http://localhost:11434/v1

  # OpenAI
  python -m pipeline.llm_chat --provider openai --model gpt-4o --api-key sk-...

  # With trained checkpoint
  python -m pipeline.llm_chat --provider openai --model gpt-4o --checkpoint checkpoints/ultrachat_1k/final
        """,
    )
    p.add_argument("--provider", type=str, default="openai", choices=["openai", "anthropic"])
    p.add_argument("--model", type=str, default="gpt-4o")
    p.add_argument("--base-url", type=str, default=None, help="Base URL for OpenAI-compatible API")
    p.add_argument("--api-key", type=str, default=None, help="API key (or set OPENAI_API_KEY / ANTHROPIC_API_KEY)")
    p.add_argument("--checkpoint", type=str, default=None, help="Path to neural checkpoint directory")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--transcript", type=str, default=None, help="Path to save conversation transcript")
    return p.parse_args()


def _print_banner() -> None:
    print("\n" + "=" * 60)
    print("  BRAIN MEMORY — Neural Memory-Augmented Chat")
    print("=" * 60)
    print("  Commands:")
    print("    /stats   — Show memory system statistics")
    print("    /memory  — Show last injected memories")
    print("    /modules — Show module specialization")
    print("    /forget  — Clear fast-weight memory")
    print("    /clear   — Clear working memory")
    print("    /save    — Save checkpoint")
    print("    /quit    — Exit")
    print("=" * 60 + "\n")


def main() -> None:
    args = parse_args()

    # Resolve API key
    api_key = args.api_key
    if not api_key:
        if args.provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY", "")
        elif args.provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    llm = LLMClient(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key=api_key or "",
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    chat = MemoryChat(
        llm=llm,
        checkpoint_path=args.checkpoint,
        save_transcript=args.transcript,
    )

    _print_banner()
    print(f"  Provider: {args.provider} | Model: {args.model}")
    if args.base_url:
        print(f"  Base URL: {args.base_url}")
    if args.checkpoint:
        print(f"  Checkpoint: {args.checkpoint}")
    print()

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input.lower().split()[0]
                if cmd == "/quit":
                    break
                elif cmd == "/stats":
                    stats = chat.get_memory_stats()
                    print("\n  Memory Stats:")
                    for k, v in stats.items():
                        print(f"    {k}: {v}")
                    print()
                    continue
                elif cmd == "/clear":
                    chat.observer.working_memory.clear()
                    print("  Working memory cleared.\n")
                    continue
                elif cmd == "/save":
                    ckpt_path = "checkpoints/chat_session"
                    chat.observer._trainer.save_checkpoint(ckpt_path)
                    print(f"  Checkpoint saved to {ckpt_path}\n")
                    continue
                elif cmd == "/memory":
                    from memory.neural_events import event_bus
                    for event in reversed(event_bus.recent(50)):
                        if event.event_type == "inject":
                            for txt in event.data.get("memory_texts", []):
                                print(f"  - {txt}")
                            break
                    else:
                        print("  No injected memories yet.")
                    print()
                    continue
                elif cmd == "/modules":
                    hopfield = chat.observer._hopfield
                    if hopfield is not None and hasattr(hopfield, "module_summary"):
                        for m in hopfield.module_summary():
                            if m["write_count"] > 0:
                                print(
                                    f"  Module {m['module_index']:2d}: "
                                    f"writes={m['write_count']:3d}  "
                                    f"w_key={m['w_key_norm']:.2f}  "
                                    f"entities={[e[0] for e in m['top_entities'][:3]]}"
                                )
                    else:
                        print("  No modular Hopfield available.")
                    print()
                    continue
                elif cmd == "/forget":
                    hopfield = chat.observer._hopfield
                    if hopfield is not None and hasattr(hopfield, "clear"):
                        hopfield.clear()
                        print("  Fast-weight memory cleared.\n")
                    else:
                        print("  No clearable memory.\n")
                    continue
                else:
                    print(f"  Unknown command: {cmd}\n")
                    continue

            # Process through memory pipeline + LLM
            response = chat.process_user_message(user_input)
            print(f"\nAssistant: {response}\n")

    except KeyboardInterrupt:
        print("\n\nInterrupted.")
    finally:
        chat.save_transcript_to_file()
        llm.close()
        print("Session ended.")


if __name__ == "__main__":
    main()
