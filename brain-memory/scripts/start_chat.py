"""Launch interactive chat with full brain memory system."""
from __future__ import annotations

import os
import sys

# Enable all neural modules before any brain-memory imports
os.environ["BRAIN_USE_PATTERN_SEPARATION"] = "true"
os.environ["BRAIN_USE_DOPAMINERGIC_GATE"] = "true"
os.environ["BRAIN_USE_HOPFIELD_MEMORY"] = "true"
os.environ["BRAIN_USE_MODULAR_HOPFIELD"] = "true"
os.environ["BRAIN_USE_FAST_WEIGHT_MEMORY"] = "true"
os.environ["BRAIN_USE_TRANSFORMER_WM"] = "true"
os.environ["BRAIN_USE_LEARNED_FORGETTING"] = "true"
os.environ["BRAIN_USE_VAE_CONSOLIDATION"] = "true"
os.environ["BRAIN_USE_SALIENCE_MLP"] = "true"
os.environ["BRAIN_USE_GNN_ACTIVATION"] = "false"  # GAT not trained for fast-weight


def main() -> None:
    from config.settings import settings
    from pipeline.llm_chat import LLMClient, MemoryChat

    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/fast_weight_1k/final"

    # Resolve LLM settings
    api_key = settings.openai_api_key or settings.anthropic_api_key or ""
    provider = settings.llm_provider
    model = settings.llm_model
    base_url = settings.llm_base_url

    print("=" * 60)
    print("  BRAIN MEMORY — Interactive Chat")
    print("  Memory: Phase 3 (Fast Weight)")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  LLM: {provider} / {model}")
    print("=" * 60)
    print()

    llm = LLMClient(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
    )

    chat = MemoryChat(llm=llm, checkpoint_path=checkpoint_path)

    # Disable incompatible training components to avoid .patterns crash
    if hasattr(chat.observer, "_trainer") and hasattr(chat.observer._trainer, "components"):
        for name in ("hopfield", "gat"):
            comp = chat.observer._trainer.components.get(name)
            if comp is not None:
                comp.enabled = False

    print("Type your messages below. Type '/quit' to exit.")
    print("Type '/stats', '/memory', '/modules', '/save', '/forget' for controls.")
    print()

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break
            if not user_input:
                continue

            if user_input.startswith("/"):
                cmd = user_input.lower().split()[0]
                if cmd in ("/quit", "/exit"):
                    break
                elif cmd == "/stats":
                    stats = chat.get_memory_stats()
                    print("\n  Memory Stats:")
                    for k, v in stats.items():
                        print(f"    {k}: {v}")
                    print()
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
                elif cmd == "/save":
                    chat.observer._trainer.save_checkpoint("checkpoints/chat_session")
                    print("  Checkpoint saved to checkpoints/chat_session/\n")
                    continue
                elif cmd == "/forget":
                    if chat.observer._hopfield is not None and hasattr(chat.observer._hopfield, "clear"):
                        chat.observer._hopfield.clear()
                        print("  Fast-weight memory cleared.\n")
                    else:
                        print("  No clearable memory.\n")
                    continue
                elif cmd == "/clear":
                    chat.observer.working_memory.clear()
                    print("  Working memory cleared.\n")
                    continue
                else:
                    print(f"  Unknown command: {cmd}\n")
                    continue

            response = chat.process_user_message(user_input)
            print(f"\nAssistant: {response}\n")

    except KeyboardInterrupt:
        print("\n\nInterrupted.")
    finally:
        chat.observer._trainer.save_checkpoint("checkpoints/chat_session")
        llm.close()
        print("Session saved. Goodbye!")


if __name__ == "__main__":
    main()
