"""Launch brain memory chat with neural injection into local LLM.

Memory activations are injected directly into the LLM's hidden states
via forward hooks — no text conversion needed.

Usage::

    python -m scripts.start_neural_chat [checkpoint_path] [model_name]
    python -m scripts.start_neural_chat checkpoints/fast_weight_1k/final Qwen/Qwen2.5-3B-Instruct
"""
from __future__ import annotations

import os
import sys
import time

# Enable all neural modules
os.environ["BRAIN_USE_PATTERN_SEPARATION"] = "true"
os.environ["BRAIN_USE_DOPAMINERGIC_GATE"] = "true"
os.environ["BRAIN_USE_HOPFIELD_MEMORY"] = "true"
os.environ["BRAIN_USE_MODULAR_HOPFIELD"] = "true"
os.environ["BRAIN_USE_FAST_WEIGHT_MEMORY"] = "true"
os.environ["BRAIN_USE_TRANSFORMER_WM"] = "true"
os.environ["BRAIN_USE_LEARNED_FORGETTING"] = "true"
os.environ["BRAIN_USE_NEURAL_INJECTION"] = "true"


def main() -> None:
    import torch

    from config.settings import settings
    from memory.neural_events import event_bus
    from memory.observer import NeuralMemoryObserver
    from pipeline.local_llm import LocalLLM

    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/fast_weight_1k/final"
    model_name = sys.argv[2] if len(sys.argv) > 2 else settings.local_llm_model

    print("=" * 60)
    print("  BRAIN MEMORY — Neural Injection Mode")
    print(f"  Local LLM: {model_name}")
    print(f"  Checkpoint: {checkpoint}")
    print("=" * 60)

    # Step 1: Load local LLM
    llm = LocalLLM(model_name=model_name)
    llm.load()

    # Step 2: Create brain memory observer
    observer = NeuralMemoryObserver()

    # Step 3: Load checkpoint
    if os.path.isdir(checkpoint):
        observer._trainer.load_checkpoint(checkpoint)
        for name in ("hopfield", "gat"):
            comp = observer._trainer.components.get(name)
            if comp is not None:
                comp.enabled = False
        print(f"  Brain checkpoint loaded: {checkpoint}")

    # Step 4: Initialize neural injection bridge
    observer.init_neural_injection(llm)
    print(f"  Neural bridge: brain[{settings.embedding_dim}] -> projection -> LLM[{llm.hidden_dim}]")
    print(f"  Injection layer: {llm._injection_layer}")

    # Step 5: Start visualization server (optional)
    try:
        from visualization.neural_viz_server import start_viz_server

        start_viz_server(event_bus, observer, port=7860)
        print("  Neural viz: http://127.0.0.1:7860")
    except Exception as e:
        print(f"  Viz server not available: {e}")

    print()
    print("  Type messages to chat. Memory is injected neurally.")
    print("  Type /compare to see neural vs text injection side by side.")
    print("  Type /modules, /stats, /save, /quit for brain commands.")
    print("=" * 60)
    print()

    conversation_history: list[dict[str, str]] = []

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if user_input.lower() in ("quit", "exit", "/quit"):
                break
            if not user_input:
                continue

            # Special commands
            if user_input.startswith("/"):
                cmd = user_input.lower().split()[0]
                if cmd in ("/quit", "/exit"):
                    break
                elif cmd == "/modules":
                    if hasattr(observer._hopfield, "module_summary"):
                        for m in observer._hopfield.module_summary():
                            if m["write_count"] > 0:
                                print(
                                    f"  Module {m['module_id']:2d}: "
                                    f"writes={m['write_count']:3.0f} "
                                    f"w_key={m['w_key_norm']:.2f}"
                                )
                    print()
                    continue
                elif cmd == "/stats":
                    print(f"  VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
                    print(f"  Injection layer: {llm._injection_layer}")
                    if observer._projection is not None:
                        print(
                            f"  Injection strength: "
                            f"{observer._projection.injection_strength}"
                        )
                    if hasattr(observer._hopfield, "total_writes"):
                        print(f"  Hopfield writes: {observer._hopfield.total_writes()}")
                    print()
                    continue
                elif cmd == "/compare":
                    print(
                        "  [Neural mode active — memory injected as hidden state modification]"
                    )
                    if llm._memory_vector is not None:
                        vec = llm._memory_vector
                        print(f"  Injection vector norm: {vec.norm().item():.4f}")
                        print(f"  Injection vector mean: {vec.mean().item():.6f}")
                    # Also show what text decode would give
                    query_emb = observer.working_memory.predict_next_embedding()
                    if query_emb is not None and observer._hopfield is not None:
                        decoded = observer._hopfield.retrieve_decoded(query_emb, top_k=3)
                        print("  Text decode would have retrieved:")
                        for d in decoded:
                            print(f"    score={d['score']:.3f}: {d.get('text', '')[:80]}")
                    print()
                    continue
                elif cmd == "/save":
                    # Sleep consolidation — intensive replay before saving
                    print("  Running memory consolidation (replay)...")
                    if hasattr(observer._hopfield, "replay_recent"):
                        for _ in range(5):
                            observer._hopfield.replay_recent(n=10, replay_strength=0.5)
                    print("  Consolidation complete")
                    observer._trainer.save_checkpoint("checkpoints/neural_session")
                    # Explicitly save decode index
                    if hasattr(observer._hopfield, "save_decode_index"):
                        observer._hopfield.save_decode_index(
                            "checkpoints/neural_session/fast_weight_decode.json"
                        )
                    if observer._projection is not None:
                        torch.save(
                            observer._projection.state_dict(),
                            "checkpoints/neural_session/projection.pt",
                        )
                    print("  Saved checkpoint + decode index + projection weights\n")
                    continue
                else:
                    print(f"  Unknown command: {cmd}\n")
                    continue

            # Process through brain memory
            observer.observe(user_input, speaker="user")

            # Prepare injection (this sets the memory vector on the LLM)
            conversation_history.append({"role": "user", "content": user_input})

            # The activate_and_inject call now sets the injection vector
            # instead of modifying the messages list
            messages_for_llm, activated = observer.activate_and_inject(
                list(conversation_history)
            )

            # Generate with local LLM (hook injects memory during forward pass)
            response = llm.generate(messages_for_llm, max_new_tokens=512, temperature=0.7)

            # Clear injection vector after generation
            llm.set_memory_vector(None)

            # Observe response
            observer.observe(response, speaker="assistant")
            conversation_history.append({"role": "assistant", "content": response})

            print(f"\nAssistant: {response}\n")

    except KeyboardInterrupt:
        print("\n\nInterrupted.")
    finally:
        # Sleep consolidation before exit
        if hasattr(observer._hopfield, "replay_recent"):
            for _ in range(5):
                observer._hopfield.replay_recent(n=10, replay_strength=0.5)
        os.makedirs("checkpoints/neural_session", exist_ok=True)
        observer._trainer.save_checkpoint("checkpoints/neural_session")
        # Explicitly save decode index
        if hasattr(observer._hopfield, "save_decode_index"):
            observer._hopfield.save_decode_index(
                "checkpoints/neural_session/fast_weight_decode.json"
            )
        if observer._projection is not None:
            torch.save(
                observer._projection.state_dict(),
                "checkpoints/neural_session/projection.pt",
            )
        print("Session saved. Goodbye!")
        llm.unload()


if __name__ == "__main__":
    main()
