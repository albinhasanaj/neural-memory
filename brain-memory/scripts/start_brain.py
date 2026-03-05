"""Launch brain memory chat + visualization dashboard together.

Usage::

    python -m scripts.start_brain [checkpoint_path]
    python -m scripts.start_brain checkpoints/fast_weight_1k/final
"""
from __future__ import annotations

import os
import sys

# Set env vars BEFORE any brain-memory imports
os.environ["BRAIN_USE_PATTERN_SEPARATION"] = "true"
os.environ["BRAIN_USE_DOPAMINERGIC_GATE"] = "true"
os.environ["BRAIN_USE_HOPFIELD_MEMORY"] = "true"
os.environ["BRAIN_USE_MODULAR_HOPFIELD"] = "true"
os.environ["BRAIN_USE_FAST_WEIGHT_MEMORY"] = "true"
os.environ["BRAIN_USE_TRANSFORMER_WM"] = "true"
os.environ["BRAIN_USE_LEARNED_FORGETTING"] = "true"
os.environ["BRAIN_USE_VAE_CONSOLIDATION"] = "true"
os.environ["BRAIN_USE_SALIENCE_MLP"] = "true"
os.environ["BRAIN_USE_GNN_ACTIVATION"] = "false"
os.environ["BRAIN_CONSOLIDATION_INTERVAL"] = "10"


def main() -> None:
    from config.settings import settings
    from memory.observer import NeuralMemoryObserver
    from memory.neural_events import event_bus

    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/fast_weight_1k/final"

    # Create shared observer
    settings.ensure_data_dirs()
    observer = NeuralMemoryObserver()

    if os.path.isdir(checkpoint):
        observer._trainer.load_checkpoint(checkpoint)
        # Disable incompatible training components
        for name in ("hopfield", "gat"):
            comp = observer._trainer.components.get(name)
            if comp is not None:
                comp.enabled = False
        print(f"  Loaded checkpoint: {checkpoint}")
    else:
        print(f"  No checkpoint at {checkpoint} — running untrained")

    # Start 3D brain visualization server in background thread
    from visualization.neural_viz_server import start_viz_server

    start_viz_server(event_bus, observer, host="127.0.0.1", port=7860)
    print("  Neural visualization: http://127.0.0.1:7860")

    # Set up LLM client
    from pipeline.llm_chat import LLMClient

    api_key = settings.openai_api_key or settings.anthropic_api_key or ""
    client = LLMClient(
        provider=settings.llm_provider,
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=api_key,
    )

    print()
    print("=" * 60)
    print("  BRAIN MEMORY — Live Neural Chat")
    print(f"  LLM: {settings.llm_provider} / {settings.llm_model}")
    print("  Neural Viz: http://127.0.0.1:7860")
    print("  Commands: /modules /memory /stats /save /forget /quit")
    print("=" * 60)
    print()

    conversation_history: list[dict[str, str]] = []

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit"):
                break

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
                                    f"  Module {m['module_index']:2d}: "
                                    f"writes={m['write_count']:3d}  "
                                    f"w_key={m['w_key_norm']:.2f}  "
                                    f"entities={[e[0] for e in m['top_entities'][:3]]}"
                                )
                    else:
                        print("  No modular Hopfield available.")
                    print()
                    continue
                elif cmd == "/stats":
                    print(f"  Episodic entries: {observer.episodic_store.active_count}")
                    print(f"  Graph nodes: {observer.graph.num_nodes}")
                    if hasattr(observer._hopfield, "total_writes"):
                        print(f"  Hopfield writes: {observer._hopfield.total_writes()}")
                    print(f"  Trainer step: {observer._trainer.global_step}")
                    print()
                    continue
                elif cmd == "/save":
                    observer._trainer.save_checkpoint("checkpoints/live_session")
                    print("  Checkpoint saved to checkpoints/live_session/\n")
                    continue
                elif cmd == "/memory":
                    for event in reversed(event_bus.recent(50)):
                        if event.event_type == "inject":
                            for txt in event.data.get("memory_texts", []):
                                print(f"  - {txt}")
                            break
                    else:
                        print("  No injected memories yet.")
                    print()
                    continue
                elif cmd == "/forget":
                    if hasattr(observer._hopfield, "clear"):
                        observer._hopfield.clear()
                        print("  Fast-weight memory cleared.\n")
                    else:
                        print("  No clearable memory.\n")
                    continue
                elif cmd == "/debug":
                    # Show internal state of each module's decode index + write history
                    print("  ── Hopfield internals ──")
                    for i, m in enumerate(observer._hopfield.modules_list):
                        di = len(m._decode_index)
                        wh = len(m._write_history)
                        if di > 0 or wh > 0:
                            print(f"  Module {i:2d}: decode_index={di} entries, write_history={wh} entries")
                            for h, meta in list(m._decode_index.items())[:2]:
                                print(f"    hash={h[:16]}..  text={meta.get('text', '')[:60]}")
                    # Try a test retrieval
                    ctx = observer.working_memory.context_vector
                    if ctx is not None:
                        import torch
                        with torch.no_grad():
                            ctx_emb = observer.working_memory.encoder.predictor(ctx)
                        results = observer._hopfield.retrieve_decoded(ctx_emb, top_k=5)
                        print(f"  Retrieval results: {len(results)}")
                        for r in results:
                            print(f"    score={r['score']:.4f}  text={r.get('text', '')[:60]}")
                    else:
                        print("  No context vector yet (say something first)")
                    print()
                    continue
                else:
                    print(f"  Unknown command: {cmd}\n")
                    continue

            # Process through brain memory
            observer.observe(user_input, speaker="user")

            # Build messages with memory injection
            conversation_history.append({"role": "user", "content": user_input})
            messages_for_llm, activated = observer.activate_and_inject(
                list(conversation_history)
            )

            # Call LLM
            try:
                response = client.chat(messages_for_llm)
            except Exception as e:
                response = f"[LLM Error: {e}]"

            # Observe assistant response
            observer.observe(response, speaker="assistant")
            conversation_history.append({"role": "assistant", "content": response})

            print(f"\nAssistant: {response}\n")

    except KeyboardInterrupt:
        print("\n\nInterrupted.")
    finally:
        observer._trainer.save_checkpoint("checkpoints/live_session")
        client.close()
        print("Session saved. Goodbye!")


if __name__ == "__main__":
    main()
