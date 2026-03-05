"""Train the memory projection layer.

Runs the local LLM in text-injection mode, collects hidden state deltas,
and trains the projection to reproduce them.

Usage::

    python -m scripts.train_projection [checkpoint] [model_name] [num_episodes]
    python -m scripts.train_projection checkpoints/fast_weight_1k/final Qwen/Qwen2.5-3B-Instruct 200
"""
from __future__ import annotations

import os
import random
import sys

os.environ["BRAIN_USE_PATTERN_SEPARATION"] = "true"
os.environ["BRAIN_USE_DOPAMINERGIC_GATE"] = "true"
os.environ["BRAIN_USE_HOPFIELD_MEMORY"] = "true"
os.environ["BRAIN_USE_MODULAR_HOPFIELD"] = "true"
os.environ["BRAIN_USE_FAST_WEIGHT_MEMORY"] = "true"
os.environ["BRAIN_USE_TRANSFORMER_WM"] = "true"


def collect_delta(llm, messages_with_memory, messages_without_memory, layer_idx):
    """Get hidden state delta between memory-injected and clean forward passes."""
    import torch

    hidden_with = []
    hidden_without = []

    def capture_hook_with(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        hidden_with.append(h[:, -1, :].detach().clone())  # Last token

    def capture_hook_without(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        hidden_without.append(h[:, -1, :].detach().clone())

    layer = llm._get_layer(layer_idx)

    # Forward with memory text
    handle = layer.register_forward_hook(capture_hook_with)
    text_with = llm.tokenizer.apply_chat_template(
        messages_with_memory, tokenize=False, add_generation_prompt=True
    )
    inputs_with = llm.tokenizer(text_with, return_tensors="pt").to(llm.model.device)
    with torch.no_grad():
        llm.model(**inputs_with)
    handle.remove()

    # Forward without memory text
    handle = layer.register_forward_hook(capture_hook_without)
    text_without = llm.tokenizer.apply_chat_template(
        messages_without_memory, tokenize=False, add_generation_prompt=True
    )
    inputs_without = llm.tokenizer(text_without, return_tensors="pt").to(
        llm.model.device
    )
    with torch.no_grad():
        llm.model(**inputs_without)
    handle.remove()

    if hidden_with and hidden_without:
        delta = hidden_with[0] - hidden_without[0]  # [1, hidden_dim]
        return delta.squeeze(0)
    return None


def main() -> None:
    import torch

    from config.settings import settings
    from memory.neural_bridge import (
        MemoryProjection,
        ProjectionReplayBuffer,
        train_projection_step,
    )
    from memory.observer import NeuralMemoryObserver
    from pipeline.local_llm import LocalLLM

    checkpoint = (
        sys.argv[1] if len(sys.argv) > 1 else "checkpoints/fast_weight_1k/final"
    )
    model_name = sys.argv[2] if len(sys.argv) > 2 else settings.local_llm_model
    num_episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 200

    print(f"Training projection: {num_episodes} episodes")

    # Load LLM
    llm = LocalLLM(model_name=model_name)
    llm.load()
    # Disable the injection hook during training (we want clean forward passes)
    if llm._hook_handle:
        llm._hook_handle.remove()
        llm._hook_handle = None

    # Load brain
    observer = NeuralMemoryObserver()
    if os.path.isdir(checkpoint):
        observer._trainer.load_checkpoint(checkpoint)
        for name in ("hopfield", "gat"):
            comp = observer._trainer.components.get(name)
            if comp is not None:
                comp.enabled = False

    # Create projection — brain_dim matches embedding_dim since retrieve() returns embedding-space tensors
    projection = MemoryProjection(brain_dim=settings.embedding_dim, llm_dim=llm.hidden_dim).to(
        llm.model.device
    )
    optimizer = torch.optim.Adam(
        projection.parameters(), lr=settings.neural_projection_lr
    )
    buffer = ProjectionReplayBuffer(maxlen=1000)

    # Training conversations
    training_turns = [
        "My name is Alex and I work at Google.",
        "I have a cat named Whiskers who is 5 years old.",
        "I'm learning Rust programming language.",
        "My favorite food is sushi, especially salmon rolls.",
        "I live in Stockholm, Sweden.",
        "I'm training a neural network for image recognition.",
        "My birthday is March 15th.",
        "I play guitar in a band called Electric Dreams.",
        "I'm planning a trip to Japan next summer.",
        "My best friend's name is Marcus.",
    ]

    queries = [
        "What's my name?",
        "Do I have any pets?",
        "What programming language am I learning?",
        "What food do I like?",
        "Where do I live?",
        "What am I working on?",
        "When is my birthday?",
        "What instrument do I play?",
        "Where am I traveling?",
        "Who is my friend?",
    ]

    print("Phase 1: Collecting hidden state deltas...")

    for episode in range(num_episodes):
        # Pick a random setup turn and query
        setup_idx = random.randint(0, len(training_turns) - 1)
        query_idx = setup_idx if setup_idx < len(queries) else random.randint(
            0, len(queries) - 1
        )

        setup = training_turns[setup_idx]
        query = queries[query_idx]

        # Feed setup through brain
        observer.observe(setup, speaker="user")
        observer.observe("Got it!", speaker="assistant")

        # Get memory retrieval for the query
        observer.observe(query, speaker="user")

        # Build messages with and without memory
        messages_no_memory = [
            {"role": "user", "content": setup},
            {"role": "assistant", "content": "Got it!"},
            {"role": "user", "content": query},
        ]

        messages_with_memory, _ = observer.activate_and_inject(
            list(messages_no_memory)
        )

        # Collect hidden state delta
        delta = collect_delta(
            llm, messages_with_memory, messages_no_memory, llm._injection_layer
        )

        if delta is not None:
            # Get the brain's raw retrieval
            query_emb = observer.working_memory.predict_next_embedding()
            if query_emb is not None and observer._hopfield is not None:
                # Try modular retrieval
                if hasattr(observer._hopfield, "modules_list"):
                    from memory.hopfield_memory import ModularFastWeightMemory

                    if isinstance(observer._hopfield, ModularFastWeightMemory):
                        routes = observer._hopfield.router.route_read(query_emb)
                        dev = query_emb.device
                        combined_output = torch.zeros(settings.embedding_dim, device=dev)
                        combined_energy = torch.tensor(0.0, device=dev)
                        for mod_idx, w in routes:
                            mod = observer._hopfield.modules_list[mod_idx]
                            out, eng = mod.retrieve(query_emb)
                            combined_output += out * w
                            combined_energy += eng * w
                        buffer.push(combined_output, combined_energy, delta)
                elif hasattr(observer._hopfield, "retrieve"):
                    out, eng = observer._hopfield.retrieve(query_emb)
                    buffer.push(out, eng, delta)

        # Train every 10 episodes
        if (episode + 1) % 10 == 0:
            for _ in range(5):  # Multiple training steps per collection
                loss = train_projection_step(projection, buffer, optimizer)
            if loss is not None:
                print(
                    f"  Episode {episode + 1}/{num_episodes}: "
                    f"loss={loss:.4f} buffer={len(buffer)}"
                )

        # Reset working memory between episodes
        observer.working_memory.clear()

    # Save projection
    os.makedirs("checkpoints/projection", exist_ok=True)
    torch.save(projection.state_dict(), "checkpoints/projection/projection.pt")
    print(f"\nProjection saved to checkpoints/projection/projection.pt")
    print(f"Final buffer size: {len(buffer)}")

    llm.unload()


if __name__ == "__main__":
    main()
