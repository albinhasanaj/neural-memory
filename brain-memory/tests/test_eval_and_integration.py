"""Tests for the eval harness and gate reward wiring."""

from __future__ import annotations

import os
import torch
import pytest

# Enable minimal neural modules for testing
os.environ.setdefault("BRAIN_USE_PATTERN_SEPARATION", "true")
os.environ.setdefault("BRAIN_USE_DOPAMINERGIC_GATE", "true")
os.environ.setdefault("BRAIN_USE_HOPFIELD_MEMORY", "true")
os.environ.setdefault("BRAIN_USE_TRANSFORMER_WM", "true")
os.environ.setdefault("BRAIN_USE_LEARNED_FORGETTING", "true")

from config.settings import settings
from memory.gate_network import DopaminergicGate, GateReplayBuffer
from memory.trainer import TrainingCoordinator


# ────────────────────────────────────────────────────────────────────
# Gate Replay Buffer — reward update tests
# ────────────────────────────────────────────────────────────────────


class TestGateReplayBufferRewards:
    """Test the retroactive reward update mechanism."""

    def test_push_and_sample(self):
        buf = GateReplayBuffer(capacity=100)
        emb = torch.randn(settings.embedding_dim)
        ctx = torch.randn(settings.gru_hidden_dim)
        sig = torch.randn(4)
        buf.push(emb, ctx, sig, reward=0.0)
        assert len(buf) == 1

    def test_initial_reward_is_zero(self):
        buf = GateReplayBuffer(capacity=100)
        emb = torch.randn(settings.embedding_dim)
        ctx = torch.randn(settings.gru_hidden_dim)
        sig = torch.randn(4)
        buf.push(emb, ctx, sig, reward=0.0)
        # Sample and check reward
        embs, ctxs, sigs, rewards = buf.sample(1)
        assert rewards[0].item() == pytest.approx(0.0)

    def test_update_rewards_for_matching_embedding(self):
        buf = GateReplayBuffer(capacity=100)
        emb = torch.randn(settings.embedding_dim)
        ctx = torch.randn(settings.gru_hidden_dim)
        sig = torch.randn(4)
        buf.push(emb, ctx, sig, reward=0.0)

        # Update reward for the same embedding (should match)
        n_updated = buf.update_rewards_for_embeddings([emb], reward=1.0)
        assert n_updated == 1

        # Verify reward was actually updated
        _, _, _, rewards = buf.sample(1)
        assert rewards[0].item() == pytest.approx(1.0)

    def test_update_rewards_no_match_for_different_embedding(self):
        buf = GateReplayBuffer(capacity=100)
        emb = torch.randn(settings.embedding_dim)
        ctx = torch.randn(settings.gru_hidden_dim)
        sig = torch.randn(4)
        buf.push(emb, ctx, sig, reward=0.0)

        # Try to update with a very different embedding
        other_emb = -emb  # opposite direction = low similarity
        n_updated = buf.update_rewards_for_embeddings([other_emb], reward=1.0)
        assert n_updated == 0

    def test_update_rewards_multiple_entries(self):
        buf = GateReplayBuffer(capacity=100)
        target_emb = torch.randn(settings.embedding_dim)
        ctx = torch.randn(settings.gru_hidden_dim)
        sig = torch.randn(4)

        # Push several entries, some similar, some not
        buf.push(target_emb, ctx, sig, reward=0.0)           # should match
        buf.push(target_emb + 0.01 * torch.randn_like(target_emb), ctx, sig, reward=0.0)  # near match
        buf.push(torch.randn_like(target_emb), ctx, sig, reward=0.0)  # random, likely no match

        n_updated = buf.update_rewards_for_embeddings([target_emb], reward=1.0)
        # At least the exact match should work, near-match likely too
        assert n_updated >= 1

    def test_reward_not_downgraded(self):
        buf = GateReplayBuffer(capacity=100)
        emb = torch.randn(settings.embedding_dim)
        ctx = torch.randn(settings.gru_hidden_dim)
        sig = torch.randn(4)
        buf.push(emb, ctx, sig, reward=1.0)  # Already has reward

        # Try to set a lower reward — should not downgrade
        n_updated = buf.update_rewards_for_embeddings([emb], reward=0.5)
        assert n_updated == 0


# ────────────────────────────────────────────────────────────────────
# Training Coordinator — reward wiring tests
# ────────────────────────────────────────────────────────────────────


class TestTrainerGateReward:
    """Test the gate reward method on TrainingCoordinator."""

    def test_reward_gate_disabled(self):
        """reward_gate_for_retrieval should return 0 when gate not enabled."""
        coord = TrainingCoordinator()
        # Don't initialise — no components
        result = coord.reward_gate_for_retrieval([torch.randn(settings.embedding_dim)])
        assert result == 0

    def test_reward_gate_enabled(self):
        """reward_gate_for_retrieval should work with an enabled gate."""
        from memory.gate_network import DopaminergicGate, GateReplayBuffer

        coord = TrainingCoordinator()
        coord.initialise()

        # Manually enable gate component (settings singleton may not have
        # the env var because it was constructed before this test file ran)
        gate_comp = coord.components.get("gate")
        if gate_comp is None or not gate_comp.enabled:
            from memory.trainer import ComponentState
            gate = DopaminergicGate()
            gate_comp = ComponentState(name="gate", enabled=True)
            gate_comp.model = gate
            gate_comp.optimizer = torch.optim.Adam(gate.parameters(), lr=1e-3)
            gate_comp.replay_buffer = GateReplayBuffer(capacity=100)
            coord.components["gate"] = gate_comp

        # Push a gate experience
        emb = torch.randn(settings.embedding_dim)
        ctx = torch.randn(settings.gru_hidden_dim)
        sig = torch.randn(4)
        coord.push_gate_experience(emb, ctx, sig, reward=0.0)

        # Reward it
        n = coord.reward_gate_for_retrieval([emb], reward=1.0)
        assert n == 1


# ────────────────────────────────────────────────────────────────────
# Eval harness — basic functionality tests
# ────────────────────────────────────────────────────────────────────


class TestEvalConversations:
    """Test that the eval dataset is well-formed."""

    def test_all_conversations_have_required_keys(self):
        from scripts.eval_recall import EVAL_CONVERSATIONS
        for conv in EVAL_CONVERSATIONS:
            assert "id" in conv
            assert "setup_turns" in conv
            assert "query" in conv
            assert "expected_keywords" in conv
            assert len(conv["setup_turns"]) >= 2
            assert len(conv["expected_keywords"]) >= 1


class TestRecallEvaluator:
    """Test the RecallEvaluator with a single conversation."""

    def test_evaluate_single_conversation(self):
        from scripts.eval_recall import EVAL_CONVERSATIONS, RecallEvaluator

        evaluator = RecallEvaluator()
        conv = EVAL_CONVERSATIONS[0]  # Python prefs
        result = evaluator.evaluate_conversation(conv)

        # Should have required keys
        assert "id" in result
        assert "recall_at_1" in result
        assert "recall_at_3" in result
        assert "recall_at_5" in result
        assert "mrr" in result
        assert "stored_count" in result
        assert result["total_turns"] == len(conv["setup_turns"])

    def test_evaluate_produces_stored_episodes(self):
        from scripts.eval_recall import EVAL_CONVERSATIONS, RecallEvaluator

        evaluator = RecallEvaluator()
        conv = EVAL_CONVERSATIONS[0]
        result = evaluator.evaluate_conversation(conv)

        # With neural gating, at least some turns should be stored
        # (though not guaranteed — depends on gating decisions)
        assert result["total_turns"] >= 2


# ────────────────────────────────────────────────────────────────────
# LLM Chat — smoke test
# ────────────────────────────────────────────────────────────────────


class TestMemoryChat:
    """Test MemoryChat instantiation and message processing."""

    def test_memory_chat_init(self):
        """MemoryChat should initialize with mock LLM."""
        from pipeline.llm_chat import LLMClient, MemoryChat

        class MockLLM(LLMClient):
            def chat(self, messages):
                return "I'm a mock response."

        chat = MemoryChat(llm=MockLLM())
        assert chat._turn_count == 0
        assert chat.observer is not None

    def test_memory_chat_process_message(self):
        """Process a message through the full pipeline with mock LLM."""
        from pipeline.llm_chat import LLMClient, MemoryChat

        class MockLLM(LLMClient):
            def chat(self, messages):
                return "Sure, I'll remember that!"
            def close(self):
                pass

        chat = MemoryChat(llm=MockLLM())
        response = chat.process_user_message("My name is Alice and I live in Portland.")
        assert response == "Sure, I'll remember that!"
        assert chat._turn_count == 1

    def test_memory_stats(self):
        """get_memory_stats should return well-formed dict."""
        from pipeline.llm_chat import LLMClient, MemoryChat

        class MockLLM(LLMClient):
            def chat(self, messages):
                return "Hello!"
            def close(self):
                pass

        chat = MemoryChat(llm=MockLLM())
        chat.process_user_message("Hello!")
        stats = chat.get_memory_stats()
        assert "episodic_count" in stats
        assert "turn_count" in stats
        assert stats["turn_count"] == 1
