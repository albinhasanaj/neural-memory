"""Tests for the intent-cue detector."""

from __future__ import annotations

import pytest

from nlp.intent_detector import IntentCueResult, detect_intent_cues


class TestDetectIntentCues:
    """Test that canonical recall phrases are detected."""

    @pytest.mark.parametrize(
        "text",
        [
            "Do you remember when we discussed the API?",
            "Remember that project we mentioned?",
            "remember how I set up Docker?",
        ],
    )
    def test_remember_when(self, text: str) -> None:
        result = detect_intent_cues(text)
        assert result.is_recall is True
        assert result.confidence >= 0.9

    @pytest.mark.parametrize(
        "text",
        [
            "Last time we talked about Redis.",
            "Earlier I mentioned my SSH key.",
            "Previously we settled on Next.js.",
        ],
    )
    def test_last_time_earlier(self, text: str) -> None:
        result = detect_intent_cues(text)
        assert result.is_recall is True

    def test_what_was_my(self) -> None:
        result = detect_intent_cues("What was my API key?")
        assert result.is_recall is True
        assert result.confidence == 1.0
        assert any("api key" in t.lower() for t in result.targets)

    def test_what_did_i(self) -> None:
        result = detect_intent_cues("What did I decide about the database?")
        assert result.is_recall is True
        assert len(result.targets) > 0

    def test_you_told_me(self) -> None:
        result = detect_intent_cues("You told me to use FastAPI instead.")
        assert result.is_recall is True

    def test_remind_me(self) -> None:
        result = detect_intent_cues("Remind me about the deployment steps.")
        assert result.is_recall is True
        assert any("deployment steps" in t.lower() for t in result.targets)

    def test_no_recall_intent(self) -> None:
        result = detect_intent_cues("I want to build a REST API with Flask.")
        assert result.is_recall is False
        assert result.confidence == 0.0
        assert result.targets == []

    def test_go_back_to(self) -> None:
        result = detect_intent_cues("Let's go back to the auth discussion.")
        assert result.is_recall is True

    def test_can_you_recall(self) -> None:
        result = detect_intent_cues("Can you find my old config?")
        assert result.is_recall is True
