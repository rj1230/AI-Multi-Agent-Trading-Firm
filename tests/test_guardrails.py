"""
Tests for the semantic coherence guardrail (Phase 8, doc section 4.4).

Covers the two real bugs found during manual debugging:
- execute_action() returning a (dict, status) tuple, not a dict directly
- GROQ_API_KEY not being loaded when config.settings isn't imported first
"""

from __future__ import annotations

import pytest

from agents.guardrails import check_signal_coherence, _check_async


@pytest.fixture
def anyio_backend():
    return "asyncio"


class TestCheckSignalCoherence:
    """Sync wrapper -- what NewsAgent and other sync call sites actually use."""

    def test_incoherent_bullish_with_bearish_rationale(self):
        coherent, reason = check_signal_coherence(
            "bullish", "stock price is crashing due to bad earnings"
        )
        assert coherent is False
        assert reason

    def test_neutral_direction_always_coherent(self):
        coherent, reason = check_signal_coherence("neutral", "mixed signals, no clear read")
        assert coherent is True
        assert reason == ""

    def test_coherent_bullish_with_supporting_rationale(self):
        coherent, reason = check_signal_coherence(
            "bullish", "strong earnings beat and raised guidance"
        )
        assert coherent is True


class TestCheckAsyncReturnShape:
    """Guards against regressing the tuple-unwrap bug specifically."""

    @pytest.mark.anyio
    async def test_check_async_returns_bool_str_tuple(self):
        coherent, reason = await _check_async(
            "bullish", "stock price is crashing due to bad earnings"
        )
        assert isinstance(coherent, bool)
        assert isinstance(reason, str)
