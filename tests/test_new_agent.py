"""
NewsAgent tests. All external calls (fetch_news, Groq, and the guardrails
coherence check) are mocked — these tests verify the agent's own logic
(zero-article handling, JSON validation, retry-then-fallback, coherence
gating), not NewsAPI, Groq, or NeMo Guardrails themselves.
"""

import json
from datetime import datetime, timezone
from unittest.mock import patch

from agents.news_agent import run_news_agent
from data_sources.schemas import DataSourceMode, NewsArticle, NewsResult


def _make_news_result(titles: list[str]) -> NewsResult:
    now = datetime.now(timezone.utc)
    return NewsResult(
        ticker="AAPL",
        articles=[
            NewsArticle(
                title=t, source="TestWire", published_at=now, url="https://example.com"
            )
            for t in titles
        ],
        source="newsapi",
        mode=DataSourceMode.LIVE,
        as_of=now,
    )


def test_zero_articles_skips_llm_and_returns_neutral():
    empty = _make_news_result([])
    with (
        patch("agents.news_agent.fetch_news", return_value=empty),
        patch("agents.news_agent._call_groq") as mock_groq,
    ):
        signal = run_news_agent("AAPL")

    mock_groq.assert_not_called()
    assert signal.direction == "neutral"
    assert signal.confidence == 0.0


def test_valid_llm_response_produces_matching_signal():
    news = _make_news_result(["Apple beats earnings expectations"])
    llm_json = json.dumps(
        {"direction": "bullish", "confidence": 0.8, "rationale": "Strong earnings beat"}
    )
    with (
        patch("agents.news_agent.fetch_news", return_value=news),
        patch("agents.news_agent._call_groq", return_value=llm_json),
        # Coherence check runs for any non-neutral direction; mock it so
        # this test doesn't make a real Groq call via the guardrails
        # action and stays deterministic like every other agent test here.
        patch("agents.news_agent.check_signal_coherence", return_value=(True, "")),
    ):
        signal = run_news_agent("AAPL")

    assert signal.direction == "bullish"
    assert signal.confidence == 0.8
    assert "earnings" in signal.rationale.lower()


def test_malformed_json_retries_then_falls_back():
    news = _make_news_result(["Some headline"])
    with (
        patch("agents.news_agent.fetch_news", return_value=news),
        patch("agents.news_agent._call_groq", return_value="not valid json"),
    ):
        signal = run_news_agent("AAPL")

    assert signal.direction == "neutral"
    assert signal.confidence == 0.0
    assert "validation" in signal.rationale.lower()


def test_out_of_range_confidence_is_rejected():
    news = _make_news_result(["Some headline"])
    bad_json = json.dumps(
        {"direction": "bullish", "confidence": 1.5, "rationale": "too confident"}
    )
    with (
        patch("agents.news_agent.fetch_news", return_value=news),
        patch("agents.news_agent._call_groq", return_value=bad_json),
    ):
        signal = run_news_agent("AAPL")

    # 1.5 is out of the 0.0-1.0 range -> validation must reject it, not clamp it
    assert signal.direction == "neutral"
    assert signal.confidence == 0.0


def test_second_attempt_succeeds_after_first_malformed():
    news = _make_news_result(["Some headline"])
    good_json = json.dumps(
        {"direction": "bearish", "confidence": 0.6, "rationale": "Weak guidance"}
    )
    with (
        patch("agents.news_agent.fetch_news", return_value=news),
        patch("agents.news_agent._call_groq", side_effect=["garbage", good_json]),
        patch("agents.news_agent.check_signal_coherence", return_value=(True, "")),
    ):
        signal = run_news_agent("AAPL")

    assert signal.direction == "bearish"
    assert signal.confidence == 0.6


def test_neutral_direction_skips_coherence_check():
    # Neutral has no directional claim to check for coherence against —
    # confirms the agent doesn't call the guardrail unnecessarily.
    news = _make_news_result(["Mixed signals, no clear catalyst"])
    llm_json = json.dumps(
        {
            "direction": "neutral",
            "confidence": 0.3,
            "rationale": "No clear catalyst either way",
        }
    )
    with (
        patch("agents.news_agent.fetch_news", return_value=news),
        patch("agents.news_agent._call_groq", return_value=llm_json),
        patch("agents.news_agent.check_signal_coherence") as mock_coherence,
    ):
        signal = run_news_agent("AAPL")

    mock_coherence.assert_not_called()
    assert signal.direction == "neutral"
    assert signal.confidence == 0.3


def test_incoherent_signal_falls_back_to_neutral():
    # Pydantic-valid shape (enum direction, in-range confidence, non-empty
    # rationale) but self-contradictory content: bullish direction paired
    # with a rationale describing bad news. This is exactly the case
    # Pydantic's schema can't catch and check_signal_coherence exists for.
    news = _make_news_result(["Earnings miss sends shares tumbling"])
    llm_json = json.dumps(
        {
            "direction": "bullish",
            "confidence": 0.8,
            "rationale": "stock price is crashing due to bad earnings",
        }
    )
    with (
        patch("agents.news_agent.fetch_news", return_value=news),
        patch("agents.news_agent._call_groq", return_value=llm_json),
        patch(
            "agents.news_agent.check_signal_coherence",
            return_value=(False, "Rationale indicates bearish sentiment"),
        ),
    ):
        signal = run_news_agent("AAPL")

    assert signal.direction == "neutral"
    assert signal.confidence == 0.0
    assert "incoherent" in signal.rationale.lower()


def test_coherent_signal_passes_through_unmodified():
    # Sanity check for the passing branch: a genuinely coherent bullish
    # call is not blocked or altered by the guardrail.
    news = _make_news_result(["Apple raises full-year guidance"])
    llm_json = json.dumps(
        {
            "direction": "bullish",
            "confidence": 0.75,
            "rationale": "raised full-year guidance on strong demand",
        }
    )
    with (
        patch("agents.news_agent.fetch_news", return_value=news),
        patch("agents.news_agent._call_groq", return_value=llm_json),
        patch("agents.news_agent.check_signal_coherence", return_value=(True, "")),
    ):
        signal = run_news_agent("AAPL")

    assert signal.direction == "bullish"
    assert signal.confidence == 0.75
