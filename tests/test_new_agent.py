"""
NewsAgent tests. All external calls (fetch_news, Groq) are mocked — these
tests verify the agent's own logic (zero-article handling, JSON validation,
retry-then-fallback), not NewsAPI or Groq themselves.
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
    ):
        signal = run_news_agent("AAPL")

    assert signal.direction == "bearish"
    assert signal.confidence == 0.6
