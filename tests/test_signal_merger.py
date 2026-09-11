import pytest
from agents.signal_merger import Signal, merge_signals


def test_agreement_averages_confidence():
    news = Signal(direction="bullish", confidence=0.7, rationale="earnings beat")
    chart = Signal(direction="bullish", confidence=0.6, rationale="RSI recovering")
    merged = merge_signals(news, chart)
    assert merged.direction == "bullish"
    assert merged.agreement is True
    assert merged.combined_confidence == pytest.approx(0.65)


def test_bearish_agreement_averages_confidence():
    """Case 2 also covers bearish/bearish agreement, not just bullish/bullish."""
    news = Signal(direction="bearish", confidence=0.5, rationale="bad guidance")
    chart = Signal(
        direction="bearish", confidence=0.3, rationale="breakdown below support"
    )
    merged = merge_signals(news, chart)
    assert merged.direction == "bearish"
    assert merged.agreement is True
    assert merged.combined_confidence == pytest.approx(0.4)


def test_disagreement_defaults_to_hold():
    news = Signal(direction="bullish", confidence=0.8, rationale="positive headline")
    chart = Signal(direction="bearish", confidence=0.9, rationale="death cross")
    merged = merge_signals(news, chart)
    assert merged.direction == "neutral"
    assert merged.agreement is False
    assert merged.combined_confidence == 0.0


def test_reverse_disagreement_also_defaults_to_hold():
    """Case 5 is symmetric -- bearish/bullish must veto the same as bullish/bearish."""
    news = Signal(direction="bearish", confidence=0.9, rationale="death cross")
    chart = Signal(direction="bullish", confidence=0.8, rationale="positive headline")
    merged = merge_signals(news, chart)
    assert merged.direction == "neutral"
    assert merged.agreement is False
    assert merged.combined_confidence == 0.0


def test_both_neutral_stays_neutral():
    news = Signal(direction="neutral", confidence=0.0, rationale="no articles")
    chart = Signal(direction="neutral", confidence=0.1, rationale="no clear setup")
    merged = merge_signals(news, chart)
    assert merged.direction == "neutral"
    # Neither side has an opinion -- not the same claim as both sides
    # independently agreeing (that's reserved for real bullish/bullish or
    # bearish/bearish agreement, Case 2 in signal_merger.py).
    assert merged.agreement is False
    assert merged.combined_confidence == 0.0


def test_news_neutral_chart_bullish_reduces_confidence_not_contradiction():
    news = Signal(direction="neutral", confidence=0.0, rationale="no headlines")
    chart = Signal(direction="bullish", confidence=0.8, rationale="uptrend")
    merged = merge_signals(news, chart)
    assert merged.direction == "bullish"
    assert merged.agreement is False
    assert merged.combined_confidence == pytest.approx(0.6)  # 0.8 * 0.75


def test_news_neutral_chart_bearish_reduces_confidence_not_contradiction():
    """Case 3's other direction -- neutral News shouldn't only work for bullish Chart."""
    news = Signal(direction="neutral", confidence=0.0, rationale="no headlines")
    chart = Signal(direction="bearish", confidence=0.6, rationale="downtrend")
    merged = merge_signals(news, chart)
    assert merged.direction == "bearish"
    assert merged.agreement is False
    assert merged.combined_confidence == pytest.approx(0.45)  # 0.6 * 0.75


def test_chart_neutral_news_bearish_reduces_confidence_not_contradiction():
    news = Signal(direction="bearish", confidence=0.6, rationale="bad earnings")
    chart = Signal(direction="neutral", confidence=0.0, rationale="no clear setup")
    merged = merge_signals(news, chart)
    assert merged.direction == "bearish"
    assert merged.agreement is False
    assert merged.combined_confidence == pytest.approx(0.45)  # 0.6 * 0.75


def test_chart_neutral_news_bullish_reduces_confidence_not_contradiction():
    """Case 4's other direction -- neutral Chart shouldn't only work for bearish News."""
    news = Signal(direction="bullish", confidence=0.8, rationale="earnings beat")
    chart = Signal(direction="neutral", confidence=0.0, rationale="no clear setup")
    merged = merge_signals(news, chart)
    assert merged.direction == "bullish"
    assert merged.agreement is False
    assert merged.combined_confidence == pytest.approx(0.6)  # 0.8 * 0.75


def test_merge_is_deterministic_same_inputs_same_output():
    news = Signal(direction="bullish", confidence=0.55, rationale="x")
    chart = Signal(direction="bullish", confidence=0.45, rationale="y")
    assert merge_signals(news, chart) == merge_signals(news, chart)
