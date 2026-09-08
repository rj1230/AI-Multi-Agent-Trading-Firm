import pytest
from agents.signal_merger import Signal, merge_signals


def test_agreement_averages_confidence():
    news = Signal(direction="bullish", confidence=0.7, rationale="earnings beat")
    chart = Signal(direction="bullish", confidence=0.6, rationale="RSI recovering")
    merged = merge_signals(news, chart)
    assert merged.direction == "bullish"
    assert merged.agreement is True
    assert merged.combined_confidence == pytest.approx(0.65)


def test_disagreement_defaults_to_hold():
    news = Signal(direction="bullish", confidence=0.8, rationale="positive headline")
    chart = Signal(direction="bearish", confidence=0.9, rationale="death cross")
    merged = merge_signals(news, chart)
    assert merged.direction == "neutral"
    assert merged.agreement is False
    assert merged.combined_confidence == 0.0


def test_both_neutral_stays_neutral():
    news = Signal(direction="neutral", confidence=0.0, rationale="no articles")
    chart = Signal(direction="neutral", confidence=0.1, rationale="no clear setup")
    merged = merge_signals(news, chart)
    assert merged.direction == "neutral"
    assert merged.agreement is True
    assert merged.combined_confidence == pytest.approx(0.05)


def test_merge_is_deterministic_same_inputs_same_output():
    news = Signal(direction="bullish", confidence=0.55, rationale="x")
    chart = Signal(direction="bullish", confidence=0.45, rationale="y")
    assert merge_signals(news, chart) == merge_signals(news, chart)
