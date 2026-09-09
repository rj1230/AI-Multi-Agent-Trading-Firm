"""
NewsAgent (Phase 4).

Fetches recent headlines for a ticker, then asks the LLM (via Groq) for a
structured {direction, confidence, rationale} read. The LLM's raw output
is validated against a strict Pydantic schema before it's allowed to
become a Signal in TradingState — malformed output never propagates
silently (see doc §4.4; full retry/guardrails layer lands in Phase 8, this
is the first, minimal version of that same principle).

Zero-articles handling (Phase 4 self-check): if fetch_news returns no
articles, we skip the LLM call entirely and return a neutral, zero-
confidence signal saying so. This is deliberate — there's no headline
for the model to reason about, so calling it anyway would just invite
a hallucinated rationale.

Coherence check (Phase 8, doc §4.4): Pydantic above only validates
*shape* (valid direction enum, confidence in range, non-empty rationale).
It cannot catch a self-contradictory but well-formed response, e.g.
direction="bullish" paired with a rationale describing bad news. That's
what agents.guardrails.check_signal_coherence exists for -- it runs
after Pydantic validation succeeds, and a failed coherence check falls
straight to neutral/hold rather than retrying (unlike the JSON-parse
retry above, retrying the same prompt would likely reproduce the same
contradiction).
"""

from __future__ import annotations

import json
import logging
import os

from pydantic import BaseModel, Field, ValidationError

from agents.guardrails import check_signal_coherence
from data_sources import fetch_news
from graph.state import Signal

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("GROQ_NEWS_MODEL", "llama-3.3-70b-versatile")

SYSTEM_PROMPT = (
    "You are a financial news analyst. Given recent headlines for a stock "
    "ticker, assess the likely short-term directional sentiment. "
    "Respond with ONLY a JSON object, no other text, matching exactly:\n"
    '{"direction": "bullish" | "bearish" | "neutral", '
    '"confidence": <float 0.0-1.0>, "rationale": "<one short sentence>"}\n'
    "If headlines are mixed, old news already priced in, or genuinely "
    'ambiguous, prefer "neutral" with lower confidence rather than '
    "forcing a directional call."
)


class LLMNewsOutput(BaseModel):
    """Strict schema the raw LLM JSON must satisfy before it becomes a
    Signal. Rejects out-of-range confidence and non-enum directions
    instead of silently coercing them."""

    direction: str = Field(pattern="^(bullish|bearish|neutral)$")
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=500)


def _build_user_prompt(ticker: str, headlines: list[str]) -> str:
    joined = "\n".join(f"- {h}" for h in headlines)
    return f"Ticker: {ticker}\n\nRecent headlines:\n{joined}"


def _call_groq(prompt: str) -> str:
    from groq import Groq

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def _parse_and_validate(raw: str) -> LLMNewsOutput | None:
    try:
        data = json.loads(raw)
        return LLMNewsOutput(**data)
    except (json.JSONDecodeError, ValidationError) as e:
        logger.warning("NewsAgent: LLM output failed validation: %s", e)
        return None


def _fallback_signal(rationale: str) -> Signal:
    return Signal(direction="neutral", confidence=0.0, rationale=rationale)


def run_news_agent(ticker: str) -> Signal:
    """Core logic, decoupled from the graph node wrapper so it's directly
    unit-testable without needing a TradingState."""
    news = fetch_news(ticker)

    if news.is_empty:
        return _fallback_signal("no recent headlines available")

    headlines = [a.title for a in news.articles]
    prompt = _build_user_prompt(ticker, headlines)

    try:
        raw = _call_groq(prompt)
    except Exception as e:
        logger.warning("NewsAgent: Groq call failed for %s: %s", ticker, e)
        return _fallback_signal(f"LLM call failed: {e}")

    parsed = _parse_and_validate(raw)
    if parsed is None:
        # One retry before giving up — malformed structured output from
        # LLMs is an expected failure mode, not an edge case (doc §4.4).
        try:
            raw_retry = _call_groq(prompt)
            parsed = _parse_and_validate(raw_retry)
        except Exception as e:
            logger.warning("NewsAgent: Groq retry failed for %s: %s", ticker, e)

    if parsed is None:
        return _fallback_signal("LLM output failed validation after retry")

    # Pydantic confirmed the *shape* is valid; now confirm the *content*
    # is self-consistent (doc §4.4). No retry here — a failed coherence
    # check means the model's reasoning was internally contradictory,
    # and retrying the identical prompt would likely reproduce it.
    if parsed.direction != "neutral":
        coherent, reason = check_signal_coherence(parsed.direction, parsed.rationale)
        if not coherent:
            logger.warning(
                "NewsAgent: incoherent signal for %s (%s): %s",
                ticker,
                parsed.direction,
                reason,
            )
            return _fallback_signal(f"incoherent signal rejected: {reason}")

    return Signal(
        direction=parsed.direction,
        confidence=parsed.confidence,
        rationale=parsed.rationale,
    )
