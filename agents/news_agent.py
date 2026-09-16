"""
NewsAgent.

Fetches historical/recent headlines for a ticker, then asks the LLM
(via Groq) for a structured directional read.

Availability semantics:

    AVAILABLE + articles
        Headlines exist -> analyze them with the LLM.

    AVAILABLE + no articles
        The source/cache exists, but no eligible headlines exist for the
        requested point in time -> neutral signal.

    UNAVAILABLE
        Historical news data does not exist -> neutral signal, while
        explicitly reporting DATA AVAILABILITY limitation.

    ERROR
        The news source/cache could not be loaded -> neutral signal with
        explicit error provenance.

Production API:

    run_news_agent_with_metadata(ticker) -> NewsAgentResult

Backward-compatible API:

    run_news_agent(ticker) -> Signal

Important backtest principle:

    Missing historical news must NEVER be silently interpreted as
    "there was no news."
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, Field, ValidationError

from agents.guardrails import check_signal_coherence
from data_sources import fetch_news
from data_sources.schemas import NewsAvailability, NewsResult
from graph.state import Signal

logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

DEFAULT_MODEL = os.getenv(
    "GROQ_NEWS_MODEL",
    "openai/gpt-oss-120b",
)


SYSTEM_PROMPT = (
    "You are a financial news analyst. Given recent headlines for a stock "
    "ticker, assess the likely short-term directional sentiment. "
    "Respond with ONLY a JSON object, no other text, matching exactly:\n"
    '{"direction": "bullish" | "bearish" | "neutral", '
    '"confidence": <float 0.0-1.0>, '
    '"rationale": "<one short sentence>"}\n'
    "If headlines are mixed, old news already priced in, or genuinely "
    'ambiguous, prefer "neutral" with lower confidence rather than '
    "forcing a directional call."
)


# ============================================================
# Structured result
# ============================================================


@dataclass(frozen=True)
class NewsAgentResult:
    """
    Structured result returned by the metadata-aware NewsAgent.

    Signal and data provenance remain separate concerns.
    """

    signal: Signal
    availability: NewsAvailability
    article_count: int
    source: str
    as_of: datetime


# ============================================================
# LLM output schema
# ============================================================


class LLMNewsOutput(BaseModel):
    """Strict schema for validated LLM output."""

    direction: str = Field(
        pattern=r"^(bullish|bearish|neutral)$",
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    rationale: str = Field(
        min_length=1,
        max_length=500,
    )


# ============================================================
# Prompt
# ============================================================


def _build_user_prompt(
    ticker: str,
    headlines: list[str],
) -> str:
    """Build the LLM prompt from validated headline titles."""

    joined = "\n".join(f"- {headline}" for headline in headlines)

    return f"Ticker: {ticker}\n\nRecent headlines:\n{joined}"


# ============================================================
# Groq
# ============================================================


def _call_groq(
    prompt: str,
) -> str:
    """Call the configured Groq model."""

    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    client = Groq(
        api_key=api_key,
    )

    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
        response_format={
            "type": "json_object",
        },
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError("Groq returned an empty response.")

    return content


# ============================================================
# Validation
# ============================================================


def _parse_and_validate(
    raw: str,
) -> LLMNewsOutput | None:
    """
    Parse and validate raw LLM JSON.

    Invalid output never propagates into TradingState.
    """

    try:
        data = json.loads(raw)

        return LLMNewsOutput(
            **data,
        )

    except (
        json.JSONDecodeError,
        ValidationError,
        TypeError,
    ) as exc:
        logger.warning(
            "NewsAgent: LLM output failed validation: %s",
            exc,
        )

        return None


# ============================================================
# Fallback
# ============================================================


def _fallback_signal(
    rationale: str,
) -> Signal:
    """Return deterministic neutral fallback."""

    return Signal(
        direction="neutral",
        confidence=0.0,
        rationale=rationale,
    )


# ============================================================
# Availability handling
# ============================================================


def _handle_news_availability(
    ticker: str,
    availability: NewsAvailability,
) -> Signal | None:
    """
    Convert data-availability states into deterministic fallbacks.

    Returns:

        Signal
            Stop processing.

        None
            News is available and should be analyzed.
    """

    if availability == NewsAvailability.UNAVAILABLE:
        logger.warning(
            "[NewsAgent] ticker=%s historical news UNAVAILABLE; "
            "no archived news dataset/cache is available for this "
            "backtest period",
            ticker,
        )

        return _fallback_signal(
            "historical news unavailable; no archived news data for this period",
        )

    if availability == NewsAvailability.ERROR:
        logger.warning(
            "[NewsAgent] ticker=%s news source returned ERROR; news analysis skipped",
            ticker,
        )

        return _fallback_signal(
            "historical news data unavailable due to source error",
        )

    if availability == NewsAvailability.AVAILABLE:
        return None

    logger.error(
        "[NewsAgent] ticker=%s received unknown news availability=%s",
        ticker,
        availability,
    )

    return _fallback_signal(
        "news availability state unsupported",
    )


# ============================================================
# News analysis
# ============================================================


def _analyze_news(
    ticker: str,
    news: NewsResult,
) -> Signal:
    """
    Analyze an already-fetched NewsResult.

    The news object is passed in so the metadata-aware API performs
    exactly one fetch.
    """

    # --------------------------------------------------------
    # Availability
    # --------------------------------------------------------

    availability_signal = _handle_news_availability(
        ticker,
        news.availability,
    )

    if availability_signal is not None:
        return availability_signal

    # --------------------------------------------------------
    # AVAILABLE source but no eligible articles
    # --------------------------------------------------------

    if not news.has_articles:
        logger.info(
            "[NewsAgent] ticker=%s news source AVAILABLE but "
            "no eligible headlines were found",
            ticker,
        )

        return _fallback_signal(
            "no eligible headlines available",
        )

    # --------------------------------------------------------
    # LLM analysis
    # --------------------------------------------------------

    headlines = [article.title for article in news.articles]

    prompt = _build_user_prompt(
        ticker,
        headlines,
    )

    try:
        raw = _call_groq(prompt)

    except Exception as exc:
        logger.warning(
            "NewsAgent: Groq call failed for %s: %s",
            ticker,
            exc,
        )

        return _fallback_signal(
            f"LLM call failed: {exc}",
        )

    # --------------------------------------------------------
    # Parse + validate
    # --------------------------------------------------------

    parsed = _parse_and_validate(raw)

    if parsed is None:
        try:
            raw_retry = _call_groq(prompt)

            parsed = _parse_and_validate(
                raw_retry,
            )

        except Exception as exc:
            logger.warning(
                "NewsAgent: Groq retry failed for %s: %s",
                ticker,
                exc,
            )

    if parsed is None:
        return _fallback_signal(
            "LLM output failed validation after retry",
        )

    # --------------------------------------------------------
    # Semantic coherence
    # --------------------------------------------------------

    if parsed.direction != "neutral":
        coherent, reason = check_signal_coherence(
            parsed.direction,
            parsed.rationale,
        )

        if not coherent:
            logger.warning(
                "NewsAgent: incoherent signal for %s (%s): %s",
                ticker,
                parsed.direction,
                reason,
            )

            return _fallback_signal(
                f"incoherent signal rejected: {reason}",
            )

    # --------------------------------------------------------
    # Final validated signal
    # --------------------------------------------------------

    return Signal(
        direction=parsed.direction,
        confidence=parsed.confidence,
        rationale=parsed.rationale,
    )


# ============================================================
# Metadata-aware production API
# ============================================================


def run_news_agent_with_metadata(
    ticker: str,
) -> NewsAgentResult:
    """
    Fetch news exactly once and return:

        Signal
        +
        availability
        +
        article count
        +
        source
        +
        timestamp

    This is the API used by the production graph.
    """

    news = fetch_news(ticker)

    logger.info(
        "[NewsAgent] ticker=%s availability=%s articles=%s source=%s mode=%s as_of=%s",
        ticker,
        news.availability.value,
        len(news.articles),
        news.source,
        news.mode.value,
        news.as_of.isoformat(),
    )

    signal = _analyze_news(
        ticker=ticker,
        news=news,
    )

    return NewsAgentResult(
        signal=signal,
        availability=news.availability,
        article_count=len(news.articles),
        source=news.source,
        as_of=news.as_of,
    )


# ============================================================
# Backward-compatible API
# ============================================================


def run_news_agent(
    ticker: str,
) -> Signal:
    """
    Backward-compatible API.

    Existing callers that only need a Signal continue to work.
    """

    return run_news_agent_with_metadata(ticker).signal
