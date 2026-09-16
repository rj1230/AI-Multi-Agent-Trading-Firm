"""
NewsAgent.

Fetches recent headlines for a ticker, then asks the LLM (via Groq) for a
structured {direction, confidence, rationale} read.

The agent explicitly distinguishes:

    AVAILABLE + articles
        Headlines exist -> analyze them with the LLM.

    AVAILABLE + no articles
        The news source/cache exists, but there are no eligible headlines
        for the requested point in time -> neutral signal.

    UNAVAILABLE
        Historical news data does not exist -> neutral signal, but clearly
        reports that the absence is a DATA AVAILABILITY limitation rather
        than claiming that there was no news.

    ERROR
        The news source/cache could not be loaded -> neutral signal and
        explicit error provenance.

The LLM's raw output is validated against a strict Pydantic schema before
it is allowed to become a Signal. Malformed output never propagates
silently.

Coherence check:
Pydantic validates shape, but not semantic consistency. A response such
as direction="bullish" with a bearish rationale can still be structurally
valid. agents.guardrails.check_signal_coherence() handles this case.

Important backtest principle:
Missing historical news must NEVER be silently interpreted as "there was
no news." This matters because an empty historical news cache can otherwise
make backtest results appear artificially neutral.
"""

from __future__ import annotations

import json
import logging
import os

from pydantic import BaseModel, Field, ValidationError

from agents.guardrails import check_signal_coherence
from data_sources import fetch_news
from data_sources.schemas import NewsAvailability
from graph.state import Signal

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv(
    "GROQ_NEWS_MODEL",
    "openai/gpt-oss-120b",
)

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
    """Strict schema for validated LLM output."""

    direction: str = Field(pattern="^(bullish|bearish|neutral)$")

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    rationale: str = Field(
        min_length=1,
        max_length=500,
    )


def _build_user_prompt(
    ticker: str,
    headlines: list[str],
) -> str:
    """Build the LLM prompt from validated headline titles."""

    joined = "\n".join(f"- {headline}" for headline in headlines)

    return f"Ticker: {ticker}\n\nRecent headlines:\n{joined}"


def _call_groq(prompt: str) -> str:
    """Call the configured Groq model."""

    from groq import Groq

    client = Groq(
        api_key=os.getenv("GROQ_API_KEY"),
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

    return response.choices[0].message.content


def _parse_and_validate(
    raw: str,
) -> LLMNewsOutput | None:
    """
    Parse and validate raw LLM JSON.

    Invalid JSON or schema violations return None rather than allowing
    malformed output to propagate into TradingState.
    """

    try:
        data = json.loads(raw)

        return LLMNewsOutput(
            **data,
        )

    except (
        json.JSONDecodeError,
        ValidationError,
    ) as e:
        logger.warning(
            "NewsAgent: LLM output failed validation: %s",
            e,
        )

        return None


def _fallback_signal(
    rationale: str,
) -> Signal:
    """Return a deterministic neutral fallback signal."""

    return Signal(
        direction="neutral",
        confidence=0.0,
        rationale=rationale,
    )


def _handle_news_availability(
    ticker: str,
    availability: NewsAvailability,
) -> Signal | None:
    """
    Convert data-availability states into deterministic fallback signals.

    Returns:
        Signal -> stop processing because the data cannot/should not be
                  sent to the LLM.

        None -> news is available and should be analyzed.
    """

    if availability == NewsAvailability.UNAVAILABLE:
        logger.warning(
            "[NewsAgent] ticker=%s historical news UNAVAILABLE; "
            "no archived news dataset/cache is available for this "
            "backtest period",
            ticker,
        )

        return _fallback_signal(
            "historical news unavailable; no archived news data for this period"
        )

    if availability == NewsAvailability.ERROR:
        logger.warning(
            "[NewsAgent] ticker=%s news source returned ERROR; news analysis skipped",
            ticker,
        )

        return _fallback_signal("historical news data unavailable due to source error")

    if availability == NewsAvailability.AVAILABLE:
        return None

    # Defensive fallback in case a future enum value is introduced.
    logger.error(
        "[NewsAgent] ticker=%s received unknown news availability=%s",
        ticker,
        availability,
    )

    return _fallback_signal("news availability state unsupported")


def run_news_agent(
    ticker: str,
) -> Signal:
    """
    Run the news-analysis agent.

    This function is intentionally decoupled from the graph node so it can
    be unit-tested independently.
    """

    news = fetch_news(ticker)

    # ------------------------------------------------------------------
    # DATA PROVENANCE / AVAILABILITY
    # ------------------------------------------------------------------
    logger.info(
        "[NewsAgent] ticker=%s availability=%s "
        "is_empty=%s num_articles=%s source=%s mode=%s as_of=%s",
        ticker,
        news.availability.value,
        news.is_empty,
        len(news.articles),
        news.source,
        news.mode.value,
        news.as_of.isoformat(),
    )

    availability_signal = _handle_news_availability(
        ticker,
        news.availability,
    )

    if availability_signal is not None:
        return availability_signal

    # ------------------------------------------------------------------
    # AVAILABLE SOURCE BUT ZERO ARTICLES
    # ------------------------------------------------------------------
    if not news.has_articles:
        logger.info(
            "[NewsAgent] ticker=%s news source AVAILABLE but "
            "no eligible headlines were found",
            ticker,
        )

        return _fallback_signal("no eligible headlines available")

    # ------------------------------------------------------------------
    # LLM NEWS ANALYSIS
    # ------------------------------------------------------------------
    headlines = [article.title for article in news.articles]

    prompt = _build_user_prompt(
        ticker,
        headlines,
    )

    try:
        raw = _call_groq(prompt)

    except Exception as e:
        logger.warning(
            "NewsAgent: Groq call failed for %s: %s",
            ticker,
            e,
        )

        return _fallback_signal(f"LLM call failed: {e}")

    # ------------------------------------------------------------------
    # PARSE + VALIDATE
    # ------------------------------------------------------------------
    parsed = _parse_and_validate(raw)

    if parsed is None:
        # One retry before giving up. Structured-output failures are an
        # expected LLM failure mode rather than an exceptional edge case.
        try:
            raw_retry = _call_groq(prompt)

            parsed = _parse_and_validate(
                raw_retry,
            )

        except Exception as e:
            logger.warning(
                "NewsAgent: Groq retry failed for %s: %s",
                ticker,
                e,
            )

    if parsed is None:
        return _fallback_signal("LLM output failed validation after retry")

    # ------------------------------------------------------------------
    # SEMANTIC COHERENCE
    # ------------------------------------------------------------------
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

            return _fallback_signal(f"incoherent signal rejected: {reason}")

    # ------------------------------------------------------------------
    # FINAL VALIDATED SIGNAL
    # ------------------------------------------------------------------
    return Signal(
        direction=parsed.direction,
        confidence=parsed.confidence,
        rationale=parsed.rationale,
    )
