"""
Semantic guardrail layer (Phase 8, doc section 4.4): catches output that
passes LLMNewsOutput's Pydantic schema but is internally incoherent --
e.g. direction="bullish" with a rationale describing bad news. Pydantic
checks type/range; this checks whether the rationale actually supports
the direction, via a real NeMo Guardrails config (config/guardrails/).

Invocation note: this calls the registered action directly through
LLMRails' action dispatcher rather than through rails.generate() --
Colang's dialog-flow intent matching needs its own LLM call to classify
messages, which isn't a fit for a single deterministic structured check.
The Colang flow in rails.co still defines the rail's policy; this just
skips paying for intent classification to reach it.

Singleton construction: guarded by an asyncio.Lock with double-checked
locking, since LLMRails(config) does non-trivial setup (loading the
Colang config) and multiple concurrent tick-runner tasks could otherwise
race to construct it simultaneously.
"""

from __future__ import annotations

import asyncio
import logging

from nemoguardrails import LLMRails, RailsConfig

logger = logging.getLogger(__name__)

_rails_singleton: LLMRails | None = None
_rails_lock = asyncio.Lock()  # guards first-construction against concurrent tasks


def _rails_sync() -> LLMRails:
    """Non-locking accessor for sync contexts where construction is
    already known-safe (kept for any existing sync call sites)."""
    global _rails_singleton
    if _rails_singleton is None:
        config = RailsConfig.from_path("config/guardrails")
        _rails_singleton = LLMRails(config)
    return _rails_singleton


async def _rails_async() -> LLMRails:
    global _rails_singleton
    if _rails_singleton is not None:
        return _rails_singleton
    async with _rails_lock:
        # Re-check inside the lock -- another task may have finished
        # construction while this one was waiting for the lock.
        if _rails_singleton is None:
            config = RailsConfig.from_path("config/guardrails")
            _rails_singleton = LLMRails(config)
    return _rails_singleton


async def _check_async(direction: str, rationale: str) -> tuple[bool, str]:
    try:
        rails = await _rails_async()
        # execute_action returns (result_dict, status), NOT the result
        # dict directly -- confirmed by direct inspection (repr shows a
        # 2-tuple). Unpacking this wrong was a real, previously-fixed bug
        # that regressed when the locking change above was built on an
        # older copy of this file -- see agents/news_agent.py's coherence
        # wiring and tests/test_guardrails.py for the regression test.
        result_dict, status = await rails.runtime.action_dispatcher.execute_action(
            "check_signal_coherence_action",
            {"direction": direction, "rationale": rationale},
        )
        if status != "success":
            logger.warning("check_signal_coherence_action returned status=%r", status)
            return True, ""
        return bool(result_dict.get("coherent", True)), str(
            result_dict.get("reason", "")
        )
    except Exception as e:
        logger.warning("Guardrails check failed open: %s", e)
        return True, ""


def check_signal_coherence(direction: str, rationale: str) -> tuple[bool, str]:
    """Sync wrapper -- news_agent.py's call sites are sync. Fails open
    (coherent=True) on any error reaching the guardrails layer itself,
    same reasoning as NewsAgent's own Groq-call failure handling: a
    guardrails outage shouldn't block every trade the way a primary LLM
    outage already does via NewsAgent's existing fallback."""
    return asyncio.run(_check_async(direction, rationale))
