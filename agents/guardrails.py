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
"""

from __future__ import annotations

import asyncio
import logging

from nemoguardrails import LLMRails, RailsConfig

logger = logging.getLogger(__name__)

_rails_singleton: LLMRails | None = None


def _rails() -> LLMRails:
    global _rails_singleton
    if _rails_singleton is None:
        config = RailsConfig.from_path("config/guardrails")
        _rails_singleton = LLMRails(config)
    return _rails_singleton


async def _check_async(direction: str, rationale: str) -> tuple[bool, str]:
    try:
        rails = _rails()
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
