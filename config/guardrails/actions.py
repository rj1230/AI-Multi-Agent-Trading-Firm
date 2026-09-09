"""
Custom action backing the "check signal coherence" output rail.

Calls Groq directly (same provider NewsAgent already uses) rather than
routing through NeMo's own `main` LLM engine -- avoids needing to
register Groq as a NeMo-recognized provider, and keeps this check on the
same model family the rest of the system already depends on.
"""

from __future__ import annotations

import json
import logging
import os

import config.settings  # noqa: F401 -- triggers load_dotenv() at import time, ensures .env is loaded before this file's os.getenv("GROQ_API_KEY") calls regardless of import order elsewhere

from nemoguardrails.actions import action

logger = logging.getLogger(__name__)

COHERENCE_CHECK_MODEL = os.getenv("GROQ_GUARDRAILS_MODEL", "openai/gpt-oss-20b")

COHERENCE_SYSTEM_PROMPT = (
    "You are a strict consistency checker. Given a directional call and its "
    "stated rationale, decide if the rationale actually supports the "
    "direction. Respond with ONLY JSON: "
    '{"coherent": true|false, "reason": "<short reason if false, empty if true>"}'
)


@action(name="check_signal_coherence_action")
async def check_signal_coherence_action(direction: str, rationale: str) -> dict:
    if direction == "neutral":
        return {"coherent": True, "reason": ""}

    try:
        from groq import Groq

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model=COHERENCE_CHECK_MODEL,
            messages=[
                {"role": "system", "content": COHERENCE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f'Direction: {direction}\nRationale: "{rationale}"',
                },
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        return {
            "coherent": bool(data.get("coherent", True)),
            "reason": str(data.get("reason", "")),
        }
    except Exception as e:
        logger.warning("Guardrails coherence action failed open: %s", e)
        return {"coherent": True, "reason": ""}
