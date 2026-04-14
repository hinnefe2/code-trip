"""LLM-based intent classifier for voice transcripts.

Runs in front of per-mode PTT dispatch in the orchestrator. Maps an
utterance to one of a small set of known intents (mode change, status
query, verbosity change). Returns ``None`` when no intent matches, so the
caller can fall through to the existing per-mode handler.
"""

from __future__ import annotations

import enum
import json
import logging
import os
from dataclasses import dataclass, field

import openai
from openai import APIConnectionError, APIError, AuthenticationError

logger = logging.getLogger(__name__)

DEFAULT_MODEL: str = "gpt-4o-mini"
ENV_API_KEY: str = "OPENAI_API_KEY"


class Intent(enum.Enum):
    LIST_TICKETS = "list_tickets"
    SWITCH_TICKET = "switch_ticket"
    STATUS = "status"
    SHIP_IT = "ship_it"
    SET_VERBOSITY = "set_verbosity"


INTENT_PROMPT: str = (
    "You classify short spoken utterances from a developer driving a coding "
    "assistant by voice. Map the utterance to ONE of these intents, or null "
    "if none fit:\n"
    '  - "list_tickets": user wants to see/browse their assigned tickets '
    '(e.g. "list my tickets", "show tickets", "what am I working on")\n'
    '  - "switch_ticket": user wants to switch to ticket number N. '
    'arg = integer N (1-based)\n'
    '  - "status": user asks for current status (e.g. "status", '
    '"what\'s the status", "where am I")\n'
    '  - "ship_it": user is ready to ship/open a PR (e.g. "ship it", '
    '"ship", "open the PR")\n'
    '  - "set_verbosity": user wants summaries shorter/longer. '
    'arg = "brief" | "detailed" | "verbose"\n'
    "\n"
    'Respond with ONLY a JSON object: {"intent": "<name>" | null, '
    '"arg": <int|string|null>}.\n'
    "Do not wrap in markdown. If no intent matches, respond "
    '{"intent": null, "arg": null}. Do not invent intents.'
)


@dataclass
class IntentResult:
    intent: Intent
    arg: int | str | None = None


class IntentClassifierError(Exception):
    """Raised when intent classification fails for a non-recoverable reason."""


@dataclass
class IntentClassifier:
    """Classifies a transcript into a known :class:`Intent` via the OpenAI API."""

    model: str = DEFAULT_MODEL
    api_key: str | None = None

    _client: openai.OpenAI = field(default=None, init=False, repr=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        resolved_key = self.api_key or os.environ.get(ENV_API_KEY)
        if not resolved_key:
            raise IntentClassifierError(
                f"No API key: set {ENV_API_KEY} or pass api_key"
            )
        self._client = openai.OpenAI(api_key=resolved_key)

    def classify(self, transcript: str) -> IntentResult | None:
        """Classify *transcript*; return ``None`` if no known intent matches."""
        text = transcript.strip()
        if not text:
            return None

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": INTENT_PROMPT},
                    {"role": "user", "content": text},
                ],
                response_format={"type": "json_object"},
            )
        except AuthenticationError as exc:
            raise IntentClassifierError(
                f"Authentication failed: {exc}"
            ) from exc
        except APIConnectionError as exc:
            raise IntentClassifierError(f"Network error: {exc}") from exc
        except APIError as exc:
            raise IntentClassifierError(f"API error: {exc}") from exc

        raw = (response.choices[0].message.content or "").strip()
        if not raw:
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Intent classifier returned non-JSON: %r", raw)
            return None
        if not isinstance(payload, dict):
            return None

        name = payload.get("intent")
        if not name:
            return None
        try:
            intent = Intent(name)
        except ValueError:
            logger.warning("Intent classifier returned unknown intent: %r", name)
            return None

        arg = payload.get("arg")
        if intent is Intent.SWITCH_TICKET:
            try:
                arg = int(arg)
            except (TypeError, ValueError):
                logger.warning("SWITCH_TICKET missing integer arg: %r", arg)
                return None
        elif intent is Intent.SET_VERBOSITY:
            if not isinstance(arg, str) or arg not in {"brief", "detailed", "verbose"}:
                logger.warning("SET_VERBOSITY invalid arg: %r", arg)
                return None
        else:
            arg = None

        return IntentResult(intent=intent, arg=arg)
