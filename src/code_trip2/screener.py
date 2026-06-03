"""Task screener: auto-handle producer output via skill agents.

Producers push Tasks into an intake queue instead of calling
:meth:`TaskQueue.add` directly. The screener loop drains the intake
queue and for each task:

1. Filters skill manifests to those declaring ``auto-handle: true`` and
   listing this task's ``kind`` under ``auto-handle-kinds``. No
   candidates → forward to the user-facing queue (no LLM cost).
2. Asks Claude (Haiku, no tools) which candidate, if any, can fully
   handle the task. Unsure → ``NONE`` → forward.
3. If a skill was named, runs it via ``run_agent`` with that skill's
   tool list. The task never enters the user-facing queue; the outcome
   is reported via ``on_outcome`` for logging.

All decision logic lives in plain functions returning a
:class:`ScreeningOutcome`. The loop function is the only thing with
lifecycle, and it's just a coroutine waiting on two awaitables.

Fail-safe principle: every error path forwards the task to the user
queue. A misbehaving screener must never silently lose work.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Literal

from code_trip2.producers.claude_mcp import ClaudeMCPClient, ClaudeMCPError
from code_trip2.skills import SkillManifest
from code_trip2.tasks import Task

logger = logging.getLogger(__name__)


ScreeningAction = Literal["forward", "handled", "failed", "dismissed"]


@dataclass(frozen=True)
class ScreeningOutcome:
    """Result of running :func:`screen` on one task.

    - ``forward``: no skill matched (no candidates, classifier
      declined, or dry-run). Caller adds the task to the user queue.
    - ``handled``: an auto-handle skill matched and the executor
      reported success. Caller does NOT add to the user queue.
    - ``failed``: an auto-handle skill matched but the executor
      raised. Task is forwarded to the user queue with the error
      annotated in body, so nothing falls through the cracks.
    - ``dismissed``: a dismiss-style skill matched. The task is
      suppressed from the user queue without an executor call —
      classifier judgement alone says it isn't worth surfacing.

    ``dry_run_nominated`` carries the classifier's pick when dry-run
    mode prevented execution / dismissal — useful for comparing
    dry-run decisions to live behavior offline. The action is still
    ``forward`` in that case (the user sees the task).
    """

    action: ScreeningAction
    task: Task
    skill: str | None = None
    summary: str | None = None
    error: str | None = None
    dry_run_nominated: bool = False
    # New tasks the skill chose to spawn (e.g. the meeting-notes archiver
    # turning a "Henry: investigate X" action item into a meeting_followup
    # task). The screener routes these through ``add_to_queue`` regardless
    # of the action on the original task — handling the parent and
    # spawning children are independent decisions.
    follow_up_tasks: tuple[Task, ...] = ()


@dataclass(frozen=True)
class AutohandleLogEntry:
    """A time-stamped screening outcome retained for TUI display.

    The TUI keeps a bounded deque of these on :class:`Context` and
    renders one line per entry under the Queue panel — gives the user
    a peripheral-vision view of what the background screener has been
    up to without disturbing the queue itself.
    """

    ts: float
    outcome: ScreeningOutcome


# --- pure helpers --------------------------------------------------------


def candidates_for(
    task: Task, manifests: Iterable[SkillManifest]
) -> list[SkillManifest]:
    """Skills that apply to this task — either as auto-handlers or as
    dismissers. The classifier picks one (or none); the screener
    dispatches based on which flag the chosen skill carries.
    """
    return [
        m for m in manifests
        if (m.auto_handle and task.kind in m.auto_handle_kinds)
        or (m.dismiss and task.kind in m.dismiss_kinds)
    ]


# Permissive — the model often wraps the answer in prose. Conservative
# fallback (no match → None → forward) keeps the failure mode safe.
# Both HANDLE: and DISMISS: route to the same parser; the chosen
# manifest's own ``auto_handle`` / ``dismiss`` flags decide the action
# in :func:`screen`. Trusting flags over prefixes means a model that
# mis-tags an auto-handle skill as DISMISS still gets the right
# behavior.
_PICK_RE = re.compile(r"(?:HANDLE|DISMISS)\s*[:= ]\s*([A-Za-z0-9_\-]+)")
# Skill executors can spawn additional tasks by emitting one line per
# task: ``FOLLOWUP_TASK: {"headline": "...", "body": "...", "topic": "…"}``.
# The line must be on its own (no leading text other than whitespace /
# code-fence backticks) — the JSON payload itself can use any of the
# documented fields. The kind defaults to ``meeting_followup``, the
# topic defaults to ``inbox``. Designed for the Gemini meeting-notes
# archiver but available to any skill.
_FOLLOWUP_RE = re.compile(
    r"^\s*`*\s*FOLLOWUP_TASK\s*:\s*(\{.*\})\s*`*\s*$", re.MULTILINE,
)
_FOLLOWUP_DEFAULT_KIND = "meeting_followup"
_FOLLOWUP_DEFAULT_TOPIC = "inbox"


def parse_classifier_reply(
    text: str, candidates: list[SkillManifest]
) -> SkillManifest | None:
    """Pick a candidate from the classifier's reply, or ``None``."""
    if not text:
        return None
    name_to_manifest = {c.name: c for c in candidates}
    m = _PICK_RE.search(text)
    if not m:
        return None
    return name_to_manifest.get(m.group(1).strip())


def parse_follow_up_tasks(summary: str | None) -> tuple[Task, ...]:
    """Pull spawned tasks from an executor's summary.

    Recognises lines like ``FOLLOWUP_TASK: {"headline": "..."}``. Lines
    with malformed JSON or a missing headline are skipped — a buggy
    skill output shouldn't poison the queue. Returns tasks with
    ``kind`` defaulted to ``meeting_followup`` unless the JSON
    specifies one.
    """
    if not summary:
        return ()
    out: list[Task] = []
    for m in _FOLLOWUP_RE.finditer(summary):
        raw = m.group(1).strip()
        try:
            payload = json.loads(raw)
        except (ValueError, TypeError):
            logger.warning("Skipping malformed FOLLOWUP_TASK JSON: %s", raw)
            continue
        if not isinstance(payload, dict):
            continue
        headline = str(payload.get("headline") or "").strip()
        if not headline:
            continue
        body = payload.get("body")
        if body is not None:
            body = str(body)
        topic = str(payload.get("topic") or _FOLLOWUP_DEFAULT_TOPIC).strip() or _FOLLOWUP_DEFAULT_TOPIC
        kind = str(payload.get("kind") or _FOLLOWUP_DEFAULT_KIND).strip() or _FOLLOWUP_DEFAULT_KIND
        source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
        out.append(Task(
            kind=kind,
            topic=topic,
            headline=headline,
            body=body,
            source=source,
        ))
    return tuple(out)


def build_classifier_prompt(
    task: Task, candidates: list[SkillManifest]
) -> str:
    def _purpose(c: SkillManifest) -> str:
        if c.auto_handle and c.dismiss:
            return "[handle or dismiss]"
        if c.auto_handle:
            return "[handle]"
        if c.dismiss:
            return "[dismiss]"
        return "[?]"

    skills_block = "\n".join(
        f"- {c.name} {_purpose(c)}: {c.description}" for c in candidates
    )
    try:
        source_json = json.dumps(task.source, default=str)
    except (TypeError, ValueError):
        source_json = "{}"
    return (
        "You are a router for a voice-driven inbox. A task just "
        "arrived. Decide whether any skill below applies. Skills "
        "tagged [handle] DO something on the user's behalf (RSVP, "
        "draft a reply, archive, etc.). Skills tagged [dismiss] mark "
        "the task as not worth surfacing — the user doesn't need to "
        "see or respond to it.\n"
        "\n"
        "Skills (name [purpose]: description):\n"
        f"{skills_block}\n"
        "\n"
        "Task:\n"
        f"  kind: {task.kind}\n"
        f"  topic: {task.topic}\n"
        f"  headline: {task.headline}\n"
        f"  body: {task.body or '(empty)'}\n"
        f"  source: {source_json}\n"
        "\n"
        "Reply with EXACTLY ONE line, in one of these formats:\n"
        "  HANDLE: <skill-name>     (skill will act on the task)\n"
        "  DISMISS: <skill-name>    (skill says this isn't worth surfacing)\n"
        "  NONE                     (user should see this task)\n"
        "\n"
        "The reply prefix must match the skill's purpose tag. Only "
        "reply HANDLE or DISMISS if you are confident the named skill "
        "applies unambiguously. When unsure, reply NONE — the user "
        "can handle it."
    )


def build_executor_prompt(task: Task, skill: SkillManifest) -> str:
    """Hand the chosen skill the task context.

    The skill body (``.claude/skills/<name>/SKILL.md``) carries the
    actual instructions; this prompt just names the skill and supplies
    the task context that the skill's instructions assume is available.
    """
    try:
        source_json = json.dumps(task.source, default=str)
    except (TypeError, ValueError):
        source_json = "{}"
    return (
        "You are auto-handling a task from a voice-driven inbox. The "
        "user has not been shown this task — they are trusting you to "
        f"complete it silently. Use the `{skill.name}` skill from "
        "`.claude/skills/` and its tools.\n"
        "\n"
        f"Task kind: {task.kind}\n"
        f"Task topic: {task.topic}\n"
        f"Task source: {source_json}\n"
        f"Task headline: {task.headline}\n"
        f"Task body:\n{task.body or '(empty)'}\n"
        "\n"
        "Don't ask for confirmation. When done, return ONE sentence "
        "describing what you did."
    )


# --- async transforms ----------------------------------------------------


async def classify(
    task: Task,
    candidates: list[SkillManifest],
    mcp: ClaudeMCPClient,
) -> SkillManifest | None:
    """Ask Claude to pick a skill, or decline.

    Empty candidates → ``None`` without an MCP call. Subprocess /
    parse / budget failures → ``None``. Caller treats ``None`` as
    "forward to user queue."
    """
    if not candidates:
        return None
    prompt = build_classifier_prompt(task, candidates)
    try:
        # Budget is sized for the context-load cost, not the model's
        # output: ``claude --print`` loads the full MCP tool catalog
        # into context every invocation (~$0.02–0.04 of cache reads),
        # and the model itself emits maybe one short line. $0.10 gives
        # headroom without enabling runaway loops.
        reply = await mcp.run_agent(
            prompt=prompt,
            allowed_tools=(),     # classifier shouldn't call any tool
            max_budget_usd=0.10,
        )
    except ClaudeMCPError as exc:
        logger.warning("Screener classifier failed: %s", exc)
        return None
    return parse_classifier_reply(reply, candidates)


async def execute(
    task: Task, skill: SkillManifest, mcp: ClaudeMCPClient,
) -> str:
    """Run ``skill`` against ``task``. Returns the agent's summary.

    Raises whatever the MCP client raises; the caller turns that into
    a ``failed`` outcome.
    """
    prompt = build_executor_prompt(task, skill)
    return await mcp.run_agent(
        prompt=prompt,
        allowed_tools=skill.allowed_tools,
    )


# --- decision composite --------------------------------------------------


async def screen(
    task: Task,
    manifests: Iterable[SkillManifest],
    mcp: ClaudeMCPClient,
    *,
    dry_run: bool = False,
) -> ScreeningOutcome:
    """Full screening pipeline on one task.

    Returns a new :class:`ScreeningOutcome`; does not mutate ``task``
    (the ``failed`` branch uses :func:`dataclasses.replace` to annotate
    the body of a copy).
    """
    candidates = candidates_for(task, manifests)
    if not candidates:
        return ScreeningOutcome("forward", task)

    chosen = await classify(task, candidates, mcp)
    if chosen is None:
        return ScreeningOutcome("forward", task)

    if dry_run:
        return ScreeningOutcome(
            "forward", task, skill=chosen.name, dry_run_nominated=True,
        )

    # Dismiss skills are pure classifier judgements — no executor call.
    # Trust the skill's own ``dismiss`` flag over the classifier's
    # HANDLE/DISMISS prefix. When a skill carries both flags
    # (unusual), prefer auto-handle since it's the more interesting
    # action.
    if chosen.dismiss and not chosen.auto_handle:
        return ScreeningOutcome("dismissed", task, skill=chosen.name)

    try:
        summary = await execute(task, chosen, mcp)
    except Exception as exc:
        logger.exception("Screener executor failed for task %s", task.id)
        annotated = replace(
            task,
            body=(
                f"{task.body or ''}\n"
                f"[auto-handle attempted ({chosen.name}): {exc}]"
            ).strip(),
        )
        return ScreeningOutcome(
            "failed", annotated, skill=chosen.name, error=str(exc),
        )
    follow_ups = parse_follow_up_tasks(summary)
    for ft in follow_ups:
        ft.parent_id = task.id
    return ScreeningOutcome(
        "handled", task, skill=chosen.name, summary=summary,
        follow_up_tasks=follow_ups,
    )


# --- runtime loop --------------------------------------------------------


async def _next_or_stop(
    intake: "asyncio.Queue[Task]", stop: asyncio.Event,
) -> Task | None:
    """Block on either the next intake task or the stop event.

    Cancels whichever awaitable lost the race so neither leaks.
    """
    getter = asyncio.create_task(intake.get())
    stopper = asyncio.create_task(stop.wait())
    try:
        done, pending = await asyncio.wait(
            {getter, stopper}, return_when=asyncio.FIRST_COMPLETED,
        )
        for p in pending:
            p.cancel()
        if getter in done:
            return getter.result()
        return None
    finally:
        for t in (getter, stopper):
            if not t.done():
                t.cancel()


async def run_screener_loop(
    *,
    intake: "asyncio.Queue[Task]",
    manifests: tuple[SkillManifest, ...],
    mcp: ClaudeMCPClient,
    add_to_queue: Callable[[Task], None],
    on_outcome: Callable[[ScreeningOutcome], None],
    allowed_kinds: frozenset[str] | None,
    dry_run: bool,
    stop: asyncio.Event,
) -> None:
    """Drain the intake queue, screen each task, dispatch the outcome.

    Serial: one in-flight screen at a time. Producer poll intervals are
    much longer than a single classify+execute round, so this is
    fine. If a screen run blocks (slow MCP), tasks queue up; that's
    backpressure, not data loss.

    ``allowed_kinds`` is a config gate. ``None`` means "no extra
    restriction beyond what manifests opt into"; a frozenset further
    restricts. An empty frozenset effectively disables auto-handling
    without changing the call sites that feed the intake queue.
    """
    while not stop.is_set():
        task = await _next_or_stop(intake, stop)
        if task is None:
            return
        if allowed_kinds is not None and task.kind not in allowed_kinds:
            outcome = ScreeningOutcome("forward", task)
        else:
            try:
                outcome = await screen(task, manifests, mcp, dry_run=dry_run)
            except Exception:
                logger.exception(
                    "Screener crashed on task %s; forwarding", task.id,
                )
                outcome = ScreeningOutcome(
                    "forward", task, error="screener-crash",
                )
        try:
            on_outcome(outcome)
        except Exception:
            logger.exception("on_outcome callback raised; continuing")
        # Only ``forward`` and ``failed`` outcomes surface the original
        # task to the user. ``handled`` (executor ran) and ``dismissed``
        # (classifier said not worth surfacing) both suppress it.
        if outcome.action in ("forward", "failed"):
            try:
                add_to_queue(outcome.task)
            except Exception:
                logger.exception(
                    "add_to_queue failed for task %s", outcome.task.id,
                )
        # Follow-up tasks ride along independently — a handled
        # meeting-notes email can still spawn a meeting_followup the
        # user needs to see.
        for ft in outcome.follow_up_tasks:
            try:
                add_to_queue(ft)
            except Exception:
                logger.exception(
                    "add_to_queue failed for follow-up task %s", ft.id,
                )
