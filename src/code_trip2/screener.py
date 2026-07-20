"""Task screener: auto-handle producer output via skill agents.

Screenable tasks enter :class:`TaskQueue` immediately in the
``screening`` state (see ``main.py``'s ``submit`` gate) and their ids
land on the screener's work queue. Because the task is *resident* in
the queue the whole time, a producer re-sighting the same object during
a screening backlog collapses into it via ``upsert`` instead of minting
a duplicate. For each ``(task_id, mode)`` work item the loop:

1. Filters skill manifests to those declaring ``auto-handle: true`` and
   listing this task's ``kind`` under ``auto-handle-kinds``. No
   candidates → release to ``pending`` (no LLM cost).
2. Asks Claude (Haiku, no tools) which candidate, if any, can fully
   handle the task. Unsure → ``NONE`` → release to ``pending``.
3. If a skill was named, runs it via ``run_agent`` with that skill's
   tool list. The outcome becomes a state transition on the resident
   task: handled → done, dismissed → dropped, forward/failed →
   pending. ``mode="reconsider"`` items are already user-visible
   tasks being re-judged; only a dismiss verdict does anything (marks
   them done).

All decision logic lives in plain functions returning a
:class:`ScreeningOutcome`. The loop function is the only thing with
lifecycle, and it's just a coroutine waiting on two awaitables.

Fail-safe principle: every error path releases the task to ``pending``.
A misbehaving screener must never silently lose work.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, replace
from typing import Awaitable, Callable, Iterable, Literal

from code_trip2.producers.claude_mcp import ClaudeMCPClient, ClaudeMCPError
from code_trip2.skills import SkillManifest
from code_trip2.tasks import (
    STATE_DONE,
    STATE_DROPPED,
    STATE_PENDING,
    STATE_SCREENING,
    Task,
    TaskQueue,
)

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
    # task). The screener routes these through ``submit_follow_up``
    # regardless of the action on the original task — handling the parent
    # and spawning children are independent decisions.
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


# Structured exit signal emitted by skills on the last (or any) line of
# their reply: ``STATUS: handled`` if the side effect happened, or
# ``STATUS: declined: <reason>`` if it didn't. This is the contract path
# — when present, it overrides :func:`summary_indicates_failure`, which
# remains a safety-net heuristic for skills that haven't yet been
# updated to emit STATUS. Tolerant of code-fence wrapping for the same
# reason ``_FOLLOWUP_RE`` is.
_STATUS_RE = re.compile(
    r"^\s*`*\s*STATUS\s*:\s*(handled|declined|skipped)\b[^\n]*`*\s*$",
    re.MULTILINE | re.IGNORECASE,
)


# Self-reported failure detection on executor summaries.
#
# The agent can return normally (no exception) while its prose says it
# couldn't actually perform the action — e.g. "I cannot complete the
# acceptance without finding the event first, so I'm skipping the
# archive step" (accept-invite), or "Not archived — Gmail tools
# encountered errors" (archive-gemini-meeting-notes). Without this
# check the screener marks the task ``handled`` and the user never
# sees it again, even though the side effect never happened.
#
# Patterns are deliberately conservative — they fire only on phrases
# that strongly imply the agent itself is acknowledging non-completion.
# False positives just mean the user sees a task they didn't strictly
# need to (annoying), while false negatives mean silent data loss
# (much worse). The bias is intentional.
_FAILURE_INDICATORS: tuple[re.Pattern[str], ...] = (
    # "I cannot" / "I can't" / "I could not" / "I couldn't"
    re.compile(r"\bI\s+(?:cannot|can(?:['‘’])?t|could\s+not|couldn(?:['‘’])?t)\b", re.IGNORECASE),
    # "unable to" / "I'm unable to" / "I am unable to"
    re.compile(r"\bunable\s+to\b", re.IGNORECASE),
    # "skipping the archive/accept/action/step/email" — paired with the
    # action verb so we don't trip on benign uses of "skip".
    re.compile(r"\bskipp?(?:ing|ed)\s+(?:the\s+)?(?:archive|accept(?:ance)?|action|step|email)\b", re.IGNORECASE),
    # "I'll skip" / "I will skip"
    re.compile(r"\bI(?:['‘’])?ll\s+skip\b", re.IGNORECASE),
    # "not archived" / "not accepting" / "did not archive" — explicit
    # negation of the action verb the skill was supposed to perform.
    re.compile(r"\bnot\s+(?:archiv(?:ing|ed)|accept(?:ing|ed)|complet(?:ing|ed)|send(?:ing)|sent)\b", re.IGNORECASE),
    re.compile(r"\bdid\s+not\s+(?:archive|accept|complete|send)\b", re.IGNORECASE),
    # Tool-call failure phrasings
    re.compile(r"\bencountered\s+(?:an?\s+)?errors?\b", re.IGNORECASE),
    re.compile(r"\bauthentication\s+errors?\b", re.IGNORECASE),
    re.compile(r"\bfailed\s+to\s+\w+", re.IGNORECASE),
)


def summary_indicates_failure(summary: str | None) -> bool:
    """Heuristically decide whether the executor's summary says the
    skill *didn't* complete its intended action.

    See :data:`_FAILURE_INDICATORS` for the patterns and rationale.
    """
    if not summary:
        return False
    return any(p.search(summary) for p in _FAILURE_INDICATORS)


def _one_line(summary: str | None, limit: int = 600) -> str:
    """Flatten an executor summary to a single capped line.

    Executor summaries are multi-line: a preamble, some working notes,
    and a trailing ``STATUS: declined: <reason>`` that carries the
    actual reason. Logged or embedded raw, the embedded newlines split
    the record across physical lines, so line-based tools (``grep`` /
    ``tail``) capture only the preamble and drop the reason — which is
    exactly what happened when a Gmail rate limit went undiagnosed.
    Collapsing whitespace keeps the whole reason on one line; the cap is
    generous enough to retain the trailing STATUS for typical summaries.
    """
    flat = " ".join((summary or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


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


def parse_skill_status(
    summary: str | None,
) -> Literal["handled", "declined"] | None:
    """Pull the structured ``STATUS:`` exit signal from an executor summary.

    Returns ``"handled"`` or ``"declined"`` if the skill emitted one,
    else ``None``. ``STATUS: skipped`` folds into ``declined`` — the
    user-visible disposition is the same (task surfaces to the queue).
    When a summary contains multiple STATUS lines (model retrying,
    quoted earlier output, etc.), the last one wins — the skill's final
    word should override anything it said while deciding.
    """
    if not summary:
        return None
    matches = _STATUS_RE.findall(summary)
    if not matches:
        return None
    last = matches[-1].lower()
    if last == "handled":
        return "handled"
    return "declined"


def follow_up_origin_key(anchor: str, headline: str) -> str:
    """Stable identity for a spawned follow-up task.

    A follow-up's ``id`` is a fresh uuid on every emit, so re-screening
    the parent email (e.g. archiving failed and it resurfaced, or a wide
    re-poll refetched it) mints duplicates. Anchoring on the parent's
    Gmail thread id plus the normalized headline yields a key that
    survives re-screens and restarts, letting
    ``TaskQueue.upsert(if_terminal="skip")`` suppress the duplicate even
    after the original was filed or dismissed.
    """
    norm = " ".join((headline or "").lower().split())
    return f"followup:{anchor}:{norm}"


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
    """Run ``skill`` against ``task``. Returns the agent's full transcript.

    ``transcript=True``: we keep every assistant text block, not just
    the final summary, so structured emissions like ``FOLLOWUP_TASK:``
    lines that the model prints before its archive tool call survive
    into :func:`parse_follow_up_tasks`. The trade-off is that the
    session log entry for ``outcome.summary`` is larger, which is fine
    — it's also more useful when diagnosing skill behavior.

    Raises whatever the MCP client raises; the caller turns that into
    a ``failed`` outcome.
    """
    prompt = build_executor_prompt(task, skill)
    return await mcp.run_agent(
        prompt=prompt,
        allowed_tools=skill.allowed_tools,
        transcript=True,
    )


# --- decision composite --------------------------------------------------


async def screen(
    task: Task,
    manifests: Iterable[SkillManifest],
    mcp: ClaudeMCPClient,
    *,
    dry_run: bool = False,
    classifier_mcp: ClaudeMCPClient | None = None,
    verify_side_effect: "Callable[[Task, str], Awaitable[bool | None]] | None" = None,
) -> ScreeningOutcome:
    """Full screening pipeline on one task.

    Returns a new :class:`ScreeningOutcome`; does not mutate ``task``
    (the ``failed`` branch uses :func:`dataclasses.replace` to annotate
    the body of a copy).

    ``classifier_mcp`` runs the skill-nomination step; the executor uses
    ``mcp``. They're separate so classification can use a stronger model
    than execution. Defaults to ``mcp`` when unset (single-model setups
    and existing call sites are unaffected).

    ``verify_side_effect`` is an optional ground-truth check run before a
    ``handled`` outcome is trusted: given the task and the chosen skill's
    declared ``verify`` type, it returns True (confirmed), False (the side
    effect did NOT happen — surface the task instead of closing it), or
    None (inconclusive — fall back to trusting the skill). Skills that
    declare no ``verify`` are unaffected.
    """
    classifier_mcp = classifier_mcp or mcp
    candidates = candidates_for(task, manifests)
    if not candidates:
        return ScreeningOutcome("forward", task)

    chosen = await classify(task, candidates, classifier_mcp)
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
    # Anchor the origin key on the parent's Gmail thread id (stable across
    # re-screens) rather than its task id (a fresh uuid each poll), so a
    # follow-up the user already filed or dismissed isn't respawned when
    # the parent meeting-notes email gets re-screened. Falls back to the
    # parent task id for non-email sources.
    anchor = (task.source or {}).get("thread_id") or task.id
    for ft in follow_ups:
        ft.parent_id = task.id
        ft.origin_key = follow_up_origin_key(anchor, ft.headline)
    # Disposition precedence:
    #   1. Explicit ``STATUS:`` from the skill wins. Updated skills emit
    #      ``STATUS: handled`` or ``STATUS: declined: <reason>``, which
    #      is the contract. ``handled`` overrides the failure heuristic
    #      so a skill that knows its prose is benign can opt out;
    #      ``declined`` overrides a success-looking summary so a skill
    #      that knows it didn't act can opt in.
    #   2. Fall back to :func:`summary_indicates_failure` for skills
    #      that haven't been updated to emit STATUS yet. Heuristic
    #      stays as the safety net until all skills are converted.
    status = parse_skill_status(summary)
    declined = status == "declined" or (
        status is None and summary_indicates_failure(summary)
    )
    if declined:
        logger.info(
            "Screener: %s returned a self-reported failure for task %s; "
            "forwarding to user. Summary: %s",
            chosen.name, task.id, _one_line(summary),
        )
        annotated = replace(
            task,
            body=(
                f"{task.body or ''}\n"
                f"[auto-handle declined ({chosen.name}): {_one_line(summary, 300)}]"
            ).strip(),
        )
        return ScreeningOutcome(
            "failed", annotated, skill=chosen.name, summary=summary,
            error="self-reported failure", follow_up_tasks=follow_ups,
        )
    # Ground-truth gate: the skill said it succeeded, but self-reports are
    # not trustworthy (observed: emails marked handled that were never
    # archived). If the skill declares a check, confirm the side effect
    # actually happened before closing the task. False = definitively not
    # done → surface it; None = couldn't check → fall back to trusting the
    # skill rather than nagging the user with a false alarm.
    if chosen.verify and verify_side_effect is not None:
        try:
            verified = await verify_side_effect(task, chosen.verify)
        except Exception:
            logger.exception(
                "Verification %r raised for task %s; trusting the skill",
                chosen.verify, task.id,
            )
            verified = None
        if verified is False:
            logger.info(
                "Screener: %s reported success but the %r check did not "
                "confirm it for task %s; surfacing to user.",
                chosen.name, chosen.verify, task.id,
            )
            annotated = replace(
                task,
                body=(
                    f"{task.body or ''}\n"
                    f"[auto-handle unverified ({chosen.name}): '{chosen.verify}' "
                    f"check failed — skill reported success but the side effect "
                    f"was not observed]"
                ).strip(),
            )
            return ScreeningOutcome(
                "failed", annotated, skill=chosen.name, summary=summary,
                error="unverified side-effect", follow_up_tasks=follow_ups,
            )
    return ScreeningOutcome(
        "handled", task, skill=chosen.name, summary=summary,
        follow_up_tasks=follow_ups,
    )


# --- runtime loop --------------------------------------------------------


async def _next_or_stop(
    work: "asyncio.Queue[tuple[str, str]]",
    stop: asyncio.Event,
) -> tuple[str, str] | None:
    """Block on the next ``(task_id, mode)`` work item, or the stop event.

    Returns ``None`` when stop fires before any item arrives. Cancels
    whichever awaitable lost the race so none leaks.
    """
    getter = asyncio.create_task(work.get())
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
    work: "asyncio.Queue[tuple[str, str]]",
    queue: TaskQueue,
    manifests: tuple[SkillManifest, ...],
    mcp: ClaudeMCPClient,
    on_outcome: Callable[[ScreeningOutcome], None],
    allowed_kinds: frozenset[str] | None,
    dry_run: bool,
    stop: asyncio.Event,
    submit_follow_up: Callable[[Task], None] | None = None,
    classifier_mcp: ClaudeMCPClient | None = None,
    verify_side_effect: "Callable[[Task, str], Awaitable[bool | None]] | None" = None,
) -> None:
    """Drain the work queue, screen each task, apply the state transition.

    Work items are ``(task_id, mode)``, mode ∈ {"intake", "reconsider"}.
    The task is already resident in ``queue`` — intake items in the
    ``screening`` state (put there by the submit gate), reconsider items
    in whatever user-visible state they hold.

    Serial: one in-flight screen at a time. Producer poll intervals are
    much longer than a single classify+execute round, so this is
    fine. If a screen run blocks (slow MCP), work items queue up; that's
    backpressure, not data loss — and because the tasks are resident,
    re-sightings during the backlog collapse into them instead of
    duplicating.

    ``allowed_kinds`` is a config gate. ``None`` means "no extra
    restriction beyond what manifests opt into"; a frozenset further
    restricts. An empty frozenset effectively disables auto-handling
    without changing the call sites that feed the work queue.

    Intake transitions only apply while the task is still ``screening``
    — if some other path retired it mid-screen (user action, a resolve
    sweep), the verdict is stale and dropped.
    """
    while not stop.is_set():
        nxt = await _next_or_stop(work, stop)
        if nxt is None:
            return
        task_id, mode = nxt
        task = queue.get(task_id)
        if task is None:
            logger.warning("Screener: task %s vanished before screening", task_id)
            continue
        if allowed_kinds is not None and task.kind not in allowed_kinds:
            outcome = ScreeningOutcome("forward", task)
        else:
            try:
                outcome = await screen(
                    task, manifests, mcp, dry_run=dry_run,
                    classifier_mcp=classifier_mcp,
                    verify_side_effect=verify_side_effect,
                )
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
        if mode == "reconsider":
            # Reconsider mode: the task is already user-visible. Only
            # ``dismissed`` does anything — mark the existing task
            # done. Everything else (forward, handled, failed) leaves
            # the task where it is.
            if outcome.action == "dismissed":
                queue.mark_done(task_id)
        else:
            current = queue.get(task_id)
            if current is not None and current.state == STATE_SCREENING:
                if outcome.action == "failed":
                    # ``screen`` annotated a copy's body (it never
                    # mutates its input); copy the annotation onto the
                    # resident task before releasing it.
                    queue.update_task(task_id, body=outcome.task.body)
                    queue.set_state(task_id, STATE_PENDING)
                elif outcome.action == "forward":
                    queue.set_state(task_id, STATE_PENDING)
                elif outcome.action == "handled":
                    queue.set_state(task_id, STATE_DONE)
                elif outcome.action == "dismissed":
                    queue.set_state(task_id, STATE_DROPPED)
        # Follow-up tasks ride along independently — a handled
        # meeting-notes email can still spawn a meeting_followup the
        # user needs to see. Same applies in reconsider mode.
        for ft in outcome.follow_up_tasks:
            try:
                if submit_follow_up is not None:
                    submit_follow_up(ft)
                else:
                    queue.upsert(ft, if_terminal="skip")
            except Exception:
                logger.exception(
                    "follow-up submit failed for task %s", ft.id,
                )
