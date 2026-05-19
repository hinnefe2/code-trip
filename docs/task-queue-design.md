# Task Queue Mode: Design Sketch

> **Status:** Brainstorming doc, written 2026-05-19 before implementation. Not committed code. Intended to be loaded into a fresh session on a new branch (`task-queue` or similar) as the starting context for the rewrite.

## Motivation

The current orchestrator routes voice input by **what app has focus** (`_dispatch_by_focus`): if the frontmost macOS app is in `config.terminal_apps`, the transcript goes to the active tmux pane; otherwise it pastes into the focused app. Mode-style state (active tmux window, ticket cursor) lives on `Context`.

This is fine for sit-down work where the laptop screen is the source of truth. It's clumsy for **away-from-screen** use, which is what this project actually exists for. While walking with earbuds, the user is doing one of these things:

- Reading what a Claude session has produced (paste from tmux capture)
- Replying to a Claude session that's awaiting input
- Reading a Slack thread / replying in Slack
- Reading a web article / Gmail message
- Asking a one-shot question to a fresh Claude session
- Doing free-form dictation into whatever app

To do any of these today, the user has to **manually cycle apps** with `nav+act` and **manually pick a tmux window** by voice. The cognitive cost of "what should I do next?" is borne every time something finishes.

The proposal: invert it. **The system tells the user what's next**, drawn from a passively-maintained pool of waiting work. The user's only job is to keep responding to whatever is presented. Mental model: an old-school paper inbox — take the top item, deal with it, take the next one.

## Tension: the scheduler must reduce context switching, not amplify it

The naive version of this idea — pure LIFO, newest task on top — would be **worse** than the current setup. It would yank the user between unrelated topics every time a new task lands. The whole reason to build this is to *reduce* context-switching cost.

Concretely:

- If the user just sent a prompt to Claude in `ticket-42` and is expected to read the reply in 30 seconds, **inserting** a Slack message above that pending reply is bad. The Slack message should land *below*, so the user finishes the Claude thread first.
- If two unrelated Claude sessions are both awaiting input, the scheduler should prefer the one matching the user's last interaction, not whichever happened to fire its Stop hook a millisecond earlier.
- Quick, related work batches well; slow context switches need to amortize across more than 15 seconds of action.

So the scheduler is the interesting design problem, not the queue type.

---

## Decision point: LIFO queue vs. tree

This is the central data-model question and the rest of the design depends on it. I think the right answer is **a flat list with topic-affinity scoring** (effectively a priority queue), not a strict LIFO and not a tree. But it's worth stating all three so the choice is deliberate.

### Option A: LIFO queue (a literal inbox stack)

Tasks are an ordered list. New tasks land on top. User always pulls from the top. Defer = move to bottom; dismiss = drop.

- **Pros:** Trivially understandable. Matches the paper-inbox metaphor exactly. Voice grammar is tiny: "next", "skip", "dismiss", "done". Easy to render: "5 in the inbox, top is …".
- **Cons:** No way to express coherence. A noisy Slack channel would constantly push real work down the stack. Insertion-order is exactly the wrong signal — the *newest* task is most likely to be the *least* related to what the user is currently thinking about.
- **Verdict:** Too simple. We'd be back to manual app-cycling within a week.

### Option B: Tree / hierarchy

Tasks belong to nodes; nodes can have children. Example tree:

```
ticket-42 (worktree, currently working)
├── claude awaiting input (the active Claude pane)
├── PR review thread from teammate
└── linked Slack thread

ticket-77 (background)
└── claude finished, summary pending review

inbox
├── slack DM from Alice (unrelated)
└── article saved earlier: "rust async patterns"
```

User navigates by "drill in", "back out", "next sibling". Finishing a subtree pops back to the parent.

- **Pros:** Expresses topic structure literally. Easy to say "park this whole ticket" or "dump everything for ticket-77". Natural fit for the way a developer actually thinks about work.
- **Cons:** Voice navigation of a tree is hard. Trees need a "you are here" pointer, and the user has to remember the structure to know what `back up` means. Producers need to know what node to attach to (Slack thread → which ticket?), which is itself an inference problem.
- **Verdict:** Powerful but premature. The structural inference (which Slack message belongs to which ticket?) requires either user effort to tag, or an LLM classifier — and we'd be paying that cost before we know whether the queue idea is even right.

### Option C (recommended): Flat priority queue with topic affinity

Tasks are a flat list. Each task carries a `topic` tag — a free-form string like `ticket-42`, `slack-general`, `linear-triage`, `web-read`. The scheduler maintains a small **topic stack** (the last 2–3 topics the user has been in). When the user asks for "next", the scheduler scores all pending tasks by:

1. **Recency-on-topic bonus**: tasks tagged with the topic the user *just* worked on get a large bonus that decays over ~60 seconds. This is what keeps the user in flow.
2. **Urgency**: tasks can mark themselves "interrupt" (e.g., production alert) for unconditional top priority.
3. **Ready time**: tasks can sit in a "not yet" state until a wall-clock time or a condition (e.g., a Claude session is expected to finish at ~T+15s; don't surface until then).
4. **Age**: oldest pending wins ties.

Topic is just metadata — there's no tree, no parent/child. But topic-affinity scoring approximates the "stay in this subtree" property of a tree without the navigation complexity.

- **Pros:** Voice grammar stays inbox-like ("next", "skip", "dismiss"). The hard problem (scheduling) is concentrated in one scoring function that's easy to tune. Tree can be retrofitted later if topic-affinity proves insufficient.
- **Cons:** Topic tagging has to be done somehow. For Claude tasks, the tmux window name is a fine topic. For Slack, channel-or-thread-id. For unrelated items, `inbox` is fine. Cross-cutting items (Slack about ticket-42) need an LLM call or a manual tag — but we can defer that and start with single-topic tagging.

**Recommendation:** Go with Option C. Design the `Task` dataclass so that adding `parent_id` later (= retrofitting a tree) is trivial. Don't build the tree until topic-affinity proves insufficient.

---

## Architecture

```
                   ┌────────────────────────────────────────┐
                   │  Producers (run in background threads) │
                   │                                        │
                   │  • Claude Stop-hook listener           │
                   │  • Slack poller / RTM                  │
                   │  • Gmail poller                        │
                   │  • Manual ("remind me to read this")   │
                   │  • Linear ticket watcher (optional)    │
                   └─────────────────┬──────────────────────┘
                                     │ push Task
                                     ▼
                   ┌────────────────────────────────────────┐
                   │              TaskQueue                 │
                   │  • thread-safe list of Task            │
                   │  • score(task, scheduler_state) → fl   │
                   │  • peek() / pull() / defer() / done()  │
                   └─────────────────┬──────────────────────┘
                                     │ pull()
                                     ▼
                   ┌────────────────────────────────────────┐
                   │           Voice loop (consumer)        │
                   │                                        │
                   │  announce(task) → TTS                  │
                   │  on PTT: dispatch(task, transcript)    │
                   │  on YES/NO/NAV taps: queue ops         │
                   │  on done: pull next, repeat            │
                   └────────────────────────────────────────┘
```

### `Task` dataclass

```python
@dataclass
class Task:
    id: str                    # uuid
    kind: str                  # "claude_reply" | "slack_msg" | "gmail" | "web" | "note"
    topic: str                 # e.g. "ticket-42", "slack-general", "inbox"
    headline: str              # one-liner for TTS announcement
    body: str | None = None    # optional larger payload (already-summarized)
    source: dict = field(default_factory=dict)
                               # producer-specific: tmux window, slack channel/ts, URL...
    created_at: float = field(default_factory=time.time)
    ready_at: float = 0.0      # don't surface before this wall-clock time
    urgency: str = "normal"    # "interrupt" | "normal" | "background"
    state: str = "pending"     # "pending" | "active" | "snoozed" | "done" | "dropped"
    parent_id: str | None = None    # reserved for future tree retrofit
```

### `TaskQueue`

Thread-safe wrapper around a list of `Task`. Public ops:

- `add(task)` — producer-side; wakes the consumer if idle
- `peek()` — return the highest-scoring pending task without consuming
- `pull()` — atomically mark the top task `active` and return it
- `defer(id, seconds)` — set `ready_at = now + seconds`, state back to `pending`
- `done(id)` / `drop(id)` — terminal states; falls off the list
- `score(task, sched_state) -> float` — pure function the consumer can call to rank

Scoring inputs (`sched_state`):

- `recent_topics`: list of (topic, last_touched_time) for the last few topics worked
- `now`: current time (vs. each task's `ready_at`)

Initial scoring formula (tunable):

```
score = base_age_score (older = higher)
      + topic_affinity_bonus (decays exponentially with time since topic last touched)
      + urgency_bonus (interrupt: +inf, background: -large)
      − ready_at_penalty (huge negative if now < ready_at)
```

### Voice grammar

Reduced from the current mode/router setup. Globals at any time:

| Phrase | Effect |
|---|---|
| "next" / "what's next" | announce top task |
| "skip" / "later" | defer top task ~5 minutes |
| "dismiss" / "drop it" | mark dropped |
| "snooze 10" / "snooze an hour" | defer for parsed duration |
| "what's in the queue" / "how many" | speak count + breakdown by kind |
| "stop" / "be quiet" | interrupt TTS playback (already exists) |
| "repeat" / "what" | replay last TTS (already exists) |

Anything that doesn't match a global is **dispatched against the active task** by kind:

- `claude_reply` → `tmux send-keys` to the source window (today's WORK behavior)
- `slack_msg` → reply in thread (new producer-side action)
- `gmail` → compose-and-send (later phase)
- `web` / `note` → speak-only; no input expected
- No active task → fall through to current focused-app dispatch (preserves today's free-form behavior)

### Two interaction modes, one binary toggle

Decided (Q5/Q8): the system has **two modes**, switched by a single chord. This is a deliberately small mode FSM — not a return to the original 5-mode design.

- **Queue mode** — idle YES/NO/ACT taps drive the queue. PTT input dispatches against the currently-active task. This is the away-from-screen default.
- **Focused-app mode** — today's behavior. YES=Enter, NO=Esc, NAV=down/per-app, PTT input goes through `_dispatch_by_focus`. The right tool when sitting at the screen doing focused work and the queue would just be noise.

Mode transition is announced with a distinct earcon (two clearly different tones for the two modes, like the existing mode-change chime) and a short spoken label ("queue mode" / "focused mode"). No risk of being unsure which mode you're in.

**Toggle binding: ACT solo tap.** ACT has no solo behavior today (it's chord-modifier-only in `chords.TAP_STROKES`), so this is a clean grab. Mode-independent — one tap toggles in either direction. Muscle memory transfers across modes.

### Macropad mapping

Today's chord layout stays. Only the **solo taps** of YES/NO/ACT change based on which mode is active:

| Key | Focused-app mode (today's behavior) | Queue mode (new) |
|---|---|---|
| PTT hold | record | record; transcript dispatched against active task |
| YES tap | Enter into focused app | "accept / engage with this task" — also pulls next if nothing active |
| NO tap | Esc into focused app | "skip / dismiss" current task |
| NAV tap | down / per-app | down / per-app (unchanged — still useful: open URL in pane, chrome new tab) |
| **ACT tap** | **mode flip** | **mode flip** |

NAV+chord combos (`nav+yes`, `nav+no`, `nav+act`, `nav+ptt`, `act+no`) are mode-independent and keep their current behavior. Playback-aware NAV/NO behavior (advance/stop chunked TTS) also stays mode-independent.

### Producers

Each producer is an independent module under `src/code_trip2/producers/`. Each runs on its own daemon thread, observes some external signal, and calls `queue.add(task)`.

**`producers/claude.py`** — extends the existing Stop-hook mechanism.

- Today's hook: `touch /tmp/claude-done-<window>` on every Claude stop.
- New hook: write a small JSON line to `/tmp/claude-events/<window>-<timestamp>.json` with `{window, finished_at, last_user_msg}`. Producer watches the directory (`watchdog` or just polling `os.listdir` every second), reads each new file, deletes it, and emits a `Task(kind="claude_reply", topic=window, ...)`. The headline can be a short capture-pane snippet.
- Migration: keep the touch-file behavior as a fallback for the active window's `wait_done` (today's blocking-wait flow), so a half-migrated state still works.

**MCP integration strategy.** Producers that need external services (Slack, Linear, later Gmail) talk to **locally-hosted MCP servers** via the official Python MCP SDK (`pip install mcp`). Each server runs as a child process of the orchestrator (stdio transport) and is given its own credentials via env vars (Slack bot token, Linear API key, etc.). No Claude agent in the loop — the orchestrator is a direct MCP client. This is more setup than going through `claude -p`, but the round-trip is fast, deterministic, and doesn't depend on parsing model output.

A small shared module (`producers/mcp_client.py`) wraps the boilerplate: start the server subprocess, hold the session, expose typed methods over `call_tool`. Each producer imports it and calls the tools it needs.

**`producers/slack.py`** — Slack via a local Slack MCP server (the project's own — e.g., the reference Slack MCP server pointed at a workspace-scoped bot token). Poll-based: `search_messages` / `conversations_history` for unreads / mentions across a configured channel list, every ~30s. New mention → `Task(kind="slack_msg", topic=f"slack-{channel}")`. Reply path is also via the same server.

**`producers/linear.py`** — Linear via a local Linear MCP server with the user's API key. Replaces today's `claude -p` + JSON-extraction hack in `_linear_refresh`. Producer polls `list_issues` for assigned + recently-updated tickets and surfaces high-priority changes as tasks. Direct MCP calls remove the need for the dedicated `linear` tmux window and the regex JSON-extraction path entirely.

**`producers/manual.py`** — voice phrases like "remind me to read this" capture the most recent URL from the active tmux pane (we already have this logic in `chords._open_last_pane_url`) and add a `web` task.

Out of scope for v1: Gmail, calendar. Same MCP pattern applies when they're added.

### What stays the same

- macOS event-tap interception of macropad keys (`window.darwin_intercept`)
- All of `chords.py` for NAV-modifier behavior
- `remote.py` send/capture/wait_done (used by Claude producer + dispatch)
- `stt_client.py`, `tts_client.py`, `earcon.py`, `session_log.py`, `macropad.py`
- Chunked playback (`_speak_chunked`) — task announcement uses the same path

### What gets replaced

- `modes.py` `_dispatch_by_focus` / `_work_voice` / `_dictate_voice` — collapsed into a single `dispatch(ctx, transcript)` that consults `ctx.current_task` first, falls back to focused-app dispatch when no task is active.
- `Context` gains: `queue: TaskQueue`, `current_task: Task | None`, `recent_topics: deque[(str, float)]`.
- LINEAR-mode voice phrases (`list tickets`, `select N`, etc.) become either (a) producers that surface tickets as tasks, or (b) a one-shot voice command that asks the LLM to populate the queue. Pick whichever is less work; either is fine for v1.

---

## Resolved decisions

These were the open questions in the draft; resolved by walking through them.

1. **Scoring weights — hand-picked constants + structured logs for offline tuning.** No live `--debug-queue` speech. Producers and the scheduler write structured events (task created, scored, pulled, deferred, dropped) to a separate JSONL log under `~/.code-trip/queue/scoring-<date>.jsonl`. A small offline script can replay logs and re-score with alternative weights to inform tuning. Keep the surface area minimal until there's actual signal to tune against.
2. **Topic inference — single topic per task, channel-only.** v1 producers tag with one string (tmux window for Claude, channel name for Slack, "inbox" for misc). No LLM at producer-time. Multi-topic / cross-cutting is a v2 problem; the `Task` dataclass keeps `topic: str` (not `list[str]`) but the field is treated as opaque by the scheduler so widening it later is mechanical.
3. **Persistence — JSONL append-only log.** Producers append every event (add / state-change / done / drop) to `~/.code-trip/queue/queue-<date>.jsonl`. On startup, replay the last 24 hours of events to rebuild in-memory state; anything older is discarded. Matches the existing `session_log.py` style; inspectable with `cat` / `jq`.
4. **Backpressure — auto-digest old per-topic items.** Per-topic soft cap (start ~5 pending). When exceeded, the older tasks for that topic are merged into a single digest task ("3 unread in #general") that lives at the position of the oldest item. Pulling the digest expands it. Preserves the signal that there's activity without forcing the user to grind through 47 individual Slack lines.
5. **Macropad — binary mode flip.** Two modes (queue / focused-app); single chord toggles between them with an earcon + spoken label. See "Two interaction modes" above.
6. **Announcement — headline only, body on demand.** Pull plays ~5–10 seconds: kind + topic + one-line headline. The body is held; user taps NAV (or says "go on") to play it via the existing chunked-playback path. Default behavior across all task kinds; specific producers can opt out if their body is the headline.
7. **Idle behavior — silence.** Empty queue = no audio. No transition chime, no heartbeat. The first producer event after empty plays as a normal announcement.
8. **Coexistence with app-focus dispatch — via the Q5 mode flip.** Focused-app mode preserves today's `_dispatch_by_focus` verbatim. Queue mode is the new behavior. No implicit fallback; mode is always explicit and announced.

## Still open, deferred to the implementation session

- **Default startup mode.** Probably queue mode (matches the "always away from screen" framing) but worth letting `config.toml` set it.
- **Scoring constants.** First-pass values will be guesses; tuning comes from the offline log replay tool once there's real usage.
- **Digest UX details.** Whether expanding a digest pulls all its items into the queue as separate tasks, or plays them in a single announcement sequence. Lean toward the latter — keeps the queue clean.
- **Which Slack/Linear MCP servers specifically.** There are multiple reference + community MCP servers per service. Pick when the producer is being built; the `mcp_client.py` wrapper isolates the choice.
- **MCP server lifecycle.** Spawn-per-producer vs. one persistent server per service for the orchestrator's life. Lean persistent — fewer subprocess starts, MCP sessions are cheap to hold open.

---

## Implementation phases (rough)

1. **Branch + skeleton.** New branch (`task-queue`). Add `tasks.py` with `Task` + `TaskQueue` + `score()`. Unit tests for scoring under various `sched_state`. No producers yet, no voice integration.
2. **Voice integration with a single producer (manual add).** A voice phrase `"add a task X"` creates a `note` task; `"next"` pulls and announces. No Claude/Slack yet — just prove the loop with manually-added tasks.
3. **Claude producer.** New hook that writes events to `/tmp/claude-events/`. Producer thread, claude task → dispatch reuses today's WORK flow. At this point the system is end-to-end useful for multi-Claude workflows.
4. **Macropad rebinding.** YES/NO/ACT idle taps gain queue semantics. Tested by walking around for a session.
5. **Slack producer.** Poll-based, configurable channel list. Reply path via MCP.
6. **Persistence + tuning.** JSONL queue log, `--debug-queue` scoring output, tuning notes captured in this doc.
7. **Stretch:** Gmail producer, Linear watcher, topic inference via LLM.

Each phase ends with a usable system; nothing is half-finished mid-phase.

---

## What this doc does *not* commit to

- Specific scoring constants (left for tuning via offline log replay)
- Specific MCP server implementations chosen for Slack and Linear
- Whether to keep the `modes.py` filename or rename to `dispatch.py`
- Tree structure (deferred unless flat-list-with-topics proves insufficient)
- Digest expansion UX (single announcement vs. exploded into N tasks)
