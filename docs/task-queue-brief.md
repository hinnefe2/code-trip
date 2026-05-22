# Task-queue branch: orientation for future Claude sessions

You're picking up work on the **`task-queue`** branch of [code-trip](../README.md), Henry Hinnefeld's STT → LLM → TTS orchestrator for "coding while walking around with earbuds." This doc is the fast-path orientation; the source-of-truth design lives in [`task-queue-design.md`](task-queue-design.md), and the broader system design (pre-branch) is in [`voice-driven-claude-design.md`](voice-driven-claude-design.md).

## What this branch is doing

Replacing the original five-mode FSM (`IDLE` / `DICTATE` / `WORK` / `NAVIGATING` / `LINEAR` / `SLACK`) with a **task-queue** interaction model. The user no longer cycles between modes to "go check Slack" or "talk to Claude." Instead:

- **Producers** (background threads) push tasks into a queue: Slack @-mentions, Gmail inbox, Claude Stop-hook events from a remote tmux session, manual voice notes.
- A **consumer thread** auto-announces the highest-scoring pending task whenever the user is idle in queue mode.
- The user **engages** via voice (PTT) or macropad chord on the announced task.
- The mental model: an old paper inbox. Take the top item, deal with it, take the next.

The whole point is **reducing cognitive context-switching** while away from a screen. The scheduler uses topic affinity so related work clusters together (don't pull the user out of a Claude session to read random Slack unless something is urgent).

## Two interaction modes (just two!)

The original FSM is gone. There are now exactly two **app-modes**, toggled by a single key:

- **Queue mode**: away from the screen, audio-driven. Macropad solo taps drive the queue.
- **Focused mode**: at the laptop. Macropad solo taps are keyboard-style (Enter/Esc/etc) into whatever app has focus.

Mode flip is **NAV solo tap**. Two distinct mode chimes — the chime alone tells you which mode you flipped into.

## Macropad reference

Physical layout (left to right): **PTT · YES · NO · ACT · NAV**

| Key | Queue mode (solo tap) | Focused mode (solo tap) |
|---|---|---|
| PTT (hold) | Record audio for STT — transcript routes to source reply (slack reply, email draft, claude reply) | Same |
| YES | Accept / expand current task (or pull next if idle) | `Enter` into focused app |
| NO | Skip current task (5-min defer) | `Esc` into focused app |
| NAV | → flip to focused mode | → flip to queue mode |
| ACT | Stop audio (if playing); else no-op | Stop audio; else per-app handler (open URL in tmux, ⌘T in Chrome) |
| ACT+PTT (hold both) | Skill mode — transcript + active task context ships to `claude --print`; Claude Code picks the matching skill from `.claude/skills/` | Same |

Chords (mostly mode-independent):

- **NAV + PTT**: speak the active macOS app's name
- **NAV + YES/NO**: per-app "next / prev" (kitty pane, Chrome tab, Slack thread)
- **NAV + ACT**: cycle through `app_cycle` apps
- **ACT + NO** (queue mode): **dismiss** current task permanently (mark done)
- **ACT + NO** (focused mode): `Ctrl+U` (clear shell line)

## Architecture sketch

```
producers (threads)         queue              consumer / dispatch
  • claude (tmux stop hook)   │
  • slack (claude.ai MCP)     │── push ──▶  TaskQueue ──▶ auto-announce
  • manual (voice add)        │              + scoring     + voice routing
  • linear (stub)             │
                                │
state on disk: ~/.code-trip/queue/queue-YYYY-MM-DD.jsonl
                              /scoring-YYYY-MM-DD.jsonl
                              /slack-state.json
                              /email-state.json
                              /logs/orchestrator.log
```

Topic-affinity scoring lives in `tasks.score()`. The consumer wakes on queue mutations + a short timer.

## Key modules to know about

`src/code_trip2/`

- **`tasks.py`** — `Task`, `TaskQueue`, `RecentTopics`, `score()`. Pure data + ranking; no I/O.
- **`queue_log.py`** — append-only JSONL persistence + 24h startup replay.
- **`dispatch.py`** — top-level `handle_voice`, `flip_mode`, `QueueConsumer`, per-kind task response. The router that replaced the old mode FSM.
- **`modes.py`** — the `Context` dataclass, focused-mode voice handlers, chunked TTS playback worker. (Name is legacy; this module no longer contains a "modes" concept.)
- **`chords.py`** — macropad solo-tap and chord dispatch.
- **`macropad.py`** — pynput listener; tracks held keys, emits taps on release if not chorded.
- **`tts_client.py`** — OpenAI TTS with a **persistent OutputStream** (avoids per-chunk audio device reinit + crackle). Fade-in/out applied to samples.
- **`summarizer.py`** — LLM that turns raw Claude pane output into audio-friendly summaries. Used by both the focused-mode WORK flow and the Claude producer.
- **`producers/`**
  - `__init__.py` — `Producer` protocol + `ProducerSupervisor`.
  - `claude.py` — watches `/tmp/claude-events/*.json` over SSH for stop-hook events from the remote tmux.
  - `slack.py` — via `claude.ai Slack` MCP (see below). Mention search + watched-channel polling. Topic = raw channel name (no `slack-` prefix). New messages in an existing thread *update* the live task rather than queue a duplicate.
  - `email.py` — via `claude.ai Gmail` MCP. Polls `search_threads` with the configured query + `after:<unix_ts>` cursor. One pending task per thread (same dedup model as Slack). Reply path creates a *draft* via `create_draft` — the claude.ai Gmail MCP has no send tool, and draft-only is safer for voice anyway.
  - `claude_mcp.py` — wrapper around `claude --print` for invoking MCP tools. Parses the `tool_result` block out of stream-json output; ignores LLM prose. The whole "MCP proxy via Claude CLI" mechanism lives here.
  - `manual.py` — voice-driven manual task add (a no-op start/stop; logic is in dispatch).
  - `linear.py` — stub.
- **`slack_state.py`** / **`email_state.py`** — per-producer cursors persisted as JSON.
- **`tui.py`** — Rich live dashboard (`--tui` flag). Also: TUI-host detection + keystroke suppression so synthesized Enter/Esc/Down doesn't scroll the alternate-screen buffer.
- **`remote.py`** — SSH + tmux helpers (`send-keys`, `capture-pane`, `wait_done`).
- **`window.py`** — macOS-only: active-app, app activation, keystroke synthesis.

`docs/task-queue-design.md` is the source-of-truth design doc. Most non-obvious architectural calls are explained there with rationale.

## Skill-based task completion

ACT+PTT routes a transcript through Claude instead of through the per-kind reply path. Used for actions like "accept and archive" on a calendar-invite email — Claude reads the task context, picks a matching skill from `.claude/skills/<name>/SKILL.md` (project-scoped, **not** `~/.claude/skills`), and executes via the MCP tools the skill references. On success the task is marked done and the orchestrator speaks Claude's one-sentence summary.

The orchestrator doesn't have a skill registry — skill discovery and matching happen entirely on the Claude side via its own `.claude/skills/` mechanism. To add a new completion action, just drop another `SKILL.md` in `.claude/skills/<name>/` and write its `description` so Claude can route to it. See `.claude/skills/accept-invite/SKILL.md` for the template.

## Important design decisions (and the *why* behind them)

1. **Slack auth goes through `claude --print`**, not a Slack app. Rationale: installing a workspace Slack app is a setup-friction wall. claude.ai already holds the user's Slack OAuth via its hosted Slack MCP. The orchestrator shells out to `claude --print --allowedTools=mcp__claude_ai_Slack__<tool>` for each call. Per-call cost ~$0.02-0.03 (the full MCP catalog gets loaded into context); the user is on a Max Pro subscription so this is moot, but worth knowing.

2. **Auto-advance through TTS chunks**, no manual "next chunk" key. Long responses are split into ~3-sentence chunks by `modes.chunk_text`; the playback worker iterates through them. ACT tap stops playback entirely. NAV used to advance chunks; that's gone.

3. **Mode flip = NAV solo tap, not ACT.** NAV is the most-accessible key on the right side of the pad. ACT is now reserved for "interrupt audio" because that needs to be reachable mid-announcement.

4. **TTS uses a persistent `sd.OutputStream`.** Repeated `sd.play()` reopens (which the original code did) cause audio-device reinit on every chunk → buffer underruns → crackle throughout playback. The persistent stream + `latency='high'` fixed it. See the docstring in `tts_client.py`.

5. **`--tui` redirects Python logging to a file.** Rich's live display can't share stdout with logging. Logs go to `~/.code-trip/logs/orchestrator.log`. `tail -f` that file when debugging.

6. **Synthesized keystrokes are suppressed when frontmost app == TUI host.** Otherwise tapping YES (Enter) into the kitty hosting the dashboard would scroll the alternate-screen buffer up by a line. Detection via `TERM_PROGRAM` → mapped to the macOS app name `osascript` reports.

## Current state at a glance

- **Working**: Task queue + scoring + persistence; mode flip; TUI dashboard with mode-aware keymap; Slack producer (@-mentions across all channels + per-channel polling for "watched" channels); Gmail producer (configurable Gmail-syntax search, draft-only reply); Claude Stop-hook producer over SSH to a remote tmux; manual voice-add; summarizer LLM; persistent-stream TTS playback. Slack + Gmail both collapse a stream of in-thread messages into one live task.
- **Stubbed**: Linear producer (no MCP wiring).
- **Known gaps**: DMs that don't @-mention the user aren't surfaced (claude.ai Slack MCP doesn't expose a clean "all DMs since X"). Slack/email reply via voice works but without LLM body-shaping (you reply with literal STT text). Gmail send goes through `create_draft` — the user has to open Gmail to actually send.

## Run / test / debug

**Run:**
```bash
python -m code_trip2.main --config config.toml --tui
```

`config.toml` is git-ignored (contains the user's OpenAI key); `config.example.toml` is the template.

**Test the live suite (the only one that passes):**
```bash
source venv-codetrip/bin/activate
python -m pytest \
  tests/test_tasks.py tests/test_queue_log.py tests/test_dispatch.py \
  tests/test_summarizer.py tests/test_modes.py tests/test_macropad.py \
  tests/test_tui.py tests/test_tts_client.py tests/test_slack.py \
  tests/test_email.py
```

**Do NOT run `pytest tests/` whole-directory** — there are several stale test files (`test_audio_recorder.py`, `test_browse.py`, `test_intent_classifier.py`, `test_keypad.py`, `test_mode_fsm.py`, `test_orchestrator.py`, `test_push_to_talk.py`, `test_remote_tmux.py`, `test_review.py`, `test_ship.py`, `test_stt_client.py`, `test_summarizer.py`'s old version, `test_work.py`) that import from the defunct `code_trip` package (note no `2`) and error on collection.

**Debug**: `~/.code-trip/logs/orchestrator.log` (TUI mode), or just run without `--tui` to get logs on stdout.

## State files the user may reset

When the user wants a "clean slate" they may delete one or more of:

- `~/.code-trip/queue/queue-*.jsonl` — clears the persisted task queue
- `~/.code-trip/slack-state.json` — clears the Slack mention/channel cursors (next poll re-pulls from 5pm of the most recent weekday)
- `~/.code-trip/email-state.json` — clears the Gmail cursor (next poll re-pulls from 5pm of the most recent weekday)
- `~/.code-trip/logs/orchestrator.log` — clears the log

These are independent; a fresh queue without resetting the Slack cursor means "I cleared my queue but new Slack messages won't backfill."

## Conventions

- Single developer; we're working on the `task-queue` branch off `main`. (The root `CLAUDE.md` says "all work on main" — that convention is from before this branch existed.)
- Python package: `src/code_trip2/` layout. (The `code_trip` package without `2` is defunct; tests against it are dead.)
- macOS-only currently (Quartz event taps, `osascript` for active app).
- No backward-compat shims — single developer, move fast.
- Don't add features the task doesn't require; don't add comments that just describe what well-named code already says.
