# Voice-Driven Claude Code: Design Document

## Problem Statement

Enable a developer workflow with Claude Code that works with **audio output only** (earbuds) and **voice + minimal physical buttons for input**, suitable for use while walking or otherwise away from a screen and full keyboard. The long-term goal is to support the full ticket-to-PR lifecycle: browsing tickets, starting work in a git worktree, running Claude in plan mode, reviewing results, iterating, and pushing a draft PR.

## Design Principles

1. **Audio-native, not audio-adapted.** Don't try to read a screen aloud. Design interactions that are *meant* to be heard.
2. **Modal interaction.** Inspired by vim: different modes with different key meanings. Reduces the number of physical buttons needed.
3. **Natural language over code.** Claude should summarize and explain, never read code aloud. Code lives in the worktree; the voice interface operates at the intent/status/decision level.
4. **Minimal hardware.** Earbuds you already own + a small custom macro pad (5 keys).
5. **Leverage existing infra.** tmux is the source of truth. The voice interface is a remote control for your existing tmux workflow — same sessions, same panes, same Claude instances. When you sit down at a screen, everything is right where you left it.
6. **Simple before fast.** Use request-response patterns everywhere. Optimize for latency only after the core loop works and you've identified actual bottlenecks.

---

## Current Status (2026-05)

The system is partially built and usable end-to-end for the core voice loop.

| Component | Status |
|---|---|
| 5-key BLE macro pad (ZMK on nice!nano, F13–F17) | **Built and in daily use** |
| PTT recording + OpenAI Whisper STT | Working |
| Local-STT mode (PTT forwards a hotkey to e.g. Superwhisper) | Working |
| OpenAI TTS (`gpt-4o-mini-tts`) | Working |
| Earcons (thinking pulse, completion, error, mode chime) | Working |
| SSH + tmux remote control (send-keys / capture-pane) | Working |
| Stop-hook completion detection (`/tmp/claude-done-<window>`) | Working (`scripts/setup-stop-hook.sh` installs it) |
| NAV-modifier chord system (next/prev/cycle/identify in frontmost app) | Working |
| ACT+NO chord (Ctrl+U clear-line) | Working |
| Solo taps (YES→Enter, NO→Esc, NAV→Down) | Working |
| Modes: IDLE, DICTATE, NAVIGATING, WORK, LINEAR | Working |
| Per-session JSONL event log (`~/.code-trip/sessions/`) | Working |
| Summarizer LLM (raw-pane → spoken English) | **Not yet implemented** — current pipeline only does ANSI/box-drawing strip + tail truncation |
| Local Whisper / Piper TTS | Not yet implemented (cloud APIs only) |
| REVIEW / SHIP modes (review tools, draft PR) | Not yet implemented |
| SLACK mode | Stubbed |
| Verbosity controls (brief / detailed / verbose) | Not yet implemented |

The biggest gap relative to the original design is the **summarizer LLM**. The orchestrator currently feeds Claude's raw pane text — minus ANSI/box-drawing characters — directly to TTS. This works for short responses but is the obvious next thing to add for longer outputs.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Local Machine                         │
│                                                         │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────┐  │
│  │ BLE Macro │   │  Orchestrator │   │  Audio Output  │  │
│  │   Pad     │──▶│  (Python)     │──▶│  (TTS Engine)  │  │
│  │ (5 keys,  │   │              │   │                │  │
│  │  F13–F17) │   │  - Mode FSM   │   │   OpenAI TTS   │  │
│  └──────────┘   │  - SSH + tmux │   │   (Piper TBD)  │  │
│                  │    commands   │   └────────────────┘  │
│  ┌──────────┐   │               │                       │
│  │ Earbuds  │   │  - STT batch  │                       │
│  │ (mic)    │──▶│  - TTS batch  │                       │
│  │          │◀──│  - State mgmt │                       │
│  └──────────┘   └──────┬───────┘                       │
│                         │ SSH                            │
└─────────────────────────┼───────────────────────────────┘
                          │
┌─────────────────────────┼───────────────────────────────┐
│                  Remote Workspace                        │
│                         │                                │
│  ┌──────────────────────▼──────────────────────────┐    │
│  │              tmux session (source of truth)      │    │
│  │                                                  │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐         │    │
│  │  │ window 1│  │ window 2│  │ window 3│  ...     │    │
│  │  │ ticket-1│  │ ticket-2│  │ general │         │    │
│  │  │ worktree│  │ worktree│  │ claude  │         │    │
│  │  │         │  │         │  │         │         │    │
│  │  │ claude  │  │ claude  │  │         │         │    │
│  │  │ (inter- │  │ (inter- │  │         │         │    │
│  │  │ active) │  │ active) │  │         │         │    │
│  │  └─────────┘  └─────────┘  └─────────┘         │    │
│  └──────────────────────────────────────────────────┘    │
│                                                          │
│  Claude Code runs interactively in tmux panes.           │
│  The orchestrator drives it via send-keys and reads      │
│  output via capture-pane. No headless mode needed.       │
└──────────────────────────────────────────────────────────┘
```

The system runs as a **local Python orchestrator** on your laptop/phone that:
- Captures voice via microphone, transcribes with STT
- Reads button presses from the BLE macro pad (shows up as a keyboard)
- Manages modal state (which mode you're in, which worktree is active)
- Communicates with the remote workspace over SSH
- **Drives Claude Code interactively via tmux** — `send-keys` to type commands, `capture-pane` to read output
- Cleans Claude's output and (eventually) passes it through a summarizer LLM before TTS playback
- **Layers onto your existing tmux workflow** — discovers existing sessions/windows, doesn't create a parallel world. When you sit down at a screen, the full conversation history is visible in the terminal.

---

## Input System

### Voice Input (Primary)

**Push-to-talk with batch transcription.** Hold PTT to record, release to transcribe and dispatch.

The orchestrator supports two STT backends, selected via `[stt] provider` in `config.toml`:

- **`provider = "openai"` (default):** PTT records a WAV via `sounddevice` and posts it to the **OpenAI Whisper API** (`/v1/audio/transcriptions`). The transcript is then dispatched through the mode router (`handle_voice`).
- **`provider = "local"`:** PTT does *not* record audio in-process. Instead, while PTT is held the orchestrator presses-and-holds a configurable forwarding key (default: `delete`, configured at `[stt.local] hotkey`). A local STT app like **Superwhisper** is bound to that key and inserts its transcript directly into the focused macOS app on release. This bypasses the orchestrator's mode router entirely and is useful for free-form dictation into any app.

A **local Whisper model via `faster-whisper`** is still on the roadmap as a third option (orchestrator-internal, no Superwhisper dependency), but is not yet implemented.

### Macro Pad (Secondary Input)

**Hardware: 5-key BLE macro pad — built and in daily use.**

Build:
- **Controller:** nice!nano (nRF52840-based, ~$25)
- **Switches:** 5x Kailh Choc low-profile switches
- **Battery:** small LiPo (weeks of life with ZMK power management)
- **Firmware:** ZMK (built via GitHub Actions, mapped to F13–F17)
- **Case:** 3D printed

The macro pad appears as a standard Bluetooth keyboard to the laptop. The orchestrator listens for the F-keys via `pynput` and — on macOS — installs a `Quartz` event-tap interceptor (`darwin_intercept`) that **drops the macropad keys at the OS level** so the focused app (e.g. kitty's CSI-u protocol) never sees them. Without this, holding NAV would spam escape sequences into the active terminal.

**Physical layout (left-to-right):**

```
┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
│ PTT │ │ YES │ │ NO  │ │ ACT │ │ NAV │
│ 🎤  │ │ ✓   │ │ ✗   │ │     │ │     │
└─────┘ └─────┘ └─────┘ └─────┘ └─────┘
  F13     F14     F15     F16     F17
```

The mapping from logical key (`ptt`/`yes`/`no`/`act`/`nav`) to physical F-key is configurable in `[macropad]`.

**Solo taps** — typed into whatever app currently has focus, via a synthesized keystroke:

| Key | Solo behavior |
|-----|---|
| **PTT** | Hold → record audio (or forward `[stt.local] hotkey` if `provider="local"`) |
| **YES** | Tap → `Enter` (default-accept in most prompts) |
| **NO** | Tap → `Esc` (cancel) |
| **NAV** | Tap → `↓` (next item in lists / completion menus). Fires only on release if NAV was *not* used as a chord modifier. |
| **ACT** | No solo behavior; ACT only acts as a chord modifier |

**Chord system.** Two modifier keys produce useful combos. NAV takes precedence over ACT when both are held.

| Chord | Action |
|-----|---|
| **NAV + PTT** | Speak the frontmost macOS app's name via TTS (debugging / situational awareness) |
| **NAV + YES** | App-specific "next" — e.g. kitty next pane (`Ctrl-b n`), Chrome next tab (`⌘⌥→`), Slack next unread (`⌥⇧↓`) |
| **NAV + NO** | App-specific "previous" — e.g. kitty prev pane, Chrome prev tab, Slack history-back (`⌘[`) |
| **NAV + ACT** | Rotate through `[macropad] app_cycle` (default: kitty, Chrome, Slack) by activating the next app in the list |
| **ACT + NO** | `Ctrl-U` — clear-line for shell / readline inputs |

The "next/prev" pair per app is defined in `chords.APP_NAV`; new apps are added by registering a `(yes_stroke, no_stroke)` pair. Pairs do not have to be symmetric — Slack deliberately combines "next unread" with "history back" because that's the useful walking workflow.

**Why a chord system instead of "modes change key meanings".** The original design pushed mode-dependent key meanings (NAV does X in BROWSE, Y in WORK, Z in REVIEW). In practice, the more useful split turned out to be: **what app has focus on the laptop**, not what mode the orchestrator is in. The orchestrator's modes still exist (see Modal System below), but the day-to-day buttons drive the focused app rather than overlaying their own meaning.

---

## Output System

### TTS Engine

**One engine, batch synthesis.** The system waits for Claude's full response before speaking, so there is no token-level streaming.

**Implemented:** **OpenAI TTS API** with model `gpt-4o-mini-tts`, voice `nova`, speed `1.15x`. Audio is requested as WAV, decoded with stdlib `wave`, and played through `sounddevice`. `TTSClient.stop()` interrupts playback mid-sentence (used by the NO-while-speaking interruption path).

**Roadmap:** **Piper TTS (local, free)** as a drop-in replacement to eliminate API cost and network dependency. Not yet implemented.

**Voice configuration:**
- Slightly faster than normal speaking rate (1.1–1.2x) — experienced screen reader users listen at 2–3x but start slower.
- Distinct earcons for mode changes, errors, completions (synthesized in code, not WAV files — see below).

### Content Rendering Strategy

**Original plan (still the right design):** Claude Code runs unconstrained, and a small fast LLM (Haiku or gpt-4o-mini) summarizes its raw pane output into spoken English before TTS:

```
Claude Code (full output) → Summarizer LLM → TTS → Audio
```

**Current implementation (placeholder):** The orchestrator's `clean_output()` performs only a mechanical strip:

1. Remove ANSI escape sequences and box-drawing characters.
2. Drop Claude's trailing `>` input prompt block.
3. Take the last ~60 non-empty lines, cap at 4000 characters.
4. Send the result directly to TTS.

This is fine for short responses ("Tests passed.") but produces unlistenable audio for anything substantive. **Adding the summarizer LLM is the highest-leverage next change** — the architecture already has a summarization seam (`clean_output` is a single function call inside `_work_voice`); the work is writing the prompt and wiring the API call.

**Planned summarizer prompt:**

```
You are converting developer tool output into spoken audio for a user wearing earbuds
who cannot see a screen. You will receive the raw output from a coding assistant.

Produce a concise spoken summary following these rules:
- NEVER include code, diffs, file contents, or raw terminal output
- NEVER use markdown formatting, bullet characters, or special symbols
- Summarize in natural spoken English, as if briefing a colleague
- For file paths, say just the filename unless the directory matters
- For errors, state the error type, the affected file, and the fix in plain English
- Keep it concise — aim for 15-30 seconds of speech
- Use ordinal markers for lists: "First... Second... Third..."
- Describe WHAT changed and WHY, not the literal code
- If the user needs to make a decision, state the options clearly
```

The orchestrator can prepend context (current mode, what was asked) to help the summarizer:

```
[Mode: WORK] [User asked: "run the tests"]
<raw claude output here>
```

**Content rendering guidelines** (for the summarizer, once built):

| Content Type | Summarizer Should Produce |
|---|---|
| **Ticket description** | Title, then 2-3 sentence summary |
| **Plan** | Numbered list of high-level steps, spoken with ordinals |
| **Status update** | Short sentence: "Step 3 of 7 complete. Now running tests." |
| **File changes** | "Modified 4 files: auth service, user model, two test files." Never read diffs. |
| **Errors** | Error type, affected file, and fix in plain English |
| **Review results** | "3 issues found. Issue 1: [category] in [file]. Issue 2: ..." |
| **Git operations** | "Branch created. 3 commits ahead of main. Ready to push." |
| **Code questions** | Natural language explanation, referencing functions by name but not their source |

### Earcons

Earcons are **synthesized in code** as short sine-wave tones via `numpy` + `sounddevice` (not WAV files). All live in `code_trip2/earcon.py`:

| Event | Sound | Where |
|---|---|---|
| Claude thinking | 660 Hz pulse, repeats every 2.5s on a background thread (`Thinking.start()` / `.stop()`) | While `_work_voice` waits on the Stop hook |
| Task complete | Rising two-tone (660 → 880 Hz) | After Claude's response is spoken |
| Error | Low 220 Hz buzz | Wrapper around `_speak_error` / `_report_error` |
| Mode change | Two-tone chime per mode (IDLE 440, NAVIGATING 523, WORK 587, LINEAR 659, SLACK 784 Hz, plus the perfect-fifth above) | `_enter_mode` |

---

## Modal System

Modes reflect **the surface being interacted with**, not workflow phase. The router (`modes.handle_voice`) checks the transcript for a mode-switch phrase first, then for a global command, then dispatches to the per-mode handler. Mode entry is always announced: a chime + spoken `"<mode> mode"`.

The current modes are: **IDLE**, **DICTATE**, **NAVIGATING**, **WORK**, **LINEAR**, **SLACK** (stub).

The original design's REVIEW and SHIP modes are not yet implemented — they require automated review tools and `gh pr create` integration that haven't been built. They remain the right shape for the long-term ticket-to-PR vision.

### Mode entry (voice phrases)

| Phrase fragment | Enters mode |
|---|---|
| "work mode", "switch to work", "start working" | WORK |
| "dictation mode", "dictate mode", "switch to dictate" | DICTATE |
| "navigation mode", "switch windows" | NAVIGATING |
| "linear mode", "list tickets", "show tickets", "my tickets" | LINEAR (also refreshes if "ticket" present) |
| "slack mode", "check slack", "read slack" | SLACK |
| "idle mode", "go idle", "exit mode" | IDLE |

### Global commands (any mode)

- **"status"** — speak `"<mode> mode. Active window: <window>."`
- **"what" / "repeat" / "say that again"** — replay last TTS (history buffer not yet implemented; currently says "Nothing to repeat.")

### Mode: IDLE

The default mode at startup if `[mode] default = "IDLE"`. PTT input that doesn't match a mode-switch phrase falls through to **WORK** for that turn (auto-promote on first utterance).

### Mode: DICTATE

Pure local dictation — pastes the STT transcript into the frontmost macOS app via `pbcopy` + `⌘V`. Requires no SSH/tmux config, so it works on a laptop without a remote workspace.

This is the mode in `config.example.toml` because it's the lowest-friction starting point.

### Mode: NAVIGATING

Voice control of the remote tmux window list.

- **"list windows" / "what windows"** — speak the count and comma-separated names
- **"switch to <name>" / "go to <name>" / "select <name>"** — `tmux select-window`, set `ctx.active_window`, auto-promote to WORK
- **"new window <name>"** — `tmux new-window`, auto-promote to WORK
- Anything else — fall back to listing windows

### Mode: WORK

Voice ↔ Claude in the active tmux pane on the remote.

For each PTT turn:

1. `tmux send-keys -t <session>:<active_window> "<transcript>" Enter`
2. Start the thinking-pulse earcon on a background thread.
3. Poll for the Stop-hook signal file `/tmp/claude-done-<window>` (timeout configurable via `[claude] wait_timeout`, default 300s).
4. `tmux capture-pane -p -S -200` to grab the response.
5. Run `clean_output()` (placeholder — to be replaced by summarizer LLM).
6. Speak the result via TTS, play the completion earcon.
7. Append a `turn` event to the JSONL session log with `{user, remote_output, spoken}`.

Errors at any step stop the thinking pulse, play the error earcon, and speak a brief explanation.

### Mode: LINEAR

Linear ticket list, fetched by running `claude -p "..."` in a dedicated tmux window (`[tmux] linear_window`, default `linear`). The prompt asks Claude to call the Linear MCP and return a JSON array of `{id, title, priority, assignee, branch}`. The orchestrator extracts the array with a regex, parses it, and caches it in `Context.tickets`.

Voice commands:

- **"refresh" / "reload" / "list"** — re-fetch tickets
- **"next" / "previous"** — step the cursor; speak `"<n> of <N>. <id>. <title>. Priority <p>. Assigned to <a>."`
- **"select <n>" / "select this" / "work on <n>"** — open a tmux window named after the ticket's `branch`, run `claude` in it, set it as `active_window`, auto-promote to WORK
- Free-text — filter cached tickets by case-insensitive substring match on title or id

### Mode: SLACK

Stub. Speaks `"Slack mode is not yet implemented."` Future: read unread channels / threads aloud, send replies via dictation. The macropad chord system already provides Slack navigation via `nav+yes` / `nav+no` when Slack is the frontmost app, which covers most of the walking-workflow need without a dedicated voice mode.

### Note: keys vs. modes

The macropad chords (NAV+YES, NAV+NO, NAV+ACT, ACT+NO) operate **independently of mode** — they always drive the focused macOS app, not the orchestrator's mode state. Mode transitions happen by **voice**, not by key.

---

## Orchestrator Design

The orchestrator is a single Python process running locally. It is the brain of the system. Source layout (`src/code_trip2/`):

```
main.py          Entry point: load config, wire components, run the listener loop
config.py        TOML loader + frozen Config dataclass (single source of defaults)
macropad.py      One pynput Listener for all 5 keys + chord/tap/PTT state machine
chords.py        Chord & tap handlers; per-app keystroke definitions (APP_NAV)
modes.py         Context dataclass + handle_voice mode router + per-mode handlers
remote.py        SSH + tmux helpers (send / capture / list/new/select-window / wait_done)
window.py        macOS-only: active_app(), activate_app(), send_keystroke(), paste_text()
stt_client.py    OpenAI Whisper wrapper
tts_client.py    OpenAI TTS wrapper (gpt-4o-mini-tts)
earcon.py        Synthesized tones + Thinking thread
session_log.py   Per-session JSONL event log → ~/.code-trip/sessions/
```

### Core state

```python
@dataclass
class Context:
    config: Config                # TOML-loaded config
    tts: TTSClient                # OpenAI TTS
    log: SessionLogger            # JSONL event sink
    thinking: earcon.Thinking     # Background thinking-pulse thread
    mode: str = ""                # "IDLE" | "DICTATE" | "NAVIGATING" | "WORK" | "LINEAR" | "SLACK"
    active_window: str = ""       # Currently selected tmux window for WORK mode
    tickets: list[dict] = []      # LINEAR mode cache
    ticket_index: int = 0         # LINEAR cursor
```

`main.run()` builds the `Context`, instantiates the `Macropad` listener with three callbacks (`on_audio`, `on_chord`, `on_tap`), and waits on a shutdown event. The macropad runs on its own pynput thread; callbacks dispatch into the modes/chords modules.

### Claude Code Integration via tmux

Claude Code runs as a **normal interactive session** inside tmux panes — not in headless mode. The orchestrator interacts with it the same way a human at a keyboard would: typing commands and reading the screen.

**Sending a command to Claude:**

```bash
# Send the transcribed voice input to the active tmux window's pane
ssh remote "tmux send-keys -t mysession:ticket-1 'implement the auth fix' Enter"
```

**Reading Claude's output:**

```bash
# Capture the current pane content (last N lines)
ssh remote "tmux capture-pane -t mysession:ticket-1 -p -S -50"
```

**The orchestrator loop:**

1. Plays a "thinking" earcon when the command is sent via `send-keys`
2. Waits for a **notification from Claude's `Stop` hook** indicating Claude has finished responding (see "Detecting Claude is done" below)
3. Runs `capture-pane` to grab Claude's output
4. Sends the captured output to the **summarizer LLM** (Haiku or gpt-4o-mini) with the audio summarization prompt and current mode context
5. Sends the summarizer's output to TTS for audio synthesis
6. Plays the resulting audio, then plays the completion earcon

**Why this approach:**

- **Conversation continuity is automatic.** Claude runs as an interactive session with full conversation history — no need to manage `session_id` or `--resume` flags.
- **Survives disconnects.** If SSH drops while walking, tmux keeps Claude running. The orchestrator reconnects and `capture-pane` picks up wherever Claude left off.
- **Screen/voice continuity.** You can start a task by voice while walking, then sit down and see the full conversation in the terminal. Or start something on screen and check on it by voice later. One workflow, two interfaces.
- **Parallel work.** Claude can be working in one tmux window while you're interacting with another via voice. `capture-pane` is non-blocking — it reads pane state without interrupting the process.
- **No headless mode needed.** The orchestrator doesn't need `--output-format json` or any special Claude flags. It reads raw terminal output and the summarizer LLM handles the translation.

**Detecting "Claude is done":**

Use **Claude Code's `Stop` hook** — a lifecycle hook that fires every time Claude finishes a response. The orchestrator polls for a per-window signal file at `/tmp/claude-done-<window_name>`; install the hook on the remote with:

```bash
bash scripts/setup-stop-hook.sh /path/to/project
```

This writes the following into `.claude/settings.json`:

```json
{
  "hooks": {
    "Stop": [
      {
        "type": "command",
        "command": "touch /tmp/claude-done-$(tmux display-message -p '#{window_name}' 2>/dev/null || echo unknown)"
      }
    ]
  }
}
```

`remote.wait_done()` clears the signal file before sending the prompt and polls every 1s (timeout 300s by default) for it to reappear. This is precise — no heuristics, no false positives from output that happens to look like a prompt.

**Worktree/window management:**

The orchestrator discovers existing tmux state rather than creating it from scratch:

```bash
# List all windows in the session
ssh remote "tmux list-windows -t mysession -F '#{window_index} #{window_name} #{pane_current_path}'"
```

This returns the window layout, names, and working directories — enough to map windows to worktrees. When starting work on a new ticket, the orchestrator creates a new tmux window in the existing session:

```bash
ssh remote "tmux new-window -t mysession -n ticket-42 -c /path/to/worktree/ticket-42"
ssh remote "tmux send-keys -t mysession:ticket-42 'claude' Enter"
```

### STT Pipeline

```
Mic (earbud) → PTT hold → record WAV → STT (API or local) → transcript → intent router
                                                                        │
                                        ┌───────────────────────────────┤
                                        ▼                               ▼
                                  Mode command                   Claude prompt
                              (handled locally)         (tmux send-keys to active pane)
```

**Intent routing:** Simple keyword-based routing for mode commands ("list tickets", "switch to", "status", "ship it"). Everything else is forwarded to Claude via `tmux send-keys`.

### SSH + tmux Management

All remote interaction goes through SSH + tmux commands. Configure SSH `ControlMaster` in `~/.ssh/config` for connection reuse:

```
Host remote
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 600
```

This gives persistent multiplexed connections with zero library dependencies. Every remote operation is an `ssh remote "tmux ..."` call:

| Operation | Command |
|---|---|
| Send input to Claude | `tmux send-keys -t session:window 'prompt text' Enter` |
| Read pane output | `tmux capture-pane -t session:window -p -S -50` |
| List windows/worktrees | `tmux list-windows -t session -F '#{window_index} #{window_name}'` |
| Create new worktree window | `tmux new-window -t session -n name -c /path/to/worktree` |
| Switch active window | `tmux select-window -t session:window` |
| Check if Claude is running | `tmux display-message -t session:window -p '#{pane_current_command}'` |

---

## Walking-Optimized UX Details

### Expected Latency

The system is request-response, not conversational. Expected round-trip:

| Phase | Estimate |
|---|---|
| PTT release → STT transcript (Whisper API) | ~1–2s |
| `tmux send-keys` → Stop hook fires | ~5–15s (depends on task complexity) |
| `capture-pane` + (planned) summarizer LLM | ~1–2s once added; ~0s today |
| Text → TTS audio ready | ~1–2s |
| **Total PTT-to-hearing-audio** | **~7–20s** |

Acceptable for a walking workflow where you're directing an agent, not having a real-time conversation. The thinking-pulse earcon plays continuously while waiting on the Stop hook so you know the system is alive.

### Interruption Handling — current state

- ✅ `TTSClient.stop()` exists; can be wired to a NO-tap during playback (not yet wired — NO currently always sends an Esc keystroke).
- ❌ PTT-while-TTS-playing interrupt: not yet wired.
- ❌ SSH auto-reconnect on drop: not yet implemented; failed SSH calls just speak an error and abort the turn. Claude itself keeps running in tmux, so no work is lost — the next PTT turn picks up wherever Claude is.

### Repetition and Navigation — planned

- ❌ "Read that again" / message-history buffer
- ❌ NAV-step through recent messages
- The voice command "repeat" / "what" is recognized but currently says "Nothing to repeat." until the buffer exists.

### Verbosity Control — planned

Once the summarizer LLM is in place, screen-reader-style verbosity levels apply by tweaking the summarizer prompt:

- **"brief"** — 1–2 sentences
- **"detailed"** — full summary (default)
- **"verbose"** — include filenames, line counts, test names

`config.example.toml` already reserves a `[openai] verbosity` field for this; the code does not yet read it.

---

## Implementation Plan

### Stage 1a: Proof of Concept — **Done**

- ✅ Listen for F13–F17 via `pynput`
- ✅ Record audio on PTT hold (or forward a hotkey for local STT)
- ✅ Send to OpenAI Whisper for transcription
- ✅ Send transcript to Claude via `ssh remote "tmux send-keys ..."`
- ✅ Detect completion via the Stop-hook signal file (no polling pane content)
- ✅ Capture pane output and speak via OpenAI TTS
- ❌ **Pending:** route capture through a summarizer LLM (currently only mechanical strip — see "Content Rendering Strategy")

### Stage 1b: Modal System — **Mostly done**

- ✅ Mode state machine: IDLE → DICTATE / NAVIGATING / WORK / LINEAR / SLACK
- ✅ Earcons for mode transitions, completion, errors, and "thinking"
- ✅ NAVIGATING mode (list / switch / new tmux window)
- ✅ LINEAR mode via `claude -p` + Linear MCP, cached ticket list with cursor
- ✅ DICTATE mode (paste STT into frontmost macOS app)
- ✅ Per-session JSONL event log
- ❌ REVIEW / SHIP modes (need automated review tools + `gh pr create` integration)
- ❌ Verbosity controls (brief / detailed / verbose)
- ❌ Message-history buffer for "repeat that"

### Stage 2: Local Models — **Not started**

- ❌ Replace OpenAI Whisper with `faster-whisper`. (Note: a local-STT *escape hatch* already works via `[stt] provider = "local"`, which forwards a hotkey to Superwhisper — but that's an external tool, not orchestrator-internal.)
- ❌ Replace OpenAI TTS with Piper.
- The summarizer (once added) stays as a cloud API call.

### Stage 3: Hardware — **Done**

- ✅ Five-key BLE macro pad on a **nice!nano** (nRF52840) controller, Kailh Choc switches, LiPo battery, ZMK firmware, F13–F17 keymap, 3D-printed case
- ✅ BLE pairing + key detection verified with the orchestrator
- ✅ macOS-side: `Quartz` event-tap interceptor drops macropad keys at the OS level so the focused app doesn't see them as text input

### Stage 4: Polish — **Partial**

- ✅ TTS voice/speed tuned (`nova` at 1.15x)
- ✅ Earcons tuned (synthesized in `earcon.py`, no WAV asset management)
- ✅ Error handling: every remote / TTS / earcon call path has a fallback to a spoken error message
- ❌ Verbosity controls
- ❌ Connection recovery / reconnection logic for SSH drops
- ❌ Summarizer prompt tuning (because the summarizer doesn't exist yet)
- ❌ launchd/systemd service for autostart

---

## Bill of Materials

### Software

| Component | Stage | Cost |
|---|---|---|
| Python orchestrator | 1 | Free |
| OpenAI Whisper API (STT) | 1 | $0.006/min |
| OpenAI TTS API | 1 | ~$15/1M chars |
| Summarizer LLM (Haiku or gpt-4o-mini) | 1 | ~$0.001-0.005 per summary |
| faster-whisper (local STT) | 2 | Free (open source) |
| Piper TTS (local) | 2 | Free (open source) |
| ZMK firmware | 3 | Free (open source) |

### Hardware (as built)

| Component | Approx. Cost |
|---|---|
| nice!nano (nRF52840) controller | $25 |
| 5x Kailh Choc switches | $5 |
| Small LiPo battery | $5 |
| Wire, diodes | $5 |
| 3D printed case | Free (own printer) |
| **Total macro pad** | **~$40** |
| Earbuds with mic (already owned) | $0 |

---

## Prior Art and Inspirations

### Accessibility Tools

- **Screen readers (NVDA, JAWS, VoiceOver, Orca):** Pioneered verbosity levels, mode-based navigation, earcons, and the principle that not everything needs to be read aloud. NVDA is open source — its navigation patterns (browse mode, focus mode, landmarks) directly inspire this design.
- **Microsoft CodeTalk:** VS Code extension for blind developers. Key insight: "Talk Points" that announce expression results during debugging, and code summaries that give high-level overviews without reading code line by line.
- **Blind developer workflows:** Blind programmers prefer CLI tools (git, terminal) because screen readers work well with pure text. They navigate by structure, not by scanning. This validates the modal/structural approach.

### Voice Coding

- **Talon Voice + Cursorless:** State-of-the-art voice coding. Talon uses a command grammar with a phonetic alphabet. Cursorless uses syntax-tree-aware navigation. Key insight: short, unambiguous commands work better than natural language for repetitive actions — but for our use case (directing an AI agent, not editing code directly), natural language is appropriate.
- **Talon's noise commands:** Talon maps mouth sounds (pop, hiss) to actions. Could be used for OK/NO if no macro pad is available.

### Voice-LLM Integration

- **Vocode:** Open-source library for building voice-based LLM apps. Good reference for the STT → LLM → TTS pipeline architecture, though it's optimized for conversational latency which is more than we need.
- **OpenAI Realtime API:** Native speech-to-speech with GPT-4o. Overkill for this use case — we're directing an agent, not having a conversation.

### References

- [Talon Voice](https://talonvoice.com/) — voice coding framework
- [Cursorless](https://www.cursorless.org/) — structural voice code editing
- [Piper TTS](https://github.com/rhasspy/piper) — fast local neural TTS
- [ZMK Firmware](https://zmk.dev/) — wireless keyboard firmware
- [Claude Code headless mode](https://code.claude.com/docs/en/headless) — programmatic CLI usage
- [Claude Code hooks](https://code.claude.com/docs/en/hooks) — lifecycle event hooks
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — fast local Whisper inference (CTranslate2)
- [OpenAI Whisper API](https://platform.openai.com/docs/guides/speech-to-text) — cloud speech-to-text fallback
- [OpenAI TTS](https://developers.openai.com/api/docs/guides/text-to-speech) — text-to-speech API
- [Microsoft CodeTalk](https://www.microsoft.com/en-us/research/blog/codetalk-rethinking-ide-accessibility/) — IDE accessibility for blind developers
- [NVDA screen reader](https://github.com/nvaccess/nvda) — open source screen reader
- [Blind Programmer Blueprint](https://blindpenguincoder.com/blind-programmer-blueprint-how-i-code-and-why-it-matters/) — how blind developers code
- [ACM: Code navigation challenges for blind developers](https://dl.acm.org/doi/10.1145/3132525.3132550) — research on accessibility barriers
- [nice!nano](https://nicekeyboards.com/nice-nano) — wireless keyboard controller
- [ESP32-BLE-Keyboard](https://github.com/T-vK/ESP32-BLE-Keyboard) — ESP32 as BLE keyboard

---

## Open Questions

1. **Phone as orchestrator?** Running the orchestrator on a phone (via Termux or a native app) would eliminate the need to carry a laptop. The phone has mic, speaker, and BLE. Trade-off: more complex to develop, but more portable.

2. **Earbud buttons as input?** Most earbuds support single/double/triple tap and long press. That's 4 gestures per ear = 8 total. Could replace the macro pad entirely, but gestures are less reliable and harder to discover. Could be a "no macro pad" fallback mode.

3. **Local vs. cloud STT?** OpenAI Whisper API is simpler but requires internet. Local `faster-whisper` avoids API costs but adds latency and uses CPU. For walking outdoors (phone tethering), the Whisper API's bandwidth is negligible.

4. **Multiple simultaneous worktrees?** Today, NAVIGATING mode handles the window list one switch at a time and WORK runs against `ctx.active_window`. Cycling through several active Claude sessions by audio alone (a la "NAV through worktrees") is plausible but cognitively heavy — start with single-active-window and add only if real usage demands it.

5. **Safety:** Claude runs interactively in tmux, so its default permission mode applies. If Claude prompts for confirmation (e.g., before running a command), the orchestrator currently has no way to detect the prompt and relay it as a voice question — it just times out on the Stop hook. Workarounds: configure Claude with `--permission-mode acceptEdits` in trusted worktrees, or add a fallback that scans `capture-pane` for the prompt pattern.

6. **Notification while idle?** Should the system proactively notify when a long-running Claude task completes (i.e. a Stop-hook fired in a window other than the active one)? Probably yes — a completion earcon + brief status. Not yet implemented.

7. **Summarizer model choice.** Haiku vs. gpt-4o-mini vs. local — once the summarizer is added, this matters less than getting the prompt right. Defer to "whatever's cheapest and fast enough" until the prompt is dialed in.
