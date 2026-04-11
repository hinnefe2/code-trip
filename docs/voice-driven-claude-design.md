# Voice-Driven Claude Code: Design Document

## Problem Statement

Enable a developer workflow with Claude Code that works with **audio output only** (earbuds) and **voice + minimal physical buttons for input**, suitable for use while walking or otherwise away from a screen and full keyboard. The system should support the full ticket-to-PR lifecycle: browsing tickets, starting work in a git worktree, running Claude in plan mode, reviewing results, iterating, and pushing a draft PR.

## Design Principles

1. **Audio-native, not audio-adapted.** Don't try to read a screen aloud. Design interactions that are *meant* to be heard.
2. **Modal interaction.** Inspired by vim: different modes with different key meanings. Reduces the number of physical buttons needed.
3. **Natural language over code.** Claude should summarize and explain, never read code aloud. Code lives in the worktree; the voice interface operates at the intent/status/decision level.
4. **Minimal hardware.** Earbuds you already own + a small custom macro pad (3-5 keys).
5. **Leverage existing infra.** tmux is the source of truth. The voice interface is a remote control for your existing tmux workflow — same sessions, same panes, same Claude instances. When you sit down at a screen, everything is right where you left it.
6. **Simple before fast.** Use request-response patterns everywhere. Optimize for latency only after the core loop works and you've identified actual bottlenecks.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Local Machine                         │
│                                                         │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────┐  │
│  │ BLE Macro │   │  Orchestrator │   │  Audio Output  │  │
│  │   Pad     │──▶│  (Python)     │──▶│  (TTS Engine)  │  │
│  │ (3-5 keys)│   │              │   │                │  │
│  └──────────┘   │  - Mode FSM   │   │  Piper (local) │  │
│                  │  - SSH + tmux │   │  or OpenAI TTS │  │
│  ┌──────────┐   │    commands   │   └────────────────┘  │
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
- Sends Claude's output through a summarizer LLM, then to TTS for audio playback
- **Layers onto your existing tmux workflow** — discovers existing sessions/windows, doesn't create a parallel world. When you sit down at a screen, the full conversation history is visible in the terminal.

---

## Input System

### Voice Input (Primary)

**Push-to-talk with batch transcription:**
- A dedicated button on the macro pad activates the microphone (hold to record)
- On release, the recorded audio is transcribed and the text is sent to the orchestrator

**Stage 1 (API-based):** Send the recorded audio to the **OpenAI Whisper API** (`/v1/audio/transcriptions`) — one HTTP POST, one transcript back. Simple, accurate, no local model setup.

**Stage 2 (local models):** Replace with a **local Whisper model via `faster-whisper`** or similar runtime. Use the ~500MB English-only model (same class as what superwhisper uses on Mac) — good accuracy, no API costs, works offline. Transcription takes ~1-3s which is negligible since Claude's response takes longer anyway.

### Macro Pad (Secondary Input)

**Hardware: 5-key BLE macro pad built with ZMK firmware**

Recommended build:
- **Controller:** Seeed XIAO nRF52840 BLE (~$10)
- **Switches:** 5x Cherry MX or Kailh Choc low-profile switches
- **Battery:** 100-300mAh LiPo (weeks of battery life with ZMK's power management)
- **Firmware:** ZMK (built via GitHub Actions, no local toolchain needed)
- **Case:** 3D printed or a simple PCB sandwich

The macro pad appears as a standard Bluetooth keyboard to the local machine. ZMK handles debouncing, power management, and BLE HID.

**Key layout and function (mode-dependent):**

```
┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
│ PTT │ │ NAV │ │ ACT │ │ OK  │ │ NO  │
│     │ │ ↑↓  │ │     │ │ ✓   │ │ ✗   │
└─────┘ └─────┘ └─────┘ └─────┘ └─────┘
  F13     F14     F15     F16     F17
```

| Key | Short press | Long press | Hold |
|-----|------------|------------|------|
| **PTT** (Push to Talk) | — | — | Mic active while held |
| **NAV** | Next item | Previous item | Cycle modes |
| **ACT** | Context-dependent action | Secondary action | — |
| **OK** | Confirm / approve / continue | "Read that again" | — |
| **NO** | Reject / skip / cancel | "Go back" | Abort current operation |

ZMK maps these to F13-F17 (unused function keys), which the orchestrator listens for via `evdev` or `pynput`.

**Alternative to building:** An off-the-shelf **Elgato Stream Deck Mini** (6 keys, USB) or even repurposing earbud button gestures (single tap, double tap, long press) for the 3 most critical actions (PTT, OK, NO).

---

## Output System

### TTS Engine

**Start simple: one engine, batch synthesis.**

Since the system waits for Claude's complete response before speaking, there is no need for token-level streaming TTS. Send the full response text to one TTS engine, get audio back, play it. No multi-engine fallback, no streaming library, no sentence boundary detection needed.

**Stage 1 (API-based):** Use the **OpenAI TTS API** — one API call per response, ~$15/1M characters. Use `gpt-4o-mini-tts` model. Simple to integrate, good voice quality, no local setup.

**Stage 2 (local models):** Replace with **Piper TTS (local, free)** — pipe text in, get WAV out, play with `mpv` or `aplay`. Fast enough on CPU (~1-2s for a paragraph). Zero API costs. Good quality with the right voice model.

**Voice configuration:**
- Slightly faster than normal speaking rate (1.1-1.2x) — experienced screen reader users listen at 2-3x but start slower
- Distinct audio tones/earcons for mode changes, errors, completions (short wav files played via `mpv`)

### Content Rendering Strategy

This is the core design challenge. The system must transform Claude's output into something meaningful in audio.

**Key principle: Claude Code runs normally — a separate summarizer model makes its output listenable.**

Claude Code should not be constrained to produce audio-friendly output. Its work quality depends on being able to reason with code blocks, diffs, file paths, and structured formatting. Instead, the orchestrator passes Claude's raw output through a small, fast LLM that produces a spoken-English summary.

**Two-stage pipeline:**

```
Claude Code (full output) → Summarizer LLM → TTS → Audio
```

**Summarizer model:** Use a cheap, fast model — Haiku or gpt-4o-mini. The task is easy (restate technical output as speech), so a small model handles it well. Cost is negligible (fractions of a cent per summary).

**Summarizer prompt:**

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

The orchestrator can prepend context to help the summarizer (current mode, what was asked):

```
[Mode: WORK] [User asked: "run the tests"]
<raw claude output here>
```

**Why this is better than constraining Claude's system prompt:**
- Claude Code works at full quality — no fighting its natural output format
- The summarizer prompt can be tuned independently without risking work quality
- Different modes can use different summarizer prompts (BROWSE wants ticket summaries, WORK wants action summaries)
- Claude's raw output is preserved for later screen review

**Content rendering guidelines** (for the summarizer, by content type):

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

**Earcons (audio cues):**

Short, distinct sounds for common events, avoiding the need for speech:

| Event | Sound |
|---|---|
| Mode change | Distinctive chime per mode (different pitch) |
| Task complete | Rising two-tone |
| Error occurred | Low buzz |
| Waiting for input | Soft pulse |
| Claude thinking | Subtle ticking (or silence — TBD based on preference) |
| New item (in browse mode) | Soft click |

These are simple WAV files played via `mpv --no-terminal` or `paplay`.

---

## Modal System

Inspired by vim's modal editing and screen reader navigation patterns.

### Mode: IDLE

The home state. Earbuds are silent.

- **PTT + voice:** Natural language command, routed by intent:
  - "List my tickets" → enters BROWSE mode
  - "What's the status?" → reads status of active worktree
  - "Switch to ticket 3" → changes active worktree
- **NAV:** Cycle through active worktrees, hear ticket title for each
- **ACT:** Enter BROWSE mode

### Mode: BROWSE

Navigate through open Linear tickets.

- **NAV short:** Read next ticket (title + priority + assignee)
- **NAV long:** Read previous ticket
- **ACT:** Select current ticket → start work (create worktree, tmux window, enter WORK mode)
- **OK:** Hear full ticket description
- **NO:** Exit to IDLE
- **PTT + voice:** Filter/search ("show me only bugs", "sort by priority")

Behind the scenes: The orchestrator uses Claude Code's Linear MCP tools to fetch tickets, caches the list locally, and navigates through it.

### Mode: WORK

Active development on a ticket. Claude is running in the worktree.

- **PTT + voice:** Talk to Claude ("implement the auth fix", "run the tests", "what's your plan?")
- **NAV:** Cycle through Claude's recent messages (re-read them)
- **ACT:** Trigger automated review tools
- **OK:** Approve Claude's current action / continue
- **NO:** Reject / tell Claude to try differently
- **NAV long hold:** Switch to different worktree (cycles through active ones)

**Sub-states within WORK mode:**

- **WORK:PLAN** — Claude is presenting its plan. NAV steps through plan items. OK approves. NO rejects with voice feedback.
- **WORK:EXECUTING** — Claude is making changes. Periodic status updates via TTS. OK is "keep going." NO is "stop and explain."
- **WORK:REVIEWING** — Review results are being read. NAV cycles through issues. PTT + voice gives feedback on each.

### Mode: REVIEW

Post-implementation review.

- **NAV:** Step through review findings one by one
- **PTT + voice:** Give feedback on current finding ("fix that", "ignore it", "explain more")
- **ACT:** Run review tools again
- **OK:** Approve all remaining → enter SHIP mode
- **NO:** Go back to WORK mode with review feedback

### Mode: SHIP

Final steps before PR.

- **OK:** Push draft PR (Claude creates it with `gh pr create`)
- **PTT + voice:** Adjust PR title/description
- **NO:** Go back to REVIEW
- After PR is pushed: reads back PR URL and returns to IDLE

### Mode Transitions

```
IDLE ──▶ BROWSE ──▶ WORK ──▶ REVIEW ──▶ SHIP ──▶ IDLE
  ▲         │         ▲  ◀──────┘         │
  └─────────┘         └──────────────────┘
```

NAV long-hold from any mode → mode selection (hear mode names, NAV to select, OK to enter).

---

## Orchestrator Design

The orchestrator is a single Python process running locally. It is the brain of the system.

### Core Components

```python
# Conceptual structure (not literal implementation)

class VoiceClaudeOrchestrator:
    mode: Mode                    # Current mode (IDLE, BROWSE, WORK, REVIEW, SHIP)
    active_window: str | None     # Currently selected tmux window
    ticket_cache: list[Ticket]    # Cached Linear tickets
    message_history: list[str]    # Recent TTS outputs for replay
    
    # Input handlers
    stt: STTClient                # Stage 1: OpenAI Whisper API; Stage 2: local faster-whisper
    keypad: KeypadListener        # Listens for F13-F17 from BLE macro pad
    
    # Output handlers  
    tts: TTSClient                # Stage 1: OpenAI TTS API; Stage 2: Piper local
    summarizer: SummarizerLLM     # Haiku/gpt-4o-mini for audio rendering
    earcons: EarconPlayer         # Short audio cues
    
    # Remote — tmux is the interface to everything
    tmux_session: str             # Name of the tmux session on remote
    last_capture: dict            # Map of window → last captured pane content
```

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

Use **Claude Code's `Stop` hook** — a lifecycle hook that fires every time Claude finishes a response. Configure it in the remote workspace's `.claude/settings.json`:

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

The hook writes a signal file when Claude finishes. The orchestrator watches for this file (via SSH) instead of polling pane content. This is precise — no heuristics, no false positives from output that happens to look like a prompt.

**Fallback for edge cases** (e.g., hook not configured, or checking on a session started before the voice interface): inspect `capture-pane` output for Claude's input prompt (the `>` character at the bottom of the pane). This is a simple string check, not polling — use it as a one-shot "is Claude waiting right now?" test when you switch to a window or reconnect.

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
| PTT release → STT transcript | ~1-2s |
| tmux send-keys → Claude complete (Stop hook fires) | ~5-15s (depends on task complexity) |
| capture-pane + summarizer LLM | ~1-2s (Haiku/gpt-4o-mini) |
| Summary → TTS audio ready | ~1-2s |
| **Total PTT-to-hearing-audio** | **~9-22s** |

This is acceptable for a walking workflow where you're directing an AI agent, not having a real-time conversation. If specific phases become bottlenecks in practice, they can be optimized individually (e.g., switch to streaming TTS, use a faster STT provider).

**While waiting for Claude:** Play a subtle "thinking" earcon so you know the system received your input.

### Interruption Handling

- **NO button while TTS is playing:** Immediately stop playback, discard remaining buffered text
- **PTT while TTS is playing:** Stop playback, start recording (interrupt to speak)
- **Connection drop:** Earcon alert. Orchestrator auto-reconnects SSH. Claude keeps running in tmux — no work lost. On reconnect, `capture-pane` picks up whatever happened while disconnected.

### Repetition and Navigation

- **OK long press:** "Read that again" — replays the last TTS output from the message history buffer
- **NAV in any mode:** Step through recent messages (last 10 kept in memory)
- **"What?" or "repeat"** as voice command: replays last message

### Verbosity Control

Inspired by screen reader verbosity levels:

- **PTT + "brief":** Responses capped at 1-2 sentences
- **PTT + "detailed":** Full responses (default)
- **PTT + "verbose":** Include file names, line counts, test names

This adjusts the summarizer prompt dynamically (not Claude's — Claude runs unconstrained).

---

## Implementation Plan

### Stage 1: Core Loop with APIs

Get the end-to-end loop working using cloud APIs for STT and TTS. No local models, no custom hardware. Use laptop mic + keyboard shortcuts as stand-ins.

**Phase 1a: Proof of Concept**

1. **Local Python script** that:
   - Listens for a keyboard shortcut (e.g., F13 via `pynput`, or a regular hotkey)
   - Records audio on key-hold, sends to **OpenAI Whisper API** for transcription
   - Sends transcribed text to Claude via `ssh remote "tmux send-keys ..."`
   - Polls `ssh remote "tmux capture-pane ..."` until Claude finishes responding
   - Passes the captured output through a **summarizer LLM** (Haiku or gpt-4o-mini) to get audio-friendly text
   - Sends the summary to **OpenAI TTS API**, plays the resulting audio
2. **Summarizer prompt** — write and tune the prompt that converts Claude's raw terminal output to spoken English
3. **Prerequisite:** An existing tmux session on the remote with Claude running interactively in at least one pane
4. Test the core loop: hold key → speak → hear Claude's summarized response

**Minimal dependencies:** `pynput`, `openai` (for Whisper + TTS), `anthropic` or `openai` (for summarizer). Everything is API calls + SSH/tmux commands — no local model setup, no special Claude flags.

**Phase 1b: Modal System**

1. Implement the mode state machine (IDLE → BROWSE → WORK → REVIEW → SHIP)
2. Add Linear ticket browsing via Claude Code's Linear MCP (run `claude -p "list my tickets"`)
3. Add worktree management (create/switch/list)
4. Add earcons for mode transitions and events
5. Test the full ticket-to-PR workflow with keyboard shortcuts

### Stage 2: Local Models

Replace cloud API calls with local models to eliminate API costs and network dependency for STT/TTS.

1. Replace OpenAI Whisper API with **`faster-whisper`** using the ~500MB English-only model (same class as superwhisper). No API costs, works offline.
2. Replace OpenAI TTS API with **Piper TTS** (local, free). Pipe text in, get WAV out.
3. Tune local model quality — voice selection for Piper, Whisper model size tradeoffs
4. The summarizer LLM stays as an API call (it's cheap and there's no good local equivalent at the quality needed)

### Stage 3: Hardware

1. Order parts: XIAO nRF52840, 5x Kailh Choc switches, LiPo battery, diodes
2. Wire switches directly to GPIO pins (5 keys = no matrix needed, direct wiring)
3. Write ZMK shield config (devicetree + keymap) mapping to F13-F17
4. Build with GitHub Actions, flash firmware
5. 3D print or hand-wire a minimal case
6. Test BLE pairing and key detection with the orchestrator

### Stage 4: Polish

1. Tune TTS voice, speed, and earcon sounds
2. Add robust error handling and connection recovery
3. Add verbosity controls
4. Fine-tune the summarizer prompt based on real usage
5. Optional: package as a systemd service that starts on login

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

### Hardware

| Component | Approx. Cost |
|---|---|
| Seeed XIAO nRF52840 BLE | $10 |
| 5x Kailh Choc switches | $5 |
| Small LiPo battery (150mAh) | $5 |
| Diodes, wire, PCB or perfboard | $5 |
| 3D printed case | $5 (or free if you have a printer) |
| **Total macro pad** | **~$30** |
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
- [XIAO nRF52840 macro pad guide](https://www.ubiqueiot.com/posts/xiao-nrf52840-zmk) — hardware build reference
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

4. **Multiple simultaneous worktrees?** The design supports it (NAV cycles through them), but cognitive load of tracking multiple tasks by audio alone may be high. Start with one active worktree and add multi-worktree support based on experience.

5. **Safety:** Claude runs interactively in tmux, so its default permission mode applies. If Claude prompts for confirmation (e.g., before running a command), the orchestrator would need to detect the prompt via `capture-pane` and relay it as a voice question. Alternatively, configure Claude with `--permission-mode acceptEdits` in worktrees where you trust it. Mitigate risk with: review mode before shipping, automated review tools (linting, type checking, tests), and the ability to reject/undo.

6. **Notification while idle?** Should the system proactively notify when a long-running Claude task completes? Probably yes — a completion earcon + brief status when Claude finishes, even if you haven't asked.
