# Voice-Driven Claude Code: Design Document

## Problem Statement

Enable a developer workflow with Claude Code that works with **audio output only** (earbuds) and **voice + minimal physical buttons for input**, suitable for use while walking or otherwise away from a screen and full keyboard. The system should support the full ticket-to-PR lifecycle: browsing tickets, starting work in a git worktree, running Claude in plan mode, reviewing results, iterating, and pushing a draft PR.

## Design Principles

1. **Audio-native, not audio-adapted.** Don't try to read a screen aloud. Design interactions that are *meant* to be heard.
2. **Modal interaction.** Inspired by vim: different modes with different key meanings. Reduces the number of physical buttons needed.
3. **Natural language over code.** Claude should summarize and explain, never read code aloud. Code lives in the worktree; the voice interface operates at the intent/status/decision level.
4. **Minimal hardware.** Earbuds you already own + a small custom macro pad (3-5 keys).
5. **Leverage existing infra.** tmux + SSH to remote workspace. Claude Code in headless/print mode with `stream-json` output. No new server infrastructure required beyond what's already running.

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
│                  │  - SSH to     │   │  or OpenAI TTS │  │
│  ┌──────────┐   │    remote     │   │  via RealtimeTTS│  │
│  │ Earbuds  │   │  - STT ingest │   └────────────────┘  │
│  │ (mic)    │──▶│  - TTS output │                       │
│  │          │◀──│  - State mgmt │                       │
│  └──────────┘   └──────┬───────┘                       │
│                         │ SSH                            │
└─────────────────────────┼───────────────────────────────┘
                          │
┌─────────────────────────┼───────────────────────────────┐
│                  Remote Workspace                        │
│                         │                                │
│  ┌──────────────────────▼──────────────────────────┐    │
│  │              tmux session                        │    │
│  │                                                  │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐         │    │
│  │  │ worktree│  │ worktree│  │ worktree│  ...     │    │
│  │  │ ticket-1│  │ ticket-2│  │ ticket-3│         │    │
│  │  │         │  │         │  │         │         │    │
│  │  │ claude  │  │ claude  │  │ claude  │         │    │
│  │  │ -p ...  │  │ -p ...  │  │ -p ...  │         │    │
│  │  └─────────┘  └─────────┘  └─────────┘         │    │
│  └──────────────────────────────────────────────────┘    │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │  Claude Code instances (headless, stream-json)   │    │
│  │  Linear MCP · Git · Review tools                  │    │
│  └──────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

The system runs as a **local Python orchestrator** on your laptop/phone that:
- Captures voice via microphone, transcribes with STT
- Reads button presses from the BLE macro pad (shows up as a keyboard)
- Manages modal state (which mode you're in, which worktree is active)
- Communicates with the remote workspace over SSH
- Runs Claude Code in headless mode (`claude -p --output-format stream-json`)
- Processes Claude's output and feeds it to a TTS engine for audio playback

---

## Input System

### Voice Input (Primary)

**Recommended: Deepgram streaming API** for real-time STT.
- Sub-300ms latency with streaming WebSocket
- Keyword boosting for technical vocabulary (project names, CLI commands, etc.)
- Nova-3 model supports custom vocabulary without retraining
- Alternative: OpenAI Realtime Transcription API (gpt-4o-transcribe) or local Whisper via WhisperLive

**Push-to-talk model** (not always-listening):
- A dedicated button on the macro pad activates the microphone
- Voice Activity Detection (Silero VAD) handles end-of-utterance detection
- This avoids ambient noise issues while walking and gives a clear "I'm talking to the system" signal

**Technical vocabulary handling:**
- Maintain a keyword boost list: project names, common commands, teammate names
- Configure via a simple text file that gets loaded at startup
- Example: `["smarthoa", "fastapi", "supabase", "RLS", "worktree", "tmux"]`

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

**Recommended: Layered approach using RealtimeTTS library**

[RealtimeTTS](https://github.com/KoljaB/RealtimeTTS) is a Python library that wraps multiple TTS backends with streaming support and automatic fallback. It accepts text incrementally (as Claude streams tokens) and begins audio playback immediately.

**Engine priority:**
1. **Piper TTS (local, free)** — 20-30ms generation on CPU, good quality, zero API costs. Best for status messages, confirmations, and navigation feedback. Install voices for your preferred accent.
2. **OpenAI TTS API (cloud, paid)** — Higher quality for longer narration. Use `gpt-4o-mini-tts` model with streaming. ~$15/1M characters. Good for reading ticket descriptions, plan summaries, review results.
3. **ElevenLabs (cloud, paid)** — Highest quality, 75ms latency with Flash v2.5. Use if you want the most natural-sounding voice for extended listening.

RealtimeTTS handles engine switching and fallback automatically. Use Piper for short/frequent outputs, cloud TTS for longer content.

**Voice configuration:**
- Slightly faster than normal speaking rate (1.1-1.2x) — experienced screen reader users listen at 2-3x but start slower
- Distinct audio tones/earcons for mode changes, errors, completions (short wav files played via `mpv`)
- SSML support (available in OpenAI and ElevenLabs) for controlling pronunciation of technical terms

### Content Rendering Strategy

This is the core design challenge. The system must transform Claude's output into something meaningful in audio.

**Key principle: Claude should never read code aloud.** Instead, the system prompt instructs Claude to communicate at the natural language level.

**Content categories and rendering approach:**

| Content Type | Rendering Strategy |
|---|---|
| **Ticket description** | Read title, then summarize description in 2-3 sentences |
| **Plan** | Read as numbered list of high-level steps |
| **Status update** | Short sentence: "Step 3 of 7 complete. Now running tests." |
| **File changes** | "Modified 4 files: auth service, user model, two test files." Never read diffs. |
| **Errors** | Read error type and message. Summarize stack trace as "Error in [file] at [function]: [message]" |
| **Review results** | "3 issues found. Issue 1: [category] in [file]. Issue 2: ..." |
| **Git operations** | "Branch created. 3 commits ahead of main. Ready to push." |
| **Code questions** | Claude explains in natural language, referencing functions by name but not reading their source |

**Implementation: System prompt augmentation**

When running in voice mode, the orchestrator appends to Claude's system prompt:

```
You are being used through a voice-only interface. The user cannot see a screen.
All of your output will be converted to speech and played through earbuds.

Rules for voice output:
- NEVER output code blocks, diffs, file contents, or raw terminal output
- NEVER use markdown formatting, bullet characters, or special symbols
- Summarize technical details in natural spoken English
- For file paths, say just the filename unless the directory matters
- For errors, state the error type, the affected file, and the fix in plain English
- Keep responses concise — aim for 15-30 seconds of speech per response
- Use ordinal markers for lists: "First... Second... Third..."
- When you make file changes, summarize WHAT you changed and WHY, not the literal code
- If the user needs to make a decision, state the options clearly and wait
- Prefix urgent information with "Note:" and Claude will add emphasis
```

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
    active_worktree: str | None   # Currently selected worktree
    worktrees: list[Worktree]     # All active worktrees
    ticket_cache: list[Ticket]    # Cached Linear tickets
    message_history: list[str]    # Recent TTS outputs for replay
    
    # Input handlers
    stt: DeepgramStreamingClient  # or WhisperLive
    keypad: KeypadListener        # Listens for F13-F17 from BLE macro pad
    
    # Output handlers  
    tts: RealtimeTTSEngine        # Piper local + OpenAI cloud fallback
    earcons: EarconPlayer         # Short audio cues
    
    # Remote
    ssh: SSHConnection            # Persistent SSH connection to workspace
    claude_sessions: dict         # Map of worktree → claude session_id
```

### Claude Code Integration

The orchestrator runs Claude Code on the remote workspace via SSH in headless mode:

```bash
ssh remote "cd /path/to/worktree && claude -p \
  --output-format stream-json \
  --verbose \
  --include-partial-messages \
  --append-system-prompt-file /path/to/voice-mode-prompt.txt \
  --allowedTools 'Bash,Read,Edit,Write,Glob,Grep' \
  --permission-mode acceptEdits \
  'implement the authentication fix described in the ticket'"
```

The orchestrator:
1. Parses the stream-json NDJSON output in real-time
2. Extracts text content from `stream_event` messages (filtering for `text_delta`)
3. Buffers text until sentence boundaries (period, question mark, exclamation)
4. Feeds complete sentences to RealtimeTTS for streaming audio
5. Tracks tool use events for status updates ("Claude is editing auth.py...")
6. Detects completion (`result` message type) and plays the completion earcon

**Conversation continuity:** The orchestrator captures `session_id` from the JSON output and uses `--resume {session_id}` for follow-up prompts in the same worktree.

### STT Pipeline

```
Mic (earbud) → PTT gate → Deepgram WebSocket → transcript → intent router
                                                                    │
                                    ┌───────────────────────────────┤
                                    ▼                               ▼
                              Mode command                   Claude prompt
                          (handled locally)              (sent to remote)
```

**Intent routing:** Simple keyword-based routing for mode commands ("list tickets", "switch to", "status", "ship it"). Everything else is forwarded to Claude as a prompt.

### SSH Management

- Use `paramiko` or `asyncssh` for a persistent SSH connection
- Multiplex commands over the connection (no reconnect overhead)
- Run Claude Code processes as background jobs; stream their output back
- Use `tmux send-keys` and `tmux capture-pane` as fallback for non-headless interactions

---

## Walking-Optimized UX Details

### Latency Budget

For a responsive feel while walking:

| Phase | Target | Approach |
|---|---|---|
| PTT → STT start | < 100ms | Pre-opened WebSocket to Deepgram |
| STT → transcript | < 500ms | Streaming interim results |
| Transcript → Claude first token | 2-5s | Unavoidable LLM latency |
| Claude token → audio start | < 200ms | RealtimeTTS streaming with Piper |
| **Total PTT-to-first-audio** | **~3-6s** | Acceptable for walking pace |

**While waiting for Claude:** Play a subtle "thinking" earcon so you know the system received your input.

### Interruption Handling

- **NO button while TTS is playing:** Immediately stop playback, discard remaining buffered text
- **PTT while TTS is playing:** Stop playback, start recording (interrupt to speak)
- **Connection drop:** Earcon alert. Orchestrator auto-reconnects SSH. Buffered commands replayed.

### Repetition and Navigation

- **OK long press:** "Read that again" — replays the last TTS output from the message history buffer
- **NAV in any mode:** Step through recent messages (last 10 kept in memory)
- **"What?" or "repeat"** as voice command: replays last message

### Verbosity Control

Inspired by screen reader verbosity levels:

- **PTT + "brief":** Responses capped at 1-2 sentences
- **PTT + "detailed":** Full responses (default)
- **PTT + "verbose":** Include file names, line counts, test names

This adjusts the system prompt dynamically.

---

## Implementation Plan

### Phase 1: Proof of Concept (1-2 days of hacking)

No custom hardware. Use laptop mic + keyboard shortcuts as stand-ins.

1. **Local Python script** that:
   - Listens for keyboard shortcuts (F13-F17 via `pynput`, or just regular hotkeys)
   - Records audio on key-hold, sends to Deepgram or OpenAI Whisper API
   - SSHs to remote, runs `claude -p --output-format stream-json`
   - Pipes Claude's text output to Piper TTS for local playback
2. **Voice mode system prompt** — write the prompt that constrains Claude to audio-friendly output
3. Test the core loop: hold key → speak → hear Claude's response

**Minimal dependencies:** `pynput`, `paramiko`, `deepgram-sdk` (or `openai`), `RealtimeTTS` + Piper.

### Phase 2: Modal System (1-2 days)

1. Implement the mode state machine (IDLE → BROWSE → WORK → REVIEW → SHIP)
2. Add Linear ticket browsing via Claude Code's Linear MCP (run `claude -p "list my tickets"`)
3. Add worktree management (create/switch/list)
4. Add earcons for mode transitions and events
5. Test the full ticket-to-PR workflow with keyboard shortcuts

### Phase 3: Hardware (1-2 weekends)

1. Order parts: XIAO nRF52840, 5x Kailh Choc switches, LiPo battery, diodes
2. Wire switches directly to GPIO pins (5 keys = no matrix needed, direct wiring)
3. Write ZMK shield config (devicetree + keymap) mapping to F13-F17
4. Build with GitHub Actions, flash firmware
5. 3D print or hand-wire a minimal case
6. Test BLE pairing and key detection with the orchestrator

### Phase 4: Polish

1. Tune TTS voice, speed, and earcon sounds
2. Add robust error handling and connection recovery
3. Add verbosity controls
4. Fine-tune the voice-mode system prompt based on real usage
5. Consider adding Whisper keyword boosting for project-specific terms
6. Optional: package as a systemd service that starts on login

---

## Bill of Materials

### Software (all free/open-source except API costs)

| Component | Cost |
|---|---|
| Python orchestrator | Free |
| Piper TTS | Free (open source) |
| RealtimeTTS | Free (open source) |
| ZMK firmware | Free (open source) |
| Deepgram STT | $0.0043/min (Nova-2) or free tier |
| OpenAI TTS (optional) | ~$15/1M chars |
| OpenAI Whisper API (alternative STT) | $0.006/min |

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

- **Vocode:** Open-source library for building voice-based LLM apps. Good reference for the STT → LLM → TTS pipeline architecture.
- **OpenAI Realtime API:** Native speech-to-speech with GPT-4o. Could be an alternative architecture where Claude's text output is fed to GPT-4o-realtime for voice synthesis, but adds complexity and cost.
- **RealtimeTTS:** Handles the streaming TTS problem well — accepts text token-by-token and begins playback at sentence boundaries.

### References

- [Talon Voice](https://talonvoice.com/) — voice coding framework
- [Cursorless](https://www.cursorless.org/) — structural voice code editing
- [Piper TTS](https://github.com/rhasspy/piper) — fast local neural TTS
- [RealtimeTTS](https://github.com/KoljaB/RealtimeTTS) — streaming multi-engine TTS library
- [ZMK Firmware](https://zmk.dev/) — wireless keyboard firmware
- [XIAO nRF52840 macro pad guide](https://www.ubiqueiot.com/posts/xiao-nrf52840-zmk) — hardware build reference
- [Claude Code headless mode](https://code.claude.com/docs/en/headless) — programmatic CLI usage
- [Claude Code hooks](https://code.claude.com/docs/en/hooks) — lifecycle event hooks
- [Deepgram STT](https://deepgram.com/product/speech-to-text) — streaming speech-to-text with keyword boosting
- [ElevenLabs streaming](https://elevenlabs.io/docs/developers/websockets) — low-latency WebSocket TTS
- [OpenAI TTS](https://developers.openai.com/api/docs/guides/text-to-speech) — streaming text-to-speech API
- [WhisperLive](https://github.com/collabora/WhisperLive) — real-time Whisper transcription
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

3. **Local vs. cloud STT?** Deepgram is faster and more accurate but requires internet. WhisperLive runs locally but adds latency and uses laptop resources. For walking outdoors (phone tethering), Deepgram's small bandwidth is fine.

4. **Multiple simultaneous worktrees?** The design supports it (NAV cycles through them), but cognitive load of tracking multiple tasks by audio alone may be high. Start with one active worktree and add multi-worktree support based on experience.

5. **Safety:** Auto-accepting edits while walking means you're trusting Claude more. Mitigate with: review mode before shipping, automated review tools (linting, type checking, tests), and the ability to reject/undo.

6. **Notification while idle?** Should the system proactively notify when a long-running Claude task completes? Probably yes — a completion earcon + brief status when Claude finishes, even if you haven't asked.
