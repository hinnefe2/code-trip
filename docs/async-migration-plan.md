# Asyncio migration plan

> **Status:** Phases 1–7 landed 2026-05-27 on the `async-migration` branch.

## Motivation

The orchestrator is fundamentally I/O-heavy: MCP subprocess calls (`claude --print` runs ~5 s), SSH polling for the Claude stop hook, OpenAI HTTP for TTS/STT/summarizer, audio playback. Today each producer is a daemon thread, the queue consumer is a daemon thread, TTS playback spawns a worker thread, and inter-thread coordination is a mix of `threading.Lock`, `threading.Event`, listener callbacks fired synchronously across threads, and `subprocess.run` blocking the calling thread.

Adding a Textual TUI on top would introduce an asyncio event loop, creating a permanent mixed-paradigm seam. Rather than live with that, this plan converts the orchestrator to a single asyncio event loop. The only threaded code left is the irreducible boundary at pynput (keyboard listener) and sounddevice (PortAudio C callbacks); both are well-defined single-source boundaries with a small bridge.

End state: one event loop, no `threading.Lock` in our code, all I/O concurrent for free, Textual integrates natively, future contributors don't have to track which side of a paradigm boundary they're on.

## Goals & non-goals

**Goals**
- Single asyncio event loop owning all orchestrator state
- All blocking I/O converted to async (HTTP, subprocess, file I/O at non-trivial volume)
- `threading.Lock` removed everywhere it currently exists
- Tests run on the event loop via pytest-asyncio
- Textual TUI integrates without paradigm-bridging

**Non-goals (stays threaded; bridged to the loop)**
- pynput macropad listener — its API is a synchronous callback model on its own thread; no async port exists. We keep it on its thread and bridge callbacks to the loop.
- sounddevice OutputStream — PortAudio's API is C-callback based; the blocking `stream.write(chunk)` is invoked from `asyncio.to_thread` inside the TTS playback coroutine.

Both boundaries are single-source: one bridge module each, one pattern, easy to test and easy to find.

## The two threaded boundaries

**pynput → asyncio.** The macropad listener thread fires `on_press` / `on_release` / `on_audio` / `on_chord` / `on_tap` / `on_ptt_press` / `on_ptt_release` callbacks. In `main.py` these will be wrapped:

```python
def _from_thread(coro_fn, *args, **kwargs):
    loop.call_soon_threadsafe(
        lambda: loop.create_task(coro_fn(*args, **kwargs))
    )

macropad = Macropad(
    on_chord=lambda name: _from_thread(handle_chord, ctx, name),
    on_tap=lambda name: _from_thread(handle_tap, ctx, name),
    ...
)
```

One helper, one pattern, used at every callback site.

**sounddevice → asyncio.** Replace `stream.write(chunk)` in `TTSClient._write_in_blocks` with:

```python
await asyncio.to_thread(stream.write, samples[start:end])
```

Each block write blocks an executor thread for ~170 ms. With one playback in flight at a time, this is fine; the default executor pool is large enough.

## Phases

Each phase ends with a working, shippable orchestrator. Run the live test suite after every phase; commit between phases.

### Phase 1 — Stand up the event loop next to existing threads (no behavior change)

- `main.py` wraps the existing `run(config)` body in `async def main_async(config)` and calls `asyncio.run(main_async(config))`.
- Signal handling moves from `signal.signal(SIGINT, …)` to `loop.add_signal_handler(SIGINT, shutdown_event.set)` where `shutdown_event = asyncio.Event()`.
- The main coroutine: kick off existing thread-based startup unchanged, then `await shutdown_event.wait()`, then teardown.
- The threading-Event `shutdown` flag used by producers/consumer remains in place for now — a small bridge sets both events when SIGINT fires.

After this phase, nothing about the rest of the orchestrator has changed. The async scaffolding is just there to be filled in.

**Risk:** none significant. Mechanical change.

### Phase 2 — Convert producers to async tasks

- `Producer` protocol: `start()` / `stop()` → `async def run(self)` + `def request_stop(self)`. (`request_stop` is sync so the supervisor and signal handler can call it without an `await`.)
- `ProducerSupervisor`: holds a `dict[name, asyncio.Task]`. `start_all()` is sync (creates tasks); `stop_all()` is async — signals stop, waits up to 2 s for voluntary exit, then cancels stragglers.
- Each producer's poll loop becomes:
  ```python
  async def run(self):
      while not self._stop.is_set():
          try:
              await self._poll_once()
          except Exception:
              logger.exception(...)
          await asyncio.wait_for(self._stop.wait(), timeout=interval)
              # except asyncio.TimeoutError pass
  ```
- `subprocess.run` in `ClaudeProducer` (SSH ls/cat/rm) → `asyncio.create_subprocess_exec` with `await proc.communicate()`. The async SSH helper lives inline as `ClaudeProducer._ssh` for now; Phase 5 hoists it into `remote.py`.
- `remote.capture` and `Summarizer.summarize` calls inside ClaudeProducer are bridged via `asyncio.to_thread` until Phases 4–5 convert them natively.
- `claude --print` calls in Slack/Email producers stay sync (`ClaudeMCPClient`); Phase 2 wraps each call site in `await asyncio.to_thread(mcp.call_tool, …)`. Phase 5 converts the MCP client and the wrappers go away.

**Per-producer trickiness**
- `ClaudeProducer` polls SSH every 1.5 s; conversion is mechanical.
- `SlackProducer` setup (`_fetch_user_id`, `_resolve_watch_channels`) is sequential async calls; conversion is mechanical.
- `EmailProducer`: same shape as Slack.
- `ManualProducer`: no thread, no change needed.
- `LinearProducer`: stub, easy.

**Test impact:** Producer tests use synchronous setup-then-poll; mark `@pytest.mark.asyncio` and `await p._poll_once()`.

### Phase 3 — Convert TaskQueue + listener pattern

- Drop `threading.Lock` from `tasks.py` and `queue_log.py`. Every method is a non-awaiting compute body, so single-loop discipline already serializes them — no lock needed. (The original plan called for `asyncio.Lock`; we don't have anywhere it'd actually do work.)
- Listeners stay **sync-only**. `_fire` just calls them inline. The original plan proposed `inspect.iscoroutinefunction` runtime detection so async listeners could be scheduled via `loop.create_task`; we don't have async listeners and runtime type-sniffing is hacky. If a listener ever needs to do async work, it can write `asyncio.create_task(...)` itself — that's explicit and obvious.
- `QueueLog.record` stays sync. The proposed `async def` + `asyncio.to_thread` adds nothing for tiny JSONL appends and would force every sync caller (e.g. `dispatch._announce_next`, which becomes async in Phase 4) to either await or `create_task` it. Phase 4 leaves it sync too.
- `QueueConsumer` becomes an async task: `async def run()`, `asyncio.Event` for wakeup, `request_stop()` for shutdown. The listener (`_on_event`) stays sync — it just sets the asyncio.Event, which is safe because every queue mutation now originates on the loop thread.
- `main.py` spawns the consumer with `asyncio.create_task(consumer.run())` and awaits it during teardown with the same wait-then-cancel pattern as the producer supervisor.

### Phase 4 — Convert dispatch + handle_voice + handle_skill

- All `handle_*`, `_respond_*`, `_announce_*` (except the fire-and-forget `_announce_headline`), `_skip_current`, `_drop_current`, `_snooze_current`, `_add_manual`, `queue_yes_tap`, `queue_no_tap`, `dismiss_current_task` become `async def`.
- `modes.handle_voice`, `_try_global_commands`, `_try_voice_phrase`, `_dispatch_by_focus`, `_dictate_voice`, `_work_voice`, `_select_window`, `_new_window`, `_announce_windows`, `_linear_refresh`, `_linear_step`, `_linear_announce`, `_linear_select`, `_speak`, `_report_error`, `_summarize_or_strip` become `async`.
- `_respond_claude` / `_respond_slack` / `_respond_email` await `remote.*` / `mcp.call_tool` via `asyncio.to_thread` until Phase 5 makes those async.
- `TTSClient.speak`, `STTClient.transcribe`, `Summarizer.summarize` → async. OpenAI calls use `AsyncOpenAI`; TTS audio playback wraps `stream.write` in `asyncio.to_thread`. `TTSClient.stop` stays sync (callable from the pynput listener thread) and the `_stop_event` stays `threading.Event` for the same reason. `_speak_lock` becomes `asyncio.Lock` to serialize between async speak callers.
- `modes.speak_chunked` stays **sync** — it's fire-and-forget (chunks go into `ctx.playback_queue`, the playback task picks them up). Callers don't want to wait for playback to finish.
- The chunked playback worker (`_playback_loop`) becomes an `async def` coroutine; `_start_playback_task` spawns it with `asyncio.create_task`. The `Context._playback_lock` / `_playback_thread` fields go away.
- `chords.handle_chord` / `handle_tap` / `_act_tap_app_aware` / `_open_last_pane_url` / `_nav` / `_cycle_app` / `_speak_active_app` / `_speak` / `_speak_error` become async. The stale `from code_trip2 import dispatch` local imports inside `handle_chord` / `handle_tap` are hoisted to module top — the cycle they claimed to avoid no longer exists.
- `main.py` introduces a `_from_thread(coro_fn, *args, **kwargs)` bridge that schedules a coroutine on the loop from the macropad's pynput thread and the stdin paste reader thread. Each spawned task gets a `done_callback` that logs uncaught exceptions so they're not silently swallowed. (Phase 6 documents this as the single seam to the threaded boundaries.)

**Risk:** many call sites. The test churn was substantial — every `MagicMock` for `ctx.tts`, `ctx.summarizer`, etc. needed `tts.speak = AsyncMock()` (etc.) because the methods are now awaited; every test that exercises a handler needed `@pytest.mark.asyncio` + `await`.

### Phase 5 — Convert MCP clients + remote helpers

- `ClaudeMCPClient.call_tool` and `.run_agent` are async; the shared subprocess plumbing lives in a module-level `_run_subprocess` helper.
- `remote.py` SSH helpers (`send`, `capture`, `wait_done`, `clear_signal`, `select_window`, `new_window`, `list_windows`, `_ssh`) are all async. `wait_done`'s polling loop uses `asyncio.sleep` against the running loop's `loop.time()` for the deadline.
- `window.py` is async: `active_app`, `activate_app`, `paste_text`, `send_keystroke` (its inter-chord delay uses `asyncio.sleep`). `_send_chord` stays sync — it's just pynput press/tap/release, no I/O.
- `ClaudeProducer`'s inline `_ssh` is removed; callers use `remote._ssh` directly. Its `_summarize_pane` drops the `asyncio.to_thread` wrappers around `remote.capture`.
- All Phase 4 `asyncio.to_thread(...)` stopgaps in `dispatch.py`, `modes.py`, `chords.py`, `producers/slack.py`, `producers/email.py` are replaced with direct awaits.
- `LinearProducer` still wraps `MCPClient.start` / `.stop` in `asyncio.to_thread` — that's a different MCP client (stdio-based, used only by Linear). It's a stub producer with no live caller; conversion can wait.

Remaining `asyncio.to_thread` uses after Phase 5:
- `TTSClient._write_in_blocks` → wraps sounddevice's blocking `stream.write` (irreducible PortAudio C boundary, intentional).
- `LinearProducer.run` → wraps the sync `MCPClient` (stub).

### Phase 6 — Replace macropad → async bridge

The bridge itself landed as part of Phase 4 (the `_from_thread` helper was needed once `handle_chord` / `handle_tap` / `handle_voice` / `handle_skill` became coroutines). Phase 6 is the audit confirming uniformity and cataloguing what threads remain.

- Macropad keeps its pynput listener thread and audio capture stream internally. Public API stays callback-based.
- `main.py` wraps every async-bound callback in `_from_thread`. One pattern, applied uniformly:
  - `on_audio` → `_from_thread(_process_audio, path, skill_mode)` (STT + dispatch on the loop)
  - `on_chord` → `_from_thread(handle_chord, ctx, name)`
  - `on_tap` → `_from_thread(handle_tap, ctx, name)`
  - Local-STT stdin reader → `_from_thread(handle_voice|handle_skill, ctx, transcript)` (the stdin reader still runs on its own thread until Phase 7)
- Two callbacks are intentionally **sync** because they must be callable from the pynput thread without a loop hop:
  - `on_ptt_press` → `stop_playback(ctx)` (sets `TTSClient._stop_event`, a `threading.Event`)
  - `on_ptt_release` → push skill-mode flag onto a `queue.Queue` for the stdin paste reader
- No per-callback `threading.Thread(target=…)` wrappers remain. Confirmed by grep across `src/`.

**Threads still alive in our code after Phase 6** (catalogued for Phase 8 cleanup):

| Thread | Where | Status |
|---|---|---|
| pynput keyboard listener | `macropad.py` | **Irreducible** — pynput C-callback API |
| sounddevice `InputStream` (mic capture) | `macropad.py` | **Irreducible** — PortAudio C-callback |
| sounddevice playback in `earcon.Thinking` | `earcon.py` | **Irreducible** — PortAudio playback boundary |
| sounddevice `OutputStream` (TTS playback, via `asyncio.to_thread`) | `tts_client.py` | **Irreducible** — executor thread per block write |
| Rich Live render thread | `tui.py` Dashboard | Phase 7 removes (Textual owns its loop) |
| Stdin paste reader (local-STT mode) | `main.py:_stdin_paste_loop` | Phase 7 removes (Textual's Input widget replaces it) |
| `threading.Event` shutdown bridge | `main.py` | Phase 8 removes once the stdin reader is gone |

The locks at `email_state.py`, `slack_state.py`, `input_buffer.py` are no-ops under single-loop discipline; Phase 8 drops them.

### Phase 7 — Migrate TUI to Textual

- `tui.py` rewritten: Rich `Live` `Dashboard` replaced by `CodeTripApp(App)`. The panel-builders (`_header`, `_current_task_panel`, `_queue_table`, `_topics_panel`, `_keymap_panel`, `_producers_panel`) stayed pure-Rich and are rendered through Textual `Static` widgets, refreshed at 2 Hz from a `set_interval` timer.
- Voice/skill input goes through a Textual `Input` widget (visible only in local-STT mode). Submission dispatches via `handle_voice` / `handle_skill`. PTT release in local-STT mode posts a `PttReleased` message from the pynput thread (`app.call_from_thread(app.post_message, …)`), which clears the field and arms an autosubmit timer so a Superwhisper paste burst dispatches without a trailing newline (matches the old `_INPUT_QUIET_S = 0.4` semantics, with a 5 s timeout).
- Removed in `main.py`: cbreak/termios setup, the stdin paste reader thread, `_BRACKETED_PASTE_RE`, `_ANSI_CSI_RE`, `ptt_release_skill_q`, `_ingest_stdin_chunk`, `_submit_input_buffer`, `_stdin_paste_loop`, the `threading.Event` shutdown bridge, the `input_buffer` wiring. The `_from_thread` helper stays (still needed for macropad's pynput thread).
- Deleted: `src/code_trip2/input_buffer.py`, `tests/test_input_buffer.py`. `modes.Context.input_buffer` field removed.
- **Breaking change in CLI semantics:** local-STT mode (`stt_provider != "openai"`) now requires `--tui`. `main_async` raises `SystemExit` with an explanatory message if the combination is misconfigured. There's no longer a non-TUI path that accepts pasted transcripts.
- Tests: existing Rich-panel assertions kept (the panel builders are unchanged); new Pilot-based tests cover Input submit → `handle_voice`, PttReleased → `handle_skill`, and the openai-mode hidden-Input case.
- `textual>=0.80` added to `dependencies` in `pyproject.toml`.

### Phase 8 — Cleanup

- Audit for any remaining `threading.Lock` / `threading.Event` / daemon threads in our code. Should be exactly two: the pynput listener and any sounddevice executor threads in flight. Document them in CLAUDE.md / brief.
- Remove the `threading.Event` shutdown-bridge added in Phase 1.
- Update `docs/task-queue-brief.md` to reflect the async architecture.
- Remove pytest-asyncio's auto-mode if we adopted strict mode (recommended) — confirm every test that needs the loop is explicitly marked.

## Per-module conversion checklist

| File | Phase | Change | Tricky? |
|---|---|---|---|
| `main.py` | 1 | `asyncio.run(main_async)`, `loop.add_signal_handler` | low |
| `producers/__init__.py` | 2 | Supervisor holds tasks not threads | low |
| `producers/claude.py` | 2 | Async poll, `asyncio.create_subprocess_exec` for SSH | medium |
| `producers/slack.py` | 2 | Async poll loop; MCP calls await | medium |
| `producers/email.py` | 2 | Same as slack | medium |
| `producers/manual.py` | 2 | No background work, trivial | none |
| `producers/linear.py` | 2 | Stub | none |
| `producers/claude_mcp.py` | 5 | `call_tool` + `run_agent` async via asyncio.subprocess | medium |
| `tasks.py` | 3 | `asyncio.Lock`, async listener fan-out | medium |
| `queue_log.py` | 3 | Async listener; JSONL append via `asyncio.to_thread` | low |
| `dispatch.py` | 4 | Every handler async; many call sites | **high** |
| `modes.py` | 4 | `handle_voice`, work/dictate, linear ops, chunked playback worker | high |
| `tts_client.py` | 4 | `AsyncOpenAI`; `stream.write` in `asyncio.to_thread` | medium |
| `stt_client.py` | 4 | `AsyncOpenAI` | low |
| `summarizer.py` | 4 | `AsyncOpenAI` | low |
| `remote.py` | 5 | Async subprocess for every SSH helper | medium |
| `window.py` | 5 | Async subprocess for osascript helpers | low |
| `macropad.py` | 6 | No internal change; main.py adds bridge | low |
| `chords.py` | 6 | `handle_chord` becomes async; pynput callback bridges | low |
| `tui.py` | 7 | Textual migration (separate plan) | high |
| `input_buffer.py` | 7 | Deleted | none |
| `earcon.py` | — | Stays sync; called via `asyncio.to_thread` from async callers | low |
| `session_log.py` | — | Tiny files; sync ok | none |
| `slack_state.py` | — | Tiny files; sync ok | none |
| `email_state.py` | — | Tiny files; sync ok | none |
| `skills.py` | — | Pure functions; no change | none |
| `config.py` | — | Sync TOML load; ok | none |

## Testing strategy

- Add `pytest-asyncio` to `[project.optional-dependencies] dev`.
- Use **strict mode** (`asyncio_mode = "strict"` in `pyproject.toml`) so every async test is explicit.
- `@pytest.mark.asyncio` on async tests. Sync tests stay sync.
- For tests that exercise components with thread bridges (macropad, TTS playback executor), use `asyncio.get_running_loop()` and run the test inside `asyncio.run`.
- Mocks for subprocess move from `mock.patch("subprocess.run")` to `mock.patch("asyncio.create_subprocess_exec")`. Helper fixture is worth writing once and reusing across producer tests.
- The live-suite command in `task-queue-brief.md` stays the same list of test files; just runs async-capable now.

## Risks & open questions

- **`AsyncOpenAI` API parity.** The openai SDK exposes async equivalents for `audio.speech.create`, `audio.transcriptions.create`, and `chat.completions.create` as of 1.x. Worth a 30-minute spike to confirm response shapes match. If not, wrap the sync client in `asyncio.to_thread` as a fallback.
- **sounddevice + asyncio.** Wrapping `stream.write` in `asyncio.to_thread` is straightforward but each block call holds an executor thread for ~170 ms. With one playback at a time the default executor (5 threads on macOS) is plenty. Document the boundary.
- **asyncio subprocess timeout semantics.** `subprocess.run(timeout=N)` kills the process when N elapses. The asyncio equivalent is `await asyncio.wait_for(proc.communicate(), timeout=N)` + a `try/except asyncio.TimeoutError` that does `proc.kill()`. Need to write a small `run_subprocess_with_timeout` helper and use it everywhere we currently use `timeout=` — this is the only place asyncio's ergonomics are noticeably worse than `subprocess.run`.
- **Signal handlers on macOS.** `loop.add_signal_handler` works on Unix. Verified.
- **Order of teardown.** Today teardown is mostly sequential: dashboard.stop, consumer.stop, supervisor.stop_all, macropad.stop, thinking.stop, log.close. The async version cancels tasks and awaits them; care needed so producers shut down cleanly (no pending subprocess hanging the loop).
- **pytest-asyncio + Rich console capture in tests.** Existing tui tests render a Rich Layout to a string. Textual tests use the pilot API. Phase 7 rewrites these.

## Effort estimate

Per phase, single developer working seriously:

| Phase | Estimate |
|---|---|
| 1. Event loop scaffold | 0.5 day |
| 2. Producers | 1–2 days |
| 3. TaskQueue + listeners | 1 day |
| 4. Dispatch + TTS/STT/summarizer | 1–2 days |
| 5. MCP + remote + window | 0.5 day |
| 6. Macropad bridge | 0.5 day |
| 7. Textual TUI | 1–2 days |
| 8. Cleanup | 0.5 day |

**Total: ~1–2 weeks** of dedicated work. Each phase ships independently and is committable on its own, so the migration can pause between phases without leaving the codebase in a half-state.

## Recommended starting point

Phase 1 + Phase 2 as a spike. After Phase 2 the producers run as concurrent async tasks instead of competing threads, and you can decide whether the conversion pattern feels right before committing to the full migration. If something is uncomfortable about the asyncio model in this codebase, you'd want to find out at Phase 2 rather than Phase 5.
