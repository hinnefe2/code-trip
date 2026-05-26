# Asyncio migration plan

> **Status:** Design doc, written 2026-05-26. Not yet started.

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

- `Producer` protocol: `start()` / `stop()` → `async def run(self)`.
- `ProducerSupervisor`: holds a list of `asyncio.Task`. `start_all()` creates them; `stop_all()` cancels and awaits them.
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
- `subprocess.run` in `ClaudeProducer` (SSH ls/cat/rm) → `asyncio.create_subprocess_exec` with `await proc.communicate()`.
- `claude --print` calls in Slack/Email producers route through `ClaudeMCPClient` (converted in Phase 5).

**Per-producer trickiness**
- `ClaudeProducer` polls SSH every 1.5 s; conversion is mechanical.
- `SlackProducer` setup (`_fetch_user_id`, `_resolve_watch_channels`) is sequential async calls; conversion is mechanical.
- `EmailProducer`: same shape as Slack.
- `ManualProducer`: no thread, no change needed.
- `LinearProducer`: stub, easy.

**Test impact:** Producer tests use synchronous setup-then-poll; mark `@pytest.mark.asyncio` and `await p._poll_once()`.

### Phase 3 — Convert TaskQueue + listener pattern

- `threading.Lock` → `asyncio.Lock` throughout `tasks.py`.
- `add_listener` accepts both sync and async callbacks; `_fire` checks with `inspect.iscoroutinefunction` and either calls directly or `await`s.
- `QueueLog.record` becomes `async def` (writes JSONL via `await asyncio.to_thread(self._append, …)` or `aiofiles` — pick `to_thread` to avoid the dep; the writes are tiny).
- `QueueConsumer` becomes an async task with `asyncio.Event` for wakeup.

**Migration trick to keep tests green incrementally:** `_fire` stays backward-compatible — sync listeners keep working until each is migrated.

### Phase 4 — Convert dispatch + handle_voice + handle_skill

- All `handle_*`, `_respond_*`, `_announce_*`, `_skip_current`, `_drop_current`, `_snooze_current`, `_add_manual` become `async def`.
- `handle_voice` and `handle_skill` await `_dispatch_task_response`.
- `_respond_claude` / `_respond_slack` / `_respond_email` await the MCP client (converted in Phase 5; for now wrap the sync call in `asyncio.to_thread` as a temporary measure).
- `TTSClient.speak`, `STTClient.transcribe`, `Summarizer.summarize` → async. OpenAI calls use `AsyncOpenAI`; TTS audio playback wraps `stream.write` in `asyncio.to_thread`.
- `modes.speak_chunked` becomes async; the chunked playback worker becomes an async task.
- `modes._work_voice` becomes async (uses `remote.send`/`wait_done`/`capture` from Phase 5).

**Risk:** many call sites. Move incrementally; keep tests passing after each handler conversion.

### Phase 5 — Convert MCP clients + remote helpers

- `ClaudeMCPClient.call_tool` and `.run_agent` switch from `subprocess.run` to `asyncio.create_subprocess_exec` + `await proc.communicate(input=…)`.
- `remote.py` SSH helpers (`send`, `capture`, `wait_done`, `select_window`, `new_window`, `list_windows`, `_ssh`) become async via `asyncio.create_subprocess_exec`.
- `window.py` macOS osascript helpers become async via `asyncio.create_subprocess_exec` — they're small, single subprocess.run calls.

After Phase 5 all subprocess calls are async; the `asyncio.to_thread` stopgap from Phase 4 can come out.

### Phase 6 — Replace macropad → async bridge

- Macropad keeps its pynput listener thread and audio recording thread internally. Public API stays callback-based.
- `main.py` wraps every callback in the `_from_thread` helper described above. One pattern, applied uniformly. Existing per-callback wrapper threads (`threading.Thread(target=handle_chord, …)`) go away.
- `on_audio` callback awaits STT then dispatches.

### Phase 7 — Migrate TUI to Textual

- Separate detailed plan; Textual's loop integrates natively with the asyncio loop set up in Phase 1.
- Delete the cbreak/termios setup, the stdin paste reader, the `InputBuffer`, the `BRACKETED_PASTE_RE` / `_ANSI_CSI_RE` code, the `on_ptt_release` FIFO. Replace with Textual's `Input` widget + a message from macropad's bridge.

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
