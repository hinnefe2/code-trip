"""Entry point: wire everything and run the PTT loop."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from pathlib import Path

from code_trip2 import earcon
from code_trip2.chords import handle_chord, handle_tap
from code_trip2.config import Config, load_config
from code_trip2.macropad import Macropad, resolve_key
from code_trip2.modes import Context, handle_voice, stop_playback
from code_trip2.session_log import SessionLogger, default_session_path
from code_trip2.stt_client import STTClient, STTClientError
from code_trip2.tts_client import TTSClient

logger = logging.getLogger(__name__)


def run(config: Config) -> None:
    log = SessionLogger(default_session_path())
    log.event("session_start", config={
        "tmux_session": config.tmux_session,
        "work_window": config.work_window,
        "linear_window": config.linear_window,
        "stt_model": config.stt_model,
        "tts_model": config.tts_model,
        "tts_voice": config.tts_voice,
        "app_cycle": list(config.app_cycle),
        "terminal_apps": list(config.terminal_apps),
    })

    stt: STTClient | None = None
    if config.stt_provider == "openai":
        stt = STTClient(api_key=config.api_key, model=config.stt_model)
    else:
        logger.info("STT provider=%s; bypassing OpenAI STT.", config.stt_provider)
    tts = TTSClient(
        api_key=config.api_key,
        model=config.tts_model,
        voice=config.tts_voice,
        speed=config.tts_speed,
    )
    thinking = earcon.Thinking()
    ctx = Context(config=config, tts=tts, log=log, thinking=thinking)

    shutdown = threading.Event()

    def _process_audio(path: Path) -> None:
        if stt is None:
            logger.warning("on_audio fired in non-openai STT mode; ignoring %s", path)
            return
        try:
            transcript = stt.transcribe(path)
        except STTClientError as exc:
            logger.exception("STT failed")
            try:
                earcon.error()
                tts.speak(f"Transcription failed: {exc}")
            except Exception:
                pass
            return
        logger.info("Transcribed: %s", transcript)
        handle_voice(ctx, transcript)

    def on_audio(path: Path) -> None:
        # Run STT + dispatch off the macropad listener thread so taps stay
        # responsive while a turn is in flight (wait_done can take 5–15s).
        threading.Thread(target=_process_audio, args=(path,), daemon=True).start()

    def on_ptt_press() -> None:
        # PTT-press while Claude is speaking should stop playback so the
        # mic input isn't talking over TTS.
        stop_playback(ctx)

    # Dispatch off the pynput listener thread — anything slow (TTS,
    # ssh capture) here would freeze the keyboard.
    def on_chord(name: str) -> None:
        threading.Thread(target=handle_chord, args=(ctx, name), daemon=True).start()

    def on_tap(name: str) -> None:
        threading.Thread(target=handle_tap, args=(ctx, name), daemon=True).start()

    ptt_forward_key = (
        resolve_key(config.stt_local_hotkey) if config.stt_provider == "local" else None
    )
    macropad = Macropad(
        keymap={
            "ptt": resolve_key(config.ptt_key),
            "act": resolve_key(config.act_key),
            "yes": resolve_key(config.yes_key),
            "no": resolve_key(config.no_key),
            "nav": resolve_key(config.nav_key),
        },
        on_audio=on_audio,
        on_chord=on_chord,
        on_tap=on_tap,
        on_ptt_press=on_ptt_press,
        ptt_forward_key=ptt_forward_key,
        sample_rate=config.sample_rate,
        device=config.audio_device,
    )

    def _handle_signal(signum: int, _frame: object) -> None:
        logger.info("Received signal %d; shutting down.", signum)
        shutdown.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    ptt_desc = (
        f"{config.ptt_key} (forwards to {config.stt_local_hotkey})"
        if ptt_forward_key is not None
        else f"{config.ptt_key} (OpenAI STT)"
    )
    logger.info(
        "Starting code-trip. PTT=%s NAV=%s (hold NAV + key for chords). Ctrl-C to quit.",
        ptt_desc,
        config.nav_key,
    )
    macropad.start()
    try:
        shutdown.wait()
    finally:
        macropad.stop()
        thinking.stop()
        log.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="code-trip")
    parser.add_argument("--config", type=Path, required=True, help="Path to TOML config")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    config = load_config(args.config)
    run(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
