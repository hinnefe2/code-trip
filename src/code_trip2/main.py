"""Entry point: wire everything and run the PTT loop."""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from pathlib import Path

from code_trip2 import earcon
from code_trip2.audio import PushToTalk, resolve_hotkey
from code_trip2.config import Config, load_config
from code_trip2.modes import Context, handle_voice
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
    })

    stt = STTClient(api_key=config.api_key, model=config.stt_model)
    tts = TTSClient(
        api_key=config.api_key,
        model=config.tts_model,
        voice=config.tts_voice,
        speed=config.tts_speed,
    )
    thinking = earcon.Thinking()
    ctx = Context(config=config, tts=tts, log=log, thinking=thinking)

    shutdown = threading.Event()

    def on_audio(path: Path) -> None:
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

    ptt = PushToTalk(
        hotkey=resolve_hotkey(config.hotkey),
        on_audio=on_audio,
        sample_rate=config.sample_rate,
        device=config.audio_device,
    )

    logger.info("Starting code-trip. Hold %s to talk.", config.hotkey)
    ptt.start()
    try:
        shutdown.wait()
    except KeyboardInterrupt:
        pass
    finally:
        ptt.stop()
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
