#!/usr/bin/env python3
"""Manual test: synthesize and play text via OpenAI TTS.

Usage:
    OPENAI_API_KEY=sk-... python scripts/test_tts.py "text to speak"
    OPENAI_API_KEY=sk-... python scripts/test_tts.py --interrupt
"""

from __future__ import annotations

import sys
import threading
import time

from code_trip.tts_client import TTSClient, TTSClientError


LONG_TEXT = (
    "This is a long test sentence intended to demonstrate interruption. "
    "You should hear me start speaking and then get cut off partway through "
    "because the main thread is about to call stop on me after two seconds. "
    "If you hear this entire sentence then interruption is not working."
)


def main() -> None:
    args = sys.argv[1:]
    tts = TTSClient()

    if args and args[0] == "--interrupt":
        print("Speaking long text in background; stopping after 2s...")
        t = threading.Thread(target=lambda: _speak_ignore(tts, LONG_TEXT))
        t.start()
        time.sleep(2.0)
        tts.stop()
        t.join()
        print("Done.")
        return

    text = " ".join(args) if args else "Hello, this is a test of text to speech."
    print(f"Speaking: {text!r}")
    tts.speak(text)
    print("Done.")


def _speak_ignore(tts: TTSClient, text: str) -> None:
    try:
        tts.speak(text)
    except TTSClientError as exc:
        print(f"speak error: {exc}")


if __name__ == "__main__":
    try:
        main()
    except TTSClientError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("\nCancelled.")
