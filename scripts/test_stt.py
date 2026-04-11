#!/usr/bin/env python3
"""Manual test: record audio then transcribe via Whisper API.

Usage:
    OPENAI_API_KEY=sk-... python scripts/test_stt.py
"""

from __future__ import annotations

from code_trip.audio_recorder import AudioRecorder, AudioRecorderError
from code_trip.stt_client import STTClient, STTClientError


def main() -> None:
    recorder = AudioRecorder()
    stt = STTClient()

    input("Press Enter to start recording...")
    recorder.start()
    print("Recording... Press Enter to stop.")
    input()
    wav_path = recorder.stop()
    print(f"Saved: {wav_path}")

    print("Transcribing...")
    text = stt.transcribe(wav_path)
    print(f"Transcription: {text}")


if __name__ == "__main__":
    try:
        main()
    except (AudioRecorderError, STTClientError) as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("\nCancelled.")
