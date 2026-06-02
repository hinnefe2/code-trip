"""Unit tests for TTSClient: audio-shaping + persistent OutputStream."""

from __future__ import annotations

import io
import threading
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from code_trip2 import audio_out, tts_client
from code_trip2.tts_client import (
    _FADE_MS,
    _SILENCE_PAD_MS,
    _WRITE_BLOCK_FRAMES,
    SilentTTSClient,
    TTSClient,
    _decode_wav,
    _shape_samples,
)


def _make_wav_bytes(samples: np.ndarray, sample_rate: int = 24_000) -> bytes:
    """Encode an int16 mono numpy array into WAV bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


# --- shape_samples ---------------------------------------------------------


def test_shape_samples_prepends_silence():
    sr = 24_000
    samples = np.full(sr, 30_000, dtype=np.int16)  # 1s of loud audio
    shaped = _shape_samples(samples, sr)
    n_pad = int(sr * _SILENCE_PAD_MS / 1000)
    assert shaped[:n_pad].max() == 0
    assert shaped[:n_pad].min() == 0


def test_shape_samples_appends_silence():
    sr = 24_000
    samples = np.full(sr, 30_000, dtype=np.int16)
    shaped = _shape_samples(samples, sr)
    n_pad = int(sr * _SILENCE_PAD_MS / 1000)
    assert shaped[-n_pad:].max() == 0
    assert shaped[-n_pad:].min() == 0


def test_shape_samples_fade_in_starts_at_zero():
    sr = 24_000
    samples = np.full(sr, 30_000, dtype=np.int16)
    shaped = _shape_samples(samples, sr)
    n_pad = int(sr * _SILENCE_PAD_MS / 1000)
    n_fade = int(sr * _FADE_MS / 1000)
    # First post-silence sample should be at (or very near) zero amplitude
    # because the fade-in envelope started at 0.
    first_audio = shaped[n_pad]
    assert abs(int(first_audio)) < 200  # within ~0.6% of full scale
    # Mid-fade should be at roughly half amplitude.
    mid = shaped[n_pad + n_fade // 2]
    assert 10_000 < abs(int(mid)) < 25_000


def test_shape_samples_fade_out_ends_at_zero():
    sr = 24_000
    samples = np.full(sr, 30_000, dtype=np.int16)
    shaped = _shape_samples(samples, sr)
    n_pad = int(sr * _SILENCE_PAD_MS / 1000)
    last_audio = shaped[-n_pad - 1]
    assert abs(int(last_audio)) < 1000  # last sample of the audio is near zero


def test_shape_samples_passthrough_empty():
    sr = 24_000
    samples = np.array([], dtype=np.int16)
    assert _shape_samples(samples, sr).size == 0


def test_shape_samples_preserves_middle_amplitude():
    sr = 24_000
    samples = np.full(sr, 30_000, dtype=np.int16)
    shaped = _shape_samples(samples, sr)
    n_pad = int(sr * _SILENCE_PAD_MS / 1000)
    n_fade = int(sr * _FADE_MS / 1000)
    # Middle of the audio section (well past both fades) should be at full
    # amplitude — the fade is only at the edges.
    mid = shaped[n_pad + n_fade + 100]
    assert int(mid) == 30_000


def test_shape_samples_handles_stereo():
    sr = 24_000
    samples = np.full((sr, 2), 30_000, dtype=np.int16)
    shaped = _shape_samples(samples, sr)
    assert shaped.shape[1] == 2  # 2 channels preserved
    n_pad = int(sr * _SILENCE_PAD_MS / 1000)
    n_fade = int(sr * _FADE_MS / 1000)
    assert shaped[0, 0] == 0
    assert shaped[0, 1] == 0
    # Past the fade-in, both channels should be at full amplitude.
    assert shaped[n_pad + n_fade + 100, 0] == 30_000
    assert shaped[n_pad + n_fade + 100, 1] == 30_000


# --- decode_wav -----------------------------------------------------------


def test_decode_wav_applies_shaping():
    sr = 24_000
    samples = np.full(sr, 30_000, dtype=np.int16)
    raw_wav = _make_wav_bytes(samples, sr)
    decoded, decoded_sr = _decode_wav(raw_wav)
    assert decoded_sr == sr
    # Decoded array is longer than the input because of silence padding.
    n_pad = int(sr * _SILENCE_PAD_MS / 1000)
    assert decoded.size == samples.size + 2 * n_pad
    assert decoded[0] == 0  # leading silence
    assert decoded[-1] == 0  # trailing silence


# --- persistent OutputStream behavior --------------------------------------


def _stub_client():
    """Build a TTSClient without touching the OpenAI/sounddevice modules."""
    c = TTSClient.__new__(TTSClient)
    c.api_key = "sk-test"
    c.model = "test-model"
    c.voice = "test-voice"
    c.speed = 1.0
    c._client = MagicMock()
    c._playing = False
    c._speak_lock = threading.Lock()
    c._stop_event = threading.Event()
    return c


@pytest.mark.asyncio
async def test_write_in_blocks_streams_full_array(monkeypatch):
    c = _stub_client()
    monkeypatch.setattr(audio_out, "stream_rate", lambda: 24_000)
    writes: list[np.ndarray] = []
    monkeypatch.setattr(audio_out, "play_blocking", lambda s: writes.append(s))
    samples = np.zeros(_WRITE_BLOCK_FRAMES * 3 + 100, dtype=np.int16)
    await c._write_in_blocks(samples, 24_000)
    # Expect 4 writes: 3 full blocks + a remainder.
    assert len(writes) == 4
    total_written = sum(len(w) for w in writes)
    assert total_written == len(samples)


@pytest.mark.asyncio
async def test_write_in_blocks_stops_when_event_set(monkeypatch):
    c = _stub_client()
    monkeypatch.setattr(audio_out, "stream_rate", lambda: 24_000)
    counter = {"calls": 0}

    def fake_write(_):
        counter["calls"] += 1
        if counter["calls"] == 2:
            c._stop_event.set()

    monkeypatch.setattr(audio_out, "play_blocking", fake_write)
    samples = np.zeros(_WRITE_BLOCK_FRAMES * 5, dtype=np.int16)
    await c._write_in_blocks(samples, 24_000)
    # Stop is honored within one extra block (loop checks at the top).
    assert counter["calls"] == 2


@pytest.mark.asyncio
async def test_write_in_blocks_resamples_to_stream_rate(monkeypatch):
    """Source at 24 kHz played through a 48 kHz stream should be
    upsampled, so the total written sample count roughly doubles."""
    c = _stub_client()
    monkeypatch.setattr(audio_out, "stream_rate", lambda: 48_000)
    writes: list[np.ndarray] = []
    monkeypatch.setattr(audio_out, "play_blocking", lambda s: writes.append(s))
    samples = np.zeros(1200, dtype=np.int16)  # 50 ms at 24 kHz
    await c._write_in_blocks(samples, 24_000)
    total = sum(len(w) for w in writes)
    # ~2400 samples at 48 kHz; allow a rounding wiggle.
    assert 2390 <= total <= 2410


def test_stop_just_sets_event():
    """stop() must not touch the shared OutputStream — leaving it
    alive across utterances is what keeps the audio device warm."""
    c = _stub_client()
    assert not c._stop_event.is_set()
    c.stop()
    assert c._stop_event.is_set()


# --- SilentTTSClient ------------------------------------------------------


@pytest.mark.asyncio
async def test_silent_tts_client_speak_is_noop():
    c = SilentTTSClient()
    # No API key, no audio device — just returns instantly.
    assert await c.speak("hello world") is None
    assert c.is_playing() is False
    c.stop()  # must not raise


# --- earcon mute (covered here because the legacy test_earcon.py
# still imports from the defunct ``code_trip`` package) -----------------


def test_earcon_set_silent_short_circuits_play(monkeypatch):
    """``set_silent(True)`` makes every earcon a no-op, including the
    ``Thinking`` background loop, so ``--silent`` actually silences
    every audio path — not just TTS."""
    from code_trip2 import earcon

    monkeypatch.setattr(audio_out, "stream_rate", lambda: 48_000)
    play_mock = MagicMock()
    monkeypatch.setattr(audio_out, "play_blocking", play_mock)
    # Sanity: when not silent, earcons hit the shared audio stream.
    earcon.set_silent(False)
    try:
        earcon.completion()
        assert play_mock.called
        play_mock.reset_mock()
        # When silent, no audio call.
        earcon.set_silent(True)
        earcon.completion()
        earcon.error()
        earcon.mode_chime("queue")
        earcon.new_task()
        earcon.thinking()
        play_mock.assert_not_called()
        # And the Thinking class declines to spawn its background thread.
        t = earcon.Thinking(interval=0.01)
        t.start()
        assert t._thread is None
        t.stop()  # idempotent
    finally:
        earcon.set_silent(False)
