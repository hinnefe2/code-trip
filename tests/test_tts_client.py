"""Unit tests for TTSClient: audio-shaping + persistent OutputStream."""

from __future__ import annotations

import io
import threading
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from code_trip2 import tts_client
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
    c._stream = None
    c._stream_rate = 0
    c._stream_channels = 0
    c._stream_lock = threading.Lock()
    c._speak_lock = threading.Lock()
    c._stop_event = threading.Event()
    return c


def test_ensure_stream_creates_once_and_caches():
    c = _stub_client()
    with patch.object(tts_client, "sd") as sd_mock:
        sd_mock.OutputStream.return_value = MagicMock()
        s1 = c._ensure_stream(24_000, 1)
        s2 = c._ensure_stream(24_000, 1)
    assert s1 is s2
    assert sd_mock.OutputStream.call_count == 1


def test_ensure_stream_recreates_on_rate_change():
    c = _stub_client()
    stream_a = MagicMock()
    stream_b = MagicMock()
    with patch.object(tts_client, "sd") as sd_mock:
        sd_mock.OutputStream.side_effect = [stream_a, stream_b]
        c._ensure_stream(24_000, 1)
        c._ensure_stream(48_000, 1)
    stream_a.close.assert_called_once()
    assert sd_mock.OutputStream.call_count == 2


def test_ensure_stream_uses_high_latency():
    c = _stub_client()
    with patch.object(tts_client, "sd") as sd_mock:
        sd_mock.OutputStream.return_value = MagicMock()
        c._ensure_stream(24_000, 1)
    kwargs = sd_mock.OutputStream.call_args.kwargs
    assert kwargs["latency"] == "high"
    assert kwargs["samplerate"] == 24_000
    assert kwargs["channels"] == 1
    assert kwargs["dtype"] == "int16"


@pytest.mark.asyncio
async def test_write_in_blocks_streams_full_array():
    c = _stub_client()
    stream = MagicMock()
    samples = np.zeros(_WRITE_BLOCK_FRAMES * 3 + 100, dtype=np.int16)
    await c._write_in_blocks(stream, samples)
    # Expect 4 writes: 3 full blocks + a remainder.
    assert stream.write.call_count == 4
    total_written = sum(len(call.args[0]) for call in stream.write.call_args_list)
    assert total_written == len(samples)


@pytest.mark.asyncio
async def test_write_in_blocks_stops_when_event_set():
    c = _stub_client()
    stream = MagicMock()
    samples = np.zeros(_WRITE_BLOCK_FRAMES * 5, dtype=np.int16)
    # Set the stop event before write 3.
    counter = {"calls": 0}
    def fake_write(_):
        counter["calls"] += 1
        if counter["calls"] == 2:
            c._stop_event.set()
    stream.write.side_effect = fake_write
    await c._write_in_blocks(stream, samples)
    # Stop should be honored within one extra block (the loop checks
    # at the top of each iteration).
    assert stream.write.call_count == 2


def test_stop_does_not_close_stream():
    """The stream must stay alive across stop()/speak() cycles so the
    audio device doesn't get reinitialized on every utterance."""
    c = _stub_client()
    c._stream = MagicMock()
    c._stream_rate = 24_000
    c._stream_channels = 1
    c.stop()
    c._stream.close.assert_not_called()
    assert c._stream is not None


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

    fake_sd = MagicMock()
    monkeypatch.setattr(earcon, "sd", fake_sd)
    # Sanity: when not silent, _play hits sounddevice.
    earcon.set_silent(False)
    try:
        earcon.completion()
        assert fake_sd.play.called
        fake_sd.play.reset_mock()
        # When silent, no audio call.
        earcon.set_silent(True)
        earcon.completion()
        earcon.error()
        earcon.mode_chime("queue")
        earcon.new_task()
        earcon.thinking()
        fake_sd.play.assert_not_called()
        # And the Thinking class declines to spawn its background thread.
        t = earcon.Thinking(interval=0.01)
        t.start()
        assert t._thread is None
        t.stop()  # idempotent
    finally:
        earcon.set_silent(False)
