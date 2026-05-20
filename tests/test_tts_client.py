"""Unit tests for TTSClient audio-shaping (fade + silence pad)."""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from code_trip2 import tts_client
from code_trip2.tts_client import (
    _FADE_MS,
    _SILENCE_PAD_MS,
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
