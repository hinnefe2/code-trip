"""Process-wide audio output: a single shared sounddevice OutputStream.

CoreAudio (and Bluetooth audio stacks) charge a real cost every time
an app opens or closes the default output device: the device wakes,
the nominal sample rate is renegotiated, and the first ~30 ms of
audio after the switch is routinely lost. That click/chop is audible
in *whichever* app plays next, including unrelated apps like Music
or a browser tab — not just ours.

We avoid that by keeping one persistent OutputStream alive for the
life of the process, opened at the device's native rate, and routing
both TTS speech and earcon chimes through it. Callers resample their
own audio to ``stream_rate()`` before handing samples in.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

try:
    import sounddevice as sd
except OSError:
    sd = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Sane fallback when the device-rate query fails (no PortAudio, tests,
# headless CI). 48 kHz is the modern Mac/Linux/Windows default.
_FALLBACK_SR: int = 48_000

_lock = threading.Lock()
# Single-writer guard around the underlying PortAudio ring buffer.
# Earcons and TTS both write here; without this lock the writer
# pointers race and segfault (see tts_client._speak_lock for the
# original incident).
_write_lock = threading.Lock()
_stream = None
_stream_rate: int = 0


def _native_rate() -> int:
    if sd is None:
        return _FALLBACK_SR
    try:
        info = sd.query_devices(kind="output")
        rate = int(info["default_samplerate"])
    except Exception:
        logger.warning("audio_out: default-device query failed", exc_info=True)
        return _FALLBACK_SR
    return rate or _FALLBACK_SR


def stream_rate() -> int:
    """Sample rate of the shared OutputStream (opens it on first call)."""
    _ensure_stream()
    return _stream_rate or _FALLBACK_SR


def write_lock() -> threading.Lock:
    """Lock callers must hold while writing in chunks, so a long
    TTS write and a short earcon don't interleave samples into the
    single-writer PortAudio ring buffer."""
    return _write_lock


def _ensure_stream():
    global _stream, _stream_rate
    if sd is None:
        return None
    with _lock:
        if _stream is not None:
            return _stream
        rate = _native_rate()
        try:
            _stream = sd.OutputStream(
                samplerate=rate,
                channels=1,
                dtype="float32",
                latency="high",
            )
            _stream.start()
            _stream_rate = rate
        except Exception:
            logger.warning(
                "audio_out: could not open shared stream at %d Hz", rate, exc_info=True,
            )
            _stream = None
            _stream_rate = 0
        return _stream


def play_blocking(samples: np.ndarray) -> None:
    """Write mono float32 samples to the shared stream and block until drained.

    Samples must already be at ``stream_rate()``; resample upstream.
    Silently no-ops when PortAudio is unavailable.
    """
    stream = _ensure_stream()
    if stream is None:
        return
    block = samples.astype(np.float32, copy=False)
    with _write_lock:
        stream.write(block)


def resample_to_stream(samples: np.ndarray, src_rate: int) -> np.ndarray:
    """Linear resample mono samples to the shared stream's rate.

    Linear interpolation is crude but voice and short sine earcons
    tolerate it; pulling in scipy for polyphase resampling isn't
    worth the dependency.
    """
    dst_rate = stream_rate()
    if src_rate == dst_rate or samples.size == 0:
        return samples
    n_src = len(samples)
    n_dst = max(1, int(round(n_src * dst_rate / src_rate)))
    x_src = np.linspace(0.0, 1.0, n_src, endpoint=False, dtype=np.float64)
    x_dst = np.linspace(0.0, 1.0, n_dst, endpoint=False, dtype=np.float64)
    return np.interp(x_dst, x_src, samples).astype(samples.dtype, copy=False)


def shutdown() -> None:
    """Stop and close the shared stream. Call once at process exit."""
    global _stream, _stream_rate
    with _lock:
        if _stream is None:
            return
        try:
            _stream.stop()
            _stream.close()
        except Exception:
            logger.warning("audio_out: shutdown failed", exc_info=True)
        _stream = None
        _stream_rate = 0
