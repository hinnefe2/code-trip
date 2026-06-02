"""Text-to-speech via the OpenAI TTS API.

Wraps the OpenAI ``/v1/audio/speech`` endpoint behind a simple
``speak(text) -> None`` interface.  Audio is requested as WAV, decoded
via the stdlib ``wave`` module, and played through ``sounddevice``
(cross-platform: Linux/macOS/Windows).  :meth:`stop` interrupts
playback mid-sentence.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import threading
import wave
from dataclasses import dataclass, field

import numpy as np
import openai
from openai import APIConnectionError, APIError, AsyncOpenAI, AuthenticationError

from code_trip2 import audio_out

logger = logging.getLogger(__name__)

# --- Module-level defaults ------------------------------------------------

DEFAULT_MODEL: str = "gpt-4o-mini-tts"
DEFAULT_VOICE: str = "nova"
DEFAULT_SPEED: float = 1.15
DEFAULT_FORMAT: str = "wav"
ENV_API_KEY: str = "OPENAI_API_KEY"

# Pop/click suppression. The OpenAI WAV doesn't always start or end at
# zero amplitude, and the audio device often goes idle between
# utterances (especially on Bluetooth earbuds), so the first sample of
# playback hits an asleep DAC and we hear a click. A short fade + a
# silence cushion lets the device wake up before the speech itself
# plays.
_FADE_MS: int = 12
_SILENCE_PAD_MS: int = 40

# Block size for writing into the shared OutputStream. Small enough
# that stop() is honored within ~85 ms at 48 kHz (~170 ms at 24 kHz),
# large enough that we're not paying GIL overhead per sample.
_WRITE_BLOCK_FRAMES: int = 4096


# --- Exceptions -----------------------------------------------------------


class TTSClientError(Exception):
    """Raised when a text-to-speech operation fails."""


# --- SilentTTSClient ------------------------------------------------------


class SilentTTSClient:
    """No-op stand-in for :class:`TTSClient`.

    Wired in by ``--silent`` so the orchestrator runs without spoken
    audio. Matches the duck-typed shape ``modes`` and ``dispatch``
    rely on (``speak``, ``stop``, ``is_playing``); ``speak`` returns
    immediately so chunked-playback loops drain in a single tick and
    queue-mode auto-announce behaves as if speech just finished.
    """

    async def speak(self, text: str) -> None:
        return

    def stop(self) -> None:
        return

    def is_playing(self) -> bool:
        return False


# --- TTSClient ------------------------------------------------------------


@dataclass
class TTSClient:
    """Synthesizes speech from text via the OpenAI TTS API.

    Args:
        api_key: OpenAI API key.  If ``None``, reads from the
            ``OPENAI_API_KEY`` environment variable.
        model: TTS model name (default ``"gpt-4o-mini-tts"``).
        voice: Voice preset (default ``"nova"``).
        speed: Playback speed multiplier (default ``1.15``).
    """

    api_key: str | None = None
    model: str = DEFAULT_MODEL
    voice: str = DEFAULT_VOICE
    speed: float = DEFAULT_SPEED

    _client: AsyncOpenAI = field(default=None, init=False, repr=False)  # type: ignore[assignment]
    _playing: bool = field(default=False, init=False, repr=False)
    # Serializes the body of speak() so two concurrent callers can't both
    # write into the shared OutputStream's ring buffer at once.
    # PortAudio's PaUtil_WriteRingBuffer is single-writer; concurrent
    # writes corrupt the head/tail pointers and segfault the process
    # (DiagnosticReports 2026-05-22 onward: PaUtil_WriteRingBuffer + 180,
    # KERN_INVALID_ADDRESS).
    _speak_lock: asyncio.Lock = field(
        default_factory=asyncio.Lock, init=False, repr=False
    )
    # threading.Event (not asyncio.Event) because stop() must be callable
    # from any thread (e.g. the macropad's pynput listener) and threading
    # primitives are cross-thread safe by default. The block writer inside
    # _write_in_blocks reads it from an executor thread.
    _stop_event: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False
    )

    def __post_init__(self) -> None:
        resolved_key = self.api_key or os.environ.get(ENV_API_KEY)
        if not resolved_key:
            raise TTSClientError(
                f"No API key: set {ENV_API_KEY} or pass api_key"
            )
        self._client = AsyncOpenAI(api_key=resolved_key)

    # --- public API -------------------------------------------------------

    async def speak(self, text: str) -> None:
        """Synthesize ``text`` and play it through the system audio.

        Awaits until playback finishes. Call :meth:`stop` (sync, from any
        thread) to interrupt mid-sentence — latency is one write block
        (~170 ms).

        Concurrent callers serialize on ``_speak_lock`` because
        PortAudio's ring buffer is single-writer; without serialization
        two coroutines' executor-thread writes would corrupt it and
        segfault the process.

        Raises:
            TTSClientError: if the text is empty or the API call fails.
        """
        text = text.strip()
        if not text:
            raise TTSClientError("Empty text cannot be synthesized")

        async with self._speak_lock:
            try:
                response = await self._client.audio.speech.create(
                    model=self.model,
                    voice=self.voice,
                    input=text,
                    speed=self.speed,
                    response_format=DEFAULT_FORMAT,
                )
            except AuthenticationError as exc:
                raise TTSClientError(f"Authentication failed: {exc}") from exc
            except APIConnectionError as exc:
                raise TTSClientError(f"Network error: {exc}") from exc
            except APIError as exc:
                raise TTSClientError(f"API error: {exc}") from exc

            audio: bytes = response.content
            if not audio:
                raise TTSClientError("Empty audio returned")

            samples, sample_rate = _decode_wav(audio)

            self._stop_event.clear()
            self._playing = True
            try:
                await self._write_in_blocks(samples, sample_rate)
            finally:
                self._playing = False

    def stop(self) -> None:
        """Interrupt any in-progress playback.

        Sync and thread-safe — sets a stop flag that the block writer
        loop checks between writes. The shared audio stream is left
        open so the next ``speak()`` doesn't pay the reopen cost.
        """
        self._stop_event.set()

    def is_playing(self) -> bool:
        """True while ``speak()`` is blocked on playback."""
        return self._playing

    # --- internals --------------------------------------------------------

    async def _write_in_blocks(self, samples: np.ndarray, src_rate: int) -> None:
        """Resample to the shared stream's rate, then push samples in
        small blocks so we can honor ``stop()`` without aborting playback.

        Each block is written from an executor thread so the event loop
        stays responsive. The shared OutputStream's write lock keeps the
        TTS writer from interleaving with earcons mid-utterance.
        """
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        float_samples = samples.astype(np.float32) / 32768.0
        resampled = audio_out.resample_to_stream(float_samples, src_rate)
        total = len(resampled)
        for start in range(0, total, _WRITE_BLOCK_FRAMES):
            if self._stop_event.is_set():
                return
            end = min(start + _WRITE_BLOCK_FRAMES, total)
            await asyncio.to_thread(audio_out.play_blocking, resampled[start:end])


# --- helpers --------------------------------------------------------------


def _decode_wav(audio: bytes) -> tuple[np.ndarray, int]:
    """Decode WAV bytes into a numpy int16 array and sample rate.

    Applies a small fade-in/out and silence pad to suppress the DAC-wakeup
    click that's especially noticeable for short utterances ("Queue is
    empty.") and on Bluetooth audio paths.
    """
    with wave.open(io.BytesIO(audio), "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())
    samples = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels)
    samples = _shape_samples(samples, sample_rate)
    return samples, sample_rate


def _shape_samples(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply fade-in/out and prepend/append silence so playback doesn't pop."""
    if samples.size == 0:
        return samples
    n_fade = int(sample_rate * _FADE_MS / 1000)
    n_pad = int(sample_rate * _SILENCE_PAD_MS / 1000)
    n = len(samples)
    if 2 * n_fade < n:
        env = np.ones(n, dtype=np.float32)
        env[:n_fade] = np.linspace(0.0, 1.0, n_fade, dtype=np.float32)
        env[-n_fade:] = np.linspace(1.0, 0.0, n_fade, dtype=np.float32)
        if samples.ndim == 1:
            faded = (samples.astype(np.float32) * env).astype(np.int16)
        else:
            faded = (samples.astype(np.float32) * env[:, np.newaxis]).astype(np.int16)
    else:
        faded = samples
    if n_pad > 0:
        if samples.ndim == 1:
            silence = np.zeros(n_pad, dtype=np.int16)
        else:
            silence = np.zeros((n_pad, samples.shape[1]), dtype=np.int16)
        return np.concatenate([silence, faded, silence])
    return faded
