"""Audio recording from microphone with WAV output.

Records audio via sounddevice and writes WAV files using the stdlib wave
module.  Designed to be driven by PushToTalk but usable standalone.
"""

from __future__ import annotations

import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import sounddevice as sd
except OSError:
    sd = None  # type: ignore[assignment]  # PortAudio not installed; tests mock this

# --- Module-level defaults ------------------------------------------------

DEFAULT_SAMPLE_RATE: int = 16_000  # 16 kHz — standard for speech / Whisper
DEFAULT_CHANNELS: int = 1  # mono
DEFAULT_DTYPE: str = "int16"  # 16-bit PCM
DEFAULT_OUTPUT_DIR: str = "/tmp/code-trip-audio"


# --- Exceptions -----------------------------------------------------------


class AudioRecorderError(Exception):
    """Raised when a recording operation fails."""

    def __init__(self, message: str, *, device: int | str | None = None) -> None:
        super().__init__(message)
        self.device = device


# --- AudioRecorder --------------------------------------------------------


@dataclass
class AudioRecorder:
    """Records audio from a microphone and saves to WAV files.

    Args:
        sample_rate: Sample rate in Hz (default 16000).
        channels: Number of audio channels (default 1, mono).
        dtype: NumPy dtype string for audio samples (default "int16").
        device: sounddevice device index or name.  ``None`` uses the
            system default input device.
        output_dir: Directory to write WAV files into.
    """

    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = DEFAULT_CHANNELS
    dtype: str = DEFAULT_DTYPE
    device: int | str | None = None
    output_dir: Path = field(default_factory=lambda: Path(DEFAULT_OUTPUT_DIR))

    # Private state (not constructor args)
    _frames: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _stream: sd.InputStream | None = field(default=None, init=False, repr=False)
    _recording: bool = field(default=False, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    # --- public API -------------------------------------------------------

    def start(self) -> None:
        """Begin recording from the configured audio device.

        Raises:
            AudioRecorderError: if already recording or the device is
                unavailable.
        """
        with self._lock:
            if self._recording:
                raise AudioRecorderError(
                    "Already recording", device=self.device
                )
            self._frames = []
            try:
                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype=self.dtype,
                    device=self.device,
                    callback=self._audio_callback,
                )
                self._stream.start()
            except Exception as exc:
                raise AudioRecorderError(
                    f"Failed to open audio stream: {exc}", device=self.device
                ) from exc
            self._recording = True

    def stop(self) -> Path:
        """Stop recording and save captured audio to a WAV file.

        Returns:
            Path to the written WAV file.

        Raises:
            AudioRecorderError: if not currently recording.
        """
        with self._lock:
            if not self._recording:
                raise AudioRecorderError("Not currently recording")
            # stream.stop() is synchronous — no more callbacks after return
            self._stream.stop()
            self._stream.close()
            self._stream = None
            self._recording = False

            audio_data = np.concatenate(self._frames) if self._frames else np.empty(
                0, dtype=self.dtype
            )
            self._frames = []

        # Write outside the lock — no contention on file I/O
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        path = self.output_dir / f"recording_{timestamp}.wav"
        self._write_wav(audio_data, path)
        return path

    @property
    def is_recording(self) -> bool:
        """Whether the recorder is currently capturing audio."""
        return self._recording

    @staticmethod
    def list_devices() -> list[dict]:
        """Return available audio devices from sounddevice."""
        return [dict(d) for d in sd.query_devices()]

    # --- internals --------------------------------------------------------

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,  # noqa: ARG002
        time_info: object,  # noqa: ARG002
        status: sd.CallbackFlags,  # noqa: ARG002
    ) -> None:
        """Accumulate incoming audio frames (runs on the audio thread)."""
        self._frames.append(indata.copy())

    def _write_wav(self, audio_data: np.ndarray, path: Path) -> None:
        """Write a numpy int16 array to a WAV file using the stdlib."""
        sampwidth = np.dtype(self.dtype).itemsize
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())
