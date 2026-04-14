"""Configuration loading for the code-trip orchestrator."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


class ConfigError(Exception):
    """Raised when a config file is missing, malformed, or incomplete."""


@dataclass(frozen=True)
class SSHConfig:
    host: str
    options: tuple[str, ...] = ()


@dataclass(frozen=True)
class TmuxConfig:
    session: str
    window: str
    browse_window: str = "browse"


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16_000
    device: int | str | None = None
    hotkey: str = "f13"


@dataclass(frozen=True)
class OpenAIConfig:
    api_key: str | None = None
    stt_model: str = "whisper-1"
    tts_model: str = "gpt-4o-mini-tts"
    tts_voice: str = "nova"
    tts_speed: float = 1.15
    summarizer_model: str = "gpt-4o-mini"
    verbosity: str = "detailed"


@dataclass(frozen=True)
class ClaudeConfig:
    wait_timeout: float = 300.0


@dataclass(frozen=True)
class Config:
    ssh: SSHConfig
    tmux: TmuxConfig
    audio: AudioConfig = field(default_factory=AudioConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    claude: ClaudeConfig = field(default_factory=ClaudeConfig)


def load_config(path: Path | str) -> Config:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {path}: {exc}") from exc

    try:
        ssh_raw = data["ssh"]
        tmux_raw = data["tmux"]
    except KeyError as exc:
        raise ConfigError(f"Missing required section: {exc.args[0]}") from exc

    try:
        ssh = SSHConfig(
            host=ssh_raw["host"],
            options=tuple(ssh_raw.get("options", ())),
        )
        tmux = TmuxConfig(
            session=tmux_raw["session"],
            window=tmux_raw["window"],
            browse_window=tmux_raw.get("browse_window", "browse"),
        )
    except KeyError as exc:
        raise ConfigError(f"Missing required field: {exc.args[0]}") from exc

    audio_raw = data.get("audio", {})
    audio = AudioConfig(
        sample_rate=audio_raw.get("sample_rate", 16_000),
        device=audio_raw.get("device"),
        hotkey=audio_raw.get("hotkey", "f13"),
    )

    openai_raw = data.get("openai", {})
    openai = OpenAIConfig(
        api_key=openai_raw.get("api_key") or os.environ.get("OPENAI_API_KEY"),
        stt_model=openai_raw.get("stt_model", "whisper-1"),
        tts_model=openai_raw.get("tts_model", "gpt-4o-mini-tts"),
        tts_voice=openai_raw.get("tts_voice", "nova"),
        tts_speed=openai_raw.get("tts_speed", 1.15),
        summarizer_model=openai_raw.get("summarizer_model", "gpt-4o-mini"),
        verbosity=openai_raw.get("verbosity", "detailed"),
    )

    claude_raw = data.get("claude", {})
    claude = ClaudeConfig(wait_timeout=claude_raw.get("wait_timeout", 300.0))

    return Config(ssh=ssh, tmux=tmux, audio=audio, openai=openai, claude=claude)
