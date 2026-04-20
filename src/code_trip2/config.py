"""Flat config loaded from TOML."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class Config:
    # ssh + tmux
    ssh_host: str
    ssh_options: tuple[str, ...] = ()
    tmux_session: str = "main"
    work_window: str = "work"
    linear_window: str = "linear"
    # audio
    sample_rate: int = 16_000
    audio_device: int | str | None = None
    # macropad
    ptt_key: str = "f13"
    act_key: str = "f14"
    yes_key: str = "f15"
    no_key: str = "f16"
    nav_key: str = "f17"
    app_cycle: tuple[str, ...] = ("kitty", "Google Chrome", "Slack")
    # openai
    api_key: str | None = None
    stt_model: str = "whisper-1"
    tts_model: str = "gpt-4o-mini-tts"
    tts_voice: str = "nova"
    tts_speed: float = 1.15
    # claude
    wait_timeout: float = 300.0


def load_config(path: Path | str) -> Config:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {path}: {exc}") from exc

    ssh = data.get("ssh", {})
    tmux = data.get("tmux", {})
    audio = data.get("audio", {})
    macropad = data.get("macropad", {})
    openai = data.get("openai", {})
    claude = data.get("claude", {})

    if "host" not in ssh:
        raise ConfigError("Missing required field: ssh.host")

    return Config(
        ssh_host=ssh["host"],
        ssh_options=tuple(ssh.get("options", ())),
        tmux_session=tmux.get("session", "main"),
        work_window=tmux.get("work_window", tmux.get("window", "work")),
        linear_window=tmux.get("linear_window", "linear"),
        sample_rate=audio.get("sample_rate", 16_000),
        audio_device=audio.get("device"),
        ptt_key=macropad.get("ptt_key", "f13"),
        act_key=macropad.get("act_key", "f14"),
        yes_key=macropad.get("yes_key", "f15"),
        no_key=macropad.get("no_key", "f16"),
        nav_key=macropad.get("nav_key", "f17"),
        app_cycle=tuple(macropad.get("app_cycle", ("kitty", "Google Chrome", "Slack"))),
        api_key=openai.get("api_key") or os.environ.get("OPENAI_API_KEY"),
        stt_model=openai.get("stt_model", "whisper-1"),
        tts_model=openai.get("tts_model", "gpt-4o-mini-tts"),
        tts_voice=openai.get("tts_voice", "nova"),
        tts_speed=openai.get("tts_speed", 1.15),
        wait_timeout=claude.get("wait_timeout", 300.0),
    )
