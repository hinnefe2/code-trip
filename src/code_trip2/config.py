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
    # ssh + tmux (only needed for modes that drive a remote Claude)
    ssh_host: str = ""
    ssh_options: tuple[str, ...] = ()
    tmux_session: str = "main"
    work_window: str = "work"
    linear_window: str = "linear"
    # audio
    sample_rate: int = 16_000
    audio_device: int | str | None = None
    # macropad — physical layout left-to-right: PTT, YES, NO, ACT, NAV
    ptt_key: str = "f13"
    yes_key: str = "f14"
    no_key: str = "f15"
    act_key: str = "f16"
    nav_key: str = "f17"
    app_cycle: tuple[str, ...] = ("kitty", "Google Chrome", "Slack")
    # Apps where voice routes to the active tmux pane (talk-to-Claude). Anything
    # else falls through to DICTATE-style paste into the focused app.
    terminal_apps: tuple[str, ...] = ("kitty",)
    # stt
    stt_provider: str = "openai"        # "openai" | "local"
    stt_local_hotkey: str = "delete"    # pynput Key name forwarded while PTT is held
    # openai
    api_key: str | None = None
    stt_model: str = "whisper-1"
    tts_model: str = "gpt-4o-mini-tts"
    tts_voice: str = "nova"
    tts_speed: float = 1.15
    # claude
    wait_timeout: float = 300.0
    # summarizer (cloud LLM that turns raw Claude pane output into spoken text)
    summarizer_model: str = "gpt-4o-mini"
    summarizer_max_chars: int = 600
    # task-queue
    startup_mode: str = "focused"           # "queue" | "focused"
    # mcp producers (skeletons in v1; need a local MCP server command to start)
    slack_channels: tuple[str, ...] = ()
    slack_mcp_command: str = ""
    slack_mcp_args: tuple[str, ...] = ()
    linear_mcp_command: str = ""
    linear_mcp_args: tuple[str, ...] = ()


def _select(src: dict, *fields: str) -> dict:
    """Pick fields from src only if present. Used when TOML keys == dataclass field names."""
    return {k: src[k] for k in fields if k in src}


def load_config(path: Path | str) -> Config:
    """Load a TOML config, letting the Config dataclass supply any missing defaults."""
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
    stt = data.get("stt", {})
    stt_local = stt.get("local", {})
    openai_ = data.get("openai", {})
    claude = data.get("claude", {})

    kw: dict[str, object] = {}

    # ssh / tmux / audio — TOML keys differ from field names
    if "host" in ssh:
        kw["ssh_host"] = ssh["host"]
    if "options" in ssh:
        kw["ssh_options"] = tuple(ssh["options"])
    if "session" in tmux:
        kw["tmux_session"] = tmux["session"]
    # "window" is a legacy alias for work_window
    if "work_window" in tmux:
        kw["work_window"] = tmux["work_window"]
    elif "window" in tmux:
        kw["work_window"] = tmux["window"]
    if "linear_window" in tmux:
        kw["linear_window"] = tmux["linear_window"]
    if "sample_rate" in audio:
        kw["sample_rate"] = audio["sample_rate"]
    if "device" in audio:
        kw["audio_device"] = audio["device"]

    # macropad — TOML keys match field names
    kw.update(_select(macropad, "ptt_key", "yes_key", "no_key", "act_key", "nav_key"))
    if "app_cycle" in macropad:
        kw["app_cycle"] = tuple(macropad["app_cycle"])
    if "terminal_apps" in macropad:
        kw["terminal_apps"] = tuple(macropad["terminal_apps"])

    # stt — renames
    if "provider" in stt:
        kw["stt_provider"] = stt["provider"]
    if "hotkey" in stt_local:
        kw["stt_local_hotkey"] = stt_local["hotkey"]

    # openai — mostly direct match; api_key has env-var fallback
    kw.update(_select(openai_, "stt_model", "tts_model", "tts_voice", "tts_speed"))
    api_key = openai_.get("api_key") or os.environ.get("OPENAI_API_KEY")
    if api_key:
        kw["api_key"] = api_key

    # claude
    if "wait_timeout" in claude:
        kw["wait_timeout"] = claude["wait_timeout"]

    summarizer = data.get("summarizer", {})
    if "model" in summarizer:
        kw["summarizer_model"] = summarizer["model"]
    if "max_chars" in summarizer:
        kw["summarizer_max_chars"] = summarizer["max_chars"]

    # task queue
    queue_cfg = data.get("queue", {})
    if "startup_mode" in queue_cfg:
        kw["startup_mode"] = queue_cfg["startup_mode"]

    slack_cfg = data.get("slack", {})
    if "channels" in slack_cfg:
        kw["slack_channels"] = tuple(slack_cfg["channels"])
    if "mcp_command" in slack_cfg:
        kw["slack_mcp_command"] = slack_cfg["mcp_command"]
    if "mcp_args" in slack_cfg:
        kw["slack_mcp_args"] = tuple(slack_cfg["mcp_args"])

    linear_cfg = data.get("linear", {})
    if "mcp_command" in linear_cfg:
        kw["linear_mcp_command"] = linear_cfg["mcp_command"]
    if "mcp_args" in linear_cfg:
        kw["linear_mcp_args"] = tuple(linear_cfg["mcp_args"])

    return Config(**kw)
