"""Flat config loaded from TOML."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
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
    # Mute all audio output (TTS + earcons) — the config-file twin of
    # the ``--silent`` CLI flag. Everything else runs normally: tasks
    # queue, the TUI updates, speak calls become no-ops.
    audio_silent: bool = False
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
    # remote ticket windows (WindowProducer). Plain ssh+tmux polls — no
    # LLM billing — so they run around the clock, ungated by the
    # active-hours window below.
    window_poll_interval: float = 1.5
    window_capture_lines: int = 350
    # summarizer (cloud LLM that turns raw Claude pane output into spoken text)
    summarizer_model: str = "gpt-4o-mini"
    summarizer_max_chars: int = 600
    # slack (via the claude.ai Slack MCP — auth piggy-backs on claude CLI)
    slack_poll_interval: float = 60.0
    # Channels to watch for *every* new message (not just @-mentions).
    # Channel names without the leading #. Resolved to IDs on first poll
    # via slack_search_channels.
    slack_watch_channels: tuple[str, ...] = ()
    # email (via the claude.ai Gmail MCP — auth piggy-backs on claude CLI)
    email_poll_interval: float = 120.0
    # Gmail search syntax. ``after:<unix_ts>`` is appended automatically per
    # poll. Defaults to "unread Primary inbox, not from me" which approximates
    # an action-needed view (excludes Promotions/Updates/Social/Forums tabs).
    email_search_query: str = "in:inbox category:primary -from:me is:unread"
    email_max_results: int = 20
    # linear (via the claude.ai Linear MCP — auth piggy-backs on claude CLI)
    linear_poll_interval: float = 180.0
    # Issue ``statusType`` values eligible for the queue. Linear's enum
    # is ``triage`` / ``backlog`` / ``unstarted`` / ``started`` /
    # ``completed`` / ``canceled``; default is "anything actionable but
    # not yet done."
    linear_state_types: tuple[str, ...] = ("triage", "unstarted", "started")
    linear_max_results: int = 50
    # poll scheduling. Producers only poll between start_hour and
    # end_hour (local time) — headless claude calls bill outside plan
    # limits, so overnight polling is pure spend for an empty queue.
    # Equal values disable the window (always active). batch_window is
    # how long the MCP batcher collects concurrent calls before
    # issuing one claude session for all of them.
    poll_start_hour: int = 7
    poll_end_hour: int = 19
    mcp_batch_window: float = 2.0
    # reconcile: how often (seconds) the ReconcileProducer re-checks
    # already-queued tasks against their source and retires ones handled
    # elsewhere (e.g. an email archived/read in Gmail). Independent of the
    # additive producer polls; respects the same active-hours window.
    reconcile_interval: float = 300.0
    # Runtime override (the ``--after-hours`` CLI flag): when true, the
    # active-hours window is ignored and producers poll around the
    # clock. Not a TOML key — set per-run so late-night work gets picked
    # up without permanently disabling the overnight cost saving.
    poll_ignore_active_hours: bool = False
    # autohandle: skill-driven silent handling of producer tasks. When
    # enabled, every task a producer emits is screened against the
    # auto-handle-eligible skills in ``.claude/skills/`` before it
    # reaches the user-facing queue. ``kinds`` is the whitelist of task
    # kinds eligible for screening — empty disables auto-handling even
    # if ``enabled`` is true. All producers route through the same
    # gate, so listing ``slack_msg`` here enables both intake screening
    # of new Slack tasks AND the reconsider (re-judge on thread
    # activity) path. ``dry_run`` runs the classifier but always
    # forwards (useful for watching decisions before trusting them).
    autohandle_enabled: bool = False
    autohandle_dry_run: bool = False
    autohandle_kinds: tuple[str, ...] = ()
    # Model for the classifier (skill nomination) step. Sonnet because
    # nominating the right skill hinges on nuanced manifest descriptions
    # (e.g. "a [bot]-posted comment is bot-authored even when its text
    # quotes substantive prose"), which a small model applies unreliably.
    autohandle_classifier_model: str = "sonnet"
    # Model for the executor (skill-running) step. Sonnet because a weak
    # executor narrates a side effect as done without reliably making the
    # tool call (observed: emails marked handled but never archived).
    autohandle_executor_model: str = "sonnet"


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
    if "silent" in audio:
        kw["audio_silent"] = bool(audio["silent"])

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

    windows_cfg = data.get("windows", {})
    if "poll_interval" in windows_cfg:
        kw["window_poll_interval"] = float(windows_cfg["poll_interval"])
    if "capture_lines" in windows_cfg:
        kw["window_capture_lines"] = int(windows_cfg["capture_lines"])

    summarizer = data.get("summarizer", {})
    if "model" in summarizer:
        kw["summarizer_model"] = summarizer["model"]
    if "max_chars" in summarizer:
        kw["summarizer_max_chars"] = summarizer["max_chars"]

    slack_cfg = data.get("slack", {})
    if "poll_interval" in slack_cfg:
        kw["slack_poll_interval"] = float(slack_cfg["poll_interval"])
    if "watch_channels" in slack_cfg:
        kw["slack_watch_channels"] = tuple(
            str(c).lstrip("#") for c in slack_cfg["watch_channels"]
        )

    email_cfg = data.get("email", {})
    if "poll_interval" in email_cfg:
        kw["email_poll_interval"] = float(email_cfg["poll_interval"])
    if "search_query" in email_cfg:
        kw["email_search_query"] = str(email_cfg["search_query"])
    if "max_results" in email_cfg:
        kw["email_max_results"] = int(email_cfg["max_results"])

    linear_cfg = data.get("linear", {})
    if "poll_interval" in linear_cfg:
        kw["linear_poll_interval"] = float(linear_cfg["poll_interval"])
    if "state_types" in linear_cfg:
        kw["linear_state_types"] = tuple(
            str(s) for s in linear_cfg["state_types"]
        )
    if "max_results" in linear_cfg:
        kw["linear_max_results"] = int(linear_cfg["max_results"])

    poll_cfg = data.get("poll", {})
    if "start_hour" in poll_cfg:
        kw["poll_start_hour"] = int(poll_cfg["start_hour"])
    if "end_hour" in poll_cfg:
        kw["poll_end_hour"] = int(poll_cfg["end_hour"])
    if "batch_window" in poll_cfg:
        kw["mcp_batch_window"] = float(poll_cfg["batch_window"])

    reconcile_cfg = data.get("reconcile", {})
    if "interval" in reconcile_cfg:
        kw["reconcile_interval"] = float(reconcile_cfg["interval"])

    autohandle_cfg = data.get("autohandle", {})
    if "enabled" in autohandle_cfg:
        kw["autohandle_enabled"] = bool(autohandle_cfg["enabled"])
    if "dry_run" in autohandle_cfg:
        kw["autohandle_dry_run"] = bool(autohandle_cfg["dry_run"])
    if "kinds" in autohandle_cfg:
        kw["autohandle_kinds"] = tuple(str(k) for k in autohandle_cfg["kinds"])
    if "classifier_model" in autohandle_cfg:
        kw["autohandle_classifier_model"] = str(autohandle_cfg["classifier_model"])
    if "executor_model" in autohandle_cfg:
        kw["autohandle_executor_model"] = str(autohandle_cfg["executor_model"])

    return Config(**kw)


def polling_active(cfg, now: "datetime | None" = None) -> bool:
    """True when producer polling is inside the configured active window.

    Takes the config as a plain attribute bag (``getattr`` with
    permissive defaults) so test doubles built from SimpleNamespace —
    which predate these fields — read as "always active" rather than
    crashing or going time-dependent. Handles windows that cross
    midnight (start > end); equal hours disable the window entirely.
    """
    if getattr(cfg, "poll_ignore_active_hours", False):
        return True
    start = getattr(cfg, "poll_start_hour", None)
    end = getattr(cfg, "poll_end_hour", None)
    if start is None or end is None or start == end:
        return True
    hour = (now or datetime.now()).hour
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end
