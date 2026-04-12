"""Unit tests for config loading."""

from __future__ import annotations

import pytest

from code_trip.config import ConfigError, load_config


def _write(tmp_path, body: str):
    path = tmp_path / "config.toml"
    path.write_text(body)
    return path


def test_load_minimal(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    path = _write(
        tmp_path,
        """
[ssh]
host = "me@host"

[tmux]
session = "claude"
window = "main"
""",
    )
    cfg = load_config(path)
    assert cfg.ssh.host == "me@host"
    assert cfg.ssh.options == ()
    assert cfg.tmux.session == "claude"
    assert cfg.tmux.window == "main"
    assert cfg.audio.hotkey == "f13"
    assert cfg.audio.sample_rate == 16_000
    assert cfg.openai.api_key is None
    assert cfg.openai.verbosity == "detailed"
    assert cfg.claude.wait_timeout == 300.0


def test_env_var_fallback_for_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    path = _write(
        tmp_path,
        """
[ssh]
host = "h"
[tmux]
session = "s"
window = "w"
""",
    )
    cfg = load_config(path)
    assert cfg.openai.api_key == "sk-env"


def test_explicit_api_key_overrides_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    path = _write(
        tmp_path,
        """
[ssh]
host = "h"
[tmux]
session = "s"
window = "w"
[openai]
api_key = "sk-file"
""",
    )
    cfg = load_config(path)
    assert cfg.openai.api_key == "sk-file"


def test_missing_section_raises(tmp_path):
    path = _write(tmp_path, '[ssh]\nhost = "h"\n')
    with pytest.raises(ConfigError, match="tmux"):
        load_config(path)


def test_missing_field_raises(tmp_path):
    path = _write(
        tmp_path,
        """
[ssh]
host = "h"
[tmux]
session = "s"
""",
    )
    with pytest.raises(ConfigError, match="window"):
        load_config(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")


def test_invalid_toml_raises(tmp_path):
    path = _write(tmp_path, "not = valid = toml")
    with pytest.raises(ConfigError, match="Invalid TOML"):
        load_config(path)


def test_ssh_options_parsed(tmp_path):
    path = _write(
        tmp_path,
        """
[ssh]
host = "h"
options = ["-p", "2222"]
[tmux]
session = "s"
window = "w"
""",
    )
    cfg = load_config(path)
    assert cfg.ssh.options == ("-p", "2222")
