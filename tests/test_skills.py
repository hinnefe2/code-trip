"""Tests for the skill allowed-tools loader."""

from __future__ import annotations

from pathlib import Path

from code_trip2.skills import _parse_allowed_tools, load_skill_allowed_tools


# --- frontmatter parser ---------------------------------------------------


def test_parse_allowed_tools_basic():
    text = (
        "---\n"
        "name: foo\n"
        "description: bar\n"
        "allowed-tools:\n"
        "  - mcp__server__tool_one\n"
        "  - mcp__server__tool_two\n"
        "---\n"
        "\n"
        "Body content.\n"
    )
    assert _parse_allowed_tools(text) == [
        "mcp__server__tool_one",
        "mcp__server__tool_two",
    ]


def test_parse_allowed_tools_accepts_underscore_variant():
    text = (
        "---\n"
        "allowed_tools:\n"
        "  - tool_a\n"
        "---\n"
    )
    assert _parse_allowed_tools(text) == ["tool_a"]


def test_parse_allowed_tools_strips_quotes():
    text = (
        "---\n"
        "allowed-tools:\n"
        '  - "tool_quoted"\n'
        "  - 'tool_single'\n"
        "---\n"
    )
    assert _parse_allowed_tools(text) == ["tool_quoted", "tool_single"]


def test_parse_allowed_tools_returns_empty_when_missing():
    text = (
        "---\n"
        "name: foo\n"
        "description: no tools\n"
        "---\n"
        "Body.\n"
    )
    assert _parse_allowed_tools(text) == []


def test_parse_allowed_tools_returns_empty_when_no_frontmatter():
    assert _parse_allowed_tools("Just a body, no frontmatter.\n") == []


# --- directory loader -----------------------------------------------------


def test_load_skill_allowed_tools_unions_across_skills(tmp_path: Path):
    (tmp_path / "skill-a").mkdir()
    (tmp_path / "skill-a" / "SKILL.md").write_text(
        "---\n"
        "allowed-tools:\n"
        "  - tool_one\n"
        "  - tool_two\n"
        "---\n"
        "body a\n"
    )
    (tmp_path / "skill-b").mkdir()
    (tmp_path / "skill-b" / "SKILL.md").write_text(
        "---\n"
        "allowed-tools:\n"
        "  - tool_two\n"  # duplicate
        "  - tool_three\n"
        "---\n"
        "body b\n"
    )
    out = load_skill_allowed_tools(tmp_path)
    assert out == ("tool_one", "tool_three", "tool_two")  # sorted, deduped


def test_load_skill_allowed_tools_missing_dir_returns_empty(tmp_path: Path):
    assert load_skill_allowed_tools(tmp_path / "nope") == ()


def test_load_skill_allowed_tools_skips_unreadable_skill(tmp_path: Path):
    """One broken skill shouldn't kill the orchestrator."""
    (tmp_path / "good").mkdir()
    (tmp_path / "good" / "SKILL.md").write_text(
        "---\nallowed-tools:\n  - good_tool\n---\n"
    )
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "SKILL.md").write_text(
        "no frontmatter, no tools\n"
    )
    assert load_skill_allowed_tools(tmp_path) == ("good_tool",)


def test_load_skill_allowed_tools_handles_skills_without_md(tmp_path: Path):
    """A skill directory missing its SKILL.md is silently ignored."""
    (tmp_path / "incomplete").mkdir()
    # No SKILL.md inside.
    assert load_skill_allowed_tools(tmp_path) == ()
