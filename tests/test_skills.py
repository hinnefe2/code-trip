"""Tests for the skill manifest + allowed-tools loaders."""

from __future__ import annotations

from pathlib import Path

from code_trip2.skills import (
    SkillManifest,
    _parse_allowed_tools,
    load_skill_allowed_tools,
    load_skill_manifests,
)


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


# --- manifest loader ------------------------------------------------------


def _write_skill(root: Path, name: str, frontmatter: str, body: str = "body\n") -> None:
    (root / name).mkdir()
    (root / name / "SKILL.md").write_text(f"---\n{frontmatter}---\n{body}")


def test_load_skill_manifests_parses_all_fields(tmp_path: Path):
    _write_skill(
        tmp_path,
        "accept-invite",
        "name: accept-invite\n"
        "description: Accept a Google Calendar invite and archive the email.\n"
        "auto-handle: true\n"
        "auto-handle-kinds:\n"
        "  - email_msg\n"
        "allowed-tools:\n"
        "  - mcp__claude_ai_Gmail__unlabel_thread\n"
        "  - mcp__claude_ai_Google_Calendar__respond_to_event\n",
    )
    manifests = load_skill_manifests(tmp_path)
    assert len(manifests) == 1
    m = manifests[0]
    assert m == SkillManifest(
        name="accept-invite",
        description="Accept a Google Calendar invite and archive the email.",
        allowed_tools=(
            "mcp__claude_ai_Gmail__unlabel_thread",
            "mcp__claude_ai_Google_Calendar__respond_to_event",
        ),
        auto_handle=True,
        auto_handle_kinds=frozenset({"email_msg"}),
    )


def test_load_skill_manifests_defaults_when_keys_absent(tmp_path: Path):
    """Skills without auto-handle metadata default to ``auto_handle=False``."""
    _write_skill(
        tmp_path,
        "manual-only",
        "name: manual-only\n"
        "description: Voice-triggered skill, never auto-handles.\n"
        "allowed-tools:\n"
        "  - tool_x\n",
    )
    [m] = load_skill_manifests(tmp_path)
    assert m.auto_handle is False
    assert m.auto_handle_kinds == frozenset()
    assert m.allowed_tools == ("tool_x",)


def test_load_skill_manifests_inline_list_form(tmp_path: Path):
    """Inline ``[a, b]`` list form parses the same as block form."""
    _write_skill(
        tmp_path,
        "inline",
        "name: inline\n"
        "description: d\n"
        "auto-handle: true\n"
        "auto-handle-kinds: [email_msg, slack_msg]\n",
    )
    [m] = load_skill_manifests(tmp_path)
    assert m.auto_handle_kinds == frozenset({"email_msg", "slack_msg"})


def test_load_skill_manifests_accepts_underscore_keys(tmp_path: Path):
    """Both ``auto-handle`` and ``auto_handle`` spellings are accepted."""
    _write_skill(
        tmp_path,
        "underscored",
        "name: underscored\n"
        "description: d\n"
        "auto_handle: true\n"
        "auto_handle_kinds:\n"
        "  - email_msg\n",
    )
    [m] = load_skill_manifests(tmp_path)
    assert m.auto_handle is True
    assert m.auto_handle_kinds == frozenset({"email_msg"})


def test_load_skill_manifests_auto_handle_false_variants(tmp_path: Path):
    """Anything other than a truthy value parses as False."""
    for spelling in ("false", "no", "0", "off", "maybe"):
        sub = tmp_path / spelling
        sub.mkdir()
        (sub / "skill").mkdir()
        (sub / "skill" / "SKILL.md").write_text(
            f"---\nname: x\ndescription: d\nauto-handle: {spelling}\n---\n"
        )
        [m] = load_skill_manifests(sub)
        assert m.auto_handle is False, f"{spelling} should be False"


def test_load_skill_manifests_skips_files_without_frontmatter(tmp_path: Path):
    """A SKILL.md without frontmatter contributes no manifest."""
    _write_skill(tmp_path, "good", "name: good\ndescription: d\n")
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "SKILL.md").write_text("no frontmatter here\n")
    manifests = load_skill_manifests(tmp_path)
    assert [m.name for m in manifests] == ["good"]


def test_load_skill_manifests_missing_dir_returns_empty(tmp_path: Path):
    assert load_skill_manifests(tmp_path / "nope") == ()


def test_load_skill_manifests_name_from_directory(tmp_path: Path):
    """Directory name is canonical, even if the frontmatter ``name`` differs.

    Mirrors Claude Code's own skill-discovery convention: the slug is
    the directory name. The frontmatter name is human-readable and
    advisory.
    """
    _write_skill(
        tmp_path,
        "accept-invite",
        "name: a-different-name\ndescription: d\n",
    )
    [m] = load_skill_manifests(tmp_path)
    assert m.name == "accept-invite"


def test_load_skill_manifests_is_sorted_by_directory(tmp_path: Path):
    _write_skill(tmp_path, "zebra", "name: zebra\ndescription: d\n")
    _write_skill(tmp_path, "alpha", "name: alpha\ndescription: d\n")
    _write_skill(tmp_path, "mango", "name: mango\ndescription: d\n")
    manifests = load_skill_manifests(tmp_path)
    assert [m.name for m in manifests] == ["alpha", "mango", "zebra"]


# --- dismiss-skill metadata -----------------------------------------------


def test_load_skill_manifests_parses_dismiss_metadata(tmp_path: Path):
    _write_skill(
        tmp_path,
        "dismiss-channel-noise",
        "name: dismiss-channel-noise\n"
        "description: Drop standup-style status updates with no action item.\n"
        "dismiss: true\n"
        "dismiss-kinds:\n"
        "  - slack_msg\n",
    )
    [m] = load_skill_manifests(tmp_path)
    assert m.dismiss is True
    assert m.dismiss_kinds == frozenset({"slack_msg"})
    # Auto-handle defaults stay off for a pure dismiss skill.
    assert m.auto_handle is False
    assert m.auto_handle_kinds == frozenset()


def test_load_skill_manifests_dismiss_defaults_false(tmp_path: Path):
    _write_skill(tmp_path, "handler-only", "name: x\ndescription: d\nauto-handle: true\n")
    [m] = load_skill_manifests(tmp_path)
    assert m.dismiss is False
    assert m.dismiss_kinds == frozenset()


def test_load_skill_manifests_dismiss_kinds_inline_form(tmp_path: Path):
    _write_skill(
        tmp_path,
        "multi",
        "name: multi\ndescription: d\ndismiss: true\n"
        "dismiss-kinds: [slack_msg, email_msg]\n",
    )
    [m] = load_skill_manifests(tmp_path)
    assert m.dismiss_kinds == frozenset({"slack_msg", "email_msg"})


