"""Read per-skill manifests from project-local ``.claude/skills/``.

Two things live here:

- :func:`load_skill_allowed_tools` — union of every skill's
  ``allowed-tools`` list, passed to ``claude --print --allowedTools`` so
  the CLI can use any tool any candidate skill might need. Used by the
  voice ACT+PTT path, which doesn't know ahead of time which skill
  Claude's discovery will pick.

- :func:`load_skill_manifests` — structured per-skill view: name (from
  the directory), description, allowed-tools, and the auto-handling
  metadata (``auto-handle``, ``auto-handle-kinds``) that the task
  screener uses to decide which skills can run a task without user
  intervention.

The frontmatter is YAML-shaped but we only need a few flat keys, so we
parse with regex instead of pulling in PyYAML. Both ``allowed-tools``
and ``allowed_tools`` are accepted (same for ``auto-handle`` /
``auto_handle`` and ``auto-handle-kinds`` / ``auto_handle_kinds``).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*$", re.DOTALL | re.MULTILINE)
_DESCRIPTION_RE = re.compile(r"^description\s*:\s*(.+?)\s*$", re.MULTILINE)
_AUTO_HANDLE_RE = re.compile(r"^auto[-_]handle\s*:\s*(\S+)\s*$", re.MULTILINE)
# List blocks: a key followed by lines starting with whitespace + "-".
_ALLOWED_TOOLS_BLOCK_RE = re.compile(
    r"^allowed[-_]tools\s*:\s*\n((?:[ \t]+-[ \t]*[^\n]+\n?)+)",
    re.MULTILINE,
)
_AUTO_HANDLE_KINDS_BLOCK_RE = re.compile(
    r"^auto[-_]handle[-_]kinds\s*:\s*\n((?:[ \t]+-[ \t]*[^\n]+\n?)+)",
    re.MULTILINE,
)
_LIST_ITEM_RE = re.compile(r"^[ \t]+-[ \t]*([^\n#]+?)\s*(?:#.*)?$", re.MULTILINE)
# Inline list form: ``auto-handle-kinds: [email_msg, slack_msg]``.
_INLINE_LIST_RE = re.compile(
    r"^(?P<key>auto[-_]handle[-_]kinds|allowed[-_]tools)\s*:\s*\[([^\]]*)\]\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class SkillManifest:
    """One ``.claude/skills/<name>/SKILL.md``, parsed.

    Frozen so callers can stash it in tuples / sets without surprises.
    ``allowed_tools`` and ``auto_handle_kinds`` are immutable
    collections; the screener treats them as read-only.
    """

    name: str
    description: str
    allowed_tools: tuple[str, ...] = ()
    auto_handle: bool = False
    auto_handle_kinds: frozenset[str] = field(default_factory=frozenset)


# --- frontmatter parsers --------------------------------------------------


def _extract_frontmatter(text: str) -> str | None:
    m = _FRONTMATTER_RE.search(text)
    return m.group(1) if m else None


def _parse_list_block(fm: str, block_re: re.Pattern[str], key_names: tuple[str, ...]) -> list[str]:
    """Parse either an inline ``key: [a, b]`` or a block ``key:\\n  - a\\n  - b``."""
    # Inline form first.
    for m in _INLINE_LIST_RE.finditer(fm):
        key = m.group("key").replace("_", "-")
        if key in key_names:
            return [
                item.strip().strip('"').strip("'")
                for item in m.group(2).split(",")
                if item.strip()
            ]
    block = block_re.search(fm)
    if not block:
        return []
    return [
        item.strip().strip('"').strip("'")
        for item in _LIST_ITEM_RE.findall(block.group(1))
        if item.strip()
    ]


def _parse_allowed_tools(text: str) -> list[str]:
    """Extract the ``allowed-tools`` list from SKILL.md frontmatter."""
    fm = _extract_frontmatter(text)
    if fm is None:
        return []
    return _parse_list_block(fm, _ALLOWED_TOOLS_BLOCK_RE, ("allowed-tools",))


def _parse_auto_handle_kinds(fm: str) -> frozenset[str]:
    return frozenset(_parse_list_block(fm, _AUTO_HANDLE_KINDS_BLOCK_RE, ("auto-handle-kinds",)))


def _parse_auto_handle(fm: str) -> bool:
    m = _AUTO_HANDLE_RE.search(fm)
    if not m:
        return False
    return m.group(1).strip().strip('"').strip("'").lower() in ("true", "yes", "1", "on")


def _parse_description(fm: str) -> str:
    m = _DESCRIPTION_RE.search(fm)
    return m.group(1).strip() if m else ""


def _parse_manifest(path: Path, text: str) -> SkillManifest | None:
    """Build a :class:`SkillManifest` from one SKILL.md.

    Returns ``None`` if the file has no frontmatter — that's how we
    silently skip placeholder / WIP skill directories. Name is taken
    from the parent directory, matching Claude Code's own skill-
    discovery convention.
    """
    fm = _extract_frontmatter(text)
    if fm is None:
        return None
    return SkillManifest(
        name=path.parent.name,
        description=_parse_description(fm),
        allowed_tools=tuple(_parse_list_block(fm, _ALLOWED_TOOLS_BLOCK_RE, ("allowed-tools",))),
        auto_handle=_parse_auto_handle(fm),
        auto_handle_kinds=_parse_auto_handle_kinds(fm),
    )


# --- public loaders -------------------------------------------------------


def load_skill_manifests(skills_dir: Path) -> tuple[SkillManifest, ...]:
    """Return the parsed manifest for every ``<skills_dir>/*/SKILL.md``.

    Missing directory → empty tuple. Unreadable or frontmatter-less
    files contribute nothing; the orchestrator stays usable even if a
    skill is broken. Order is sorted by directory name so callers can
    rely on deterministic iteration.
    """
    if not skills_dir.exists() or not skills_dir.is_dir():
        return ()
    out: list[SkillManifest] = []
    for path in sorted(skills_dir.glob("*/SKILL.md")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            logger.warning("Could not read %s", path, exc_info=True)
            continue
        manifest = _parse_manifest(path, text)
        if manifest is not None:
            out.append(manifest)
    return tuple(out)


def load_skill_allowed_tools(skills_dir: Path) -> tuple[str, ...]:
    """Return the sorted union of ``allowed-tools`` across all skills.

    Suitable for passing directly to :meth:`ClaudeMCPClient.run_agent`
    on the ACT+PTT path, where we don't know which skill Claude's own
    discovery will pick.
    """
    tools: set[str] = set()
    for manifest in load_skill_manifests(skills_dir):
        tools.update(manifest.allowed_tools)
    return tuple(sorted(tools))
