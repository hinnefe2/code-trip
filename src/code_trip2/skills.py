"""Read per-skill allowed-tools from project-local ``.claude/skills/``.

Claude Code's skill discovery picks which skill to run when we invoke
``claude --print`` — we don't know ahead of time which one will fire. To
keep ``--allowedTools`` meaningful, we union the ``allowed-tools`` lists
declared in every skill's ``SKILL.md`` frontmatter and pass that union
to the CLI. Net effect: Claude can use any tool that ANY candidate
skill declares it needs, but nothing else.

The frontmatter is YAML-shaped but we only need three list-of-strings
under a single key, so we parse with a regex instead of pulling in
PyYAML. Both ``allowed-tools`` and ``allowed_tools`` are accepted.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*$", re.DOTALL | re.MULTILINE)
_ALLOWED_TOOLS_BLOCK_RE = re.compile(
    r"^allowed[-_]tools\s*:\s*\n((?:[ \t]+-[ \t]*[^\n]+\n?)+)",
    re.MULTILINE,
)
_LIST_ITEM_RE = re.compile(r"^[ \t]+-[ \t]*([^\n#]+?)\s*(?:#.*)?$", re.MULTILINE)


def _parse_allowed_tools(text: str) -> list[str]:
    """Extract the ``allowed-tools`` list from a SKILL.md frontmatter."""
    fm = _FRONTMATTER_RE.search(text)
    if not fm:
        return []
    block = _ALLOWED_TOOLS_BLOCK_RE.search(fm.group(1))
    if not block:
        return []
    return [
        item.strip().strip('"').strip("'")
        for item in _LIST_ITEM_RE.findall(block.group(1))
        if item.strip()
    ]


def load_skill_allowed_tools(skills_dir: Path) -> tuple[str, ...]:
    """Return the sorted union of ``allowed-tools`` across all skills.

    Walks ``<skills_dir>/*/SKILL.md``. Missing directory or unreadable
    file = silently contributes nothing (the orchestrator stays usable
    even if a skill is broken). The result is suitable for passing
    directly to :meth:`ClaudeMCPClient.run_agent`.
    """
    tools: set[str] = set()
    if not skills_dir.exists() or not skills_dir.is_dir():
        return ()
    for path in sorted(skills_dir.glob("*/SKILL.md")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            logger.warning("Could not read %s", path, exc_info=True)
            continue
        for tool in _parse_allowed_tools(text):
            tools.add(tool)
    return tuple(sorted(tools))
