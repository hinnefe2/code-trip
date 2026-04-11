#!/usr/bin/env bash
# Setup the Claude Code Stop hook for code-trip completion detection.
#
# This script writes a Stop hook into .claude/settings.json that touches a
# signal file each time Claude finishes responding.  The signal file is:
#
#   /tmp/claude-done-<window_name>
#
# where <window_name> comes from the tmux window the Claude session is running
# in.  The code-trip orchestrator watches for this file to know when Claude is
# done.
#
# Usage:
#   Run this on the REMOTE host where Claude Code runs (inside the project
#   directory, or pass a target directory as $1).
#
#   bash scripts/setup-stop-hook.sh            # uses current directory
#   bash scripts/setup-stop-hook.sh /path/to   # uses specified directory

set -euo pipefail

TARGET_DIR="${1:-.}"
SETTINGS_DIR="${TARGET_DIR}/.claude"
SETTINGS_FILE="${SETTINGS_DIR}/settings.json"

HOOK_JSON='{
  "hooks": {
    "Stop": [
      {
        "type": "command",
        "command": "touch /tmp/claude-done-$(tmux display-message -p '"'"'#{window_name}'"'"' 2>/dev/null || echo unknown)"
      }
    ]
  }
}'

mkdir -p "$SETTINGS_DIR"

if [ -f "$SETTINGS_FILE" ]; then
    echo "WARNING: $SETTINGS_FILE already exists."
    echo "Please merge the following hook config manually:"
    echo ""
    echo "$HOOK_JSON"
    echo ""
    echo "The Stop hook command should be:"
    echo '  touch /tmp/claude-done-$(tmux display-message -p '"'"'#{window_name}'"'"' 2>/dev/null || echo unknown)'
    exit 1
fi

echo "$HOOK_JSON" > "$SETTINGS_FILE"
echo "Created $SETTINGS_FILE with Stop hook for code-trip."
