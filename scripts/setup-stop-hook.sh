#!/usr/bin/env bash
# Setup the Claude Code Stop hook for code-trip completion detection.
#
# This script writes a Stop hook into .claude/settings.json that fires
# every time Claude finishes responding. It writes TWO things:
#
#   1. /tmp/claude-done-<window_name>     (touch-file)
#      Used by code_trip2.remote.wait_done() for synchronous waits in
#      focused-mode WORK flow.
#
#   2. /tmp/claude-events/<window>-<ts>.json     (JSON event)
#      Used by code_trip2.producers.claude.ClaudeProducer in queue mode
#      to surface a new task each time a Claude session finishes.
#
# Both are kept for backward compatibility — the orchestrator can run in
# either app-mode and the hook serves both.
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

# The hook command is intentionally a single shell expression so the
# Stop hook only invokes one command. It (a) records the window name,
# (b) touches the legacy signal file, and (c) writes a JSON event file.
HOOK_CMD='WIN=$(tmux display-message -p '"'"'#{window_name}'"'"' 2>/dev/null || echo unknown); touch /tmp/claude-done-$WIN; mkdir -p /tmp/claude-events; printf '"'"'{"window":"%s","finished_at":%s}'"'"' "$WIN" "$(date +%s)" > /tmp/claude-events/$WIN-$(date +%s%N).json'

HOOK_JSON="{
  \"hooks\": {
    \"Stop\": [
      {
        \"type\": \"command\",
        \"command\": \"${HOOK_CMD}\"
      }
    ]
  }
}"

mkdir -p "$SETTINGS_DIR"

if [ -f "$SETTINGS_FILE" ]; then
    echo "WARNING: $SETTINGS_FILE already exists."
    echo "Please merge the following hook config manually:"
    echo ""
    echo "$HOOK_JSON"
    echo ""
    echo "The Stop hook command should be:"
    echo "  ${HOOK_CMD}"
    exit 1
fi

echo "$HOOK_JSON" > "$SETTINGS_FILE"
echo "Created $SETTINGS_FILE with Stop hook for code-trip."
