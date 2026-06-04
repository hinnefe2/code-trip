---
name: trigger-remote-do-ticket
description: Auto-spawn a remote /do-ticket Claude session for a Linear ticket that has just transitioned into the "Todo" status. This is the LOCAL trigger skill; the actual work is done by the /do-ticket Claude Code skill installed on the remote workspace. ONLY pick this skill when source.status is exactly "Todo". Do NOT pick for tickets in "In Progress", "Backlog", "Triage", or any other status — those should reach the user as normal queue tasks.
auto-handle: true
auto-handle-kinds:
  - linear_issue
allowed-tools:
  - Bash
---

# Trigger the remote /do-ticket skill for a "Todo" Linear ticket

You're being invoked on a `linear_issue` task. Your job is to SSH to the user's remote workspace, open a fresh tmux window in the `dev` session, start Claude in it, and run the remote `/do-ticket` skill against the ticket identifier. **You do not do the work yourself** — you just kick off the remote skill that does.

1. **Gate on status.** Read `task.source.status`. If it is not exactly the string `"Todo"`, return one sentence:

   `Skipping <ID>: status is "<status>", not "Todo".`

   and stop. Do not SSH. Do not spawn anything.

2. **Read the identifier.** It's `task.source.identifier` (e.g. `ENGAGE-1234`). Call it `<ID>` below.

3. **Spawn the remote window.** Run via `Bash`:

   ```
   ssh coder.dec-8 'tmux new-window -t dev -n <ID> \; send-keys -t dev:<ID> "claude" Enter \; run-shell "sleep 2" \; send-keys -t dev:<ID> "/do-ticket <ID>" Enter'
   ```

   - Replace `<ID>` everywhere with the actual ticket identifier from step 2.
   - `coder.dec-8` is the SSH host alias (see `~/.ssh/config`); ControlMaster keeps the per-call cost low.
   - `dev` is the tmux session name (matches `[tmux] session` in the user's config; the same session the user works in day-to-day, with one window per active ticket).
   - The `\;` separators are tmux command separators; quote the whole tmux invocation in single quotes so the local shell doesn't eat them.
   - The `sleep 2` between `claude` and `/do-ticket` gives the Claude TUI time to come up before we type the slash command.

4. **Return** one sentence:

   `Spawned /do-ticket on dev:<ID>.`

   If the `ssh` call fails (non-zero exit), let the error surface — do not retry. The screener will mark the task as `failed` and forward it to the user queue with the error annotated, which is the right behavior (user sees the ticket and can take over manually).

## Two important things to NOT do

- **Do not retry.** A failed spawn must surface, not get re-attempted. Mid-flight retries risk creating duplicate tmux windows.
- **Do not edit the ticket.** This skill is fire-and-forget; the remote `/do-ticket` skill itself is responsible for changing the Linear status to "In Progress" (or whatever convention it uses). If you change status here you'll race with the remote skill.
