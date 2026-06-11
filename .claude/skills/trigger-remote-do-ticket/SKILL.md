---
name: trigger-remote-do-ticket
description: Auto-trigger the remote /do-ticket Claude Code skill for a Linear ticket that has just transitioned into the "Todo" status. This is the LOCAL trigger skill; the actual worktree + window creation is done by the /do-ticket skill running inside the user's standing Claude session on the remote. ONLY pick this skill when source.status is exactly "Todo". Do NOT pick for tickets in "In Progress", "Backlog", "Triage", or any other status — those should reach the user as normal queue tasks.
auto-handle: true
auto-handle-kinds:
  - linear_issue
allowed-tools:
  - Bash
---

# Trigger the remote /do-ticket skill for a "Todo" Linear ticket

You're being invoked on a `linear_issue` task. Your job is to send a single `/do-ticket <ID>` slash command to the user's standing remote Claude session — which is always running in tmux window `dev:0` on the remote workspace. That remote Claude session, on receiving the slash command, runs its own `/do-ticket` skill, which is what actually creates the worktree and the per-ticket tmux window. **You do not start Claude, create windows, or do the worktree work yourself.**

1. **Gate on status.** Read `task.source.status`. If it is not exactly the string `"Todo"`, return one sentence:

   `Skipping <ID>: status is "<status>", not "Todo".`

   and stop. Do not SSH. Do not send any keys.

2. **Read the identifier.** It's `task.source.identifier` (e.g. `AI-1332`). Call it `<ID>` below.

3. **Send the slash command.** Run via `Bash` exactly one SSH call, **with the entire remote command wrapped in double quotes and the slash-command payload single-quoted inside**:

   ```
   ssh coder.dec-8 "tmux send-keys -t dev:0 '/do-ticket <ID>' Enter"
   ```

   - Replace `<ID>` with the literal identifier from step 2 (e.g. so the inner payload reads `'/do-ticket AI-1332'`).
   - `coder.dec-8` is the SSH host alias.
   - `dev:0` is the tmux session:window where the user keeps Claude always running. **Do not** create a new window, do not start Claude, do not chain commands with `\;`. One `tmux send-keys` call, done.
   - The `Enter` argument to `send-keys` is the literal token `Enter` (tmux interprets it as the Return key) — leave it outside the single quotes; it is not part of the string being typed.

   **Critical SSH-quoting note** — the nested quoting matters. Plain `ssh host tmux send-keys -t dev:0 "/do-ticket <ID>" Enter` looks right locally but doesn't work: the local shell strips the double quotes, ssh forwards `tmux send-keys -t dev:0 /do-ticket <ID> Enter` to the remote, the remote shell re-tokenizes the slash-command into two args, and tmux `send-keys` concatenates them with no space — the remote Claude then sees `/do-ticket<ID>` and errors with `Unknown command`. The double-quoted-outer, single-quoted-inner form above preserves the space across the round-trip.

4. **Return** one sentence:

   `Sent /do-ticket <ID> to dev:0.`

   If the `ssh` call fails (non-zero exit), let the error surface — do not retry. The screener will mark the task as `failed` and forward it to the user queue with the error annotated.

5. **End your reply with one of:**
   - `STATUS: handled` — the `ssh` call succeeded and the slash command was sent.
   - `STATUS: declined: <one-sentence reason>` — you skipped (status wasn't "Todo") or otherwise didn't send. The user will see the task with your reason annotated. (If the `ssh` itself errored, let the exception propagate — don't emit STATUS in that case.)

## Two important things to NOT do

- **Do not retry.** A failed send must surface, not get re-attempted. Retries risk sending the slash command twice into the standing Claude session, which would queue two `/do-ticket` runs.
- **Do not edit the ticket.** This skill is fire-and-forget; the remote `/do-ticket` skill itself is responsible for changing the Linear status to "In Progress" (or whatever convention it uses). If you change status here you'll race with the remote skill.
