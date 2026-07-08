---
name: archive-linear-status-change
description: Archive Linear notification emails (sender notifications@linear.app) whose entire substance is routine issue status changes the user can already see in Linear's UI. Covers TWO shapes — (1) a single status-change notice ("changed status to Done/In Progress/In Review/Canceled/Backlog/Todo", "moved <ID> to <status>", "marked <ID> as Done/Canceled/Duplicate"), and (2) a digest/roundup email (subject like "You have N updates on Linear" / "N new updates") whose body is ENTIRELY status-change lines for one or more issues. Both are auto-generated state notices with no human-authored message directed at the user, so the email is redundant noise. Pick this skill when the email is from notifications@linear.app AND every item in it is a bare status transition with nothing else the user must act on. SKIP (so the user sees it) when the email — or, for a digest, ANY single item in it — is another kind of Linear notification: a human comment or reply on an issue, an @-mention of the user, a new assignment of an issue TO the user, a review/approval request, a due-date or SLA reminder, or any human-authored prose beyond the bare "X changed status to Y" auto-text. A digest that mixes even one comment/mention/assignment in with the status changes is NOT routine — forward the whole email. When unsure whether the sender is Linear's notification mailer, or whether the body carries anything beyond status transitions, forward to the user.
auto-handle: true
auto-handle-kinds:
  - email_msg
allowed-tools:
  - mcp__claude_ai_Gmail__get_thread
  - mcp__claude_ai_Gmail__unlabel_thread
last-updated-by: Henry Hinnefeld
last-updated-date: 2026-06-24
---

# Archive routine Linear status-change notification

You're being invoked on an email task that looks like a Linear notification whose substance is routine issue status changes. It comes in one of two shapes: a single status-change notice, or a digest ("You have N updates on Linear") that bundles several. Archive it from the user's inbox only when EVERYTHING in it is a bare status change.

1. **Confirm the sender is Linear's notification mailer.** Check `source.sender_email` — it must be `notifications@linear.app` (or another `@linear.app` automated address). If the sender is a real person at any other domain, or you can't tell it's Linear's mailer, do NOT archive. Return: `Not archived — sender is not Linear's notification mailer.`

2. **Identify the shape, and read the full body if it's a digest.** The task body is a truncated snippet of the email (`subject. snippet`). Decide which shape you have from the subject + snippet:
   - **Single notice** — the snippet's first line carries one status transition and an issue identifier (e.g. "AI-1458 changed status to Done"). The snippet is enough; do NOT fetch the thread.
   - **Digest** — subject is a roundup like "You have N updates on Linear" / "N new updates", and the body lists several per-issue items. The snippet will almost always be truncated mid-list, so you CANNOT judge a digest from the snippet alone. Fetch the full body with `mcp__claude_ai_Gmail__get_thread` (`threadId=<source.thread_id>`, `messageFormat="FULL_CONTENT"`) and read every item before deciding. If the fetch errors, do not archive — return: `Not archived — could not read the full digest body.`

   Confirm that EVERY item (the single notice, or all N items in the digest) is a bare status transition:
   - "changed status to Done / In Progress / In Review / Canceled / Backlog / Todo / Triage"
   - "moved <ID> to <status>"
   - "marked <ID> as Done / Canceled / Duplicate"

   STOP and do not archive if you see ANY of the following — and for a digest, a single offending item poisons the whole email (forward it, don't try to archive part):
   - A human comment, reply, or @-mention of the user — anything a person typed, beyond the bare "X changed status to Y" auto-text. Return: `Not archived — this involves a human comment or mention.`
   - A new assignment of an issue TO the user, a review/approval request, or a due-date / SLA reminder — these may need the user's action. Return: `Not archived — this may need the user's attention.`
   - Any item that isn't a status change at all (new issue created, project update, etc.). Return: `Not archived — contains a non-status-change update.`

   A status change to Todo is still routine noise here — the separate `trigger-remote-do-ticket` skill handles the Todo transition off the Linear API task (`kind: linear_issue`), not off this email, so archiving the email does not suppress that path.

3. **Archive the email.** Use `mcp__claude_ai_Gmail__unlabel_thread` with `threadId=<the email's thread_id from the task source>` and `labelIds=["INBOX"]`. The tool's required argument is **`threadId` (camelCase)** — the task source's field is `thread_id` (snake_case), so rename when passing. Passing `thread_id` makes the MCP reject with a misleading "Invalid label" error.

Don't ask for confirmation. When archived, return ONE sentence: `Archived Linear status-change notification: <brief subject>.`

End your reply with one of:
- `STATUS: handled` — you archived the email.
- `STATUS: declined: <one-sentence reason>` — you didn't archive. Use this whenever you hit one of the STOP conditions (wrong sender, human comment, assignment, etc.) or the tool errored. The user will see the task with your reason annotated.

This skill runs in two modes:
- **ACT+PTT (voice):** the user is holding the active task and explicitly asked to archive it. Their intent is the trigger.
- **Auto-handle (screener):** no user instruction; you're invoked because the screener classifier picked this skill. Apply the same sender + body check before archiving — when in doubt, leave it.
