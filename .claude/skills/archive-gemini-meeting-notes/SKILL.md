---
name: archive-gemini-meeting-notes
description: Archive auto-generated meeting-notes emails produced by Google Meet's Gemini summarizer. These are noreply emails (typical senders include meet-recordings-noreply@google.com or noreply@google.com) whose subject contains phrasing like "Notes from your meeting with", "Meeting notes for", "Gemini notes for", or similar, and whose body is a machine-generated meeting summary, attendee list, transcript link, and bulleted action items. Pick this skill only when the entire body is auto-generated meeting-notes content — never for a human-authored email about meeting notes, follow-up emails from attendees, calendar invites (those go to accept-invite), or any email where someone has personally written commentary alongside the notes. When unsure whether the email is fully auto-generated or contains human commentary, forward to the user.
auto-handle: true
auto-handle-kinds:
  - email_msg
allowed-tools:
  - mcp__claude_ai_Gmail__unlabel_thread
---

# Archive Gemini meeting-notes email

You're being invoked on an email task that looks like an auto-generated meeting-notes email from Google Meet / Gemini. Archive it.

1. **Sanity-check the body.** Confirm:
   - Sender is a Google noreply address (e.g. `meet-recordings-noreply@google.com`, `noreply@google.com`, or similar Google-internal noreply).
   - Subject describes meeting notes / Gemini notes / a Meet summary.
   - Body is machine-generated: meeting title, attendee list, time, a summary section, action items section, transcript/recording link. No personal commentary from a human.

   STOP and do not archive if you see ANY of:
   - A human-written paragraph alongside the auto-generated content (e.g. someone forwarded the notes and added their own remarks).
   - A reply thread where a human is responding to the notes.
   - A calendar invitation auto-message (that's a different skill — return so the user sees it).

   In either stop case, return one sentence describing why and skip the archive.

2. **Archive the email.** Use `mcp__claude_ai_Gmail__unlabel_thread` with `threadId=<the email's thread_id from the task source>` and `labelIds=["INBOX"]`. The tool's required argument is **`threadId` (camelCase)** — the task source's field is `thread_id` (snake_case), so rename when passing. Passing `thread_id` makes the MCP reject with a misleading "Invalid label" error.

Don't ask for confirmation. When archived, return ONE sentence: `Archived Gemini meeting notes: <brief meeting title>.`

This skill runs in two modes:
- **ACT+PTT (voice):** the user is holding the active task and explicitly asked to archive it. Their intent is the trigger.
- **Auto-handle (screener):** no user instruction; you're invoked because the screener classifier picked this skill. Apply the same body-check before archiving — when in doubt, leave it.
