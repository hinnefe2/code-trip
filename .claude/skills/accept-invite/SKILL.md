---
name: accept-invite
description: Accept a Google Calendar invitation email and archive it. The prototype case is any Google Calendar invitation auto-email — subject begins with "Invitation:" and the body contains "You have been invited by". Pick this skill for ALL such invites from internal/colleague senders, regardless of meeting type — 1:1s, team meetings, standups, syncs, demos, presentations, reviews, planning meetings. Internal-colleague invites default to "yes, will attend." Only skip when the sender is clearly external (an unknown company or vendor) or the email is not actually a Google Calendar invitation auto-message (e.g. a personal "want to grab coffee?" note).
auto-handle: true
auto-handle-kinds:
  - email_msg
allowed-tools:
  - mcp__claude_ai_Google_Calendar__list_events
  - mcp__claude_ai_Google_Calendar__respond_to_event
  - mcp__claude_ai_Gmail__get_thread
  - mcp__claude_ai_Gmail__unlabel_thread
---

# Accept invite + archive email

You're being invoked on an email task that contains a calendar invite. Complete it in two steps:

1. **Accept the calendar event.**
   - From the email `subject` and `body`, identify the event title and approximate date/time.
   - Use `mcp__claude_ai_Google_Calendar__list_events` with a narrow time window (e.g. ±1 day around the event date) and `fullText` set to a substring of the event title to locate the `eventId`.
   - Call `mcp__claude_ai_Google_Calendar__respond_to_event` with that `eventId` and `responseStatus="accepted"`.

2. **Archive the email.**
   - Use `mcp__claude_ai_Gmail__unlabel_thread` with the email's `thread_id` (from the task source) and `labelIds=["INBOX"]`.

Don't ask for confirmation. When both steps are done, return ONE sentence: `Accepted "<event title>" and archived the email.`

If you can't find the calendar event (e.g. the email isn't actually an invite, or the event is too far in the past/future to locate), say so in one sentence and skip the archive — don't guess.

This skill runs in two modes:
- **ACT+PTT (voice):** the user is holding the active task and has spoken an instruction. Their words are the trigger.
- **Auto-handle (screener):** no user instruction; you're invoked because the screener classifier picked this skill. Apply the same logic — the task source contains everything you need.
