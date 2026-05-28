---
name: accept-invite
description: Handle Google Calendar invitation emails — accept and archive most, or archive without RSVPing for office-hours-style drop-in meetings. The prototype case is any Google Calendar invitation auto-email (subject begins with "Invitation:", body contains "You have been invited by"). Pick this skill for ALL such invites from internal/colleague senders, regardless of meeting type — 1:1s, team meetings, standups, syncs, demos, presentations, reviews, planning meetings, office hours, drop-ins. Internal-colleague invites default to "yes, will attend" EXCEPT for office-hours-style meetings (event title or body explicitly mentions "office hours", "office hour", or describes a drop-in / optional-attendance format with no specific agenda for the user) — those are archived without RSVPing because attendance is not expected. Only skip the skill entirely when the sender is clearly external (an unknown company or vendor) or the email is not actually a Google Calendar invitation auto-message.
auto-handle: true
auto-handle-kinds:
  - email_msg
allowed-tools:
  - mcp__claude_ai_Google_Calendar__list_events
  - mcp__claude_ai_Google_Calendar__respond_to_event
  - mcp__claude_ai_Gmail__get_thread
  - mcp__claude_ai_Gmail__unlabel_thread
---

# Handle calendar invite + archive email

You're being invoked on an email task that contains a calendar invite. Decide which of two flows applies, then run it.

**Office-hours flow** (archive without RSVPing). Triggers:
- Event title or body explicitly mentions "office hours" or "office hour" (case-insensitive).
- OR the description makes clear it's a drop-in / optional-attendance format with no specific agenda for the user (e.g. "ask me anything", "open to the team", "no agenda — bring questions").

In this flow, **skip the calendar accept entirely** — attendance is not expected and RSVPing yes would imply a commitment the user hasn't made.
1. Use `mcp__claude_ai_Gmail__unlabel_thread` with the email's `thread_id` and `labelIds=["INBOX"]` to archive.
2. Return ONE sentence: `Archived office-hours invite "<event title>" without RSVPing.`

**Default flow** (accept and archive). For every other invite type — 1:1s, team meetings, standups, syncs, demos, presentations, reviews, planning meetings:

1. **Accept the calendar event.**
   - From the email `subject` and `body`, identify the event title and approximate date/time.
   - Use `mcp__claude_ai_Google_Calendar__list_events` with a narrow time window (e.g. ±1 day around the event date) and `fullText` set to a substring of the event title to locate the `eventId`.
   - Call `mcp__claude_ai_Google_Calendar__respond_to_event` with that `eventId` and `responseStatus="accepted"`.

2. **Archive the email.**
   - Use `mcp__claude_ai_Gmail__unlabel_thread` with the email's `thread_id` (from the task source) and `labelIds=["INBOX"]`.

Return ONE sentence: `Accepted "<event title>" and archived the email.`

Don't ask for confirmation. If you can't find the calendar event in the default flow (e.g. the event is too far in the past/future to locate), say so in one sentence and skip the archive — don't guess. The office-hours flow doesn't need to find the event since it never RSVPs.

This skill runs in two modes:
- **ACT+PTT (voice):** the user is holding the active task and has spoken an instruction. Their words are the trigger.
- **Auto-handle (screener):** no user instruction; you're invoked because the screener classifier picked this skill. Apply the same logic — the task source contains everything you need.
