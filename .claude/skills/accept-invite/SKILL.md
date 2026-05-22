---
name: accept-invite
description: Accept a Google Calendar invite that arrived as an email and archive the email. Use when the user wants to accept / RSVP yes / confirm attendance for an invite that's currently sitting in their inbox.
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
   - Use `mcp__claude_ai_Gmail__unlabel_thread` with the email's `thread_id` and `labelIds=["INBOX"]`.

Don't ask for confirmation. When both steps are done, return ONE sentence: `Accepted "<event title>" and archived the email.`

If you can't find the calendar event (e.g. the email isn't actually an invite, or the event is too far in the past/future to locate), say so in one sentence and skip the archive — don't guess.
