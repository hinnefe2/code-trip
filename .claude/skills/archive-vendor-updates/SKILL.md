---
name: archive-vendor-updates
description: Archive mass marketing-style emails AND unsolicited cold sales outreach. Two archive cases. (1) Bulk product-update / marketing email from a vendor or service — feature announcements ("Introducing X", "What's new in Y"), product newsletters and digests, webinar / virtual-event invitations, "tips and tricks" / how-to emails, sales / promotional offers, release-notes broadcasts — typically sent to many recipients with no personalized content, from a `noreply@`, `marketing@`, `news@`, `updates@`, `team@`, `hello@`, or similar bulk sender. (2) Cold B2B sales / prospecting outreach pitching a product, tool, or service — even when sent from a person-named address (e.g. `shane@vendor.ai`) and addressed to the user by name — when it reads as a generic template with a sales call-to-action (book a demo, hop on a call, start a trial, reply to learn more) and makes no reference to the user's actual projects or any prior relationship. SKIP (so the user sees it) ANY of: a genuine personalized human email that references the user's specific work / company by real detail, continues an existing relationship or thread, is a warm introduction, or responds to something the user actually did; an account / billing / security notification (password reset, MFA prompt, login alert, payment receipt, invoice, subscription change, usage warning, security advisory the user must act on); a calendar invitation (those go to accept-invite); a delivery receipt or shipping notification; a thread the user has been replying in. The tell for cold-sales-vs-genuine is specificity: a pitch that would read identically to a thousand other recipients is cold outreach (archive); a note that only makes sense sent to THIS user is genuine (forward). When unsure, forward.
auto-handle: true
auto-handle-kinds:
  - email_msg
allowed-tools:
  - mcp__claude_ai_Gmail__unlabel_thread
last-updated-by: Henry Hinnefeld
last-updated-date: 2026-06-24
---

# Archive vendor update / marketing / cold-sales email

You're being invoked on an email task that looks like either a mass marketing / product-update email or an unsolicited cold sales pitch. Archive it.

1. **Confirm it's one of the two archive shapes.** Either:
   - **Bulk marketing / product update:** from a bulk sender (`noreply@`, `marketing@`, `news@`, `updates@`, `team@`, `hello@`, `community@`, `events@`, `info@`), with broadcast content — a product / feature announcement, a webinar or event invitation, a newsletter digest, a "tips and tricks" / how-to email, a sales / promotional offer, a release-notes broadcast — clearly sent to many recipients with no personalized reference to the user's specific work.
   - **Cold sales / prospecting outreach:** an unsolicited pitch for a product, tool, or service — even from a person-named address (e.g. `shane@vendor.ai`) and even addressed "Hi <Name>," — that reads as a generic template, references nothing about the user's actual projects or any prior relationship, and pushes a sales call-to-action (book a demo, hop on a call, start a trial, reply to learn more).

   STOP and do not archive if you see ANY of:
   - A **genuine personal email** from a real human that references the user's specific work / company situation by real detail, continues an existing relationship or thread, is a warm introduction, or responds to something the user actually did. The tell for cold-sales-vs-genuine is **specificity**: a generic pitch that would read identically to a thousand other recipients is cold outreach (archive); a note that only makes sense sent to *this* user is genuine (skip). When the two are genuinely hard to tell apart, skip.
   - An account / billing / security notification: password reset, MFA prompt, login from a new device, payment receipt, invoice, subscription change, plan upgrade required, usage limit warning, security advisory the user must act on.
   - A calendar invitation (subject `Invitation:` + body `You have been invited by`). Forward — accept-invite handles those.
   - A shipping notification, delivery receipt, or order status update.
   - A thread the user has been actively replying in.

   In any stop case, return one sentence describing why and skip the archive.

2. **Archive the email.** Use `mcp__claude_ai_Gmail__unlabel_thread` with `threadId=<the email's thread_id from the task source>` and `labelIds=["INBOX"]`. The tool's required argument is **`threadId` (camelCase)** — the task source's field is `thread_id` (snake_case), so rename when passing. Passing `thread_id` makes the MCP reject with a misleading "Invalid label" error.

Don't ask for confirmation. When archived, return ONE sentence: `Archived <vendor update | cold sales outreach> from <sender>: <brief subject>.`

End your reply with one of:
- `STATUS: handled` — you archived the email.
- `STATUS: declined: <one-sentence reason>` — you didn't archive. Use this whenever you hit one of the STOP conditions above or the tool errored. The user will see the task with your reason annotated.

This skill runs in two modes:
- **ACT+PTT (voice):** the user is holding the active task and explicitly asked to archive it. Their intent is the trigger.
- **Auto-handle (screener):** no user instruction; you're invoked because the screener classifier picked this skill. Apply the same body-check before archiving — when in doubt, leave it.
