---
name: archive-vendor-updates
description: Archive mass marketing-style emails from vendors / SaaS companies / services the user uses — feature announcements ("Introducing X", "What's new in Y"), product newsletters and digests, webinar / virtual-event invitations, "tips and tricks" / how-to-use-our-product emails, sales/promotional offers, release notes broadcasts. These emails are sent to many recipients, have no personalized content, and typically come from a `noreply@`, `marketing@`, `news@`, `updates@`, `team@`, `hello@`, or similar bulk sender. SKIP (so the user sees it) ANY of: a personal email from a real human at a vendor (sales rep, support engineer, founder reaching out individually), an account / billing / security notification (password reset, MFA prompt, login alert, payment receipt, invoice, subscription change, usage warning, security advisory the user must act on), a calendar invitation (those go to accept-invite), a delivery receipt or shipping notification, a thread the user has been replying in, an email addressed only to the user with content that responds to something the user did. When unsure whether an email is bulk marketing or an actionable account notice, forward.
auto-handle: true
auto-handle-kinds:
  - email_msg
allowed-tools:
  - mcp__claude_ai_Gmail__unlabel_thread
---

# Archive vendor update / marketing email

You're being invoked on an email task that looks like a mass marketing or product-update email from a vendor. Archive it.

1. **Sanity-check the body.** Confirm:
   - Sender address looks like a bulk / no-reply sender (e.g. `noreply@`, `marketing@`, `news@`, `updates@`, `team@`, `hello@`, `community@`, `events@`, `info@`).
   - Body has the shape of broadcast content: a product / feature announcement, a webinar or event invitation, a newsletter digest, a "tips and tricks" / how-to email, a sales / promotional offer, a release-notes broadcast.
   - The email is clearly sent to many recipients — no personalized references to the user's specific work, no question for the user, no requested action with a deadline.

   STOP and do not archive if you see ANY of:
   - A personal email from a real human (sales rep, support engineer, founder, account manager, etc.) — even if hosted on a vendor's domain. Look for first-person voice, a specific reference to the user, and a real reply-to address.
   - An account / billing / security notification: password reset, MFA prompt, login from a new device, payment receipt, invoice, subscription change, plan upgrade required, usage limit warning, security advisory the user must act on.
   - A calendar invitation (subject `Invitation:` + body `You have been invited by`). Forward — accept-invite handles those.
   - A shipping notification, delivery receipt, or order status update.
   - A thread the user has been actively replying in.

   In any stop case, return one sentence describing why and skip the archive.

2. **Archive the email.** Use `mcp__claude_ai_Gmail__unlabel_thread` with the email's `thread_id` (from the task source) and `labelIds=["INBOX"]`.

Don't ask for confirmation. When archived, return ONE sentence: `Archived vendor update from <sender>: <brief subject>.`

This skill runs in two modes:
- **ACT+PTT (voice):** the user is holding the active task and explicitly asked to archive it. Their intent is the trigger.
- **Auto-handle (screener):** no user instruction; you're invoked because the screener classifier picked this skill. Apply the same body-check before archiving — when in doubt, leave it.
