---
name: dismiss-resolved-slack-thread
description: Pick this skill ONLY when source.messages shows that Henry (the user) has just sent a message (the last entry has is_self == true) AND the thread reads as wrapped — Henry's last message is a conclusion, acknowledgment, answer, or thanks; there is no outstanding @-mention of him after his last reply; no one has asked him a question that he hasn't answered. Do NOT pick if anyone else replied after Henry, if there's an unanswered direct question to Henry mid-thread, if Henry's last message is itself a question, or if the thread is ambiguous. When in doubt, do not pick — the user can still dismiss manually.
dismiss: true
dismiss-kinds:
  - slack_msg
---

# Mark a Slack thread done because Henry has wrapped it

This skill carries no executor body — it's a pure classifier judgment.
See the description above for the criteria; the screener doesn't read
this body, so any text here is documentation only.
