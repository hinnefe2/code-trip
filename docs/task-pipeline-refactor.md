# Task Ingestion & Lifecycle Pipeline Refactor

Status: implemented 2026-07-08 (all five migration steps)
Author: design review (Fable architect agent), captured 2026-07-08
Scope: producers → screener → queue pipeline — production, screening/auto-handle,
dedup/collapse, queue entry, retraction/reconciliation, re-judgment.

## Motivation

A user hit three identical pending Linear tasks for one ticket (TRI-240). Root cause
analysis showed this is not a one-off but a structural gap, on top of several
overlapping identity/dedup/lifecycle mechanisms that have accreted. This plan fixes
the bug class and consolidates the cruft into a single coherent model — deliberately
small (single-developer voice app, one asyncio loop, queues of tens of tasks).

## 1. Current-state map

Pipeline (as wired in `main.py`):

```
LinearProducer ──┐ (new tasks)                       ┌──> queue.add ──> TaskQueue ──> ranked() ──> voice loop
EmailProducer ───┤──> intake_q ──> screener loop ────┤    (forward/failed)
                 │    (serial,     classify+execute  └──> nothing (handled/dismissed)
                 │     backlogs)   claude --print,        + follow_ups ──> queue.add
                 │                  up to 300s each)
SlackProducer ───┼──> queue.add DIRECTLY (no intake!)  (slack.py:795)
ClaudeProducer ──┼──> queue.add directly
ManualProducer ──┘
SlackProducer ──> reconsider_q ──> screener (dismiss-only) ──> queue.mark_done
ReconcileProducer ──> EmailProducer.reconcile_inbox ──> queue.mark_done
LinearProducer._mark_closed_task (inline in poll) ──> queue.mark_done
                                                       QueueLog (JSONL) replay on restart
```

Only email and linear go through the screener intake (`main.py:275-283`); Slack calls
`queue.add` directly and only touches the screener via reconsider.

Identity / dedup / lifecycle mechanisms:

| # | Mechanism | Location | Identity | Verdict |
|---|-----------|----------|----------|---------|
| 1 | Email thread collapse | `email.py` `_find_pending_thread_task` | `source.thread_id`, PENDING only | goal right, wrong layer (races intake) |
| 2 | Slack thread collapse | `slack.py` `_find_pending_thread_task` | `(channel_id, thread_ts)`, live | 2nd reimpl; no race only because Slack bypasses intake |
| 3 | Linear issue collapse | `linear.py:329-338` `_find_live_issue_task` | `source.identifier`, live | 3rd reimpl; races intake — the TRI-240 bug |
| 4 | Claude window collapse | `claude.py` `_find_pending_window_task` | `source.window`, live | 4th reimpl |
| 5 | Email `_recent_keys` | `email.py:124,211,225` | `thread_id:msg_id` | legit cursor dedup (cap regression fixed 2026-07-08) |
| 6 | Linear `_recent_keys` | `linear.py:77,172,188` | identifier | DEAD — written/capped, never read |
| 7 | Slack `_recent_ids` + cursors | `slack.py` | message ts | legit |
| 8 | `dedup_key` any-state suppression | `tasks.py:269-282`; set `screener.py` | free-form, only `meeting_followup` | bandaid: right layer, wrong semantics for non-followups |
| 9 | `subject_key` clustering | `tasks.py:80,192-225` | "what it's about" | principled & DISTINCT — keep separate from dedup |
| 10 | Reconsider queue | `main.py:211,243-252`, `screener.py:605-641` | task in queue | judged retraction; 2nd screener input w/ race; Slack-only |
| 11 | Reconcile sweep | `reconcile.py`, `email.py` | full-listing diff | authoritative retraction; email-only |
| 12 | Linear retire-on-poll | `linear.py:154-166,388-399` | identifier, PENDING | 3rd retraction path |
| 13 | Replay drop rules | `main.py:114-124`, `queue_log.py:27-35` | kind-based | principled concept, scattered special-cases |

## 2. Core diagnosis

1. Task identity is defined four times, privately, per-producer (mechanisms 1–4),
   each a linear scan over `TaskQueue` with different state filters, plus a fifth
   notion (`dedup_key`) in the queue with different semantics and one consumer.
2. Dedup runs against an incomplete view. Real pipeline state is
   `queue ∪ intake_q ∪ (task being screened)`, but every collapse check consults
   only `queue`. The intake buffer (screener) was added without extending the
   identity check's view, so any screening backlog opens a duplication window. This
   is the entire TRI-240 bug; the inclusive Linear cursor just makes re-emission
   happen every poll rather than occasionally.
3. Lifecycle transitions (retract / re-surface / suppress) are ad hoc per producer:
   three retraction mechanisms (#10–12), two replay special-case sets (#13), no
   single authority.

The fix is not more dedup — it is moving identity and lifecycle authority into the
queue and making the intake buffer *part of the queue* (a state) rather than a
shadow pipeline in front of it.

Aggravating facts verified in code:
- Screener is serial (`screener.py:584`); a single executor run has a 300s deadline
  floor (`claude_mcp.py:219`) while `linear_poll_interval` defaults to 180s
  (`config.py:70`) — the backlog window is structurally larger than the poll
  interval, so the race is near-guaranteed whenever an executor runs.
- Linear incremental cursor is inclusive (`linear.py:239-241`), no client-side
  strict-inequality guard (email has one at `email.py:217`) — boundary ticket
  re-returns every poll and its `created_at` is reset each time (never ages up).
- Slack bypasses intake (`slack.py:795`): `slack_msg` is never screened on arrival;
  the config kind only arms reconsider (`main.py:248-252`).
- Doc/behavior contradiction: `tasks.py:267` claims `dedup_key` suppression survives
  restart, but `queue_log.replay()` skips done/dropped (`queue_log.py:150-157`), so
  cross-restart follow-up suppression does not actually work (masked because the
  parent email is usually archived and never re-screened).

## 3. Proposed architecture

### 3.1 Two identity fields, crisply defined

- `origin_key: str | None` (generalize today's `dedup_key`) — "which real-world
  object produced this task", producer-owned, namespaced:
  - `email:<thread_id>`, `slack:<channel_id>:<thread_ts>`, `linear:<IDENTIFIER>`,
    `claude:<window>`, `followup:<gmail_thread_id>:<normalized headline>`
  - `None` for manual notes → singleton, never deduped.
- `subject_key` — unchanged; "what the task is about", used only for `ranked()`
  adjacency. A Linear-notification email keeps `origin_key=email:...` +
  `subject_key=linear:TRI-240`. **Do not merge these two.**

### 3.2 The queue is the single identity authority: `TaskQueue.upsert`

Add an `origin_key → task_id` index inside `TaskQueue` and one method replacing all
four producer collapse implementations:

```python
def upsert(self, task, *, if_terminal="new") -> Task:
    # live (pending/active/snoozed/screening) task with same origin_key -> update-in-place
    # terminal (done/dropped):
    #   if_terminal="new"  -> mint fresh task (re-surface)   [linear, slack, email, claude]
    #   if_terminal="skip" -> return existing, no event      [meeting_followup]
    # none -> add
```

Subsumes `_find_live_issue_task`, both `_find_pending_thread_task`,
`_find_pending_window_task`, `_find_by_dedup_key`, and `add`'s `dedup_key` branch.
The state-filter inconsistencies (email pending-only vs others live) collapse into
one rule.

### 3.3 A `screening` state instead of a shadow intake queue — the race fix

Invert screening vs queue entry. Producers call one `submit(task)`:

1. `queue.upsert(task)` immediately; state `STATE_SCREENING` when the kind is
   screenable (sync check: `kind in allowed_kinds` and non-empty
   `candidates_for(task, manifests)` — both pure, `screener.py:95-106`), else
   straight to `pending`.
2. If screening, push the task id to the screener work queue.
3. Screener outcome becomes a state transition on the resident task:
   - `forward`/`failed` → `screening → pending` (annotate body on failed)
   - `handled` → `screening → done` (delete the synthetic `queue_log.record`
     hack at `main.py:239-241` — now a real state event)
   - `dismissed` → `screening → dropped`
4. `score()` already returns `-inf` for non-pending (`tasks.py:171`); `pending()`
   unaffected.

Why it kills the bug: a re-poll during a backlog now *finds* the in-flight task via
`upsert` (screening is live) and updates in place. No window where a task exists but
is invisible to identity checks, regardless of `claude --print` latency. Bonus: the
sync gate lets unscreenable tasks (most Slack, all `claude_reply`) skip the screener
entirely, fixing a latency issue and making it safe to route all producers through
`submit()` uniformly (removes the Slack asymmetry).

### 3.4 One retraction concept: `resolve`

`TaskQueue.resolve_by_origin(origin_key)` — mark done iff pending/snoozed; never
touch active (preserves the deliberate rule at `linear.py:351-358`,
`email.py:270-273`).
- Linear retire-on-poll → `resolve_by_origin(f"linear:{identifier}")`.
- Email reconcile sweep keeps its snapshot/complete-listing rails, ends in
  `resolve_by_origin`. `ReconcileProducer` stays as the timer shell.
- Slack reconsider is judged (LLM) retraction, not authoritative — its plumbing
  merges into the screener: replace the two queues and the three-way `asyncio.wait`
  race in `_next_or_stop` (`screener.py:516-550`) with one work queue of
  `(task_id, mode)`, mode ∈ {intake, reconsider}.

### 3.5 Lifecycle semantics (the contract)

| Event | Rule |
|-------|------|
| Object sighted, no task for `origin_key` | add (via screening gate) |
| Object sighted, live task | update-in-place; never a new task |
| Object sighted, terminal task | re-surface fresh (`if_terminal="new"`); needs exclusive cursor so "sighted" means "changed" |
| Follow-up spawn, terminal task same key | suppress (`if_terminal="skip"`) |
| Source says object resolved | `resolve_by_origin`: pending/snoozed → done; active untouched |
| Screener judges task moot (reconsider) | state transition → done |
| Restart | source-of-truth kinds dropped, first wide poll rebuilds (`main.py:114`); `screening` → `pending` (fail-safe); terminal durable-kind tasks loaded as terminal records so `if_terminal="skip"` works across restarts |

## 4. Scope bounds — what NOT to build

- No identity-registry/`IdentityIndex` class — a dict + one `upsert` method.
- No generic Reconciler protocol forced on every producer — Slack can't cheaply
  enumerate its live set; keep authoritative retraction where the source supports it
  (email, linear) and judged retraction (reconsider) where it doesn't.
- No sqlite/durable dedup store — JSONL replay + wide first polls already give
  correct restart semantics; only fix needed is loading terminal durable tasks.
- No concurrency machinery — the bug was view incompleteness, not locking.
  Single-loop discipline stays.
- No config-driven lifecycle DSL — `if_terminal` is a two-value literal at the call
  site.
- No parallel screener — serial is fine once unscreenable tasks bypass it.
- Deletions outnumber additions: 4 collapse helpers, `add`'s `dedup_key` branch,
  `_next_or_stop`'s dual-queue race, linear `_recent_keys`, the synthetic autohandle
  log record, optionally email `_recent_keys` (cursor + idempotent upsert make it
  redundant). Net additions: one state constant, one index, `upsert`,
  `resolve_by_origin`, a ~30-line `submit` gate.

## 5. Migration path (ordered, test-backed; TRI-240 fixed early)

1. **Cursor + dead code (small, immediate relief).** Client-side strict
   `updatedAt >` guard in `LinearProducer._poll_once` (mirror `email.py:217`);
   delete linear `_recent_keys` (`linear.py:77,172,188-189`). Test: boundary ticket
   not re-emitted. Effect: every-poll re-emission stops; triplication near-impossible
   before any architecture change.
2. **`origin_key` + `upsert` + `STATE_SCREENING` (structural fix).** Add field
   (`to_dict`/`from_dict` reading legacy `dedup_key`), index, `upsert`, state, and
   the `submit` gate in `main.py`; convert Linear + Email; screener transitions
   states. Delete `_find_live_issue_task`, `_find_pending_issue_task`, email
   `_find_pending_thread_task`. Test: same identifier emitted twice while a fake
   screener stalls → exactly one task (the TRI-240 regression test).
3. **Convert Slack + Claude collapse to `upsert`; route all producers through
   `submit`.** Delete `slack._find_pending_thread_task`,
   `claude._find_pending_window_task`. Document that `slack_msg` in `autohandle_kinds`
   now also enables intake screening (behavior change; dismiss skill guards make it
   safe).
4. **Fold `dedup_key` into `origin_key`; fix cross-restart follow-up suppression.**
   Follow-up spawns set `origin_key`, land via `upsert(if_terminal="skip")`. Delete
   `_find_by_dedup_key` and `add`'s suppression branch. In `replay`, load terminal
   durable-kind tasks; demote `screening` → `pending`. Fix docstring at
   `tasks.py:81-89`.
5. **Unify retraction plumbing.** `resolve_by_origin`; convert `_mark_closed_task`
   and `reconcile_inbox`'s retire loop. Merge intake/reconsider into one
   `(task_id, mode)` queue; delete `_next_or_stop`'s third waiter.

Each step is independently shippable; steps 1–2 alone fix the reported bug class.

## 6. Risks / tradeoffs

- **Reopened-ticket re-surface.** With the exclusive cursor, an incremental sighting
  means the ticket changed, so terminal→fresh is correct. Edges accepted: (a) user
  marks task done while ticket stays In Progress → later activity resurrects it
  (arguably desired); (b) after restart, source-of-truth drop resurrects it anyway —
  existing documented behavior (`main.py:108-113`), unchanged here.
- **Restart + `screening`.** Demote to pending (fail-safe) rather than re-screen on
  boot (which risks double side effects, e.g. double-RSVP).
- **claude-print latency stays the dominant constant.** Design removes the
  *correctness* dependence on the 300s-vs-180s ratio, not the throughput one; sync
  gate keeps unscreenable tasks fast.
- **`created_at` refresh on update-in-place** resets age score; post-cursor-fix only
  happens on genuine activity (matches intent) — keep, as one documented decision in
  `upsert` instead of four copies.
- **Topic-cap interplay:** `_enforce_topic_cap` counts pending only; re-run the cap
  on the `screening → pending` transition or it can transiently exceed.
- **JSONL compat:** `from_dict` tolerates missing fields; read legacy `dedup_key`
  into `origin_key` during the transition window.

## Critical files

- `src/code_trip2/tasks.py` — Task fields, `upsert`, `STATE_SCREENING`, `resolve_by_origin`
- `src/code_trip2/producers/linear.py` — cursor fix, dead `_recent_keys`, collapse → upsert
- `src/code_trip2/screener.py` — state-transition outcomes, unified work queue, follow-up keys
- `src/code_trip2/main.py` — `submit` gate, wiring, delete synthetic autohandle record
- `src/code_trip2/queue_log.py` — replay: screening demotion, terminal durable records
