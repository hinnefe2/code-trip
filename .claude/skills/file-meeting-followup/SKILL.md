---
name: file-meeting-followup
description: Convert a meeting_followup task into a Linear backlog ticket assigned to the user. Used by the ACT+YES chord — the user has decided the follow-up is real work that needs tracking. The skill always files into a project — defaulting to the project the user has been working in most recently (inferred from their most-recently-updated assigned issues) and overriding only when the follow-up's content clearly matches a different project — and picks the team to match.
auto-handle: false
allowed-tools:
  - mcp__claude_ai_Linear__list_issues
  - mcp__claude_ai_Linear__list_teams
  - mcp__claude_ai_Linear__list_projects
  - mcp__claude_ai_Linear__save_issue
last-updated-by: Henry Hinnefeld
last-updated-date: 2026-06-24
---

# File a meeting follow-up as a Linear backlog ticket

You're being invoked on a `meeting_followup` task that the user has decided is real work. Create a Linear issue for it, **always inside a project**. The default home is the project the user has been working in most recently; only override that when the follow-up's content clearly belongs somewhere else. The routing decision is the whole point of this skill — don't just dump everything into one fixed bucket, but don't leave the ticket project-less either.

1. **Read the follow-up.**
   - `task.headline` is the proposed ticket title (a short imperative).
   - `task.body` has the context — usually a quoted action item plus surrounding meeting context.
   - `task.source.meeting` names the meeting it came from.
   - `task.source.topic` is a short slug the producer assigned (often the meeting title kebab-cased).

2. **Find the recently-active project (the default home).**
   - Call `mcp__claude_ai_Linear__list_issues` with `assignee="me"` and `limit=20`. It defaults to `orderBy="updatedAt"`, so the issues come back most-recently-touched first — that's the user's current center of gravity.
   - Walk the list from the top and take the **first issue that has a project**; that issue's `project` is the recently-active project, and that same issue's `team` is the team to file under (the pair is known-valid together). Remember both.
   - If none of the returned issues has a project, fall back to `mcp__claude_ai_Linear__list_projects` with `member="me"` (default `orderBy="updatedAt"`) and take the top project, plus a team it belongs to. If that also comes back empty, you have no recent signal — go to step 3 and rely entirely on a content match.

3. **Check for a clear content override.**
   - Decide whether the follow-up obviously belongs to a *different* project than the recently-active one — i.e. the meeting title, headline, or a domain word in the body clearly names another project (e.g. a "Billing revamp" follow-up when you've recently been in "Onboarding").
   - Only override on a clear match. To find the override target, call `mcp__claude_ai_Linear__list_projects` (no filter) and pick the project whose name plainly matches; use a team that project belongs to. If you're not confident, do NOT override — keep the recently-active project from step 2.
   - Never invent a team or project — only use names/IDs returned by the list tools. If you need a team name and only have a project, `mcp__claude_ai_Linear__list_teams` lists the workspace's teams.

4. **Create the issue.** Call `mcp__claude_ai_Linear__save_issue` with:
   - `project`: the project chosen in step 3 (override) or step 2 (recently-active default). Include a project whenever you have one — only omit it in the rare case where steps 2 and 3 both came up empty.
   - `team`: the team paired with that project. (`team` is required by `save_issue`; make sure it's a team the project belongs to.)
   - `title`: the task headline verbatim.
   - `description`: the task body. Append a final note when your routing was a guess rather than a clear signal:
     - `_(Filed in your recent project — move if it belongs elsewhere.)_` when you used the step-2 default without a content match.
     - `_(Project/team inferred — reassign if wrong.)_` when you had to fall back to a weak guess.
   - `assignee`: `"me"`.
   - `state`: `"Backlog"`.

5. **Return** one sentence. The orchestrator speaks this verbatim:
   - `Filed in <team> / <project>: <title>.` — the normal case (a project was set).
   - `Filed in <team>: <title>.` — only when no project could be determined at all.

Don't ask for confirmation. The user already chose to file this via ACT+YES; your job is to route it well — into a project — and report what you did.
