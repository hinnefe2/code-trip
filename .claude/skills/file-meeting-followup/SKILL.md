---
name: file-meeting-followup
description: Convert a meeting_followup task into a Linear backlog ticket assigned to the user. Used by the ACT+YES chord — the user has decided the follow-up is real work that needs tracking. The skill picks the right team (and project, when an obvious match exists) based on the follow-up's content and the user's Linear workspace.
auto-handle: false
allowed-tools:
  - mcp__claude_ai_Linear__list_teams
  - mcp__claude_ai_Linear__list_projects
  - mcp__claude_ai_Linear__save_issue
---

# File a meeting follow-up as a Linear backlog ticket

You're being invoked on a `meeting_followup` task that the user has decided is real work. Create a Linear issue for it. The user is working across several different teams / projects, so the routing decision is the whole point of this skill — don't just dump everything into one default bucket.

1. **Read the follow-up.**
   - `task.headline` is the proposed ticket title (a short imperative).
   - `task.body` has the context — usually a quoted action item plus surrounding meeting context.
   - `task.source.meeting` names the meeting it came from.
   - `task.source.topic` is a short slug the producer assigned (often the meeting title kebab-cased).

2. **Pick the team.**
   - Call `mcp__claude_ai_Linear__list_teams` (no `query` argument — we want the whole workspace so the match is informed).
   - Choose the team whose name most plausibly owns this work, based on the meeting title and the follow-up body. Look for word overlap with the meeting name, headline, or any project/domain language in the body (e.g. a "Planning Sync" follow-up about retention metrics → a team named something like "Retention", "Analytics", or "Data"; a follow-up about portal infrastructure → "Platform" or "Infra").
   - If no team's name is a clear match, pick the team that looks like the user's primary engineering team (a generic name like "AI", "Engineering", "Platform"). Avoid teams whose name implies a different function (Marketing, Sales, Design) unless the follow-up obviously belongs there.
   - Never invent a team — only use IDs/names that came back from `list_teams`.

3. **Optionally pick a project.**
   - Call `mcp__claude_ai_Linear__list_projects` (no `query`, no team filter — pull what's available; if the list is long, the relevant project usually has the meeting/topic word in its name).
   - Pick a project ONLY when its name clearly matches the meeting title, the follow-up body, or a domain word from the headline. If nothing matches clearly, leave the issue project-less — better than wrong routing.

4. **Create the issue.** Call `mcp__claude_ai_Linear__save_issue` with:
   - `team`: the team name or ID from step 2.
   - `project`: from step 3 if you picked one; omit otherwise.
   - `title`: the task headline verbatim.
   - `description`: the task body. If the team/project you chose wasn't an obvious match (you fell back to a generic team or guessed from weak signal), append a final line `_(Team inferred — reassign if wrong.)_` so the user notices on their next Linear pass.
   - `assignee`: `"me"`.
   - `state`: `"Backlog"`.

5. **Return** one sentence. The orchestrator speaks this verbatim:
   - `Filed in <team>: <title>.` if you didn't pick a project.
   - `Filed in <team> / <project>: <title>.` if you did.

Don't ask for confirmation. The user already chose to file this via ACT+YES; your job is to route it well and report what you did.
