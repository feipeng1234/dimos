# github-issue-to-pr

Operating manual for an AI coding agent invoked by the `github-issue-worker`
daemon (whose code lives next to this file, under `scripts/worker.py`) to turn
a GitHub issue into a Pull Request.

This file is read by the agent (claude / cursor-agent / codex) at the start of
each issue. The worker's prompt template injects the path to this file; the
agent is expected to read it before editing anything.

## Purpose

Given a GitHub Issue (title + body) the worker has placed in front of you,
implement the smallest correct change that resolves it, in the working
directory and on the branch the worker has already prepared.

## What the worker has already done for you

- `cd` into the repository working directory.
- `git fetch` and `git reset --hard` to a clean copy of the base branch.
- `git checkout -B agent/issue-<N>-<slug>` so you are on a fresh branch.
- Composed a prompt containing the issue title, body, repo, and a pointer to
  this file.

## What the worker will do AFTER you exit

- `git diff` to detect whether you actually changed anything.
- If `AGENT_NEEDS_HUMAN.md` exists at the repo root → relabel the issue
  `agent-needs-human` and stop.
- If no diff → relabel the issue `agent-needs-human`.
- Otherwise: `git add -A` + `git commit --no-verify` + `git push` +
  `gh pr create --base <base>` + comment the PR URL on the issue + relabel
  `agent-done`.

So **your only job is to edit files**. Do not run git, do not commit, do not
push, do not open a PR.

## Workflow

1. Read the issue title and body that the worker pasted into the prompt.
2. Inspect the repo before editing — at minimum:
   - Project root layout (`ls`, look for `README.md`, `AGENTS.md`,
     `package.json`, `pyproject.toml`).
   - Any file the issue explicitly names.
3. Decide the smallest change that satisfies the issue.
4. If the issue is unclear, ambiguous, or you would have to guess at intent —
   STOP. Write `AGENT_NEEDS_HUMAN.md` at the repo root explaining what is
   unclear and what options you considered, then exit. Do not edit anything
   else.
5. Otherwise, edit only the files you must edit.
6. Try to run the project's checks if they exist and are cheap. The worker
   will also run `ruff check .` and `pytest -q --maxfail=1` after you exit
   and report the result on the PR; you do not need to repeat that, but if
   you wrote new code, at least sanity-check the file you touched compiles
   / parses.
7. Exit normally (return code 0). Do not leave background processes.

## Hard rules

The worker treats these as load-bearing. Break them and your work gets thrown
away.

- **No git.** The worker owns branch / commit / push / PR. **Do not run `git
  add`, `git commit`, `git push`, `git checkout`, `git reset`, or anything
  else that mutates git state.** If you have a "test plan" or "verification"
  step that involves git: skip it. Just edit files and exit. cursor-agent
  in particular has been observed running `git commit` against this rule —
  the worker now silently accepts that commit and pushes it as-is, but
  every such commit makes the audit trail lie about who did the work.
- **Edits stay inside the working directory.** Do not write outside it, do
  not modify other repos, do not modify the user's home dir.
- **Forbidden files.** Do not create, modify, or delete:
  - `.env`, `.env.*`
  - anything matching `*secret*`, `*credential*`, `*deploy_key*`
  - `.github/workflows/**` (CI config)
  - any file outside the issue's stated scope just because it "looked
    wrong"
- **Smallest correct diff.** No drive-by refactors. No reformatting unrelated
  files. No "while I'm here" cleanups. If the issue is "fix typo in README",
  change exactly that typo.
- **No new top-level dependencies** unless the issue explicitly asks for them.
  If you genuinely need one, write it in `AGENT_NEEDS_HUMAN.md` instead.
- **No deleting tests** to make them pass.
- **Honest reporting.** If a check fails, leave it failing — the worker will
  surface it on the PR. Do not paper over it.

## When to bail to AGENT_NEEDS_HUMAN.md

Write this file (and only this file) when any of these are true:

- The issue is two sentences and you do not know what success looks like.
- The change touches authentication, payment, database migrations, or
  anything in the AGENTS.md "禁止事项" list of the target repo.
- The fix would require breaking a public API or message contract.
- You would need to add a new heavyweight dependency (a new framework,
  a new build tool, a new database driver).
- The issue asks for something the codebase clearly does not support yet
  and would require an architecture decision.

The contents of `AGENT_NEEDS_HUMAN.md` should be plain markdown:

```markdown
# Why I did not act on issue #<N>

## What's unclear
- ...

## Options I considered
1. ... (pros / cons)
2. ...

## What I would need from a human
- ...
```

The worker will paste the first ~4000 chars of this file as a comment on the
issue and relabel it `agent-needs-human`.

## Done criteria (for you)

You are done when ALL of the following are true:

- You read the issue.
- You either:
  - made the smallest reasonable code change that resolves the issue, OR
  - wrote `AGENT_NEEDS_HUMAN.md` and made no other change.
- You did not run git.
- You did not modify forbidden files.
- You exited with return code 0.

The worker takes it from here.
