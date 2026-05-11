---
name: dimos-pr-flow
description: Walk user-written local changes through to a merged PR on the feipeng1234/dimos fork. Inspects the working tree, infers branch / commit / PR title + body from the diff, confirms once with the user, then commits (if needed), pushes via the bundled pre-push hook, opens a PR to dev, enables squash auto-merge, and babysits until merged. Self-contained — ships its own `_gh.py`, `install_hooks.py`, and `hooks/pre-push`; the only external dependency is the `babysit` skill for the merge-readiness loop. Use when the user has already written the code locally and just wants the PR pipeline run end-to-end. Trigger words: 走一下PR的流程, 帮我提个PR, 帮我推一下代码, single-PR, walk PR flow, ship this branch, open the PR.
---

# dimos-pr-flow

You drive a **single-PR walk-through** for code the user has already written
locally. The user is at the keyboard, so one confirmation up front is OK;
everything else is inferred from the diff + git log and executed without
further questions.

This skill is **self-contained**. All the load-bearing pieces live inside
`.cursor/skills/dimos-pr-flow/`:

- `scripts/_gh.py` — the only sanctioned `gh`/git wrapper. Enforces
  `--repo feipeng1234/dimos`, branch-prefix, PR `--base dev`,
  `--force-with-lease`, squash auto-merge. Never call `gh` or `git push`
  directly; always go through `_gh.py`.
- `scripts/install_hooks.py` + `hooks/pre-push` — the `pre-push` git hook
  that blocks pushes to non-fork remotes / dev / main / bad branch
  prefixes. `install_hooks.py install` is idempotent; safe to run every
  time. If you also have `dimos-agentic-harness` installed and it already
  linked its own `pre-push`, `install_hooks.py` will print `[warn]` and
  leave the existing symlink alone — that's expected; the two hooks are
  functionally equivalent.

The only external dependency is `~/.cursor/skills-cursor/babysit/SKILL.md`
for the merge-readiness loop.

---

## Hard rules

These mirror the harness's mechanically enforced rules (M1–M2) plus a few
honor-system rules adapted for an interactive single-PR session.

1. **Push only to `feipeng1234/dimos`, branch must be prefixed** with
   `feat/|fix/|refactor/|docs/|test/|chore/|perf/`. Both the pre-push hook
   and `_gh.push_branch` enforce this; you should also pick a compliant
   branch name up front so neither has to reject you.
2. **PR base is always `dev`**, never `main`. `_gh.pr_create` forces it.
3. **No raw `gh ...` or `git push ...`** — always go through `_gh.py`.
4. **No `--amend`** unless the HEAD commit was made in this session AND
   not yet pushed (per the global Git Safety Protocol).
5. **Never commit secrets** (`.env`, API keys, credentials). If you spot
   any in the diff, stop and warn the user before committing.
6. **Conflict policy = rebase-and-regenerate.** If the branch needs to be
   updated, `git fetch origin && git rebase origin/dev`, re-resolve hunks
   based on intent — never blindly `-X theirs`/`-X ours`.

---

## Phase 0 — Preflight (always, idempotent)

Run these four commands in parallel, batched in one tool message:

```bash
cd /home/lenovo/dimos
gh api user --jq .login
gh api repos/feipeng1234/dimos --jq '{auto:.allow_auto_merge,squash:.allow_squash_merge}'
git ls-remote --heads origin dev | grep -c refs/heads/dev
```

Then install the pre-push hook (idempotent):

```bash
.venv/bin/python .cursor/skills/dimos-pr-flow/scripts/install_hooks.py install
```

**Abort and report to the user** if any of these are wrong:

- `gh api user` ≠ `feipeng1234` → suggest `gh auth switch -u feipeng1234`.
- `allow_auto_merge` or `allow_squash_merge` is `false` → suggest
  `gh repo edit feipeng1234/dimos --enable-auto-merge --enable-squash-merge --delete-branch-on-merge`.
- `origin` does not have a `dev` branch → suggest
  `git fetch upstream && git push origin upstream/main:dev --force-with-lease`.

Do NOT try to fix these automatically — they typically need interactive
auth or repo-admin permissions.

---

## Phase 1 — Inspect

Run in parallel, one tool call each, batched in a single message:

- `git status --branch --short`
- `git log --oneline origin/dev..HEAD` (commits ahead of fork:dev — may be empty)
- `git diff origin/dev...HEAD --stat` (combined diff vs fork:dev)
- `git diff --stat` (unstaged)
- `git diff --cached --stat` (staged)

From the outputs, classify which state the user is in:

| State | Signal | Phase-3 actions |
|-------|--------|-----------------|
| **A. Branch + commits + clean tree** | on a `feat/…`-prefixed branch, commits ahead of `origin/dev`, nothing modified or staged | push → PR → auto-merge → babysit |
| **B. Branch + uncommitted changes** | on a `feat/…`-prefixed branch, modified/staged files present | commit → push → PR → auto-merge → babysit |
| **C. Wrong branch + uncommitted changes** | on `main`/`dev`/non-prefixed branch with local changes | create branch → commit → push → PR → auto-merge → babysit |
| **D. Wrong branch + commits** | on `main`/`dev`/non-prefixed branch, local commits ahead of `origin/dev` | move commits to a new branch → push → PR → auto-merge → babysit |
| **E. Nothing to ship** | clean tree, no commits ahead of `origin/dev` | report "nothing to do" and stop |

If state E, stop immediately. Don't push an empty branch.

---

## Phase 2 — Plan (infer, then confirm once)

Infer the following from the inspection output. Do not ask the user for
each one separately — just produce one plan and confirm.

### Inference rules

- **Branch name** (state C/D, or A/B if user wants to rename):
  short kebab-case from the dominant directory or filename in the diff.
  Pick the prefix from the dominant change type:
  - Adds a new feature / new file: `feat/<slug>`
  - Fixes a bug / corrects behavior: `fix/<slug>`
  - Pure refactor / rename / no behavior change: `refactor/<slug>`
  - Docs only: `docs/<slug>`
  - Tests only: `test/<slug>`
  - Tooling / config / CI: `chore/<slug>`
  - Performance: `perf/<slug>`
- **Commit subject** (state B/C): imperative mood, ≤ 72 chars, derived
  from the most impactful file change. No emojis.
- **PR title**: equal to the commit subject if there is exactly one new
  commit; otherwise a short summary of the branch.
- **PR body**: follow the template the global system prompt mandates:
  ```
  ## Summary
  - <bullet describing the "why", not the "what">
  - …

  ## Test plan
  - [ ] <one concrete checklist item>
  ```

### Confirmation

Print the entire plan to the user in one block, then call `AskQuestion`
exactly once with these options:

- **Proceed** — execute the plan as-is.
- **Edit plan** — the user will tell you what to change (branch, title,
  body, files-to-add). Loop back through Phase 2 with their edits.
- **Abort** — stop the session, do nothing.

Only ask additional questions if the diff is **genuinely ambiguous** (e.g.
touches two unrelated areas and there is no clear single subject) — in
that case offer 2–3 alternative plans inside the same `AskQuestion`.

---

## Phase 3 — Execute the plan

Run only the steps needed for the detected state. Do not over-execute.

### 3a. Re-point the branch (state C only)

```bash
git switch -c <new-branch>
```

### 3b. Move commits to a new branch (state D only)

```bash
git branch <new-branch>              # new branch points at HEAD
git switch <new-branch>
git switch -                          # back to the wrong branch
git reset --hard origin/<wrong-branch>
git switch <new-branch>
git log --oneline -3                  # verify before pushing
```

### 3c. Commit (state B/C only)

Stage only the files that belong to this PR — never `git add -A` blindly.
Use a heredoc commit message (the global Git Safety Protocol requires it):

```bash
git add <files-that-belong>
git commit -m "$(cat <<'EOF'
<subject>

<optional 1–3 sentence body explaining the "why">
EOF
)"
```

If pre-commit modifies files (ruff-format, etc.), inspect with `git diff`
and create a **new** follow-up commit. Do NOT pass `--no-verify`.

### 3d. Push

```bash
.venv/bin/python .cursor/skills/dimos-pr-flow/scripts/_gh.py push <branch>
```

The wrapper does `--force-with-lease` and asserts the fork+prefix
invariants before invoking git. The pre-push hook is a second line of
defense.

### 3e. Open the PR

```bash
.venv/bin/python .cursor/skills/dimos-pr-flow/scripts/_gh.py pr-create <branch> "<title>" "$(cat <<'EOF'
## Summary
- ...

## Test plan
- [ ] ...
EOF
)"
```

The wrapper forces `--repo feipeng1234/dimos --base dev`. Stdout is
`{"number": <n>, "url": "..."}`. Capture both.

### 3f. Enable squash auto-merge

```bash
.venv/bin/python .cursor/skills/dimos-pr-flow/scripts/_gh.py pr-merge-auto <pr-number> squash
```

Exit 0 → auto-merge armed. Exit 1 → review/conflict blocks immediate
auto-merge; continue to Phase 4 anyway (the babysitter will resolve).

---

## Phase 4 — Babysit until merged

Read `~/.cursor/skills-cursor/babysit/SKILL.md` and follow it inline (no
subagent — the user is awake; if a human review needs a decision, surface
it to them). Prepend the same v0.2 override the harness uses:

```
PARENT-AGENT-OVERRIDE for dimos-pr-flow:
1. bot review comments (codecov / bugbot / coderabbit / etc): apply suggested fixes silently and reply with a short ack.
2. human review comments: do NOT auto-accept. Surface the feedback to the user and wait for their direction.
3. merge conflicts: rebase onto origin/dev, re-implement conflicting hunks based on the original commit intent (not prefer-base, not prefer-theirs). After 2 consecutive rebase-and-regenerate failures, stop and surface to the user.
4. NEVER call `gh api .../reviews/<id>/dismissals`. Dismissing reviews is forbidden.
5. NEVER raw `gh` or `git push` — use `.cursor/skills/dimos-pr-flow/scripts/_gh.py` for all push and PR-mutating calls.
```

Poll roughly every 30–60s via `gh pr view <num> --json state,mergeable,mergeStateStatus,reviewDecision,statusCheckRollup,mergedAt`. Don't poll faster — GitHub rate-limits and the mergeable state recomputes asynchronously.

**Stop conditions:**

- `state == MERGED` → print `Merged: <pr-url>` and end.
- `reviewDecision == CHANGES_REQUESTED` from a human reviewer → print
  `Awaiting human: <pr-url> — <reason>` and end. (Bot CHANGES_REQUESTED
  is auto-resolved per override rule 1.)
- 2 consecutive rebase failures → print `Blocked: <pr-url> — unresolvable conflict after 2 rebases` and end.
- 10 babysit iterations without progress → print
  `Awaiting human: <pr-url> — no progress after 10 iterations` and end.

---

## Final report

End the session with exactly one of:

```
Merged: <pr-url>
```

```
Awaiting human: <pr-url> — <one-line reason>
```

```
Blocked: <pr-url> — <one-line reason>
```

No extra commentary, no celebratory prose.

---

## Files in this skill

- `SKILL.md` — this file (playbook).
- `scripts/_gh.py` — trimmed `gh`/git wrapper (push, pr-create,
  pr-merge-auto). Self-contained; no cross-skill imports.
- `scripts/install_hooks.py` — installs the `pre-push` hook into
  `.git/hooks/` via a symlink to `hooks/pre-push`. Idempotent.
- `hooks/pre-push` — git hook: blocks push to non-fork remotes / dev /
  main / bad branch prefixes. Chains to a `pre-push.harness-backup-*` if
  one is found (preserves prior hooks like `git-lfs install`'s).

External (only one):

- `~/.cursor/skills-cursor/babysit/SKILL.md` — read inline during Phase 4
  to drive the merge-readiness loop.
