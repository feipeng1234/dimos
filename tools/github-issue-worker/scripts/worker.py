#!/usr/bin/env python3
# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""github-issue-worker: 把带 agent-todo label 的 GitHub Issue 自动变成 PR。

设计原则 (跟 feishu-to-github-issue 保持一致):
- 单文件搞定 MVP, 不做臆测性抽象。
- GitHub / git 操作走系统的 `gh` 和 `git` CLI, 不引入 PyGithub 之类的重依赖。
- AI agent 走 subprocess, 把命令模板做成 plug-in (claude / cursor-agent / codex / dry)。
- 状态机靠 GitHub label 实现, 没有外部数据库。
- 失败优先于隐瞒: 异常一律标 agent-failed + 评论, 绝不让 issue 卡在 agent-running。

分工 (重要):
- 本进程负责 git: clone / checkout / 建分支 / 检查 diff / commit / push / 建 PR。
- AI agent 只改文件, 不动 git。这样 agent 不可能误 push 主分支。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("worker")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


DEFAULT_AGENT_TEMPLATES: dict[str, str] = {
    # claude code: -p 模式非交互运行, prompt 通过 argv 传。
    # 注意: 真实使用很可能要加 --dangerously-skip-permissions, 通过 WORKER_AGENT_CMD 覆盖。
    "claude": "claude -p {prompt}",
    # cursor-agent --print mode: --force --trust let it write files without
    # interactive permission prompts. Without these the subprocess hangs
    # forever waiting for a TTY-only confirmation it cannot get.
    "cursor-agent": "cursor-agent -p {prompt} --output-format text --force --trust",
    "codex": "codex exec {prompt}",
    # dry: 端到端冒烟用。不调任何 AI, 只在工作区写一行, 让后续 commit/push/PR 链路能跑通。
    # `_ {prompt}` 把 prompt 喂给 bash 当 $1, 脚本里不引用所以无副作用; 必须有 {prompt}
    # 是因为 validator 要求所有 agent 模板都有这个占位符。
    "dry": "bash -c 'echo dry-run > AGENT_RAN.txt' _ {prompt}",
}


@dataclass
class Labels:
    todo: str
    running: str
    done: str
    failed: str
    human: str


@dataclass
class Config:
    repo: str
    base_branch: str
    workspace: Path
    poll_interval: int
    agent: str
    agent_cmd_template: str
    agent_timeout: int
    skill_path: str
    git_ssh_host_alias: str  # e.g. "github-feipeng1234" if user has SSH config aliases
    labels: Labels
    once: bool = False
    target_issue: int | None = None
    dry_pr: bool = False

    @property
    def repo_name(self) -> str:
        return self.repo.split("/", 1)[1] if "/" in self.repo else self.repo

    @property
    def repo_dir(self) -> Path:
        return self.workspace / self.repo_name


def load_config(args: argparse.Namespace) -> Config:
    load_dotenv()

    repo = args.repo or os.getenv("WORKER_REPO", "").strip()
    if not repo or "/" not in repo:
        sys.exit("错误: 必须设 WORKER_REPO=owner/repo (或用 --repo 传)")

    workspace = Path(
        os.path.expanduser(os.getenv("WORKER_WORKSPACE", "~/agent-workspace"))
    ).resolve()

    agent = args.agent or os.getenv("WORKER_AGENT", "claude").strip()
    if agent not in DEFAULT_AGENT_TEMPLATES:
        sys.exit(f"错误: WORKER_AGENT={agent!r} 未知, 支持: {sorted(DEFAULT_AGENT_TEMPLATES)}")

    agent_cmd = (os.getenv("WORKER_AGENT_CMD") or "").strip() or DEFAULT_AGENT_TEMPLATES[agent]
    if "{prompt}" not in agent_cmd:
        sys.exit(f"错误: WORKER_AGENT_CMD 必须含占位符 {{prompt}}, 当前: {agent_cmd!r}")

    return Config(
        repo=repo,
        base_branch=args.base_branch or os.getenv("WORKER_BASE_BRANCH", "main").strip(),
        workspace=workspace,
        poll_interval=int(os.getenv("WORKER_POLL_INTERVAL", "60")),
        agent=agent,
        agent_cmd_template=agent_cmd,
        agent_timeout=int(os.getenv("WORKER_AGENT_TIMEOUT", "1800")),
        skill_path=os.getenv("WORKER_SKILL_PATH", "").strip(),
        git_ssh_host_alias=os.getenv("WORKER_GIT_SSH_HOST_ALIAS", "").strip(),
        labels=Labels(
            todo=os.getenv("WORKER_LABEL_TODO", "agent-todo").strip(),
            running=os.getenv("WORKER_LABEL_RUNNING", "agent-running").strip(),
            done=os.getenv("WORKER_LABEL_DONE", "agent-done").strip(),
            failed=os.getenv("WORKER_LABEL_FAILED", "agent-failed").strip(),
            human=os.getenv("WORKER_LABEL_HUMAN", "agent-needs-human").strip(),
        ),
        once=args.once,
        target_issue=args.issue,
        dry_pr=args.dry_pr,
    )


# ---------------------------------------------------------------------------
# subprocess helpers
# ---------------------------------------------------------------------------


def run(
    cmd: list[str],
    cwd: Path | None = None,
    capture: bool = True,
    check: bool = True,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    log.debug("run: %s (cwd=%s)", shlex.join(cmd), cwd)
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=capture,
        text=True,
        check=check,
        timeout=timeout,
    )


def gh_json(args: list[str]) -> Any:
    """运行 `gh ...` 并解析 JSON 输出。"""
    proc = run(["gh", *args])
    return json.loads(proc.stdout) if proc.stdout.strip() else None


def gh(args: list[str], check: bool = True) -> str:
    proc = run(["gh", *args], check=check)
    return proc.stdout


# ---------------------------------------------------------------------------
# GitHub
# ---------------------------------------------------------------------------


def find_next_issue(cfg: Config) -> dict[str, Any] | None:
    if cfg.target_issue is not None:
        log.info("--issue %s 模式, 跳过 label 过滤", cfg.target_issue)
        return get_issue(cfg, cfg.target_issue)

    issues = (
        gh_json(
            [
                "issue",
                "list",
                "--repo",
                cfg.repo,
                "--label",
                cfg.labels.todo,
                "--state",
                "open",
                "--limit",
                "1",
                "--json",
                "number,title,body,url,labels",
            ]
        )
        or []
    )
    return issues[0] if issues else None


def get_issue(cfg: Config, number: int) -> dict[str, Any]:
    return gh_json(
        [
            "issue",
            "view",
            str(number),
            "--repo",
            cfg.repo,
            "--json",
            "number,title,body,url,labels",
        ]
    )


def transition_label(cfg: Config, number: int, *, remove: str | None, add: str | None) -> None:
    if not remove and not add:
        return
    args = ["issue", "edit", str(number), "--repo", cfg.repo]
    if remove:
        args += ["--remove-label", remove]
    if add:
        args += ["--add-label", add]
    gh(args, check=False)


def comment_on_issue(cfg: Config, number: int, body: str) -> None:
    subprocess.run(
        ["gh", "issue", "comment", str(number), "--repo", cfg.repo, "--body-file", "-"],
        input=body,
        text=True,
        check=False,
    )


def create_pr(cfg: Config, branch: str, title: str, body: str) -> str:
    proc = subprocess.run(
        [
            "gh",
            "pr",
            "create",
            "--repo",
            cfg.repo,
            "--base",
            cfg.base_branch,
            "--head",
            branch,
            "--title",
            title,
            "--body-file",
            "-",
        ],
        input=body,
        text=True,
        capture_output=True,
        check=True,
    )
    return (proc.stdout or "").strip().splitlines()[-1] if proc.stdout else ""


def repo_https_url(repo: str) -> str:
    return f"https://github.com/{repo}"


# ---------------------------------------------------------------------------
# git workspace
# ---------------------------------------------------------------------------


def ensure_repo_clone(cfg: Config) -> Path:
    cfg.workspace.mkdir(parents=True, exist_ok=True)
    repo_dir = cfg.repo_dir
    if (repo_dir / ".git").exists():
        _maybe_align_remote_to_ssh_alias(cfg, repo_dir)
        return repo_dir
    if repo_dir.exists():
        sys.exit(f"错误: {repo_dir} 已存在但不是 git 仓库, 请清理后重试")
    log.info("clone %s -> %s", cfg.repo, repo_dir)
    run(["gh", "repo", "clone", cfg.repo, str(repo_dir)])
    _maybe_align_remote_to_ssh_alias(cfg, repo_dir)
    return repo_dir


def _maybe_align_remote_to_ssh_alias(cfg: Config, repo_dir: Path) -> None:
    """If WORKER_GIT_SSH_HOST_ALIAS is set, rewrite origin to use it.

    `gh repo clone` always uses `git@github.com:owner/repo.git`. On hosts where
    the user has multiple GitHub accounts and routes them via per-account SSH
    aliases in ~/.ssh/config (Host github-foo HostName github.com IdentityFile
    ~/.ssh/foo_ed25519), the default Host github.com SSH key may belong to the
    wrong account, causing `git push` to be rejected with a confusing 'denied
    to <other-account>' error. Setting WORKER_GIT_SSH_HOST_ALIAS=github-foo
    lets the worker route push through the correct key.
    """
    alias = cfg.git_ssh_host_alias
    if not alias:
        return
    desired = f"git@{alias}:{cfg.repo}.git"
    proc = run(["git", "remote", "get-url", "origin"], cwd=repo_dir, check=False)
    current = (proc.stdout or "").strip()
    if current == desired:
        return
    log.info("rewriting origin %s -> %s (per WORKER_GIT_SSH_HOST_ALIAS)", current, desired)
    run(["git", "remote", "set-url", "origin", desired], cwd=repo_dir)


def reset_to_base(cfg: Config, repo_dir: Path) -> None:
    """每次拿活之前都把工作树拉回干净的 base_branch。"""
    log.info("git fetch + reset to %s", cfg.base_branch)
    run(["git", "fetch", "origin", "--prune"], cwd=repo_dir)
    run(["git", "checkout", cfg.base_branch], cwd=repo_dir, check=False)
    run(["git", "reset", "--hard", f"origin/{cfg.base_branch}"], cwd=repo_dir)
    run(["git", "clean", "-fdx"], cwd=repo_dir)


def make_branch(repo_dir: Path, branch: str) -> None:
    run(["git", "checkout", "-B", branch], cwd=repo_dir)


def has_changes(repo_dir: Path) -> bool:
    proc = run(["git", "status", "--porcelain"], cwd=repo_dir)
    return bool(proc.stdout.strip())


def diffstat_last_commit(repo_dir: Path) -> str:
    """Diffstat of the latest commit. Must be called AFTER commit_all."""
    proc = run(["git", "diff", "--stat", "HEAD~1..HEAD"], cwd=repo_dir, check=False)
    out = (proc.stdout or "").strip()
    return out or "(no diffstat available)"


def commits_ahead_of_base(repo_dir: Path, base_branch: str) -> int:
    """Number of commits the current branch is ahead of origin/<base_branch>.

    Used to detect the case where the agent disregarded its no-git instruction
    and committed work itself: working tree is clean (`has_changes` False) but
    the branch is ahead of base. We accept the agent's commit and push it
    rather than throwing the work away."""
    proc = run(
        ["git", "rev-list", "--count", f"origin/{base_branch}..HEAD"],
        cwd=repo_dir,
        check=False,
    )
    try:
        return int((proc.stdout or "0").strip())
    except ValueError:
        return 0


def commit_all(repo_dir: Path, message: str) -> None:
    run(["git", "add", "-A"], cwd=repo_dir)
    run(["git", "commit", "-m", message, "--no-verify"], cwd=repo_dir)


def push_branch(repo_dir: Path, branch: str) -> None:
    run(["git", "push", "origin", branch, "--force-with-lease"], cwd=repo_dir)


# ---------------------------------------------------------------------------
# Branch / slug
# ---------------------------------------------------------------------------


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str, max_len: int = 40) -> str:
    s = _SLUG_RE.sub("-", text.lower()).strip("-")
    return (s[:max_len].rstrip("-")) or "task"


def branch_name_for(issue: dict[str, Any]) -> str:
    return f"agent/issue-{issue['number']}-{slugify(issue.get('title', ''))}"


# ---------------------------------------------------------------------------
# Agent invocation
# ---------------------------------------------------------------------------


PROMPT_TEMPLATE_FILE = Path(__file__).parent.parent / "templates" / "agent_prompt.md"
PR_BODY_TEMPLATE_FILE = Path(__file__).parent.parent / "templates" / "pr_body.md"


def render_prompt(cfg: Config, issue: dict[str, Any], repo_dir: Path, branch: str) -> str:
    template = PROMPT_TEMPLATE_FILE.read_text()
    skill_pointer = (
        f"Read this file BEFORE editing anything; it has the project-specific operating manual:\n"
        f"  {cfg.skill_path}"
        if cfg.skill_path
        else "(no project-specific operating manual configured)"
    )
    return template.format(
        number=issue["number"],
        repo=cfg.repo,
        title=issue.get("title", ""),
        body=issue.get("body") or "(empty body)",
        workdir=str(repo_dir),
        branch=branch,
        base_branch=cfg.base_branch,
        repo_url=repo_https_url(cfg.repo),
        issue_url=issue.get("url", ""),
        skill_pointer=skill_pointer,
    )


def build_agent_argv(cfg: Config, prompt: str) -> list[str]:
    """把 WORKER_AGENT_CMD 模板里的 {prompt} 替换成实际 prompt, 返回 argv。

    用 shlex 解析以兼容引号。占位符必须独立成一个 token, 避免 shell 注入。
    """
    tokens = shlex.split(cfg.agent_cmd_template)
    out: list[str] = []
    replaced = False
    for tok in tokens:
        if tok == "{prompt}":
            out.append(prompt)
            replaced = True
        elif "{prompt}" in tok:
            sys.exit(
                f"错误: agent 命令模板里 {{prompt}} 必须独立成一个 token, 不能跟其他字符拼在一起: {tok!r}"
            )
        else:
            out.append(tok)
    if not replaced:
        sys.exit(f"错误: agent 命令模板缺 {{prompt}}: {cfg.agent_cmd_template!r}")
    return out


class AgentTimeoutError(RuntimeError):
    """Agent ran out the wall clock. Worker should still inspect the worktree
    for changes — cursor-agent in particular often writes files then refuses to
    exit, so a timeout is not the same as 'agent did nothing'."""


def _kill_pgroup(pid: int, sig: int) -> None:
    try:
        os.killpg(os.getpgid(pid), sig)
    except (ProcessLookupError, PermissionError):
        pass


def invoke_agent(cfg: Config, repo_dir: Path, prompt: str) -> None:
    """Run the AI backend in a fresh process group; capture output to a file
    (not a pipe), so orphan grandchildren that inherit fds can't hold the
    parent hostage. Always kill the whole pgroup on exit to reap orphans
    (cursor-agent leaves a `worker-server` node behind every run)."""
    argv = build_agent_argv(cfg, prompt)
    agent_log = repo_dir.parent / f"{repo_dir.name}.agent.log"
    log.info(
        "invoking agent (%s) in %s, timeout=%ss, agent stdout/err -> %s",
        cfg.agent,
        repo_dir,
        cfg.agent_timeout,
        agent_log,
    )
    log.info("agent argv[0..2]: %s", argv[:3])

    proc: subprocess.Popen[bytes] | None = None
    try:
        with open(agent_log, "ab") as logf:
            logf.write(
                f"\n\n=== {time.strftime('%F %T')} agent run, argv[0..2]={argv[:3]} ===\n".encode()
            )
            logf.flush()
            proc = subprocess.Popen(
                argv,
                cwd=str(repo_dir),
                stdin=subprocess.DEVNULL,
                stdout=logf,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        try:
            rc = proc.wait(timeout=cfg.agent_timeout)
        except subprocess.TimeoutExpired:
            log.warning("agent did not exit in %ss; killing pgroup", cfg.agent_timeout)
            _kill_pgroup(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _kill_pgroup(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)
            raise AgentTimeoutError(f"agent timed out after {cfg.agent_timeout}s")
        if rc != 0:
            raise RuntimeError(f"agent exited non-zero: rc={rc}")
    except FileNotFoundError as e:
        raise RuntimeError(f"agent binary not found: {e}")
    finally:
        if proc is not None:
            _kill_pgroup(proc.pid, signal.SIGTERM)
            time.sleep(0.5)
            _kill_pgroup(proc.pid, signal.SIGKILL)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def run_checks(repo_dir: Path) -> str:
    """跑项目的 lint/test。MVP 版只跑 ruff + pytest, 都失败就报告失败但不阻断 PR。"""
    lines: list[str] = []
    for label, cmd in [
        ("ruff check", ["ruff", "check", "."]),
        ("pytest -q", ["pytest", "-q", "--maxfail=1"]),
    ]:
        if shutil.which(cmd[0]) is None:
            lines.append(f"- `{label}`: skipped (binary not on PATH)")
            continue
        try:
            run(cmd, cwd=repo_dir, check=True, timeout=600)
            lines.append(f"- `{label}`: ok")
        except subprocess.TimeoutExpired:
            lines.append(f"- `{label}`: TIMEOUT")
        except subprocess.CalledProcessError as e:
            lines.append(f"- `{label}`: FAILED (rc={e.returncode})")
    return "\n".join(lines) if lines else "(no checks run)"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def render_pr_body(
    cfg: Config, issue: dict[str, Any], branch: str, diffstat_text: str, checks_text: str
) -> str:
    template = PR_BODY_TEMPLATE_FILE.read_text()
    return template.format(
        number=issue["number"],
        title=issue.get("title", ""),
        diffstat=diffstat_text,
        checks=checks_text,
        agent=cfg.agent,
        branch=branch,
        base_branch=cfg.base_branch,
        issue_url=issue.get("url", ""),
    )


def process_issue(cfg: Config, issue: dict[str, Any]) -> None:
    n = int(issue["number"])
    title = issue.get("title", "")
    log.info("=== issue #%s: %s ===", n, title)

    transition_label(cfg, n, remove=cfg.labels.todo, add=cfg.labels.running)

    try:
        repo_dir = ensure_repo_clone(cfg)
        reset_to_base(cfg, repo_dir)
        branch = branch_name_for(issue)
        make_branch(repo_dir, branch)

        prompt = render_prompt(cfg, issue, repo_dir, branch)
        try:
            invoke_agent(cfg, repo_dir, prompt)
        except AgentTimeoutError as e:
            # cursor-agent often writes files then refuses to exit. Don't bail
            # — fall through and let the diff check below decide. If the agent
            # actually wrote nothing, the no-changes branch will mark
            # agent-needs-human anyway.
            log.warning("%s; checking worktree for any salvageable diff", e)
            comment_on_issue(
                cfg,
                n,
                f"Agent process timed out after {cfg.agent_timeout}s and was killed. "
                f"Worker is salvaging any files the agent wrote before exit.",
            )

        if (repo_dir / "AGENT_NEEDS_HUMAN.md").exists():
            note = (repo_dir / "AGENT_NEEDS_HUMAN.md").read_text()
            comment_on_issue(
                cfg,
                n,
                f"Agent declined to act and asked for human review:\n\n```\n{note[:4000]}\n```",
            )
            transition_label(cfg, n, remove=cfg.labels.running, add=cfg.labels.human)
            return

        ahead = commits_ahead_of_base(repo_dir, cfg.base_branch)
        worktree_dirty = has_changes(repo_dir)
        if not worktree_dirty and ahead == 0:
            comment_on_issue(
                cfg, n, "Worker ran the agent but no files changed. Marking for human review."
            )
            transition_label(cfg, n, remove=cfg.labels.running, add=cfg.labels.human)
            return

        if worktree_dirty:
            commit_all(repo_dir, f"agent: implement issue #{n}\n\n{title}")
        else:
            log.warning(
                "agent committed %d commit(s) itself (violates skill no-git rule); accepting and pushing",
                ahead,
            )
        ds = diffstat_last_commit(repo_dir)
        checks_text = run_checks(repo_dir)

        if cfg.dry_pr:
            log.info("--dry-pr 模式: 已 commit 但跳过 push/PR。diffstat:\n%s", ds)
            comment_on_issue(
                cfg,
                n,
                f"Dry-run: agent produced changes but worker is in --dry-pr mode.\n\n```\n{ds}\n```",
            )
            transition_label(cfg, n, remove=cfg.labels.running, add=cfg.labels.human)
            return

        push_branch(repo_dir, branch)
        pr_title = f"agent: {title}".strip() or f"agent: implement issue #{n}"
        pr_body = render_pr_body(cfg, issue, branch, ds, checks_text)
        pr_url = create_pr(cfg, branch, pr_title, pr_body)

        comment_on_issue(cfg, n, f"Worker opened PR: {pr_url}")
        transition_label(cfg, n, remove=cfg.labels.running, add=cfg.labels.done)
        log.info("done #%s -> %s", n, pr_url)

    except Exception as e:
        log.exception("issue #%s failed", n)
        comment_on_issue(cfg, n, f"Worker failed:\n\n```\n{type(e).__name__}: {e}\n```")
        transition_label(cfg, n, remove=cfg.labels.running, add=cfg.labels.failed)


def main_loop(cfg: Config) -> None:
    log.info(
        "worker starting: repo=%s base=%s agent=%s workspace=%s",
        cfg.repo,
        cfg.base_branch,
        cfg.agent,
        cfg.workspace,
    )

    while True:
        try:
            issue = find_next_issue(cfg)
            if issue is None:
                if cfg.once or cfg.target_issue is not None:
                    log.info("no matching issue; --once mode -> exiting")
                    return
                log.info("idle; sleeping %ss", cfg.poll_interval)
                time.sleep(cfg.poll_interval)
                continue
            process_issue(cfg, issue)
            if cfg.once or cfg.target_issue is not None:
                log.info("--once mode -> exiting after one issue")
                return
            time.sleep(5)
        except KeyboardInterrupt:
            log.info("interrupted, bye")
            return


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="github-issue-worker: poll → AI agent → PR")
    p.add_argument("--repo", help="owner/repo (覆盖 WORKER_REPO)")
    p.add_argument("--base-branch", help="PR base 分支 (覆盖 WORKER_BASE_BRANCH)")
    p.add_argument(
        "--agent", choices=sorted(DEFAULT_AGENT_TEMPLATES), help="AI 后端 (覆盖 WORKER_AGENT)"
    )
    p.add_argument("--once", action="store_true", help="只处理一个 issue 然后退出 (适合定时任务)")
    p.add_argument(
        "--issue", type=int, help="只处理这个具体的 issue 号 (跳过 label 过滤, 用于调试)"
    )
    p.add_argument("--dry-pr", action="store_true", help="跑完 agent 也 commit, 但不 push 不建 PR")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    cfg = load_config(args)

    # 跳过 LFS 拉取。worker 跑在 dimos 这种 LFS 大仓库里, 默认 clone/checkout 会
    # 触发 git-lfs pull, 动辄几 GB, 直接把 worker 卡死。绝大多数 agent 任务 (改
    # 文档/源代码) 用占位符就够; 真要操作 LFS 文件可以让 agent 显式 git lfs pull。
    os.environ.setdefault("GIT_LFS_SKIP_SMUDGE", "1")

    # 保证 git commit 有 author。worker 跑在专用工作区, 不依赖宿主 ~/.gitconfig。
    # 用户可以通过环境变量覆盖 (e.g. 想让 commit author 是真人邮箱用于 GitHub
    # 链接到账号)。
    os.environ.setdefault("GIT_AUTHOR_NAME", "github-issue-worker")
    os.environ.setdefault("GIT_AUTHOR_EMAIL", "github-issue-worker@local")
    os.environ.setdefault("GIT_COMMITTER_NAME", "github-issue-worker")
    os.environ.setdefault("GIT_COMMITTER_EMAIL", "github-issue-worker@local")

    def _sigterm(_signo: int, _frame: object) -> None:
        log.info("SIGTERM received, exiting after current issue")
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _sigterm)
    main_loop(cfg)


if __name__ == "__main__":
    main()
