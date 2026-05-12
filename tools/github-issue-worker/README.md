# github-issue-to-pr

把带 `agent-todo` label 的 GitHub Issue → AI agent 改代码 → 自动开 PR 的最小可跑
**worker daemon**, 跟它的 SKILL.md 操作手册打成一个**自包含的 skill**。

> ## 目录里都是什么
>
> ```
> .cursor/skills/github-issue-to-pr/
> ├── SKILL.md              ← 给 worker 拉起的 AI agent 看的操作手册
> ├── README.md             ← 你现在在看的这个
> ├── .env.example
> ├── requirements.txt
> ├── scripts/
> │   ├── worker.py         ← 后台 daemon 本体
> │   └── setup_labels.sh
> └── templates/
>     ├── agent_prompt.md   ← worker 喂给 AI 的任务模板
>     └── pr_body.md        ← PR 正文模板
> ```
>
> 整个目录拷走就是完整的产品; 删除目录也不留尾巴。
>
> 跟 `tools/feishu-to-github-issue` 配合可以走完整链路:
>
> ```
> 飞书消息  ──[feishu-to-github-issue]──>  GitHub Issue  ──[本 skill]──>  PR
> ```
>
> 参考来源: `cursor_24_h_autonomous_worker_architecture.md` 第 4 / 7 / 8 / 12 节。

## 1. 总览

每隔 60 秒做一遍这件事:

```text
gh issue list --label agent-todo --state open
        │
        ▼
 拿一个 issue
        │
        ├─→ label: agent-todo → agent-running
        │
        ▼
 git fetch + reset 到 base 分支 → checkout -B agent/issue-N-slug
        │
        ▼
 起 claude / cursor-agent / codex 子进程, 把 prompt 塞进去
 (agent 只改文件, 不动 git)
        │
        ▼
 git diff 检查
   ├─ 没改动            → label: agent-needs-human, 评论
   ├─ 写了 AGENT_NEEDS_HUMAN.md → label: agent-needs-human, 评论
   └─ 改了              → commit + push + gh pr create
                         → label: agent-done, 评论 PR 链接
        │
        ▼
 出错              → label: agent-failed, 评论 traceback
```

**Worker 管 git, agent 只改文件**。这是有意的安全分工, agent 没法误 push 主分支。

## 2. 安装

```bash
cd .cursor/skills/github-issue-to-pr
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
# 没有 uv 就用: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

系统侧前置依赖:

| 工具 | 用途 | 检查命令 |
|------|------|----------|
| `gh` | 所有 GitHub API 调用 (list issue / 改 label / 评论 / 建 PR) | `gh auth status` |
| `git` | 克隆 / 切分支 / commit / push | `git --version` |
| `claude` | 默认 AI 后端 (可换) | `claude --version` |

`gh` 必须是已登录状态 (`gh auth login`), 因为 worker 直接调 `gh issue` / `gh pr`。

## 3. 配置

```bash
cp .env.example .env
$EDITOR .env
```

至少改两项:

| 变量 | 取值 | 说明 |
|------|------|------|
| `WORKER_REPO` | `your_name/your_repo` | worker 监听哪个仓库 |
| `WORKER_BASE_BRANCH` | `dev` 或 `main` | PR 的目标分支 |

可选 (常见):

| 变量 | 何时改 |
|------|--------|
| `WORKER_AGENT` | 默认 `claude`; 想用 cursor-agent / codex / dry 在这里改 |
| `WORKER_AGENT_CMD` | 想给 agent 加额外参数 (比如 `claude --dangerously-skip-permissions -p {prompt}`) |
| `WORKER_AGENT_TIMEOUT` | 默认 1800 秒, 任务复杂可调大 |
| `WORKER_WORKSPACE` | worker 把仓库 clone 到这里, 默认 `~/agent-workspace` |
| `WORKER_SKILL_PATH` | 给 agent 看的操作手册路径 (默认 `.cursor/skills/github-issue-to-pr/SKILL.md`) |

## 4. 一次性: 给仓库建 5 个 label

```bash
./scripts/setup_labels.sh                    # 用 .env 里的 WORKER_REPO
./scripts/setup_labels.sh owner/repo         # 显式指定
```

会建:

```
agent-todo          worker 该接的活
agent-running       worker 正在干
agent-done          已经开了 PR
agent-failed        出错了, 看评论
agent-needs-human   worker 不敢动, 求人介入
```

## 5. 本地冒烟测 (不调真 AI, 不真 push)

最小回路用 `--agent dry --dry-pr` 跑一遍, 验证 worker 能正确 clone / 切分支 /
检测 diff / 改 label, 但不调真 AI 不真建 PR:

```bash
# 1. 在 GitHub 上手动建一个测试 issue, 打上 agent-todo
gh issue create \
  --repo "$WORKER_REPO" \
  --title "Smoke test" \
  --body "Worker 自检, 不要合并" \
  --label agent-todo

# 2. 用 dry agent + dry-pr 跑一次 (会 commit, 但不 push 不建 PR)
.venv/bin/python scripts/worker.py --once --agent dry --dry-pr

# 期望:
#  - 仓库被 clone 到 ~/agent-workspace/<repo_name>
#  - 新建了 agent/issue-N-smoke-test 分支
#  - 工作树里多了一个 AGENT_RAN.txt
#  - 已经 git commit, 但没 push
#  - issue label 变成 agent-needs-human (因为 --dry-pr 模式不建 PR)
```

调通这一步, 就可以换成真 AI:

```bash
# 3. 真 AI + 还是不 push (再次确认 prompt 工程)
.venv/bin/python scripts/worker.py --once --agent claude --dry-pr

# 4. 真 AI + 真 push + 真建 PR
.venv/bin/python scripts/worker.py --once --agent claude
```

## 6. 跑成 24h daemon

最简单粗暴: 直接挂在 tmux / nohup 后面。

```bash
nohup .venv/bin/python scripts/worker.py > logs/worker.log 2>&1 &
```

要 systemd:

```ini
# /etc/systemd/system/github-issue-worker.service
[Unit]
Description=github-issue-worker
After=network.target

[Service]
Type=simple
User=lenovo
WorkingDirectory=/home/lenovo/dimos/tools/github-issue-worker
ExecStart=/home/lenovo/dimos/tools/github-issue-worker/.venv/bin/python scripts/worker.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now github-issue-worker
journalctl -u github-issue-worker -f
```

## 7. 关于 AI agent 后端

Worker 把每个后端建模成一个命令模板, `{prompt}` 占位符会在运行时被替换成完整任务
描述。内置默认:

| `WORKER_AGENT` | 默认命令 | 说明 |
|----------------|----------|------|
| `claude` | `claude -p {prompt}` | Anthropic Claude Code, 通常需要加 `--dangerously-skip-permissions` 才能写文件 |
| `cursor-agent` | `cursor-agent -p {prompt} --output-format text` | Cursor CLI |
| `codex` | `codex exec {prompt}` | OpenAI Codex CLI |
| `dry` | `bash -c 'echo dry-run > AGENT_RAN.txt'` | 不调任何 AI, 端到端冒烟用 |

要换命令: 在 `.env` 里设 `WORKER_AGENT_CMD`, 必须包含 `{prompt}` 占位符, 例如:

```bash
WORKER_AGENT_CMD=claude --dangerously-skip-permissions -p {prompt}
```

> **注意**: claude 在子进程里一般不会自动允许写文件, 必须显式给权限, 否则 agent
> 会"想改但改不了", 最后被 worker 标 `agent-needs-human`。第一次跑前先用
> `--once --dry-pr` 确认权限对了。

## 8. 标签状态机

```text
agent-todo
   │   (worker 接活)
   ▼
agent-running
   │
   ├─ 成功 (有 diff + 建了 PR)        → agent-done
   ├─ agent 没改东西                  → agent-needs-human
   ├─ agent 写了 AGENT_NEEDS_HUMAN.md → agent-needs-human
   ├─ --dry-pr 模式                   → agent-needs-human
   └─ 任何异常                        → agent-failed
```

想让 worker 重新接一个失败的 issue: 把 label 从 `agent-failed` 改回 `agent-todo`
就行, worker 下一轮会再捡起来。

## 9. 常见问题

| 现象 | 原因 / 解决 |
|------|-------------|
| `worker.py` 启动报 `gh: command not found` | 装 GitHub CLI: `sudo apt install gh && gh auth login` |
| issue 一直卡 `agent-running` | worker 进程崩了/被 kill 但 label 没回滚。手动改回 `agent-todo` 即可 |
| agent 跑完没 diff (一直被标 `agent-needs-human`) | claude 没拿到写文件权限。改 `WORKER_AGENT_CMD` 加 `--dangerously-skip-permissions` |
| `git push` 报 `non-fast-forward` | 分支被别人改过, worker 已经用 `--force-with-lease`, 一般会自动恢复 |
| PR 标题/body 看着不对 | 改 `templates/pr_body.md` (worker 启动时读, 改完重启) |
| 想给 agent 看更详细的项目规则 | 改 `.cursor/skills/github-issue-to-pr/SKILL.md`, worker 会把它的路径塞进 prompt |

## 10. 安全边界

按原架构第 11 节, worker 默认遵守:

- 永远不 push `main` / `master` (worker 只 push 到 `agent/issue-N-...` 分支, base 是
  你 `WORKER_BASE_BRANCH` 指定的, 一般是 `dev`)
- 给 agent 的 prompt 里硬编码: 不许动 `.env` / secrets / deploy keys / CI 文件
- agent 完全没权碰 git (worker 自己管)
- `commit` 用 `--no-verify` 跳过 pre-commit hooks (避免被 hook 卡死), CI 在 PR
  上还是会跑, 不影响最终质量门

GitHub 端额外强烈建议:

- `main` / `dev` 开分支保护, 强制 PR review
- worker 用的 token (或 `gh auth login` 用的账号) 只给最小权限: Issues / PR /
  Contents 写, 其他只读
