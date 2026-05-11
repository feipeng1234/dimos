#!/usr/bin/env bash
# 一次性给仓库创建 worker 用的 5 个 label。
# 用法:
#   ./scripts/setup_labels.sh                       # 用 .env 里的 WORKER_REPO
#   ./scripts/setup_labels.sh owner/repo            # 显式指定仓库
#
# 已存在的 label 会被跳过 (gh label create 报错时忽略)。

set -euo pipefail

if [ -n "${1:-}" ]; then
  REPO="$1"
else
  if [ -f .env ]; then
    # shellcheck disable=SC1091
    source <(grep -E '^WORKER_REPO=' .env | sed 's/^/export /')
  fi
  REPO="${WORKER_REPO:-}"
fi

if [ -z "$REPO" ]; then
  echo "需要一个 owner/repo 参数, 或在 .env 设 WORKER_REPO" >&2
  exit 1
fi

echo "Creating labels in $REPO ..."

create_or_skip() {
  local name="$1" color="$2" desc="$3"
  if gh label create "$name" --color "$color" --description "$desc" --repo "$REPO" 2>/dev/null; then
    echo "  + $name"
  else
    echo "  = $name (already exists, skipped)"
  fi
}

create_or_skip agent-todo        0e8a16 "Worker should pick this up"
create_or_skip agent-running     fbca04 "Worker is currently working on this"
create_or_skip agent-done        1d76db "Worker finished and opened a PR"
create_or_skip agent-failed      d73a4a "Worker errored out, see issue comments"
create_or_skip agent-needs-human b60205 "Worker bailed; human review required"

echo "Done."
