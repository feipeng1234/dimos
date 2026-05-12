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

"""FastAPI server: 飞书消息 -> GitHub Issue。

设计原则:
- 单文件搞定 MVP, 不做臆测性抽象。
- 飞书签名校验在 Encrypt Key 配置时启用, 否则跳过 (兼容"不开加密"的简单场景)。
- 解析极简: 只支持 /repo /title /body 三个指令, 缺啥就用默认。
- GitHub 调用失败直接抛 HTTP 500, 让飞书侧能在事件订阅后台看到失败。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
import requests

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()
DEFAULT_REPO = os.getenv("DEFAULT_REPO", "").strip()
FEISHU_VERIFICATION_TOKEN = os.getenv("FEISHU_VERIFICATION_TOKEN", "").strip()
FEISHU_ENCRYPT_KEY = os.getenv("FEISHU_ENCRYPT_KEY", "").strip()
ISSUE_LABEL = os.getenv("ISSUE_LABEL", "agent-todo").strip()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("feishu-to-issue")

app = FastAPI(title="feishu-to-github-issue")


# ---------- GitHub ----------


def create_github_issue(repo: str, title: str, body: str) -> dict[str, Any]:
    if not GITHUB_TOKEN:
        raise HTTPException(500, "GITHUB_TOKEN 未配置, 看 .env.example")
    if "/" not in repo:
        raise HTTPException(400, f"repo 格式错误, 需要 owner/repo, 收到: {repo!r}")

    url = f"https://api.github.com/repos/{repo}/issues"
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        json={"title": title, "body": body, "labels": [ISSUE_LABEL]},
        timeout=20,
    )
    if resp.status_code >= 300:
        log.error("github api %s: %s", resp.status_code, resp.text[:500])
        raise HTTPException(502, f"github api {resp.status_code}: {resp.text[:200]}")
    return resp.json()


# ---------- 飞书签名 ----------


def verify_lark_signature(
    timestamp: str | None, nonce: str | None, signature: str | None, raw_body: bytes
) -> bool:
    """启用 Encrypt Key 时校验签名; 未启用则一律放行。

    算法: sha256(timestamp + nonce + encrypt_key + body), hex 输出。
    参考: https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/event-subscription-guide/event-subscription-configure-/encrypt-key-encryption-configuration
    """
    if not FEISHU_ENCRYPT_KEY:
        return True
    if not (timestamp and nonce and signature):
        log.warning("缺少签名头, Encrypt Key 已启用但请求未带签名")
        return False
    h = hashlib.sha256()
    h.update(f"{timestamp}{nonce}{FEISHU_ENCRYPT_KEY}".encode())
    h.update(raw_body)
    return h.hexdigest() == signature


# ---------- 飞书消息解析 ----------


def extract_text(message: dict[str, Any]) -> str:
    """从飞书消息体提取纯文本。content 字段是 JSON 字符串。"""
    content = message.get("content", "")
    if not content:
        return ""
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return content
    return (parsed.get("text") or "").strip()


def strip_at_mentions(text: str) -> str:
    """飞书 @人 在 text 里是 @_user_1 这种占位, 去掉它们让指令解析更稳。

    必须按行处理, 保留换行结构, 否则 /title 和 /body 会被拼成一行解析错。
    """
    out = []
    for line in text.splitlines():
        tokens = [tok for tok in line.split(" ") if not tok.startswith("@_user_")]
        out.append(" ".join(tokens).strip())
    return "\n".join(out)


def parse_command(text: str) -> tuple[str, str, str]:
    """支持的指令格式:

        /repo owner/repo
        /title 标题
        /body 正文 (可多行)

    缺省策略:
        - repo 默认 DEFAULT_REPO
        - title 缺失 -> 取首行的前 80 字
        - body 缺失 -> 取整段消息
    """
    repo = DEFAULT_REPO
    title: str | None = None
    body_lines: list[str] = []
    body_mode = False

    for line in text.splitlines():
        s = line.strip()
        if s.startswith("/repo "):
            repo = s[len("/repo ") :].strip()
            body_mode = False
        elif s.startswith("/title "):
            title = s[len("/title ") :].strip()
            body_mode = False
        elif s.startswith("/body "):
            body_lines = [s[len("/body ") :].strip()]
            body_mode = True
        elif body_mode:
            body_lines.append(line)

    body = "\n".join(body_lines).strip() if body_lines else text.strip()
    if not title:
        first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        title = first[:80] or "(untitled feishu task)"
    return repo, title, body


# ---------- 路由 ----------


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "ok": True,
        "github_token_set": bool(GITHUB_TOKEN),
        "default_repo": DEFAULT_REPO or None,
        "encrypt_key_set": bool(FEISHU_ENCRYPT_KEY),
        "verification_token_set": bool(FEISHU_VERIFICATION_TOKEN),
        "issue_label": ISSUE_LABEL,
    }


@app.post("/test/issue")
async def test_issue(request: Request) -> dict[str, Any]:
    """绕过飞书, 直接喂 {"text": "..."} 测整条创建链路。"""
    payload = await request.json()
    text = (payload.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "缺 text 字段")
    repo, title, body = parse_command(text)
    issue = create_github_issue(repo, title, f"{body}\n\n---\nCreated via /test/issue.")
    return {"ok": True, "issue_number": issue["number"], "issue_url": issue["html_url"]}


@app.post("/feishu/events")
async def feishu_events(request: Request) -> dict[str, Any]:
    raw = await request.body()

    if not verify_lark_signature(
        request.headers.get("X-Lark-Request-Timestamp"),
        request.headers.get("X-Lark-Request-Nonce"),
        request.headers.get("X-Lark-Signature"),
        raw,
    ):
        raise HTTPException(401, "lark signature mismatch")

    try:
        payload: dict[str, Any] = json.loads(raw or b"{}")
    except json.JSONDecodeError:
        raise HTTPException(400, "invalid json")

    # 1. URL 验证 (订阅事件时飞书会先打一次)
    if payload.get("type") == "url_verification":
        return {"challenge": payload.get("challenge")}

    # 2. 老格式 verification_token 校验 (v1 事件); v2 事件用签名, 这里跳过
    if "type" in payload and FEISHU_VERIFICATION_TOKEN:
        if payload.get("token") and payload["token"] != FEISHU_VERIFICATION_TOKEN:
            raise HTTPException(401, "verification token mismatch")

    # 3. 提取消息
    header = payload.get("header") or {}
    event_type = header.get("event_type") or payload.get("event", {}).get("type")
    event = payload.get("event") or {}
    message = event.get("message") or {}

    if event_type and event_type != "im.message.receive_v1":
        log.info("忽略事件类型: %s", event_type)
        return {"ok": True, "ignored": event_type}

    text = strip_at_mentions(extract_text(message))
    if not text:
        log.info("空消息, 忽略")
        return {"ok": True, "ignored": "empty"}

    repo, title, body = parse_command(text)
    log.info("creating issue: repo=%s title=%s", repo, title)

    sender_id = (event.get("sender") or {}).get("sender_id", {}).get("user_id", "?")
    issue = create_github_issue(
        repo=repo,
        title=title,
        body=f"{body}\n\n---\nCreated from Feishu by user `{sender_id}`.",
    )
    return {
        "ok": True,
        "issue_number": issue["number"],
        "issue_url": issue["html_url"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
