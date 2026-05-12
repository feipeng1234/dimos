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

"""本地冒烟测: 模拟飞书事件 / URL 验证, 不需要真飞书。

用法:
    python send_test_event.py verify
    python send_test_event.py message "/repo me/repo\\n/title hi\\n/body body text"
    python send_test_event.py issue "/repo me/repo\\n/title hi"   # 走 /test/issue, 真建 issue

默认打 http://localhost:8000, 改 BASE 环境变量可改地址。
"""

from __future__ import annotations

import json
import os
import sys

import requests

BASE = os.getenv("BASE", "http://localhost:8000")


def cmd_verify() -> None:
    r = requests.post(
        f"{BASE}/feishu/events",
        json={"type": "url_verification", "challenge": "abc123", "token": "x"},
        timeout=10,
    )
    print(r.status_code, r.text)


def cmd_message(text: str) -> None:
    payload = {
        "schema": "2.0",
        "header": {"event_type": "im.message.receive_v1", "event_id": "test-1"},
        "event": {
            "sender": {"sender_id": {"user_id": "test_user"}},
            "message": {
                "message_type": "text",
                "content": json.dumps({"text": text}),
            },
        },
    }
    r = requests.post(f"{BASE}/feishu/events", json=payload, timeout=20)
    print(r.status_code, r.text)


def cmd_issue(text: str) -> None:
    r = requests.post(f"{BASE}/test/issue", json={"text": text}, timeout=20)
    print(r.status_code, r.text)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    cmd = sys.argv[1]
    if cmd == "verify":
        cmd_verify()
    elif cmd == "message":
        cmd_message(sys.argv[2] if len(sys.argv) > 2 else "/title hello")
    elif cmd == "issue":
        cmd_issue(sys.argv[2] if len(sys.argv) > 2 else "/title hello")
    else:
        print(f"unknown cmd: {cmd}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
