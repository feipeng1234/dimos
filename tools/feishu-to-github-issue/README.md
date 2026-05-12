# feishu-to-github-issue 使用手册

把飞书消息 → GitHub Issue 的最小可用 **工具 (tool)**。

> ## 这不是什么 / 这是什么
>
> - **不是** Cursor Agent Skill。它跟 `.cursor/skills/` 下的东西没关系, Cursor IDE 里的
>   AI agent 不会读它、也不会调用它。
> - **是** 一个独立的 HTTP 后台服务: 起在 `localhost:8000`, 接收飞书 webhook,
>   调 GitHub API 建 issue。和 dimos 主代码完全隔离, 自带 venv, 自带依赖。
> - 放在 `tools/` 下, 跟其它"会跑起来的小工具"一类。
>
> 参考来源: `cursor_24_h_autonomous_worker_architecture.md` 第 6 节。原文里把它叫
> "Skill 1", 但那是文档作者的叫法 — 在我们这个仓库的语境里, 它就是一个 tool。

## 1. 安装

```bash
cd tools/feishu-to-github-issue
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
# 没有 uv 就用: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

## 2. 配置凭证

```bash
cp .env.example .env
$EDITOR .env
```

至少填两项:

| 变量             | 取值                                                                  | 说明                       |
| ---------------- | --------------------------------------------------------------------- | -------------------------- |
| `GITHUB_TOKEN`   | https://github.com/settings/tokens 生成 (经典 token, 勾 `repo` 权限) | 不填没法建 issue           |
| `DEFAULT_REPO`   | `your_name/your_repo`                                                 | 飞书消息没指定 `/repo` 时用 |

可选:

| 变量                        | 何时填                                                                                 |
| --------------------------- | -------------------------------------------------------------------------------------- |
| `FEISHU_ENCRYPT_KEY`        | 飞书后台 "事件订阅 → 加密策略" 设置了 Encrypt Key, 这里要填一致, 否则签名失败          |
| `FEISHU_VERIFICATION_TOKEN` | 老 v1 事件用, 新版 v2 事件 (有 `header.event_type`) 用签名, 一般不用填                 |
| `ISSUE_LABEL`               | 默认 `agent-todo`, 给 worker loop 识别                                                 |
| `PORT`                      | 默认 8000                                                                              |

## 3. 启动

```bash
python scripts/server.py
# 或
uvicorn scripts.server:app --host 0.0.0.0 --port 8000 --reload
```

看到这行就 OK:

```
Uvicorn running on http://0.0.0.0:8000
```

## 4. 本地冒烟测 (不需要飞书也能验证)

开一个新终端:

```bash
source .venv/bin/activate

# 4.1 看配置是否到位
curl -s localhost:8000/healthz | python -m json.tool

# 4.2 模拟飞书 url_verification (订阅事件时飞书会先打这个)
python scripts/send_test_event.py verify
# 期望: 200 {"challenge":"abc123"}

# 4.3 模拟一条飞书消息走完整链路 (会真的建 issue!)
python scripts/send_test_event.py message "/title smoke test from feishu sim"
# 期望: 200 {"ok":true,"issue_number":N,"issue_url":"..."}

# 4.4 直接绕过飞书走 /test/issue (同样会真建 issue)
python scripts/send_test_event.py issue "/title direct test"
```

## 5. 暴露公网 (让飞书能打到你)

本地开发用 ngrok / cloudflared / frp 都行:

```bash
ngrok http 8000
# 拿到类似 https://abc123.ngrok-free.app
```

## 6. 接到飞书

1. 飞书开放平台 https://open.feishu.cn/app 创建自建应用 (或用已有的)
2. 左侧 "事件与回调" → "事件订阅":
   - 请求地址: `https://你的公网地址/feishu/events`
   - (可选) 启用加密策略, 把 Encrypt Key 复制到 `.env` 的 `FEISHU_ENCRYPT_KEY`, 重启服务
3. 添加事件: **接收消息 v2.0** (`im.message.receive_v1`)
4. 左侧 "权限管理" 开权限:
   - `im:message` (接收消息)
   - `im:message.group_at_msg` (群里 @机器人)
   - `im:message.p2p_msg` (单聊收消息)
5. 创建版本并发布, 等待审核 (内测可直接发布)
6. 把机器人拉进群, @它 发:

   ```
   @机器人 /title 测试任务
   /body 这是从飞书来的任务描述
   ```

   你的 GitHub 仓库就会出现一个带 `agent-todo` label 的新 issue。

## 7. 常见问题

| 现象                                  | 原因 / 解决                                                              |
| ------------------------------------- | ------------------------------------------------------------------------ |
| 飞书后台保存订阅时报"验证失败"        | 服务没起 / ngrok 地址换了 / 启用了加密但 `.env` 没填 `FEISHU_ENCRYPT_KEY` |
| `/healthz` 显示 `github_token_set: false` | `.env` 没生效, 检查路径和 `python-dotenv` 是否装了                         |
| 建 issue 返回 401                     | `GITHUB_TOKEN` 过期 / 没勾 `repo` 权限                                   |
| 建 issue 返回 404                     | `DEFAULT_REPO` 写错 (大小写也敏感) 或 token 没权限访问该仓库             |
| 建 issue 返回 422 提到 label          | 仓库里没有 `agent-todo` label, 先 `gh label create agent-todo` 一下       |

## 8. 下一步

跑通这一段之后, 才是文档里的 Skill 2 (`github-issue-to-pr`) 和 Skill 3
(`pr-review-and-fix`)。本仓库目前只实现了 Skill 1 这一段。
