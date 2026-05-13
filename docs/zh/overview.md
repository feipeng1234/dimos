# DimOS 项目概览

按 [`AGENTS.md`](../../AGENTS.md) 的表述，DimOS 是面向 **generalist robotics（通用机器人）** 的 **agentic operating system（智能体原生操作系统）**：各**模块（Module）**通过类型化的流，经 LCM、ROS2、DDS 等**传输（Transport）**互联；**蓝图（Blueprint）**把模块装配成可运行的机器人栈；**技能（Skill）**让智能体能够调用抓取、追踪、跳跃等具象到硬件上的动作。

本文为中文读者的**导读**；详尽设计与术语仍以仓库内英文文档与源码为准。文内到站点的 Markdown 链接均相对于本文件所在目录 `docs/zh/` 书写，便于在 GitHub 与本地预览中正确跳转。

---

## DimOS 解决什么问题？

- **一体化**：免去自行拼凑 ROS、仿真、厂商 SDK、模型推理的步骤；DimOS 用统一的**模块**与**蓝图**描述子系统怎样部署与连接。
- **Agent 原生**：智能体可作为系统中的原生组件运行——订阅传感器与状态流，并通过 **技能** 与 **MCP** 驱动底层栈。
- **多形态载体**：内置四足（如 Unitree Go2）、人形（如 G1）、机械臂（如 xArm）、无人机（MAVLink / DJI）等蓝图与示例；各平台成熟度见英文 [`README.md`](../../README.md) 硬件表。

入门说明亦见仓库根目录英文 [`README.md`](../../README.md)。

---

## 核心概念（导读）

| 概念 | 含义（简述） | 延伸阅读 |
|------|----------------|----------|
| **模块（Module）** | 并行运行的 Python 子系统，通过声明式输入/输出流协作 | [模块](../usage/modules.md) |
| **流（Stream）** | 模块之间的发布/订阅通信 | [传感器流](../usage/sensor_streams/README.md)、[数据流](../usage/data_streams/README.md) |
| **蓝图（Blueprint）** | 模块组合与连线（含 `autoconnect` 等） | [蓝图](../usage/blueprints.md) |
| **RPC** | 模块间远程过程调用（参数序列化） | [蓝图文档 RPC 小节](../usage/blueprints.md#calling-the-methods-of-other-modules) |
| **技能（Skill）** | 可被 AI 智能体当作工具调用的 RPC（常用 `@skill`） | [蓝图文档 Skills](../usage/blueprints.md#defining-skills) |
| **传输（Transport）** | LCM、共享内存、ROS2、DDS 等 | [传输总览](../usage/transports/index.md) |

概念索引页（英文）：[`docs/usage/README.md`](../usage/README.md)。

---

## 智能体、技能与 MCP

典型流程：蓝图内同时挂载 **MCP 服务端**（对外暴露技能）与 **MCP 客户端**（连接 LLM），再用 CLI 启停栈、发指令与调试。**MCP 仅在蓝图包含 `McpServer` 时可用**（与 [`AGENTS.md`](../../AGENTS.md) 一致）。

下列命令摘自 [`AGENTS.md`](../../AGENTS.md) 中 **Quick Start** 与 **Tools available to you (MCP)**（无硬件时可用 `--replay` / `--simulation`，见该文档）：

```bash
dimos --replay run unitree-go2-agentic --daemon
dimos status
dimos log -f
dimos agent-send "walk forward 2 meters then wave"
dimos mcp list-tools
dimos mcp call move --arg x=0.5 --arg duration=2.0
dimos stop
```

能力与演示索引（英文）：[`docs/capabilities/agents/readme.md`](../capabilities/agents/readme.md)。  
CLI 参考：[命令行文档](../usage/cli.md)。  
编码智能体约定（`@skill`、pytest、分支前缀等）：[`AGENTS.md`](../../AGENTS.md)。

---

## 安装与上手

```bash
uv sync --extra all
dimos list
```

环境与依赖说明：[`docs/requirements.md`](../requirements.md)；安装变体与其它入口见 [`README.md`](../../README.md) **Installation**。  
离线/仿真入门示例仍以前述英文 README **Featured Runfiles** 与各蓝图命名为准。

---

## 仓库地图

- **`dimos/`**：模块系统、协调与 Worker、`dimos/agents/`（智能体与 MCP）及各机器人栈实现。
- **`docs/`**：使用说明（[`docs/usage/`](../usage/README.md)）、能力文档（[`docs/capabilities/`](../capabilities/agents/readme.md) 等为入口）、开发与测试（[`docs/development/testing.md`](../development/testing.md) 等）。
- **`examples/`**：示例与语言互操作演示。

英文 README 含架构插图与 Demo 动图。

---

## 参与贡献与协作约定

- 测试与 pytest：[`docs/development/testing.md`](../development/testing.md)。
- 文档撰写：[`docs/development/writing_docs.md`](../development/writing_docs.md)。
- 分支与 PR（例如目标分支 `dev`、命名前缀）：[`AGENTS.md`](../../AGENTS.md) 文末 **Git Workflow**。

---

## 延伸阅读（英文）

| 主题 | 文档 |
|------|------|
| 蓝图组合与实践 | [`docs/usage/blueprints.md`](../usage/blueprints.md) |
| 全局配置 | [`docs/usage/configuration.md`](../usage/configuration.md) |
| 可视化（Rerun / Foxglove 等） | [`docs/usage/visualization.md`](../usage/visualization.md) |
| 导航（原生 / ROS） | [`docs/capabilities/navigation/native/index.md`](../capabilities/navigation/native/index.md) |
| 感知 | [`docs/capabilities/perception/readme.md`](../capabilities/perception/readme.md) |

中文导读若与英文文档或源码不符，**以英文与源码为准**；欢迎指正并提交修正。
