# DimOS 简体中文概览

本文面向初次接触代码库的读者，用高层视角说明 DimOS（Dimensional OS）是什么、由哪些部分组成，以及如何入门与延伸阅读。**权威英文说明仍以仓库根目录的 [README.md](/README.md) 为准**。

## 项目定位

DimOS 是一套面向通用机器人的 **模块化运行时与编排框架**：感知、导航、操控等能力拆成独立的 **模块（Module）**，通过类型化的 **数据流（Streams）** 连接；一组预先连线的模块构成 **蓝图（Blueprint）**，由 **`dimos` CLI** 一键拉起。上层可以再接 **AI Agent**，通过 **技能（Skills）** 和可选的 **MCP 服务** 用自然语言驱动硬件。

目标是：**尽量用 Python 组合机器人栈**，不强依赖 ROS；同时可按需在 ROS/DDS/LCM 等传输之间衔接。

## 核心概念（术语表）

| 概念 | 简要说明 |
|------|----------|
| **模块（Module）** | 并行运行的 Python 类，声明输入/输出流；是部署的基本单元。详见 [模块](/docs/usage/modules.md)。 |
| **流（Streams）** | 模块之间的发布/订阅通道，承载传感器数据、命令等。参见 [传感器流](/docs/usage/sensor_streams/README.md)。 |
| **蓝图（Blueprint）** | 把多个模块组合并声明连接关系；详见 [蓝图文档](/docs/usage/blueprints.md)（其中的 `autoconnect()` 可按名称与类型自动连线）。 |
| **RPC** | 模块间远程调用（参数序列化）；与其他模块协作时常配合 Spec 注入。 |
| **技能（Skill）** | 一类特殊的 RPC：既可被代码调用，也可暴露给 LLM 作为工具；需规范的 docstring 与类型注解。详见 [蓝图 · 技能](/docs/usage/blueprints.md#defining-skills)。 |
| **Agent** | 带有目标、能订阅流数据并调用技能的智能体栈；能力与具体蓝图有关。入门可参考 [Agents 能力](/docs/capabilities/agents/readme.md)。 |
| **MCP** | Model Context Protocol：蓝图若包含 MCP 服务端，可把技能暴露给外部客户端（例如 IDE / 其它 Agent）。详见仓库根目录 README 中的「Agent CLI & MCP」一节及 [`dimos mcp`](/docs/usage/cli.md)。 |
| **`dimos` CLI** | 启动/停止蓝图、查看日志、发送 Agent 文本、`mcp` 子命令等。完整列表见 [CLI 参考](/docs/usage/cli.md)。 |

贡献者与自动化 Agent 的仓库约定摘要见 [AGENTS.md](/AGENTS.md)。

## 架构摘要（心智模型）

1. **定义模块**：声明 `In[T]` / `Out[T]`，在回调或 RPC 里处理数据。
2. **编写蓝图**：用 `autoconnect(module_a(), module_b(), …)` 组合栈。
3. **运行**：`dimos run <blueprint>`（可加 `--replay`、`--simulation`、`--daemon` 等全局选项）。
4. **智能化**：在蓝图里加入 Agent +（可选）McpServer/McpClient，用 `@skill` 暴露可调动作。

更细的模块与编排说明：[模块](/docs/usage/modules.md)、[蓝图](/docs/usage/blueprints.md)。

## 安装与环境

按平台准备系统依赖：

- [Ubuntu 22.04 / 24.04](/docs/installation/ubuntu.md)
- [NixOS / 通用 Linux](/docs/installation/nix.md)
- [macOS](/docs/installation/osx.md)

环境要求总览：[系统要求](/docs/requirements.md)。

从源码参与开发时，仓库内常用 **`uv sync`** 安装 Python 依赖（可选 extras，例如 `uv sync --extra all`）；README 中还展示了基于 **`uv pip install 'dimos[...]'`** 的消费方式。任选其一与你的场景匹配即可。

## 常用命令示例

下列命令假定已在虚拟环境中安装好 CLI（参见根目录 [README · Installation](/README.md#installation)）。

```bash
# 列出可运行的蓝图
dimos list

# 回放数据运行（无需真机；首次可能下载回放资源）
dimos --replay run unitree-go2

# 仿真中的智能化示例（依赖对应可选组件）
dimos --simulation run unitree-go2-agentic

# 查看运行状态与日志
dimos status
dimos log -n 50

# 若蓝图包含 MCP 服务端（示例蓝图见 README）
dimos mcp list-tools
```

全局选项与命令详解：[CLI 参考](/docs/usage/cli.md)、[配置](/docs/usage/configuration.md)。

## 延伸阅读（英文深度文档）

- **概念索引**：[用法文档索引](/docs/usage/README.md)（Modules / Streams / Blueprints / Skills）
- **开发与测试**：[测试指南](/docs/development/testing.md)、[文档链接约定](/docs/agents/docs/doclinks.md)
- **可视化**：[可视化](/docs/usage/visualization.md)
- **面向 Agent 的贡献指南**：[风格指南](/docs/agents/style.md)

若你只关心快速上手命令与蓝图表格，请以根目录 [README.md](/README.md) 为准。
