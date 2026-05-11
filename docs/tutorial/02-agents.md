# 02 · dimos.agents 子模块详解

> 系列教材的第三篇。本章带你认识 `dimos.agents/` 这个目录里到底
> 装了什么，并把"用自然语言指挥机器人"这件事拆成你能动手复现的三块
> 积木：`Agent`、`@skill`、以及 `McpServer / McpClient` 这一对 MCP
> 桥梁。

## 前置阅读

读这一章之前，请先确认你已经看完前两章：

<!-- doclinks-ignore-start -->
- [`./00-overview.md`](./00-overview.md)：dimos 整体架构总览。你需要
  从那一章里建立"模块（Module）≈ 厨房里的小家电、蓝图（Blueprint）≈
  接线菜谱、传输层（Transport）≈ 中间的电线"这种直觉。
- [`./01-core.md`](./01-core.md)：`dimos/core/` 子模块详解。本章会大量
  使用 `Module` 基类、`In[T]` / `Out[T]` 类型化数据流、`autoconnect`
  自动连线、以及 `@rpc` 装饰器（remote procedure call，"远程过程调用"
  的简称，意思是：另一个进程里的代码可以像调用本地函数一样调用它）。
  如果你看到这些词还不太确定它们在干嘛，请先回到 01 章。
<!-- doclinks-ignore-end -->

本章假设你已经知道：

- `Module` 是一个长在独立 forkserver 进程里的"小机器人"；
- 蓝图 = 一组 `Module` + 它们之间的接线 + `autoconnect()` 调用；
- `@rpc` 让一个方法"可以被别的模块远程调用"，但默认 LLM 看不见它。

如果以上三句话里有任何一句让你皱眉，请回到 01 章再过一遍，再来读
本章会顺很多。

---

## 一、什么是 dimos.agents（What）

`dimos.agents` 这个 Python 包做的事，可以用一句话概括：

> 把一个大语言模型（LLM，Large Language Model，比如 GPT-4o）塞进
> 机器人里，并且只允许它说"能落地成一个真实动作"的话。

它由三个核心概念组成。下面分别讲清楚每一个是什么、长什么样、
对应的代码文件在哪。

### 1.1 `Agent`：一个继承自 `Module` 的"LLM 大脑"

`Agent` 在 dimos 里 **不是一个独立的进程外部服务**，而是一个普通的
`Module` 子类——和摄像头模块、感知模块完全平级。它跟一般 Module
的不同只在于：

- 它订阅一条**文字流**（人类说的话或者别的模块抛出来的文本指令）；
- 它内部持有一个 LLM 客户端，把订到的文本喂给 LLM；
- LLM 决定"现在要不要调用某个 skill"，如果要，`Agent` 就帮它把
  调用真的发出去；调完之后再把结果（一个字符串）喂回 LLM，让它
  继续推理。

类比：把 ChatGPT 装进机器人脑袋，再用一根胶带把它的"输出渠道"
封死，只留一个洞——这个洞里只能掉出"动作卡牌"，不能掉出闲聊。

在 dimos 仓库里，最常见的 `Agent` 实现就是 `McpClient`
（见 `dimos/agents/mcp/mcp_client.py`）：它通过 MCP 协议跟一个
`McpServer` 通信，把 server 那边暴露出来的 skill 当成 LLM 的
"工具集合"使用。后面 1.3 节会详细讲。

> 小白提示：本章后面所有"Agent"、"LLM 那一侧"、"大脑"，指的都是
> 同一个东西——运行在你电脑上的一个 Python `Module`，里面调用了
> OpenAI 之类的 LLM API。它**不是**云端服务，也**不是**一个独立的
> 可执行程序。

### 1.2 `@skill` 装饰器：把一个 Python 方法标成"LLM 可调用的工具"

源文件：`dimos/agents/annotation.py`。这个装饰器干的事很简单：

- 它 **隐含地** 给方法套上了 `@rpc`（也就是说："这个方法可以被
  别的进程远程调用"），所以你在带 `@skill` 的方法上面**不要再写**
  一层 `@rpc`，否则就是叠加装饰，行为会出问题。详见 `AGENTS.md` 的
  原话："`@skill`: implies `@rpc` AND exposes method to the LLM as
  a tool. **Do not stack both.**"
- 它再给方法打一个 `__skill__ = True` 的标记。dimos 启动时会扫描
  所有 `Module`，把所有带这个标记的方法收集起来——这个集合就是
  "LLM 看得见、能调用的工具列表"。

类比：你拿一支马克笔，在某个 Python 方法上画了一张"动作卡牌"——
卡牌上有方法名、文档字符串、参数表。LLM 拿到手的就是这张卡牌，
而不是源代码。

为什么这一步重要？因为这意味着：**LLM 看到的"机器人能做什么"完全
取决于你画了几张卡牌**。没画的方法，LLM 完全感知不到；画错的方法
（比如忘写文档字符串、忘写类型注解），后面 § 二 会讲它会怎么坏。

### 1.3 `McpServer` / `McpClient`：把 skill 通过 HTTP 暴露给 LLM

MCP = Model Context Protocol（模型上下文协议）的缩写，由 Anthropic
等公司推动，是一个让 LLM 客户端发现和调用外部工具的开放协议。
dimos 把这套协议落地为两个普通的 `Module`：

- **`McpServer`**：源文件 `dimos/agents/mcp/mcp_server.py`，类签名
  是 `class McpServer(Module):`。它在自己的 forkserver 进程里启动
  一个 HTTP 服务，把当前蓝图里**所有**带 `@skill` 标记的方法都打包
  成 MCP 工具，挂在一个 URL 下。其他 `Module` 想要拿到这些工具，
  只要 HTTP 请求就行——既可以是同一台电脑上的 `McpClient`，也可以
  是跑在另一台电脑、甚至另一种语言（比如 TypeScript）写的 MCP
  客户端。
- **`McpClient`**：源文件 `dimos/agents/mcp/mcp_client.py`，类签名
  是 `class McpClient(Module):`。这就是 § 1.1 说的那个"LLM 大脑"
  的常见实现——它启动后会去连 `McpServer`，把对方暴露出来的工具
  全拉一份过来注册到自己的 LangChain agent 里，然后开始监听人类
  文本输入，每收到一条就交给 LLM 去推理、调用工具。

默认地址是 `http://localhost:9990/mcp`。这个端口号 **来自
`GlobalConfig.mcp_port`**，**不要在你自己的代码里硬编码 9990**——
请永远引用 `GlobalConfig` 里的字段（`AGENTS.md` 的 "Pre-commit &
Code Style" 段落明确写了："Don't hardcode ports/URLs — use
`GlobalConfig` constants."）。

为什么要拆成 server + client 两个 `Module`，而不是把 LLM 直接
塞进同一个进程？答案在 § 二的第一个问题里。

---

## 二、为什么这么设计（Why）

这一节解释 dimos.agents 里几个**初学者最容易困惑**的设计选择。
每条都是真的踩过坑才定下来的，不是花哨的工程美学。

### 2.1 为什么 LLM 不直接 import skill 函数？

最朴素的写法是这样的：

```python
from my_robot import grab, follow_object

prompt = "用户说：把可乐递给我"
if "递给我" in prompt:
    grab("coke")
```

这种写法在玩具 demo 里能跑，但在真实机器人栈里有三个无法绕过的
问题，正是这三个问题逼着 dimos 走 MCP-over-HTTP 这条路：

1. **进程边界**：dimos 的每个 `Module` 都跑在独立的 forkserver
   worker 进程里。LLM 客户端和摄像头模块、运动控制模块**根本不
   在同一个 Python 解释器里**，没法直接 `import` 一个函数。
2. **跨语言 / 跨设备**：今天的 MCP 客户端是 LangChain 写的 Python，
   明天你也许想换成一个跑在 NVIDIA Jetson 上的 C++ 客户端，或者
   一个跑在用户手机上的 TypeScript 客户端。**HTTP + JSON Schema**
   是最朴素的"哪种语言都能讲"的协议。
3. **可远程部署**：在仿真里，你可能希望 LLM 跑在一台 GPU 服务器
   上，机器人本体跑在另一台机器上，两边只有一根 WiFi 连着。
   HTTP 天然是"网络透明"的；进程内函数调用不是。

把这三条一加起来，就只剩下"用一个标准化的网络协议把 skill 暴露
出去"这一条路了。MCP 正好是为这件事设计的开放协议，所以 dimos
直接采用。

### 2.2 为什么 `@skill` 的规则这么严？

`@skill` 装饰器的"严"主要体现在四条规则上。下面这张表 **完全照搬**
自仓库根目录的 `AGENTS.md`（"Schema generation rules" 一节），
配合中文解释：

| 规则 | 一句话 | 违反了会怎样 |
|------|--------|--------------|
| **Docstring is mandatory** | 必须写文档字符串 | 启动时直接 `ValueError` —— 整个 skill container 注册不上来，LLM 一张卡牌都拿不到 |
| **Type-annotate every param** | 每个参数都要写类型注解 | 缺注解的参数在 schema 里就没有 `"type"` 字段 —— LLM 看不到这个参数是什么类型，几乎一定会传错 |
| **Return `str`** | 必须返回字符串 | 返回 `None` 时 agent 会幻觉一句固定话术："It has started. You will be updated later." 然后默认你的动作还在执行——后果就是它接着发下一条指令，一团乱 |
| **Full docstring verbatim in `description`** | 完整 docstring 会一字不差地塞进 MCP 工具的 `description` | 所以 `Args:` 这一段务必精炼——它会跟着**每一次** tool 调用提示一起重新喂给 LLM，越长越花 token |

为什么要这么严？因为 LLM 看你的 skill **只能** 通过 schema。schema
里没的字段，对 LLM 来说就是不存在；schema 里写的话，对 LLM 来说
就是金科玉律。所以"docstring 缺了"= 工具直接消失；"return None"
= LLM 永远在等一个不会来的结果。

### 2.3 为什么参数类型只能用那几种？

`AGENTS.md` 里写得很清楚，`@skill` 方法的参数 **只允许**这几种
Python 类型：

> Supported param types: `str`, `int`, `float`, `bool`, `list[str]`,
> `list[float]`. Avoid complex nested types.

也就是字符串、整数、浮点、布尔、字符串列表、浮点列表。
**就这六种**。原因是：

- MCP 协议的工具参数 schema 是基于 JSON Schema 的，本身就是为
  "扁平的、好序列化的数据"设计的；
- 嵌套字典、自定义类、`numpy.ndarray` 这类东西没法跨进程跨语言
  无损传递；
- LLM 自己生成嵌套结构的能力也很糟糕，强行让它生成 `{"pose":
  {"position": {...}, "orientation": {...}}}` 这种结构，错误率
  会肉眼可见地飙升。

实践上的结论：如果你想把"3D 位姿"传给一个 skill，不要直接收
`Pose`，而是收 `x: float, y: float, z: float`。把"翻译成 Pose
对象"的活留在 skill 内部完成。

### 2.4 为什么不能 `@rpc + @skill` 一起堆？

因为 `@skill` 内部已经替你调用过 `@rpc` 了——见
`dimos/agents/annotation.py` 第 69 行附近的 `wrapped =
rpc(context_wrapper)`。如果你再外面手动套一层 `@rpc`，就等于让
同一个函数被注册成两遍 RPC 入口，行为是未定义的（你也许会撞上
"重名"错误，也许会让 RPC 路由表里出现两个指向同一个函数但行为
不一致的项）。

写法上的简单结论是：

```python
# ✅ 暴露给 LLM 的方法
@skill
def move(self, x: float, duration: float = 2.0) -> str:
    """..."""

# ✅ 只内部模块之间互调，不让 LLM 看到
@rpc
def reset_odometry(self) -> bool:
    ...

# ❌ 永远不要这样叠
@rpc
@skill
def something(self) -> str:
    ...
```

### 2.5 顺便：为什么 Agent 要订阅一条文字流，而不是有一个 `agent.run()` 方法？

因为 dimos 里**所有东西都是 Module**——Module 通信的方式只有一种：
通过 `In[T]` / `Out[T]` 数据流。这样做的好处是统一：人类的语音
转文字 → 文字流；另一个 Agent 把"我观察到了一只猫"写成文字 →
文字流；测试脚本 `dimos agent-send "say hello"` 也是把字符串
推到这条文字流里。订阅方完全不关心文字是从哪儿来的。

<!-- doclinks-ignore-start -->
如果你忘了 `In[T]` / `Out[T]` 是什么，请回 [`./01-core.md`](./01-core.md)。
<!-- doclinks-ignore-end -->

---

## 三、如何使用（How）

下面四个小节按"从最小可运行的例子 → 接进真实蓝图 → 命令行操作 →
切换 system prompt"的顺序展开。每一步都对应仓库里能跑起来的
真实代码 / 真实命令。

### 3.1 写一个最小的 skill container

下面这段代码 **完全照搬**自 `AGENTS.md` 的 "Minimal correct skill"
小节。我们一行一行解释。

```python
from dimos.agents.annotation import skill
from dimos.core.core import rpc
from dimos.core.module import Module


class MySkillContainer(Module):
    @rpc
    def start(self) -> None:
        super().start()

    @rpc
    def stop(self) -> None:
        super().stop()

    @skill
    def move(self, x: float, duration: float = 2.0) -> str:
        """Move the robot forward or backward.

        Args:
            x: Forward velocity in m/s. Positive = forward, negative = backward.
            duration: How long to move in seconds.
        """
        return f"Moving at {x} m/s for {duration}s"


my_skill_container = MySkillContainer.blueprint
```

**逐行解释：**

- `from dimos.agents.annotation import skill`：导入 § 1.2 讲过的
  `@skill` 装饰器。
- `from dimos.core.core import rpc`：导入 `@rpc` 装饰器，因为
  `start` / `stop` 这两个生命周期方法需要被 dimos 框架"远程"
  调起来。
- `from dimos.core.module import Module`：所有 dimos 模块的基类，
  详见 01 章。
- `class MySkillContainer(Module):`：继承 `Module`。一个 skill
  container 本质上就是"装了一堆 `@skill` 方法的 `Module`"。
- `@rpc def start(self) -> None: super().start()`：模块启动时
  dimos 会调用 `start()`。这里我们什么都不需要做，所以只调用父类
  实现就够了——但还是必须写出来，且要带 `@rpc`，否则 dimos
  框架找不到这个生命周期入口。
- `@rpc def stop(self) -> None: super().stop()`：同理，停止
  生命周期。
- `@skill def move(self, x: float, duration: float = 2.0) -> str:`
  ——这才是 LLM 看得见的那张"动作卡牌"。注意：
    - 参数都加了类型注解（`x: float`、`duration: float`）—— § 二
      规则 2，必须；
    - 默认值 `duration: float = 2.0` 会让 LLM 在 schema 里看到
      "这个参数可省略、默认 2 秒"；
    - 返回类型是 `str` —— § 二 规则 3，必须；
    - **写了 docstring** —— § 二 规则 1，否则启动报 `ValueError`；
    - 函数体只是返回一句"模拟动作"的字符串。在真实项目里你会在
      这里调用机器人控制模块的 RPC，把 `x` 翻译成实际动作。
- `my_skill_container = MySkillContainer.blueprint`：把这个类
  转成一个 `Blueprint`，方便后面被 `autoconnect(...)` 引用。
  详见 01 章关于 `autoconnect` 的解释。

> 小贴士：`AGENTS.md` 的 "Adding a New Skill" 一节给出了完整的
> 添加流程清单：1) 选对 container（机器人专属的或者放到
> `dimos/agents/skills/`）；2) 写 `@skill` + docstring + 类型
> 注解；3) 如果要调别的模块，用 `Spec` 协议（也在 `AGENTS.md`
> 里）；4) 返回描述性字符串；5) 更新 system prompt；6) 把
> container 暴露成 `my_container = MySkillContainer.blueprint`。

### 3.2 把 skill container 接进 agentic 蓝图

光有 skill container 不会自动让 LLM 看到它——你必须把它和
`McpServer` + `McpClient` 一起放进同一个蓝图里 `autoconnect`
起来。仓库里已经有一个真实的样本可以照抄：

源文件：`dimos/robot/unitree/go2/blueprints/agentic/unitree_go2_agentic.py`

```python
from dimos.agents.mcp.mcp_client import McpClient
from dimos.agents.mcp.mcp_server import McpServer
from dimos.core.coordination.blueprints import autoconnect

unitree_go2_agentic = autoconnect(
    unitree_go2_spatial,    # 机器人本体栈：感知 / 建图 / 控制
    McpServer.blueprint(),  # HTTP MCP server —— 把所有 @skill 暴露在 9990 端口
    McpClient.blueprint(),  # LLM 那一侧 —— 从 McpServer 拉工具，监听人类输入
    _common_agentic,        # skill 容器们（包含 NavigationSkillContainer 等）
)
```

**关键：`McpServer.blueprint()` 和 `McpClient.blueprint()` 必须
两个都加。** 缺哪一个都不行：

- 只加 `McpServer` 没 `McpClient`：HTTP 服务起来了、卡牌挂出去了，
  但没有 LLM 那一侧主动来拉这些工具、也没人监听文字输入——你的
  `dimos agent-send "..."` 命令会发到一个没人听的地方。
- 只加 `McpClient` 没 `McpServer`：LLM 客户端起来了，但它去连
  `http://localhost:9990/mcp` 的时候是 connection refused——
  没有任何一个进程在那个端口上响应。

直觉地讲，`McpServer` 是"机器人那一侧的工具柜"，`McpClient` 是
"LLM 那一侧的工人"。少了任何一边，整套系统就不工作。

<!-- doclinks-ignore-start -->
> 蓝图相关的更多语法（`autoconnect`、`.blueprint()`、流自动
> 连线规则）请回 [`./01-core.md`](./01-core.md)。
<!-- doclinks-ignore-end -->

### 3.3 命令行演示：跑一个 agentic 蓝图

下面这组命令 **严格照抄**自 `AGENTS.md` 的 "Tools available to
you (MCP)" 小节。每一条都给出"它做了什么、什么时候用"的解释。

第一步：把带 MCP 的蓝图后台跑起来。

```bash
dimos --replay run unitree-go2-agentic --daemon
```

- 做什么：以**回放数据**模式（`--replay`）启动 `unitree-go2-agentic`
  这个蓝图，并把它后台化（`--daemon`）。后台化的好处是当前 shell
  立刻能继续敲下一条命令；前台日志可以另开一个终端用 `dimos log -f`
  看。
- 什么时候用：本地开发时——你不需要真机器人，回放数据就足以让
  整套 perception + skill + agent 链路启动起来。

第二步：让 LLM 那一侧告诉你它现在能调哪些 skill。

```bash
dimos mcp list-tools
```

- 做什么：通过 MCP 列出当前蓝图里**所有**能被 LLM 调的 skill，输出
  是一段 JSON：每个工具有 name、description、参数 schema。
- 什么时候用：你刚加完一个新 skill，想确认它确实被 `McpServer`
  注册了；或者排查"为什么 LLM 不调用我的 skill"——很多时候就是
  这一条返回里**根本没有**那个 skill（说明 `@skill` 没写对、
  docstring 缺了、container 没接进蓝图等等）。

第三步：手动调一个 skill，绕开 LLM。

```bash
dimos mcp call move --arg x=0.5 --arg duration=2.0
```

- 做什么：直接通过 MCP 调 `move` 工具，参数用 `--arg key=value`
  的方式传。
- 什么时候用：你怀疑某个 skill **本身**有 bug，想绕开 LLM 单独
  测它。如果手动调成功而 LLM 调失败，说明问题在 LLM 那一侧
  （prompt、参数生成）；反之就是 skill 自己有 bug。

如果参数比较复杂（比如有列表），用 JSON 形式更稳：

```bash
dimos mcp call move --json-args '{"x": 0.5, "duration": 2.0}'
```

- 做什么：和上一条等价，但参数用 JSON 字符串传。
- 什么时候用：参数包含列表（`list[str]`、`list[float]`）或者带
  特殊字符时，`--json-args` 比 `--arg` 鲁棒。

第四步：查看 MCP server 的运行状态。

```bash
dimos mcp status
```

- 做什么：返回当前 MCP server 的 PID、注册的 module 列表、注册的
  skill 列表。
- 什么时候用：怀疑 MCP server 没起来、或者起来了但跟你预期的
  module 集合不一样。

第五步：查看 module → skill 的映射关系。

```bash
dimos mcp modules
```

- 做什么：把"哪个 module 提供了哪些 skill"这张表打印出来。
- 什么时候用：你的蓝图里有多个 skill container（例如 Navigation +
  Speak + 自定义的 MySkillContainer），想看清每个 container 各
  贡献了哪些工具。

第六步（也是最常用的一步）：用人话指挥机器人。

```bash
dimos agent-send "walk forward 2 meters then wave"
```

- 做什么：把这一句英文（或者中文，看 system prompt 怎么写）通过
  LCM 推到 Agent 的 `human_input` 文字流上。Agent 收到之后会让
  LLM 推理、决定要调哪些 skill、按什么顺序调。
- 什么时候用：你想测试**端到端**的"自然语言 → 真实动作"链路。
  顺带提一句：这条命令**不依赖 `McpServer`**——只要蓝图里有
  `McpClient`（或者别的 `Agent`-类模块）订阅了 `human_input`
  流就行。但 LLM 真正能调到 skill 当然还是要靠 `McpServer`。

### 3.4 切换 system prompt：默认 prompt 是 Go2 专用的

`McpClient` 在创建时会用一个 system prompt（系统提示词）作为
LLM 的"出厂设定"。dimos 默认 prompt 在
`dimos/agents/system_prompt.py` 里，导出符号叫 `SYSTEM_PROMPT`，
内容**完全是为 Unitree Go2 四足机器人写的**——里面写死了
"You are Daneel...controls a Unitree Go2 quadruped robot"，并且
列出了 Go2 特有的 skill 名字（`navigate_with_text`、
`execute_sport_command` 等）。

如果你跑的是 G1 人形机器人，**必须显式传 G1 专用的 prompt**，
否则 LLM 会看着自己的 system prompt 说："我有 `RecoveryStand`
这个动作"——然后真的发出 `execute_sport_command("RecoveryStand")`
工具调用——但实际上 G1 的 skill container 里**根本没有这个
方法**，于是 LLM 就在幻觉一个不存在的 skill。`AGENTS.md` 里
明确说："The default prompt is Go2-specific; using it on G1
causes hallucinated skills."

正确的写法：

```python
from dimos.agents.mcp.mcp_client import McpClient
from dimos.robot.unitree.g1.system_prompt import G1_SYSTEM_PROMPT

McpClient.blueprint(system_prompt=G1_SYSTEM_PROMPT)
```

下面这张表抄自 `AGENTS.md` 的 "System Prompts" 一节，列出了
当前仓库里现成的两个 prompt 文件：

| 机器人 | 文件 | 变量名 |
|--------|------|--------|
| Go2（默认） | `dimos/agents/system_prompt.py` | `SYSTEM_PROMPT` |
| G1 人形 | `dimos/robot/unitree/g1/system_prompt.py` | `G1_SYSTEM_PROMPT` |

新增一种机器人时，按这个表的格式：在机器人自己的目录里加一个
`system_prompt.py`，导出一个大写命名的字符串变量，然后通过
`McpClient.blueprint(system_prompt=...)` 传进去。

---

## 四、与其他模块的关系

### 4.1 一张图看懂整条调用链

下面这张 ASCII 图描述的是"用户说一句话 → 机器人做一个动作"的
完整数据流。每一格代表一个**进程**（forkserver worker）；箭头上
的标签是数据 / 协议类型。

```
┌──────────────────────┐
│ 用户终端              │
│ dimos agent-send "…" │
└──────────┬───────────┘
           │  LCM 上的 human_input 文字流（str）
           ▼
┌──────────────────────────────────────┐
│ McpClient（LLM 那一侧）               │
│  - 订阅 In[str] human_input          │
│  - 调用 LLM（默认 GPT-4o）            │
│  - 把 LLM 决定的工具调用通过 MCP 发出 │
└──────────┬───────────────────────────┘
           │  HTTP / JSON-RPC（MCP over HTTP）
           │  默认 http://localhost:9990/mcp
           │  端口来自 GlobalConfig.mcp_port
           ▼
┌──────────────────────────────────────┐
│ McpServer                            │
│  - 启动一个 FastAPI 服务             │
│  - 把蓝图里所有 @skill 方法注册      │
│    成 MCP 工具，挂在 /mcp 下         │
│  - 通过 RPC 触发对应的 skill         │
└──────────┬───────────────────────────┘
           │  dimos 内部 RPC（@rpc / @skill）
           ▼
┌──────────────────────────────────────┐
│ 各个 skill container（Module 子类）  │
│  - UnitreeSkillContainer（Go2）      │
│  - NavigationSkillContainer（通用）  │
│  - 你自己写的 MySkillContainer        │
└──────────┬───────────────────────────┘
           │  dimos 内部 RPC / 数据流
           ▼
┌──────────────────────────────────────┐
│ 机器人本体模块                        │
│  - 运动控制 / 感知 / 建图等          │
└──────────────────────────────────────┘
```

读这张图时，请记住三件事：

1. **每一格都是一个独立 forkserver 进程**，所以箭头跨越的不是
   "函数调用"而是"进程间消息"——这正是 § 二第 1 题的答案。
2. **箭头上的协议**有意混合：人 → Agent 走 LCM（dimos 默认的
   pub/sub 消息总线），Agent → MCP 走 HTTP，MCP → skill 走 dimos
   内部 RPC。每一段都是经过权衡之后选的最合适协议——LCM 适合
   广播文字、HTTP 适合跨语言工具调用、`@rpc` 适合本地高速的方法
   调用。
3. **`@skill` 是图最右边那一段的"门面"**：你写的方法只要打了
   `@skill`，它就自动出现在最右边那个框里、并通过中间这串管道
   被 LLM 看到、调用、收到结果。

### 4.2 skill container 和具体机器人模块的关系

仓库里 `@skill` 容器**通常按机器人专属来组织**。常见布局：

- `dimos/robot/unitree/unitree_skill_container.py`：Go2 的 skill
  容器（包含 `relative_move`、`wait`、`current_time`、
  `execute_sport_command`、`dance` 等真实 `@skill` 方法）。
- `dimos/robot/unitree/g1/`：G1 人形机器人的 skill / blueprint /
  system prompt 等所有 G1 专属代码。
- `dimos/robot/drone/`：无人机相关的连接、相机、视觉伺服等
  Module，对应章节会在后续教材里展开。

通用的、不依赖具体机器人的 skill 放在 `dimos/agents/skills/`
（例如 `NavigationSkillContainer`、`SpeakSkill` 等），它们可以
被任何机器人的 agentic 蓝图复用。

判断"我应该把新 skill 放哪儿？"的简单规则：

- 这个 skill 只对某种机器人有意义（比如 Go2 的 "dance"）→ 放
  机器人自己的目录，例如 `dimos/robot/unitree/...`。
- 这个 skill 任何机器人都能用（比如 "speak"、"set_navigation_goal"）
  → 放 `dimos/agents/skills/`。

### 4.3 `Agent` / `McpServer` / `McpClient` 都是 `Module`

强调一遍重点：本章里出现的 `McpServer`、`McpClient`、以及未来
会接触的各类 `Agent`，**它们本身都是 `dimos.core.module.Module`
的子类**。所以：

- 它们在蓝图里通过 `autoconnect(...)` 接进系统；
- 它们之间的消息通过 `In[T]` / `Out[T]` 数据流传递；
- 它们的生命周期通过 `@rpc` 标记的 `start()` / `stop()` 管理；
- 它们运行时各自占一个 forkserver worker 进程。

也就是说，**01 章 `dimos.core` 里学的所有规则在 dimos.agents 里
全都成立**。dimos.agents 没有引入新的"东西"，只是在 `Module`
的基础上多加了一个标签——`@skill`——以及一对实现了 MCP 协议的
具体 `Module`（`McpServer` / `McpClient`）。

### 4.4 后续推荐章节

下面这些章节本系列后续会陆续补齐，链接是占位符（写出来 CI 不会
失败，等对应 PR 合入后链接就生效）：

<!-- doclinks-ignore-start -->
- [`./03-robot-unitree-go2.md`](./03-robot-unitree-go2.md)：Unitree
  Go2 四足的连接、skill 容器、感知/建图栈、agentic 蓝图怎么
  组装。
- [`./04-robot-unitree-g1.md`](./04-robot-unitree-g1.md)：Unitree
  G1 人形机器人的真机 + MuJoCo 仿真蓝图，G1 专属 system prompt
  的写法。
- [`./05-robot-drone.md`](./05-robot-drone.md)：MAVLink 连接、DJI
  相机、视觉伺服与目标跟踪，怎么把无人机也接进 agentic 体系。
<!-- doclinks-ignore-end -->

读完本章你应该具备的能力是：在任何一个 `dimos` 蓝图里识别出
"哪一段是 Agent / 哪一段是 skill 容器 / 数据流是怎么连的"，并且
能照着 § 3.1 自己写一个最小可运行的 skill container，把它通过
§ 3.2 的写法接进一个新的 agentic 蓝图，然后用 § 3.3 的命令验证
LLM 真的看到并调用了它。
