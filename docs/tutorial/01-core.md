# 01 · dimos.core 子模块详解

<!-- doclinks-ignore-start -->
> 前置阅读：[`./00-overview.md`](./00-overview.md)
<!-- doclinks-ignore-end -->
> 假设你已经知道 Module、Blueprint、autoconnect、@rpc 这几个词的存在。

## 一、什么是 dimos.core（What）

`dimos.core` 是整个 DimOS 的"主板层"。它不关心具体的机器人、不关心摄像
头型号、不关心 LLM 模型，它只负责一件事：**把一组独立的 Python 进程用
类型化的数据流接起来，让它们看起来像一个整体**。

如果把 `dimos run unitree-go2-agentic` 跑起来的整个系统比作一台 PC，那
`dimos.core` 就是主板加电源加机箱：它本身什么也不"做"，但所有"会做事"
的子系统（感知、导航、底盘控制、LLM 智能体）都必须插在它定义的卡槽
（`In[T]` / `Out[T]`）里，遵循它定义的供电规则（`GlobalConfig`），通过
它定义的总线（`Transport`）通信。

`dimos.core` 一共有 6 个核心子件，本章按这个顺序展开：

1. **`Module` 基类** — 一个独立运行单元，对应一个 forkserver 工作进程。
2. **`In[T]` / `Out[T]` 类型化数据流** — Module 之间唯一允许的通信方式。
3. **`Transport`** — 数据流底下真正搬比特的物理总线，共 6 种实现。
4. **`Blueprint` 与 `autoconnect`** — 把一堆 Module 拼成一台完整机器人
   的"焊接图"。
5. **`GlobalConfig`** — 全局配置单例，决定 `robot_ip` / `simulation` /
   `viewer` 这种"运行时开关"。
6. **`@rpc` 装饰器** — 把一个普通方法暴露成可以被外部进程调用的端点。

### 1. `Module` 基类 ≈ 主板上的一块独立板卡

`Module` 是 dimos 里的最小"会做事"的单位。一个 Module 一旦 `start()`
就会被部署到一个独立的 forkserver 工作进程里，和你的主进程**真物理隔
离**——它崩了不会拖死别人，它的 GIL 也不会影响别人。

它解决的问题是："Python 单进程跑机器人会爆炸"。当你同时要做相机解码、
点云处理、规划、模型推理、底盘下发，单进程一个 GIL 会把所有 latency 拖
到一起；任何一段代码 segfault 都让整个机器人挂掉。`Module` 把每个职责
切成独立进程，再用接下来要讲的"接线端子"把它们重新连起来。

### 2. `In[T]` / `Out[T]` ≈ 板卡上的接线端子

每个 Module 上的 `In[T]` 是一个**输入端子**（订阅一种类型为 `T` 的数据
流），`Out[T]` 是一个**输出端子**（发布一种类型为 `T` 的数据流）。这
里的 `T` 是 Python 的真实类型（比如 `Image`、`PoseStamped`），不是字
符串话题名。

它解决的问题是："ROS topic 的 string-typed 黑魔法"。在 ROS 里你订阅一个
`/camera/color` topic，类型对不对要等运行时报错才知道；在 dimos 里
`In[Image]` 写在类属性上，IDE 直接给你补全 `.subscribe(...)`，类型不
匹配在 `autoconnect` 阶段（也就是 `build()` 之前）就直接报错。

### 3. `Transport` ≈ 主板上的物理总线

`In[T]` / `Out[T]` 只定义"逻辑接口"，真正把字节从一个进程搬到另一个进
程的，是 `Transport`。dimos 一共提供 6 种 Transport 实现，对应不同的
"总线带宽 / 跨进程能力 / 跨机器能力"组合（详见 3.3 节的对照表）。

它解决的问题是："业务代码不应该关心传输方式"。你写感知模块的时候不
需要知道下游消费者在同机还是跨机；你只管 `self.processed.publish(img)`，
具体走 LCM 还是共享内存还是 ROS bridge，由 blueprint 配置决定。换言
之，**传输是部署时决定的，不是开发时决定的**。

### 4. `Blueprint` + `autoconnect` ≈ 主板的焊接图

`Blueprint` 是一份**部署蓝图**，说明哪些 Module 一起跑、它们的 `Out`
接谁的 `In`。`autoconnect(...)` 是默认的连线策略——按 `(端子名, 类型)`
匹配，省去手写"a.color_image -> b.color_image"这种胶水。

它解决的问题是："多模块系统必然要拼接，不要让拼接代码超过业务代码"。
ROS launch 文件能写到几百行；asyncio 项目里写一堆 `Queue` 互相 wire
更恐怖。Blueprint 把"谁连谁"压到一个声明式 Python 表达式：
`autoconnect(perception, navigator, agent)`。

### 5. `GlobalConfig` ≈ 主板上的总闸 / 拨码开关

`GlobalConfig` 是一个**单例配置对象**，里面挂着所有"运行期开关"：
`robot_ip` / `simulation` / `replay` / `viewer` / `n_workers` /
`mcp_port` 等等。它有一套明确的覆盖顺序（defaults → `.env` → `DIMOS_*`
环境变量 → blueprint 内显式赋值 → 命令行 flag），细节在 3.5 节。

它解决的问题是："同一份代码要在仿真 / 实机 / 回放三种模式下跑"。如果
没有一个集中开关，你就会在十几个文件里散落 `if simulation:` 分支。
`GlobalConfig` 把这些"环境变量级"的差异收敛到一处，模块代码只读不写。

### 6. `@rpc` ≈ 给外部脚本暴露的小开关

`@rpc` 装饰器（`dimos/core/core.py`）把一个 Module 方法标记为"可以被
外部进程调用"。最常见的用法是 `start()` / `stop()`：协调器（module
coordinator）用 RPC 调你的 `start()`，把模块拉起来。也可以用来在
Module 之间互相调用——配合 3.6 节会讲到的 Spec Protocol，编译期就能
验证调用链。

它解决的问题是："数据流是单向流，但有时候我们就是要 request/response"。
比如导航模块要告诉规划模块"取消当前目标"，这件事用 publish 一条消息
不直观；直接 `self._navigator.cancel_goal()` 才符合直觉。`@rpc` 让这
种调用看起来像普通方法调用，背后却是跨进程的。

## 二、为什么这么设计（Why）

本节回答一个问题：**为什么 dimos 要发明 Module / Blueprint / Transport
这一套，而不是用现成的方案？** 我们对比三种"显而易见的替代方案"，
看每一种在机器人场景下分别死在哪。

### 二.1 vs. 一坨 Python 脚本（"先跑起来再说"）

最初你可能会写一个 `main.py`，里面 `while True:` 拉相机帧、跑模型、
发底盘指令。这个方案在两件事上必然崩：

- **GIL 把 latency 焊到一起。** 模型推理一占 200ms，相机帧就丢 200ms；
  底盘心跳超时机器人就保护性停机。
- **一处崩，全盘崩。** 一个 numpy view 越界 segfault，整台机器人下线
  ——包括本来不依赖那段代码的电机急停链路。

`dimos.core` 用 forkserver 工作进程把每个 Module 隔离开。一个模块崩了
另一个继续跑；模型推理慢不影响底盘心跳。这是"用 Python 写机器人系统"
绕不开的硬约束。

### 二.2 vs. 直接用 ROS

第二种很容易想到的方案是："进程隔离 ROS 已经做了，直接 ROS2 不行
吗？"——能跑，但代价两件：

- **类型不安全。** ROS topic 是字符串名字 + msg 类型，类型对不对要等
  运行时 callback 才知道。dimos 的 `In[Image]` 是 Python 类型，IDE
  和 mypy 提前帮你校验。
- **Python 胶水太多。** 写一个 ROS2 Python 节点要继承 `rclpy.Node`、
  写 launch 文件、维护 package.xml；调试时 `colcon build` 来回 30 秒。
  dimos 一个 Module 就是一个继承自 `Module` 的 Python 类，blueprint
  就是一个 Python 表达式。

dimos 并不是反 ROS——`ROSTransport` 就是一个把 dimos 流桥到 ROS 话题的
出口。它的立场是："你写 dimos 代码时不用碰 ROS 的繁琐，但要对接 ROS
生态时随时能桥过去。"

### 二.3 vs. asyncio + 自定义消息总线

第三种思路是："那我自己用 asyncio 写一个 Queue 总线？"——这条路看起来
最 pythonic，但实际上你会一步一步重新发明 dimos：

- **asyncio 不解决 GIL。** 模型推理一旦阻塞事件循环，相机帧依然丢。
  你最终还是要 `ProcessPoolExecutor` 或者多进程，但那时你已经在自己
  写一套简陋的 forkserver。
- **类型提示丢失。** 自定义 `Bus.publish("camera", img)` 这种调用，IDE
  无法补全也无法静态检查；维护一年后没人记得 `"camera"` 这个名字到底
  是 `Image` 还是 `CompressedImage`。
- **进程边界硬编码。** 当你想把"感知模块"从同机移到另一台 GPU 机器
  时，你的 asyncio 代码要全改一遍；dimos 只要换一种 `Transport`。

总结一句：**dimos.core ≈ "类型化的接线 + 进程级隔离 + 多种传输切换"**。
这三件事任意去掉一个，机器人系统迟早会出问题；三件都自己实现，你会
得到一个比 dimos 更难维护的 dimos。

### 二.4 一句话设计目标

> **写业务模块的人不应该思考"这条数据怎么过去"，部署模块的人不应该
> 思考"这个模块到底在做什么"。**

`In[T]` / `Out[T]` 切开了这两类关心点；`Transport` 让部署可换；
`Blueprint` 让组装可读；`GlobalConfig` 让运行期参数可控；`@rpc`
让"必须 request/response"的边角场景也有一致的写法。这就是 `dimos.core`
全部要做的事——你接下来在第三节看到的所有 API 都只是这五件事的具体
落地。

## 三、如何使用（How）

### 3.1 Module 基类

先看 `AGENTS.md` 给出的最小 Module 例子（**这是仓库根 `AGENTS.md` 的
原文，建议你打开对照**）：

```python
from dimos.core.module import Module
from dimos.core.stream import In, Out
from dimos.core.core import rpc
from dimos.msgs.sensor_msgs import Image

class MyModule(Module):
    color_image: In[Image]
    processed: Out[Image]

    @rpc
    def start(self) -> None:
        super().start()
        self.color_image.subscribe(self._process)

    def _process(self, img: Image) -> None:
        self.processed.publish(do_something(img))
```

逐行读：

- `class MyModule(Module)`：继承自 `dimos.core.module.Module`。继承
  这个基类，你的模块就拿到了"被部署到 forkserver 工作进程"的能力。
- `color_image: In[Image]`：声明一个**输入端子**。注意这是**类属性 +
  PEP 526 类型注解**的写法，不是 `__init__` 里赋值——这是为了让
  blueprint 在 `build()` 之前就能读到端子定义并匹配。
- `processed: Out[Image]`：声明一个**输出端子**。
- `@rpc def start(self)`：把 `start` 标成 RPC 端点，由 module
  coordinator 在部署时调用。`super().start()` 调到基类做"流注册"等
  公共动作，**不要忘记调**。
- `self.color_image.subscribe(self._process)`：在 `start()` 里订阅。
  数据到来时，dimos 会在这个 Module 的进程里调用 `_process(img)`。
- `self.processed.publish(do_something(img))`：发布到 `Out[Image]`。
  下游谁订阅、用什么 Transport 走过去，这里完全不关心——那是
  blueprint 的事。

> 注：`super().start()` 之外，常见还要重写 `stop()`（见
> `AGENTS.md` 的 "Minimal correct skill" 例子，里面同样写了
> `@rpc def stop(self)`）。生命周期对称是良好习惯。

### 3.2 In / Out 流

`In[T]` 和 `Out[T]` 看起来像 typing 的泛型，实际上是**带状态的运行时
对象**，由 Module 基类在创建实例时把"类属性上的 `In[Image]` 注解"展开
成"实例属性上的真实输入端子"。

两个端子上你常用的 API 就两个：

- **`Out[T].publish(value: T)`**：把一帧数据丢进当前模块的输出端子。
  `T` 必须和声明一致；类型不匹配在类型检查阶段就能抓到。
- **`In[T].subscribe(callback: Callable[[T], None])`**：注册一个回
  调，每当上游有新数据到达时被调用一次。回调在本 Module 的工作进程里
  执行。

至于 "`a.processed` 究竟连到 `b.color_image` 还是 `c.color_image`"
这个问题，**Module 自己不知道也不关心**。这是 blueprint 通过 `(端子
名, 类型)` 自动匹配（`autoconnect`）解决的——见 3.4 节。

这种"声明端子 + 由 blueprint 连线"的设计有一个直接好处：**同一个
Module 可以在不同 blueprint 里复用而不改一行代码**。比如一个
`PerceptionModule` 既能在仿真 blueprint 里接 sim 相机，也能在实机
blueprint 里接 Realsense 相机——它只看到 `In[Image]`。

### 3.3 Transports 对照表

下表列出 `AGENTS.md` 中明确给出 class 名的 6 种 Transport。**严格按
此清单**——其它名字（包括散见于 prose 里的 "Jpeg" 之类）不在 dimos.core
公开 transport 列表里，本教材不引入。

| Transport       | 用途 / 何时用 |
|-----------------|---------------|
| `LCMTransport`  | **默认 transport，多播 UDP**。同机 / 同子网内的常规消息流（命令、状态、轻量结构体）走它。 |
| `SHMTransport`  | **共享内存**。给图像、点云这种"大块、零拷贝"数据用，避开序列化和 socket 拷贝。 |
| `pSHMTransport` | **Pickled 共享内存**。和 `SHMTransport` 一样走共享内存，但允许传任意可 pickle 的 Python 对象——给那种结构复杂、又必须省拷贝的中间数据用。 |
| `pLCMTransport` | **Pickled LCM**。把任意可 pickle 的 Python 对象塞进 LCM 帧——给"复杂 Python 对象，不在 LCM IDL 里"的场景用。 |
| `ROSTransport`  | **ROS topic 桥**。把 dimos 的 `Out[T]` 桥到 ROS 话题，或把 ROS 话题桥到 dimos 的 `In[T]`——和 ROS 节点互通时用（源码：`dimos/core/transport.py`）。 |
| `DDSTransport`  | **DDS pub/sub**。要跨机器、跨语言走工业级 DDS 总线时用；只有 `DDS_AVAILABLE` 时可用，需 `uv sync --extra dds` 安装（源码：`dimos/protocol/pubsub/impl/ddspubsub.py`）。 |

记忆口诀：

- **同机轻量 → `LCMTransport`**（默认就行，不用自己选）。
- **同机大块 → `SHMTransport` / `pSHMTransport`**（图像、点云）。
- **跨语言 / 跨机器 → `ROSTransport` / `DDSTransport`**（出 dimos 边界）。
- **结构怪 → 加 `p` 前缀的 pickled 版**。

具体某个 blueprint 用哪个 transport，是在 blueprint 的连线处决定的，
模块代码不用动。

### 3.4 Blueprint + autoconnect

最简单的 blueprint 长这样（**抄自 `AGENTS.md`，原样**）：

```python
from dimos.core.coordination.blueprints import autoconnect

my_blueprint = autoconnect(module_a(), module_b(), module_c())
```

`autoconnect` 做的事：扫描所有传进来的 Module，按 **`(端子名, 类型)`**
两元组配对——`module_a.processed: Out[Image]` 会自动连到任意一个声明
了 `processed: In[Image]` 的下游 Module。换句话说，**端子名是契约
的一部分**，重命名要小心。

如果同名端子的类型不匹配（比如一边是 `Out[Image]` 一边是
`In[CompressedImage]`），`autoconnect` 在 `build()` 阶段就会报错——
不会等到运行时才发现。

要直接从 Python 跑起来，照 `AGENTS.md` 的写法（**原样**）：

```python
autoconnect(module_a(), module_b(), module_c()).build().loop()
```

- `.build()` 把所有模块部署到 forkserver 工作进程，**并完成连线**。
- `.loop()` 阻塞主线程，直到收到 Ctrl-C / SIGTERM。

把这条表达式赋值到一个**模块级变量**（比如 `unitree_go2_agentic`），
`dimos run unitree-go2-agentic` 才能找到它。新增或重命名 blueprint 后
要跑一次 `pytest dimos/robot/test_all_blueprints_generation.py` 让自动
生成的 `dimos/robot/all_blueprints.py` 同步到位（这条规则在 `AGENTS.md`
里被反复强调）。

### 3.5 GlobalConfig 级联

`GlobalConfig` 是单例，整个进程一份。它的字段直接对应 CLI flag——
`AGENTS.md` 原话："Every `GlobalConfig` field is a CLI flag"——所以
你不会看到一份"CLI flag 列表"和一份"配置项列表"两份文档分裂。

#### 覆盖顺序（来自 `AGENTS.md`）

> defaults → `.env` → env vars → blueprint → CLI flags

也就是：

1. **defaults**：`GlobalConfig` 类内字段定义的默认值（最低优先级）。
2. **`.env` 文件**：项目根的 `.env` 加载进来。
3. **环境变量**：以 `DIMOS_` 为前缀，例如 `DIMOS_ROBOT_IP=...`、
   `DIMOS_SIMULATION=1`。
4. **blueprint 内显式赋值**：blueprint 在自己的代码里写
   `GlobalConfig.set(...)` 这种就属于这一层。
5. **CLI flag**：`--robot-ip 192.168.123.161` 等，**最高优先级**，永远
   覆盖前面所有。

记忆方法：**"越靠近用户的越优先"**。你在命令行手敲的 flag 永远赢，文件
里写死的输给环境变量，环境变量输给命令行——这和 12-factor 的 config
传统一致。

#### 关键字段（`AGENTS.md` 里点名的）

`robot_ip`、`simulation`、`replay`、`viewer`、`n_workers`、`mcp_port`。
其它字段也都是 `--xxx` 风格的 flag。

#### 看当前生效配置

```bash
dimos show-config
```

这条命令把"defaults + .env + 环境变量 + blueprint + CLI"合并后的最终
值打印出来。调试"为什么我的 `--simulation` 没生效"时第一步就是它。

### 3.6 `@rpc` vs `@skill`

这是 dimos.core 和 dimos.agents 之间最常被搞混的一对装饰器。
`AGENTS.md` 给出的判别规则非常明确，照抄如下：

- **`@rpc` 单独使用**：方法可被 RPC 调用，**不暴露给 LLM**。
- **`@skill` 单独使用**：**隐含 `@rpc`**，并且把方法暴露给 LLM 当作工具。
- **不要把 `@rpc` 和 `@skill` 叠加**——`AGENTS.md` 原话 "Do not stack both."

可以用一张表收住：

| 装饰器 | RPC 可调 | 暴露给 LLM | 典型用法 |
|--------|----------|------------|----------|
| `@rpc`            | ✅ | ❌ | `start()` / `stop()`、模块间内部调用 |
| `@skill`          | ✅（隐含） | ✅ | 给 LLM agent 的工具方法（`move`、`grab` 等） |
| `@rpc` + `@skill` | ❌ 别这么写 | ❌ 别这么写 | — |

```python
class MyModule(Module):
    @rpc
    def start(self) -> None:           # 内部 lifecycle
        super().start()

    @skill
    def move(self, x: float) -> str:    # 给 LLM 用的工具
        """Move forward."""
        return f"moving {x}"
```

`@skill` 还有更细的"docstring 必填、参数必须类型注解、返回值必须是
`str`"等规则。**这些不属于 `dimos.core` 的关心范围**——它们关于"如何
让 LLM 看懂工具"，应该在
<!-- doclinks-ignore-start -->
[`./02-agents.md`](./02-agents.md)
<!-- doclinks-ignore-end -->
里展开。本章只做一句话提醒：**先写 `@skill`，不要叠加 `@rpc`。**

## 四、与其他模块的关系

`dimos.core` 是**整个仓库唯一的"基础层"**。所有其它顶层包（`dimos.robot`、
`dimos.navigation`、`dimos.perception`、`dimos.agents` 等）都建立在它
之上：它们的 Module 全部继承自 `dimos.core.module.Module`，它们的端
子全部用 `dimos.core.stream.In` / `Out`，它们的入口最终都是某个用
`autoconnect(...)` 拼出来的 blueprint。

简单画一下依赖方向：

```
                    ┌──────────────────────────────┐
                    │           dimos.core         │
                    │  Module / In / Out / RPC     │
                    │  Transport / Blueprint       │
                    │  GlobalConfig                │
                    └──────────────┬───────────────┘
                                   │  (everyone depends on it)
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
        ▼                          ▼                          ▼
 dimos.perception         dimos.navigation             dimos.agents
 (相机/检测/跟踪)         (建图/规划/exploration)       (LLM/MCP/skills)
        │                          │                          │
        └─────────────┬────────────┴────────────┬─────────────┘
                      │                         │
                      ▼                         ▼
                              dimos.robot
                  (Go2 / G1 / xArm / drone 的 blueprint)
```

读法：

- 箭头方向 = "依赖"。`dimos.robot` 依赖感知 / 导航 / 智能体；它们依赖
  `dimos.core`；`dimos.core` 不依赖任何业务包。
- `dimos.robot.*` 只是把上面三层拼成具体的机器人 blueprint。比如
  `unitree-go2-agentic`（见 `AGENTS.md` 的 Quick Start）就是
  `unitree_go2_spatial + McpServer + McpClient + _common_agentic` 用
  `autoconnect` 拼出来的成品。

所以本章读完，你应该建立两个直觉：

1. **想看一台机器人是怎么组成的，去 `dimos.robot` 找对应的 blueprint
   定义**——它就是一行 `autoconnect(...)`。
2. **想理解任何一个业务模块（`PerceptionModule`、`NavigatorModule`、
   `AgentModule` ……）的"插槽"，回到本章重读 `In[T] / Out[T]` 那
   节就够了**——它们的"长相"全都一样。

---

<!-- doclinks-ignore-start -->
下一章：[`./02-agents.md`](./02-agents.md) — dimos.agents 子模块。
<!-- doclinks-ignore-end -->
