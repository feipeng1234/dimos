# 未来优化方向（提案）

本文回应诸如「请列一下未来的优化方向」一类的问题：**只列出基于本仓库已有注释、依赖声明与文档措辞的方向**，方便评审对照 Issue；**不是路线图承诺**，条目优先级均为维护者可调整的**提案**。

*This page lists **proposal-only** optimization directions grounded in this repository (comments, dependency notes, docs). It is **not** a committed roadmap.*

---

## 依赖体积与可选组件 *(proposal)*

- **降低核心安装对可视化栈的硬性绑定**：`pyproject.toml` 中注明 `rerun-sdk` 目前在核心依赖里、`rerun` 尚无法真正可选；注释写明希望在核心路径上把 rerun 变为可选后再移除该依赖。*（EN: shrink mandatory visualization footprint for headless / minimal installs.）*
- **恢复或替换 Jetson 专用 wheel 源**：`pyproject.toml` 中原计划通过注释掉的 **`jetson-jp6-cuda126`** 块提供 Jetson JP6/CUDA 12.6 wheel；该块因上游 URL 404 / 不可用的 NOTE 而整段注释；后续可在有可下载地址时恢复或改用其他分发方式。*（EN: re-enable embedded GPU installs once artifact URLs are reliable.）*

## 导航与其它子系统收敛 *(proposal)*

- **合并多套导航说明/实现入口**：`docs/capabilities/navigation/readme.md` 写明 Nav Stack 与 Simple Nav「将来会合并为一个系统」；文档与示例结构的收敛可减少用户选型成本。*（EN: unify navigation docs and composition paths per existing note.）*

## 传输层与 IPC 性能路径 *(proposal)*

- **共享内存后端补齐 GPU/CUDA 路径**：`dimos/protocol/pubsub/impl/shmpubsub.py` 中 `SharedMemoryConfig.prefer` 注释为 `"auto" | "cpu"` 且 `TODO: "cuda"`，另有 `is_cuda` / CUDA IPC 初始化相关 TODO；`dimos/protocol/pubsub/shm/ipc_factory.py` 含 `# TODO: Implement the CUDA version of creating this factory`。*（EN: complete CUDA-capable shared-memory pub/sub where those TODOs mark gaps.）*
- **蓝图级全局传输切换**：`dimos/robot/unitree/go2/blueprints/basic/unitree_go2_basic.py` 注释提到需要蓝图 / 全局配置层面的传输开关，以减少逐蓝图手工覆盖的成本。*（EN: central transport toggles in GlobalConfig / blueprint.）*

## 消息、几何与 TF 一致性 *(proposal)*

- **`PointCloud2` 编解码覆盖完整字段集**：`dimos/msgs/sensor_msgs/PointCloud2.py` 中 TODO 指向需支持更广的 pointcloud2 字段 spectrum。*（EN: broaden encode/decode beyond current subset.）*
- **导航侧改用 TF 取位姿**：`dimos/navigation/replanning_a_star/module.py` 中 `odom: In[PoseStamped]` 带 `# TODO: Use TF.`。*（EN: consume TF instead of a dedicated odometry stream where annotated.）*
- **TF 服务的 transport 无关化**：`dimos/protocol/tf/tflcmcpp.py` 类文档写明理想情况是 generic pub/sub 管理变换，从而 **transport agnostic**（标注为 TODO）；当前说明因 `tf_lcm_py`/C++ 缓冲等原因未做。*（EN: longer-term decouple TF from LCM-specific wiring per that comment.）*

## 可视化与调试链路 *(proposal)*

- **Rerun bridge 行为与配置**：`dimos/visualization/rerun/bridge.py` 多处 TODO（外部标注、TF 处理、配置归属、无显示环境判定等）指向可视化管线仍可收紧边界与配置。*（EN: clarify rerun bridge responsibilities and headless behavior.）*
- **文档与 Foxglove/Rerun 叙述对齐**：`docs/usage/transforms.md` 注明需针对 rerun 更新 Foxglove 相关说明。*（EN: keep viz docs accurate across viewers.）*

## Agent、MCP 与 Web 服务可靠性 *(proposal)*

- **MCP HTTP 服务的线程安全与实现整洁度**：`dimos/agents/mcp/mcp_server.py` 中源码注释写明当前做法 `a bit hacky`，且 `not thread-safe`。*（EN: harden concurrent access patterns.）*
- **FastAPI / Web 线程与流生命周期**：`dimos/web/fastapi_server.py`、`dimos/web/dimos_interface/api/server.py` 等处 TODO 指向理清线程模型与流的启停。*（EN: predictable start/stop for streaming endpoints.）*

## 内存与时间序列模块 *(proposal)*

- **`memory2` 查询与存储语义**：`dimos/memory2/module.py` TODO 涉及按峰值聚类排序结果、以及 store reset API 尚未实现；完善后可改善长时间运行与会话边界场景。*（EN: clustering / reset semantics for temporal memory.）*

## 机器人连接与相机约定 *(proposal)*

- **统一的相机约定（含链路与 CameraInfo）**：`dimos/robot/unitree/go2/connection.py` 在 `_camera_info_static` 与静态相机 mount 链（`BASE_TO_OPTICAL`）处有 `# TODO we need a standardized way to specify this for all cameras in dimos`。*（EN: one standardized way to specify camera mount + intrinsic metadata across robots, per that TODO.）*

## 开发者体验与文档 *(proposal)*

- **模块配置的易打印 / 可观测性**：`docs/usage/modules.md` TODO 希望增加便于打印配置的途径。*（EN: better config introspection for modules.）*
- **可执行示例与 md-babel**：`docs/usage/data_streams/advanced_streams.md` 注明示例应真正可执行；`docs/development/writing_docs.md` 建议使用 `md-babel-py` 校验文档代码块。*（EN: close the gap between prose examples and runnable snippets.）*

## 测试与反馈速度 *(proposal)*

- **保持「默认套件要快」的分层策略**：`docs/development/testing.md` 将默认套件与 `self_hosted` / `mujoco` / `tool` 等标记区分开；后续可继续在默认套件中覆盖关键路径，把重型依赖留在标记测试中。*（EN: preserve fast inner loop vs heavy CI buckets.）*
- **跨实现的网格测试**：`docs/development/grid_testing.md` 描述 `Case` + capability tags；例如 `dimos/protocol/pubsub/test_pattern_sub.py` 已用 `Case` 参数化多套 pub/sub 行为；类似模式可推广到其它多后端特性。*（EN: reuse the grid `Case` pattern where several implementations share test logic.）*

## 静态分析与代码卫生 *(proposal)*

- **逐步收紧 Ruff 忽略项**：`pyproject.toml` 的 `[tool.ruff.lint] ignore` 列表附带注释，说明这些规则应先修、但分批提交 autofix 更容易。*（EN: reduce permanently ignored rules over time.）*
- **遗留模块清理**：例如 `dimos/robot/robot.py` 标明 `TODO: Delete`；清理可减少入口混淆。*（EN: delete or migrate deprecated entrypoints explicitly called out in source.）*

## 性能剖析工作流（已有文档）*(proposal)*

- **建立改动前后的剖析习惯**：`docs/development/profiling_dimos.md` 已给出 `py-spy` + speedscope 的命令示例；在具体性能项立项后，用同一流程做前后对比即可。*（EN: adopt documented py-spy workflow when tackling hotspots.）*

---

## 验证（文档变更后）

纯文档改动不改变 Python 行为。`docs/development/testing.md` 写明默认套件的快捷方式是 `./bin/pytest-fast`，并与 `pytest --numprocesses=auto dimos` 等价；`bin/pytest-fast` 还会在运行前执行 `. .venv/bin/activate`（即激活仓库根目录下的 `.venv`）。
