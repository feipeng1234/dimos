# Future optimization opportunities (DimOS)

> **中文概要**：下列内容汇总 DimOS 后续可优先改进的方向，依据仓库内代码注释（`TODO`）、架构取舍与运维现状整理而成，便于评审与排期；并非承诺路线图。

This note lists directions where DimOS can improve next. Items are drawn from in-repo `TODO` comments, known tech debt, and operator-facing friction (for example long CI runs described in `AGENTS.md`). It is a working backlog sketch, not a committed roadmap.

## Transports, IPC, and performance

- **CUDA-capable shared-memory IPC**: `dimos/protocol/pubsub/shm/ipc_factory.py` and `dimos/protocol/pubsub/impl/shmpubsub.py` call out a future CUDA path for zero-copy style data paths on GPU-heavy stacks.
- **Transport selection ergonomics**: blueprints today benefit from a clearer global toggle for default transports (`unitree_go2_basic.py` notes a desired global transport switch in blueprint / `GlobalConfig`).
- **Framework-agnostic TF**: `dimos/protocol/tf/tflcmcpp.py` notes moving TF helpers toward transport-agnostic APIs.

## Core runtime and workers

- **Worker error reporting**: `dimos/core/coordination/python_worker.py` matches coarse `AttributeError` strings; richer, structured errors would speed debugging across forked workers.
- **Decorators and resource lifecycle**: `dimos/mapping/costmapper.py` disables `@timed()` due to a thread leak — fixing the decorator is preferable to leaving timing disabled.

## Agents, MCP, and skills

- **MCP server robustness**: `dimos/agents/mcp/mcp_server.py` flags thread-safety and “hacky” wiring; hardening here improves reliability when many tools are registered.
- **REST skill surface**: `dimos/skills/rest/rest.py` — extend support for query parameters, bodies, and headers for richer tool integrations.
- **Skill implementation gaps**: e.g. `dimos/skills/manipulation/manipulate_skill.py` has explicit “TODO: Implement” paths to complete.

## Visualization and operator UX

- **Rerun bridge**: `dimos/visualization/rerun/bridge.py` — better TF handling, out-of-tree annotation publishing, and config for magic constants.
- **Foxglove → Rerun**: `dimos/manipulation/blueprints.py` still carries `FoxgloveBridge` with notes to migrate visualization to Rerun for consistency with the rest of the stack.

## Web stack

- **FastAPI / threading lifecycle**: `dimos/web/fastapi_server.py` and `dimos/web/dimos_interface/api/server.py` — clarify threading model and cleanly start/stop streaming endpoints.

## Navigation, mapping, and perception messages

- **TF instead of ad-hoc odom wiring**: `dimos/navigation/replanning_a_star/module.py` TODOs for TF-based pose inputs and navigation state publishing.
- **Point cloud and camera metadata**: `dimos/msgs/sensor_msgs/PointCloud2.py` and `CameraInfo.py` — broader field coverage and moving calibration defaults closer to emitting modules.
- **Mapping utilities**: path resampling (`dimos/mapping/occupancy/path_resampling.py`), OSM query simplification, and `memory2` reset / store APIs called out in-module.

## Robot integration (Unitree and cameras)

- **Camera configuration**: `dimos/robot/unitree/go2/connection.py` — standardized way to describe intrinsics / topics for all DimOS cameras.
- **Deprecation cleanup**: remove or replace legacy pieces such as `dimos/robot/unitree/rosnav.py` and stale abstractions noted in `dimos/robot/robot.py`.

## Streams, replay, and video

- **Frame pipeline organization**: `dimos/stream/frame_processor.py` — consolidate with related video operator types; improve recording folder naming.
- **ROS video rate control**: `dimos/stream/ros_video_provider.py` — throttle using existing reactive operators.

## Testing, CI, and hygiene

- **Test design**: some Unitree tests reach too deeply into connection internals (`b1/test_connection.py`); shallower fixtures would reduce breakage churn.
- **Point cloud equality in tests**: `dimos/mapping/test_voxels.py` — proper `PointCloud2` comparison would shrink flaky assertions.
- **CI duration**: `AGENTS.md` notes ~hour-long self-hosted CI; sharding, better markers, or narrower slow suites remain an optimization for contributor velocity.
- **`agents_deprecated/`**: several tokenizer / memory TODOs remain; either finish migrations into `dimos/agents/` or delete unused paths to reduce confusion.

## How to use this document

When picking work, prefer issues that cite a concrete module and observable pain (latency, flakes, operator steps). Replace or remove `TODO` comments as items are addressed so the source of truth stays in code.
