# Future optimization directions

**中文摘要：** 本文列举 DimOS 日后可探索的优化方向，按主题分组；内容为维护者参考用的思路梳理，不构成排期或承诺。（相关讨论示例：列举未来优化方向一类的问题，如 issue #78。）

This note enumerates **non-binding** themes for improving DimOS over time. It is maintainer-facing context for roadmap-style discussions; nothing here is a commitment or dated roadmap.

DimOS centers on **modules** communicating via typed streams, **blueprints** that compose stacks, **skills** exposed to agents, and transports such as LCM, ROS2, and DDS (see `AGENTS.md` and `README.md`). The items below sit on that baseline.

---

## Packaging and optional dependencies

- **Lean core installs and optional visualization** — Core dependencies currently include `rerun-sdk` with an inline `TODO` that rerun should not be required forever (`pyproject.toml`). Progress here would shrink install surfaces and clarify which stacks require visualization viewers.

## Transports and data path efficiency

- **Transport and serialization efficiency** — DimOS supports multiple transports (including shared-memory style paths for heavy payloads). Future work could systematically profile hot topics, reduce copies, and document transport choice per workload so latency- and bandwidth-sensitive stacks stay predictable.

## Module runtime and lifecycle

- **Forkserver workers and lifecycle hardening** — Modules run in worker processes with coordinated startup and shutdown. Improvements could focus on faster cold start, clearer failure modes when a worker dies, and more uniform graceful teardown across blueprints.

## Blueprints and cross-module wiring

- **Cross-module RPC and blueprint ergonomics** — Spec-based injection already fails at build time when wiring is wrong. Further optimization could reduce boilerplate, improve error messages for ambiguous matches, and extend patterns for larger heterogeneous graphs without sacrificing type safety.

## Agents, skills, and safety

- **Agent tooling quality and safety** — Skills rely on docstrings, annotations, and structured schemas for LLM tools. Directions include richer validation, safer defaults for physical actions, better alignment between system prompts and available skills per robot, and tighter feedback loops when tool calls fail.

## MCP and external integration

- **MCP and external integration** — The HTTP MCP surface exposes skills to clients. Optimizations might cover authentication, rate limiting, versioning of tool schemas, and smoother discovery for multi-agent or remote operator setups.

## Observability

- **Observability across processes** — Today operators use CLI commands such as `dimos log`, `dimos status`, and optional visualization. A cohesive story for correlating logs across workers, tracing stream backpressure, and exporting metrics would speed up field debugging.

## Testing and CI feedback

- **Testing pyramid and CI feedback** — The default pytest `addopts` in `pyproject.toml` include `-m 'not (tool or self_hosted or mujoco)'`, so plain `pytest dimos` (or `uv run pytest`) skips those markers unless overridden. `./bin/pytest-fast` runs `pytest --numprocesses=auto "$@" dimos` (see `bin/pytest-fast`; same parallelism pattern as `docs/development/testing.md`). Future work could keep the fast loop fast while making selective heavier runs easier to trigger locally and in automation.

## Documentation and onboarding

- **Documentation and onboarding paths** — The `docs/` tree covers modules, blueprints, transports, agents, and platforms. Improvements could add more vertical “day one” paths per robot class and cross-link capability docs (`docs/capabilities/`) with concrete blueprint names where stable.

## Profiling and regression detection

- **Performance profiling workflows** — Profiling guidance exists (`docs/development/profiling_dimos.md`). Extending it with recurring benchmarks for representative blueprints would make regressions easier to spot early.

## Hardware, simulation, and connectivity

- **Reliability on real hardware and flaky networks** — Robot connectivity, replay, and simulation flags (`README.md`, `AGENTS.md`) underpin development workflows. Hardening reconnection, timeouts, and operator-visible diagnostics would reduce downtime when links degrade.

## Perception, spatial memory, and product pillars

- **Spatial memory and perception pipelines** — README highlights navigation/mapping, perception, and spatial memory as product pillars. Optimization directions include latency budgets end-to-end, memory footprint for long runs, and reproducible evaluation harnesses for perception-backed skills.

---

## Non-goals (for this note)

- This is **not** a dated roadmap, priority stack rank, or commitment to deliver any item above.
- It does **not** replace feature-specific design notes, ADRs, or issue threads where tradeoffs get decided.
- Nothing here prescribes **implementation** choices; those belong in focused proposals and code review.
