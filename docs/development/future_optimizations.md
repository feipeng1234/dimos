# Future optimization directions

This note answers [Issue #62](https://github.com/dimensionalOS/dimos/issues/62); canonical repository URLs appear under `[project.urls]` in [`pyproject.toml`](../../pyproject.toml). It is a **roadmap-style** list of concrete directions maintainers might pursue—high-level only; benchmarks and implementations belong in follow-up work.

**Prioritization tags** (subjective):

| Tag | Meaning |
|-----|---------|
| **Impact** | `H` high / `M` medium / `L` low for users and contributors |
| **Effort** | `S` small / `M` medium / `L` large |
| **Risk** | regressions, compatibility breaks, or operational complexity |

Use this as a **backlog sieve**, not a commitment order.

---

## Runtime, memory, and transports

1. **Narrow mandatory dependencies where the codebase already acknowledges coupling** — `Impact: H`, `Effort: M`, `Risk: M`  
   *Why:* Smaller installs and faster cold starts help everyone from CI to laptops.  
   *Lever:* `pyproject.toml` already tracks that `rerun-sdk` should not be unconditionally required (“remove this once rerun is optional in core”).  
   *Links:* [`pyproject.toml`](../../pyproject.toml) (dependencies + comment), [`docs/usage/visualization.md`](../usage/visualization.md).

2. **Profiling-driven hot-path work on real blueprints** — `Impact: M`, `Effort: M`, `Risk: L`  
   *Why:* End-to-end stacks (workers, transports, perception) dominate latency more than micro-optimizations in isolation.  
   *Lever:* `dimos run …` blueprints (`dimos/core/coordination/`, transports under `dimos/core/`).  
   *Links:* [`profiling_dimos.md`](profiling_dimos.md).

3. **Transport and serialization choices per stream type** — `Impact: H`, `Effort: L`, `Risk: M`  
   *Why:* Images and dense geometry need different transports than small control messages.  
   *Lever:* `In`/`Out` wiring and transport overrides documented under usage.  
   *Links:* [`docs/usage/transports/index.md`](../usage/transports/index.md), [`docs/usage/blueprints.md`](../usage/blueprints.md).

4. **Worker pool and IPC efficiency** — `Impact: M`, `Effort: L`, `Risk: M`  
   *Why:* Forkserver workers and stream fan-out affect CPU and memory at scale.  
   *Lever:* `dimos/core/coordination/` (orchestration), module lifecycle.  
   *Non-goals:* Rewriting the whole process model without measurement.

---

## Reliability and observability

5. **Consistent visualization entry points (`vis_module`)** — `Impact: M`, `Effort: M`, `Risk: L`  
   *Why:* One pattern reduces “works in rerun but broken in foxglove” drift and aligns with CLI `--viewer`.  
   *Lever:* Blueprint composition; conventions already prefer `vis_module` over wiring `RerunBridge` directly.  
   *Links:* [`conventions.md`](conventions.md), [`docs/usage/visualization.md`](../usage/visualization.md).

6. **Per-module rerun layout and config** — `Impact: M`, `Effort: L`, `Risk: M`  
   *Why:* Blueprint-level `rerun_config` / `rrb` complicates reusable modules; per-module hooks scale better as blueprints grow.  
   *Links:* [`conventions.md`](conventions.md).

7. **Structured logs and failure modes across workers** — `Impact: H`, `Effort: M`, `Risk: L`  
   *Why:* Cross-process failures are hard to correlate without consistent context (`structlog` is already in use).  
   *Lever:* Logging helpers under `dimos/utils/`, run registry paths described in repo docs.

---

## Developer experience and tooling

8. **`GlobalConfig` / heavy imports** — `Impact: M`, `Effort: S`, `Risk: L`  
   *Why:* Accidental transitive imports slow CLI and tests.  
   *Lever:* Prefer thin config modules alongside heavy implementations (see conventions).  
   *Links:* [`conventions.md`](conventions.md), [`docs/usage/configuration.md`](../usage/configuration.md).

9. **Documentation examples that stay runnable** — `Impact: M`, `Effort: S`, `Risk: L`  
   *Why:* Drift between prose and APIs wastes contributor time; CI already exercises doc code paths in some workflows.  
   *Links:* [`writing_docs.md`](writing_docs.md), [`bin/run-doc-codeblocks`](../../bin/run-doc-codeblocks), [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) (`md-babel` job).

10. **Expand grid-style coverage for multi-backend behavior** — `Impact: M`, `Effort: M`, `Risk: L`  
    *Why:* Pub/sub and transport stacks benefit from the same test matrix pattern.  
    *Links:* [`grid_testing.md`](grid_testing.md), [`docs/agents/testing.md`](../agents/testing.md).

---

## CI, quality gates, and release cost

11. **Default vs self-hosted test split** — `Impact: H`, `Effort: S`, `Risk: L`  
    *Why:* Fast default feedback loops vs heavy `self_hosted` / `mujoco` / `tool` markers are central to velocity. Keeping that boundary sharp avoids either burning CI time or starving integration coverage.  
    *Facts:* `[tool.pytest.ini_options]` → `addopts` in [`pyproject.toml`](../../pyproject.toml) sets `-m 'not (tool or self_hosted or mujoco)'`; `./bin/pytest-fast` invokes `pytest` with `--numprocesses=auto` and does not override `-m`, so those defaults apply. `./bin/pytest-slow` runs `pytest --numprocesses=auto … -m 'not (tool or mujoco)'`, so `self_hosted` tests are included while `tool` and `mujoco` stay excluded. The `tests` CI job runs `uv run pytest … -m 'not (tool or self_hosted or mujoco)'` (same marker expression as `addopts`; see workflow file).  
    *Links:* [`testing.md`](testing.md), [`pyproject.toml`](../../pyproject.toml) (`[tool.pytest.ini_options]`), [`bin/pytest-fast`](../../bin/pytest-fast), [`bin/pytest-slow`](../../bin/pytest-slow).

12. **`tests` matrix cost (multiple Python versions)** — `Impact: M`, `Effort: M`, `Risk: L`  
    *Why:* The `tests` job in CI runs pytest across several Python versions with a fixed `--numprocesses=3`; tuning that value vs matrix width affects wall-clock time for every PR.  
    *Links:* [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml).

13. **Self-hosted job reliability** — `Impact: H`, `Effort: M`, `Risk: M`  
    *Why:* `self-hosted-tests` uses labelled runners, containers (Linux), and installs `tests-self-hosted`; failures tie up scarce hardware longer than GitHub-hosted jobs. Disk and environment parity are recurring themes.  
    *Links:* [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) (`self-hosted-tests` job).

14. **Markdown-only workflows and validation** — `Impact: L`, `Effort: S`, `Risk: L`  
    *Why:* `paths-ignore: '**.md'` on push/PR means doc edits do not trigger the main `ci` workflow; optional local or scheduled checks (e.g. link hygiene) can close the gap if needed.  
    *Links:* [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) (`on:` section).

---

## Agents, MCP, and safety

15. **Tooling surface area vs prompt quality** — `Impact: H`, `Effort: M`, `Risk: M`  
    *Why:* Every `@skill` appears in prompts; schemas and robot-specific prompts must stay aligned to avoid hallucinated tools or dangerous calls.  
    *Lever:* `dimos/agents/`, MCP server/client under `dimos/agents/mcp/`; defaults such as `GlobalConfig.mcp_port` (`9990`).  
    *Links:* [`AGENTS.md`](../../AGENTS.md) (agent system rules), [`docs/agents/index.md`](../agents/index.md).

16. **MCP and network boundaries** — `Impact: H`, `Effort: M`, `Risk: M`  
    *Why:* Exposing robot skills over HTTP raises auth, tenancy, and rate-limit questions for real deployments.  
    *Lever:* MCP server blueprint wiring; configuration via CLI / `GlobalConfig`.  
    *Non-goals:* This doc does not prescribe a threat model—that belongs to a dedicated security design.

---

## Multi-robot, simulation, and data

17. **Blueprint registry and discoverability** — `Impact: M`, `Effort: S`, `Risk: L`  
    *Why:* Generated registries drift if conventions are violated; `_` prefix conventions keep helper compositions out of `dimos list`.  
    *Links:* [`AGENTS.md`](../../AGENTS.md) (`all_blueprints.py` regeneration), [`docs/usage/blueprints.md`](../usage/blueprints.md).

18. **LFS-heavy assets and onboarding** — `Impact: M`, `Effort: S`, `Risk: L`  
    *Why:* Large recordings and models dominate clone time unless contributors use documented LFS workflows.  
    *Links:* [`large_file_management.md`](large_file_management.md).

19. **`mujoco` tests and simulator integration** — `Impact: M`, `Effort: L`, `Risk: M`  
    *Why:* Marked tests are excluded from default CI flows today; bridging them safely into repeatable CI (or nightly) would tighten sim coverage without slowing every PR.  
    *Links:* [`testing.md`](testing.md) (`mujoco` marker).

---

## Operations and deployment

20. **Container images and reproducible ROS environments** — `Impact: M`, `Effort: M`, `Risk: L`  
    *Why:* Self-hosted ROS jobs already rely on pinned images (`ghcr.io/dimensionalos/ros-dev:dev`); extending that pattern to local dev lowers “works on my machine” variance.  
    *Links:* [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml), [`docker.md`](docker.md).

---

### Non-goals for this document

- Replacing detailed guides (testing, transports, agents); use the links above.  
- Promising timelines or ownership—track those in GitHub issues or a project board.  
- Duplicating profiler setup; follow [`profiling_dimos.md`](profiling_dimos.md) instead.
