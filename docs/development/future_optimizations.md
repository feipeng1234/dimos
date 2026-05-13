# Future optimization directions

This note answers [upstream issue #62](https://github.com/dimensionalOS/dimos/issues/62). The canonical Issues URL is listed under `[project.urls]` in [`pyproject.toml`](../../pyproject.toml). Forks may mirror the same number (e.g. [feipeng1234/dimos#62](https://github.com/feipeng1234/dimos/issues/62)). The lists below are **candidates**—high-level only; benchmarks and designs belong in follow-up issues or RFCs.

**Prioritization tags** (subjective):

| Tag | Meaning |
|-----|---------|
| **Impact** | `H` high / `M` medium / `L` low for users and contributors |
| **Effort** | `S` small / `M` medium / `L` large |
| **Risk** | regressions, compatibility breaks, or operational complexity |

Use this as a **backlog sieve**, not a commitment order.

---

## Engineering and technical debt

Internal quality, CI, platforms, and contributor experience.

### Runtime, memory, and transports

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

### Reliability and observability

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

### Developer experience and tooling

8. **`GlobalConfig` / heavy imports** — `Impact: M`, `Effort: S`, `Risk: L`  
   *Why:* Accidental transitive imports slow CLI and tests.  
   *Lever:* Prefer thin config modules alongside heavy implementations (see conventions).  
   *Links:* [`conventions.md`](conventions.md), [`docs/usage/configuration.md`](../usage/configuration.md).

9. **Documentation examples that stay runnable** — `Impact: M`, `Effort: S`, `Risk: L`  
   *Why:* Drift between prose and APIs wastes contributor time; CI exercises doc code blocks via the `md-babel` job.  
   *Lever:* Local runs use [`bin/run-doc-codeblocks`](../../bin/run-doc-codeblocks); CI runs `./bin/run-doc-codeblocks --ci --no-cache` after `uv sync --group tests --frozen`.  
   *Links:* [`writing_docs.md`](writing_docs.md), [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) (job `md-babel`).

10. **Expand grid-style coverage for multi-backend behavior** — `Impact: M`, `Effort: M`, `Risk: L`  
    *Why:* Pub/sub and transport stacks benefit from the same test matrix pattern.  
    *Links:* [`grid_testing.md`](grid_testing.md), [`docs/agents/testing.md`](../agents/testing.md).

### CI, quality gates, and release cost

11. **Default vs self-hosted test split** — `Impact: H`, `Effort: S`, `Risk: L`  
    *Why:* Fast default feedback loops vs heavy `self_hosted` / `mujoco` / `tool` markers are central to velocity. Keeping that boundary sharp avoids either burning CI time or starving integration coverage.  
    *Facts (verified in this tree):*  
    - `[tool.pytest.ini_options]` → `addopts` in [`pyproject.toml`](../../pyproject.toml) includes `-m 'not (tool or self_hosted or mujoco)'`.  
    - [`bin/pytest-fast`](../../bin/pytest-fast) runs `pytest --numprocesses=auto "$@" dimos`; with no extra `-m`/`--override-ini`, that `addopts` marker filter applies.  
    - [`bin/pytest-slow`](../../bin/pytest-slow) runs `pytest --numprocesses=auto "$@" -m 'not (tool or mujoco)' dimos`, so `self_hosted` tests are included while `tool` and `mujoco` stay excluded.  
    - The `tests` job in [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) runs `uv run pytest --numprocesses=3 --cov=dimos/ --junitxml=junit.xml -m 'not (tool or self_hosted or mujoco)'`.  
    - The `self-hosted-tests` job runs pytest with `-m '(…) and not (tool or mujoco)'` where the first clause is `self_hosted or skipif_no_ros` on Linux and `self_hosted` on macOS, per the workflow matrix.  
    - The `ci-complete` job runs `re-actors/alls-green@release/v1` with `allowed-skips: self-hosted-tests`, so a **`skipped`** `self-hosted-tests` job (when that job’s `if:` is false—draft PRs or PRs from forks per [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml)) still satisfies the aggregate gate. A **`cancelled`** job or run (for example cancellation after another job fails: the workflow’s **`fail-fast`** job runs `gh run cancel` after `tests`) is not equivalent to skipped and may still fail `ci-complete`; see branch protection and runner capacity in practice.  
    *Links:* [`testing.md`](testing.md), [`pyproject.toml`](../../pyproject.toml), [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml).

12. **`tests` matrix cost (multiple Python versions)** — `Impact: M`, `Effort: M`, `Risk: L`  
    *Why:* The `tests` job runs pytest across several Python versions with `--numprocesses=3`; tuning that value vs matrix width affects wall-clock time for every PR.  
    *Links:* [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) (job `tests`).

13. **Self-hosted job reliability** — `Impact: H`, `Effort: M`, `Risk: M`  
    *Why:* `self-hosted-tests` uses labelled runners, optional Linux containers, and `uv sync --group tests-self-hosted --frozen`; failures tie up scarce hardware longer than GitHub-hosted jobs. Disk and environment parity are recurring themes.  
    *Links:* [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) (job `self-hosted-tests`).

14. **Markdown-only changes vs the main CI workflow** — `Impact: L`, `Effort: S`, `Risk: L`  
    *Why:* The `ci` workflow `on.push` (to `main`) / `on.pull_request` use `paths-ignore: '**.md'`, so a changeset that touches **only** `*.md` files does not start the workflow at all—including `lint`, `tests`, and `md-babel`. Any non-markdown path in the same push/PR (for example [`pyproject.toml`](../../pyproject.toml), [`.gitignore`](../../.gitignore), or files under [`.github/workflows/`](../../.github/workflows/)) still triggers it. Optional local or scheduled checks (e.g. link hygiene) can close gaps if needed.  
    *Links:* [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) (`on:` section).

### Operations and deployment

15. **Container images and reproducible ROS environments** — `Impact: M`, `Effort: M`, `Risk: L`  
    *Why:* The Linux matrix row for `self-hosted-tests` uses the CI image `ghcr.io/dimensionalos/ros-dev:dev` (see workflow); extending that pattern to local dev lowers “works on my machine” variance.  
    *Links:* [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml), [`docker.md`](docker.md).

---

## Product and capability roadmap (candidates)

User-facing stacks, agent behavior, simulation coverage, and discoverability.

### Agents, MCP, and safety

16. **Tooling surface area vs prompt quality** — `Impact: H`, `Effort: M`, `Risk: M`  
    *Why:* Every `@skill` appears in prompts; schemas and robot-specific prompts must stay aligned to avoid hallucinated tools or dangerous calls.  
    *Lever:* `dimos/agents/`, MCP server/client under `dimos/agents/mcp/`; defaults such as `GlobalConfig.mcp_port` (`9990`).  
    *Links:* [`AGENTS.md`](../../AGENTS.md) (agent system rules), [`docs/agents/index.md`](../agents/index.md).

17. **MCP and network boundaries** — `Impact: H`, `Effort: M`, `Risk: M`  
    *Why:* Exposing robot skills over HTTP raises auth, tenancy, and rate-limit questions for real deployments.  
    *Lever:* MCP server blueprint wiring; configuration via CLI / `GlobalConfig`.  
    *Non-goals:* This doc does not prescribe a threat model—that belongs to a dedicated security design.

### Multi-robot, simulation, and data

18. **Blueprint registry and discoverability** — `Impact: M`, `Effort: S`, `Risk: L`  
    *Why:* Generated registries drift if conventions are violated; `_` prefix conventions keep helper compositions out of `dimos list`.  
    *Links:* [`AGENTS.md`](../../AGENTS.md) (`all_blueprints.py` regeneration), [`docs/usage/blueprints.md`](../usage/blueprints.md).

19. **LFS-heavy assets and onboarding** — `Impact: M`, `Effort: S`, `Risk: L`  
    *Why:* Large recordings and models dominate clone time unless contributors use documented LFS workflows.  
    *Links:* [`large_file_management.md`](large_file_management.md).

20. **`mujoco` tests and simulator integration** — `Impact: M`, `Effort: L`, `Risk: M`  
    *Why:* Tests marked `mujoco` are excluded from default `addopts` / the `tests` CI marker filter today; bridging them safely into repeatable CI (or nightly) would tighten sim coverage without slowing every PR.  
    *Links:* [`testing.md`](testing.md) (`mujoco` marker).

---

### Non-goals for this document

- Replacing detailed guides (testing, transports, agents); use the links above.  
- Promising timelines or ownership—track those in GitHub issues or a project board.  
- Duplicating profiler setup; follow [`profiling_dimos.md`](profiling_dimos.md) instead.

---

### Suggested comment for GitHub issue #62 (copy-paste)

**EN:** A maintained, repo-aligned backlog of future optimization **candidates** (engineering vs product sections) lives at `docs/development/future_optimizations.md` (linked from `docs/README.md`, [`AGENTS.md`](../../AGENTS.md), and [`docs/development/testing.md`](testing.md)). Prefer the [upstream tracker](https://github.com/dimensionalOS/dimos/issues/62); maintainers can reprioritize or file labeled issues from the doc.

**ZH:** 已在仓库中补充面向维护者的「未来可优化方向」清单（工程债与产品能力分开），见 `docs/development/future_optimizations.md`（`docs/README.md`、`AGENTS.md`、`docs/development/testing.md` 均有入口）；以 [upstream issue #62](https://github.com/dimensionalOS/dimos/issues/62) 为主；欢迎在此文件基础上调整优先级或拆成独立 issue。

*After merge, paste the bilingual block above on GitHub issue #62 (upstream or fork); the repository cannot post that comment automatically.*
