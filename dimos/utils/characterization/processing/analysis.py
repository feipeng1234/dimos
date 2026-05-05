# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2025-2026 Dimensional Inc.
# Licensed under the Apache License, Version 2.0.

"""Recipe-specific session-level analyses: E8 deadtime + E7 cross-coupling.

Both analyses share the same shape: walk a session's run dirs, filter
to the runs that match the recipe family, load measured signals via
``validate._load_measured_for_run``, compute a small stat per run, then
aggregate. Kept together because they're each ~150 L of glue around the
same primitives.

Outputs:
  - ``deadtime_stats.json`` from ``deadtime_stats_session(session_dir)``
  - ``coupling_stats.json`` from ``coupling_stats_session(session_dir)``
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from dimos.utils.characterization.processing.validate import _load_measured_for_run

# =============================================================================
# E8 — deadtime statistics
#
# For each E8 short-step run, find:
#   - cmd edge time (when commanded leaves the pre-roll zero band)
#   - response onset (first time |measured| exceeds K × σ_noise on that channel)
#   - deadtime = response_onset - cmd_edge
#
# Aggregate across all matching runs in the session: mean, median, p95, jitter.
# Uses raw measured signal (not cleaned) — onset timing is exactly the kind
# of thing the cleaning rule says to keep raw.
# =============================================================================

_DEFAULT_THRESHOLD_K = 3.0  # threshold = K × σ_noise


def deadtime_stats_session(
    session_dir: Path,
    *,
    threshold_k: float = _DEFAULT_THRESHOLD_K,
) -> dict[str, Any]:
    """Compute deadtime per E8-style run and aggregate."""
    from dimos.utils.characterization.scripts.analyze import (
        reconstruct_body_velocities,
    )

    session_dir = Path(session_dir).expanduser().resolve()
    run_dirs = sorted(p for p in session_dir.iterdir() if p.is_dir() and p.name[0].isdigit())

    per_run: list[dict[str, Any]] = []
    for rd in run_dirs:
        run_json = rd / "run.json"
        if not run_json.exists():
            continue
        meta = json.loads(run_json.read_text())
        recipe = meta["recipe"]
        if recipe.get("test_type") != "step":
            continue
        # Heuristic: short hold (<= 1.0 s) means it's an E8-style run.
        if recipe.get("duration_s", 99) > 1.0:
            continue

        cmd_t, cmd_v, channel, target = _load_cmd_channel(rd, meta)
        if cmd_t.size == 0:
            continue

        meas_ts, meas_x, meas_y, meas_yaw = _load_measured_for_run(rd, meta)
        if meas_ts.size < 3:
            per_run.append(
                {
                    "run_id": meta["run_id"],
                    "recipe": recipe["name"],
                    "deadtime_s": None,
                    "reason": "no measured samples",
                }
            )
            continue
        vx, vy, wz = reconstruct_body_velocities(meas_ts, meas_x, meas_y, meas_yaw)
        meas_v = {"vx": vx, "vy": vy, "wz": wz}[channel]

        sigma = (meta.get("noise_floor") or {}).get(channel, {}).get("std")
        if not isinstance(sigma, (int, float)) or sigma <= 0:
            per_run.append(
                {
                    "run_id": meta["run_id"],
                    "recipe": recipe["name"],
                    "deadtime_s": None,
                    "reason": "missing noise_floor — run process_session validate first",
                }
            )
            continue

        threshold = threshold_k * sigma
        edge_t = _cmd_edge_time(cmd_t, cmd_v)
        onset_t = _onset_time(meas_ts, meas_v, threshold, after=edge_t, sign=np.sign(target))

        if edge_t is None or onset_t is None:
            per_run.append(
                {
                    "run_id": meta["run_id"],
                    "recipe": recipe["name"],
                    "deadtime_s": None,
                    "edge_t": edge_t,
                    "onset_t": onset_t,
                    "threshold": threshold,
                    "reason": "edge or onset not detected",
                }
            )
            continue

        per_run.append(
            {
                "run_id": meta["run_id"],
                "recipe": recipe["name"],
                "channel": channel,
                "target": float(target),
                "sigma": float(sigma),
                "threshold": round(threshold, 5),
                "edge_t_wall": round(edge_t, 4),
                "onset_t_wall": round(onset_t, 4),
                "deadtime_s": round(onset_t - edge_t, 4),
            }
        )

    deadtimes = np.array(
        [r["deadtime_s"] for r in per_run if isinstance(r.get("deadtime_s"), (int, float))],
        dtype=float,
    )
    summary: dict[str, Any]
    if deadtimes.size == 0:
        summary = {"n": 0, "reason": "no usable runs"}
    else:
        summary = {
            "n": int(deadtimes.size),
            "mean_s": round(float(np.mean(deadtimes)), 4),
            "median_s": round(float(np.median(deadtimes)), 4),
            "p95_s": round(float(np.percentile(deadtimes, 95)), 4),
            "p5_s": round(float(np.percentile(deadtimes, 5)), 4),
            "std_s": round(float(np.std(deadtimes, ddof=1)), 4) if deadtimes.size > 1 else 0.0,
            "min_s": round(float(np.min(deadtimes)), 4),
            "max_s": round(float(np.max(deadtimes)), 4),
        }

    out = {
        "session_dir": str(session_dir),
        "threshold_k": threshold_k,
        "summary": summary,
        "per_run": per_run,
    }
    (session_dir / "deadtime_stats.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def _load_cmd_channel(
    run_dir: Path, meta: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, str, float]:
    """Load cmd timestamps + amplitudes for the dominant channel.

    Returns (t_wall, values, channel_name, target_amplitude).
    """
    rows = [
        json.loads(line)
        for line in (run_dir / "cmd_monotonic.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        return np.array([]), np.array([]), "vx", 0.0
    vx = np.array([r["vx"] for r in rows], dtype=float)
    vy = np.array([r["vy"] for r in rows], dtype=float)
    wz = np.array([r["wz"] for r in rows], dtype=float)
    amps = {"vx": np.max(np.abs(vx)), "vy": np.max(np.abs(vy)), "wz": np.max(np.abs(wz))}
    channel = max(amps, key=lambda k: amps[k])
    arr = {"vx": vx, "vy": vy, "wz": wz}[channel]
    target_amp = float(arr[np.argmax(np.abs(arr))]) if arr.size else 0.0
    t = np.array([r["tx_wall"] for r in rows], dtype=float)
    return t, arr, channel, target_amp


def _cmd_edge_time(t: np.ndarray, v: np.ndarray, *, eps: float = 1e-3) -> float | None:
    nonzero = np.flatnonzero(np.abs(v) > eps)
    if nonzero.size == 0:
        return None
    return float(t[nonzero[0]])


def _onset_time(
    t: np.ndarray, v: np.ndarray, threshold: float, *, after: float, sign: float
) -> float | None:
    """First sample after ``after`` whose value exceeds threshold in ``sign`` direction."""
    if v.size == 0:
        return None
    if sign >= 0:
        cond = v > threshold
    else:
        cond = v < -threshold
    cond = cond & (t > after)
    idx = np.argmax(cond)
    if not cond[idx]:
        return None
    return float(t[idx])


# =============================================================================
# E7 — cross-coupling aggregation
#
# For each E7 run (constant-hold of pure wz or pure vx):
#   - active window = phase=="active" rows from cmd_monotonic.jsonl
#   - leak = mean abs(measured value on the *other* channel) during active
#   - leak as % of commanded primary channel
#   - leak as multiples of σ_noise on that channel
#
# Decision rule:
#     leak% < 5%  → SISO is fine
#     leak% 5-10% → flag, note in report
#     leak% > 10% → MIMO concern, controller needs cross-axis terms
# =============================================================================


def coupling_stats_session(session_dir: Path) -> dict[str, Any]:
    """Aggregate cross-coupling per E7 run, group by (test, amplitude, direction)."""
    from dimos.utils.characterization.scripts.analyze import (
        reconstruct_body_velocities,
    )

    session_dir = Path(session_dir).expanduser().resolve()
    run_dirs = sorted(p for p in session_dir.iterdir() if p.is_dir() and p.name[0].isdigit())

    per_run: list[dict[str, Any]] = []
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for rd in run_dirs:
        run_json = rd / "run.json"
        if not run_json.exists():
            continue
        meta = json.loads(run_json.read_text())
        recipe = meta["recipe"]
        if recipe.get("test_type") != "constant":
            continue
        # Heuristic: E7 recipe names start with "e7"
        if not recipe.get("name", "").lower().startswith("e7"):
            continue

        cmd = _load_cmd(rd)
        active = cmd["phase"] == "active"
        if not active.any():
            continue
        cmd_vx = cmd["vx"][active]
        cmd_wz = cmd["wz"][active]
        t_active_lo = float(cmd["tx_wall"][active][0])
        t_active_hi = float(cmd["tx_wall"][active][-1])

        # Determine primary channel: the one with non-zero commanded amplitude.
        primary = "vx" if abs(cmd_vx).max() > abs(cmd_wz).max() else "wz"
        primary_amp = float(np.mean({"vx": cmd_vx, "wz": cmd_wz}[primary]))

        meas_ts, meas_x, meas_y, meas_yaw = _load_measured_for_run(rd, meta)
        if meas_ts.size < 3:
            continue
        vx_meas, vy_meas, wz_meas = reconstruct_body_velocities(meas_ts, meas_x, meas_y, meas_yaw)
        in_active = (meas_ts >= t_active_lo) & (meas_ts <= t_active_hi)
        if int(in_active.sum()) < 3:
            continue

        meas_primary = {"vx": vx_meas, "wz": wz_meas}[primary][in_active]
        cross_channels = [c for c in ("vx", "vy", "wz") if c != primary]
        sigmas = meta.get("noise_floor") or {}

        leaks: dict[str, dict[str, Any]] = {}
        for ch in cross_channels:
            arr = {"vx": vx_meas, "vy": vy_meas, "wz": wz_meas}[ch][in_active]
            mean_abs_leak = float(np.mean(np.abs(arr)))
            peak_abs_leak = float(np.max(np.abs(arr)))
            sigma = (sigmas.get(ch) or {}).get("std")
            leaks[ch] = {
                "mean_abs": round(mean_abs_leak, 4),
                "peak_abs": round(peak_abs_leak, 4),
                "leak_pct_of_primary": (
                    round(100.0 * mean_abs_leak / abs(primary_amp), 2)
                    if abs(primary_amp) > 1e-6
                    else None
                ),
                "sigma_multiple": (
                    round(mean_abs_leak / sigma, 2)
                    if isinstance(sigma, (int, float)) and sigma > 0
                    else None
                ),
            }

        primary_steady = float(np.mean(meas_primary)) if meas_primary.size else 0.0
        run_summary = {
            "run_id": meta["run_id"],
            "recipe": recipe["name"],
            "primary_channel": primary,
            "primary_cmd": round(primary_amp, 3),
            "primary_meas_steady": round(primary_steady, 4),
            "leaks": leaks,
        }
        per_run.append(run_summary)
        by_group[recipe["name"]].append(run_summary)

    groups: list[dict[str, Any]] = [_aggregate_group(name, runs) for name, runs in by_group.items()]
    overall_decision = _overall_decision(groups)

    out = {
        "session_dir": str(session_dir),
        "decision_rule": "siso<5% / flag5-10% / mimo>10%",
        "overall_decision": overall_decision,
        "n_runs": len(per_run),
        "groups": groups,
        "per_run": per_run,
    }
    (session_dir / "coupling_stats.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def _load_cmd(run_dir: Path) -> dict[str, np.ndarray]:
    rows = [
        json.loads(line)
        for line in (run_dir / "cmd_monotonic.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return {
        "tx_wall": np.array([r["tx_wall"] for r in rows], dtype=float),
        "phase": np.array([r["phase"] for r in rows]),
        "vx": np.array([r["vx"] for r in rows], dtype=float),
        "vy": np.array([r["vy"] for r in rows], dtype=float),
        "wz": np.array([r["wz"] for r in rows], dtype=float),
    }


def _aggregate_group(name: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {"recipe": name, "n_runs": 0}

    primary = runs[0]["primary_channel"]
    leak_channels = list(runs[0]["leaks"].keys())
    leak_pcts: dict[str, list[float]] = {ch: [] for ch in leak_channels}
    sigma_mults: dict[str, list[float]] = {ch: [] for ch in leak_channels}
    primary_meas: list[float] = []

    for r in runs:
        for ch in leak_channels:
            v = r["leaks"][ch].get("leak_pct_of_primary")
            if isinstance(v, (int, float)):
                leak_pcts[ch].append(float(v))
            s = r["leaks"][ch].get("sigma_multiple")
            if isinstance(s, (int, float)):
                sigma_mults[ch].append(float(s))
        primary_meas.append(r["primary_meas_steady"])

    return {
        "recipe": name,
        "n_runs": len(runs),
        "primary_channel": primary,
        "primary_meas_steady_mean": round(float(np.mean(primary_meas)), 4)
        if primary_meas
        else None,
        "primary_meas_steady_std": (
            round(float(np.std(primary_meas, ddof=1)), 4) if len(primary_meas) > 1 else 0.0
        ),
        "leak_pct_mean": {
            ch: round(float(np.mean(v)), 2) if v else None for ch, v in leak_pcts.items()
        },
        "leak_pct_std": {
            ch: round(float(np.std(v, ddof=1)), 2) if len(v) > 1 else 0.0
            for ch, v in leak_pcts.items()
        },
        "sigma_multiple_mean": {
            ch: round(float(np.mean(v)), 2) if v else None for ch, v in sigma_mults.items()
        },
        "decision": _group_decision(leak_pcts),
    }


def _group_decision(leak_pcts: dict[str, list[float]]) -> str:
    """Worst-case across cross channels."""
    worst = 0.0
    for v in leak_pcts.values():
        if v:
            worst = max(worst, float(np.mean(v)))
    if worst < 5.0:
        return "SISO"
    if worst <= 10.0:
        return "FLAG"
    return "MIMO"


def _overall_decision(groups: list[dict[str, Any]]) -> str:
    decisions = [g.get("decision") for g in groups if g.get("decision")]
    if not decisions:
        return "no data"
    if "MIMO" in decisions:
        return "MIMO"
    if "FLAG" in decisions:
        return "FLAG"
    return "SISO"


__all__ = ["coupling_stats_session", "deadtime_stats_session"]
