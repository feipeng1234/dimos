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

"""Per-run data validation + noise-floor estimation.

Validation
----------
Non-destructive: writes a new ``validation.json`` next to ``run.json``;
does not modify anything else. Checks:

  1. Commanded sample-rate consistency (jsonl tx_mono cadence vs recipe rate)
  2. Monotonic timestamps on commanded + measured streams
  3. No gaps > 2x nominal period in either stream
  4. Cross-correlation peak between cmd_vx and meas_vx at *positive* lag
     (a peak at 0 or negative lag means timestamps are wrong)

Noise floor
-----------
The pre-roll is recorded with all-zero commanded velocities, so the
measured signal during pre-roll *is* the noise floor (drift, IMU bias,
odometry quantization). Computed sigma values are appended to ``run.json``
under a top-level ``noise_floor`` key (additive — never overwrites
existing fields). Safe to rerun. Downstream thresholds (deadtime onset,
coupling leak) read this number instead of carrying magic constants.

CLI: ``python -m dimos.utils.characterization.scripts.process_session validate``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: dict[str, Any]


def validate_run(run_dir: Path) -> dict[str, Any]:
    """Run all per-run validation checks. Returns a dict; also writes
    ``<run_dir>/validation.json``.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    meta = json.loads((run_dir / "run.json").read_text())
    cmd_jsonl = run_dir / "cmd_monotonic.jsonl"
    cmd = _load_cmd_jsonl(cmd_jsonl)
    meas_ts, meas_x, meas_y, meas_yaw = _load_measured_for_run(run_dir, meta)

    recipe_rate = float(meta["recipe"]["sample_rate_hz"])
    expected_meas_rate = 10.0  # real Go2 nominal odom rate; tolerated above

    checks: list[CheckResult] = [
        _check_sample_rate(cmd["tx_mono"], recipe_rate, name="cmd_sample_rate"),
        _check_monotonic(cmd["tx_mono"], name="cmd_ts_monotonic"),
        _check_no_gaps(cmd["tx_mono"], 2.0 / recipe_rate, name="cmd_no_gaps_2x"),
    ]
    if meas_ts.size:
        checks.append(_check_monotonic(meas_ts, name="meas_ts_monotonic"))
        checks.append(_check_no_gaps(meas_ts, 2.0 / expected_meas_rate, name="meas_no_gaps_2x"))
        checks.append(_check_xcorr_positive_lag(cmd, meas_ts, meas_x, meta))
    else:
        checks.append(CheckResult("meas_present", False, {"meas_count": 0}))

    passed = all(c.passed for c in checks)
    out: dict[str, Any] = {
        "run_id": meta["run_id"],
        "recipe": meta["recipe"]["name"],
        "passed": passed,
        "checks": [asdict(c) for c in checks],
    }
    (run_dir / "validation.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def validate_session(session_dir: Path) -> dict[str, Any]:
    """Validate every run in a session. Writes a summary at the session root."""
    session_dir = Path(session_dir).expanduser().resolve()
    run_dirs = sorted(p for p in session_dir.iterdir() if p.is_dir() and p.name[0].isdigit())
    results: list[dict[str, Any]] = []
    for rd in run_dirs:
        try:
            results.append(validate_run(rd))
        except Exception as e:  # pragma: no cover
            results.append({"run_id": rd.name, "passed": False, "error": str(e)})

    summary = {
        "session_dir": str(session_dir),
        "total_runs": len(results),
        "passed": sum(1 for r in results if r.get("passed")),
        "failed": sum(1 for r in results if not r.get("passed")),
        "failures": [
            {
                "run_id": r["run_id"],
                "failed_checks": [c["name"] for c in r.get("checks", []) if not c["passed"]],
            }
            for r in results
            if not r.get("passed")
        ],
    }
    (session_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


# ---------------------------------------------------------------------------- internal


def _load_cmd_jsonl(path: Path) -> dict[str, np.ndarray]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {
        "seq": np.array([r["seq"] for r in rows], dtype=int),
        "tx_mono": np.array([r["tx_mono"] for r in rows], dtype=float),
        "tx_wall": np.array([r["tx_wall"] for r in rows], dtype=float),
        "phase": np.array([r["phase"] for r in rows]),
        "vx": np.array([r["vx"] for r in rows], dtype=float),
        "vy": np.array([r["vy"] for r in rows], dtype=float),
        "wz": np.array([r["wz"] for r in rows], dtype=float),
    }


def _load_measured_for_run(
    run_dir: Path, meta: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (ts, x, y, yaw) sliced to the run's ts_window_wall."""
    from dimos.memory2.store.sqlite import SqliteStore
    from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped

    session_db_rel = meta.get("session_db_path")
    win = meta.get("ts_window_wall") or {}
    if session_db_rel:
        db = (run_dir / session_db_rel).resolve()
    else:
        db = run_dir / "recording.db"
    if not db.exists():
        return (np.array([]),) * 4  # type: ignore[return-value]

    store = SqliteStore(path=str(db))
    store.start()
    try:
        ts: list[float] = []
        x: list[float] = []
        y: list[float] = []
        yaw: list[float] = []
        ts_lo = win.get("start")
        ts_hi = win.get("end")
        for obs in store.stream("measured", PoseStamped).to_list():
            t = float(obs.ts)
            if session_db_rel:
                if ts_lo is not None and t < ts_lo:
                    continue
                if ts_hi is not None and t > ts_hi:
                    continue
            ts.append(t)
            x.append(float(obs.data.x))
            y.append(float(obs.data.y))
            yaw.append(float(obs.data.yaw))
    finally:
        store.stop()
    return (
        np.array(ts, dtype=float),
        np.array(x, dtype=float),
        np.array(y, dtype=float),
        np.array(yaw, dtype=float),
    )


def _check_sample_rate(ts: np.ndarray, expected_hz: float, *, name: str) -> CheckResult:
    """Median dt should match 1/expected_hz within 5%."""
    if ts.size < 3:
        return CheckResult(name, False, {"reason": "too few samples", "n": int(ts.size)})
    dt = np.diff(ts)
    median_dt = float(np.median(dt))
    actual_hz = 1.0 / median_dt if median_dt > 0 else 0.0
    err = abs(actual_hz - expected_hz) / expected_hz
    return CheckResult(
        name,
        passed=bool(err < 0.05),
        detail={
            "expected_hz": expected_hz,
            "actual_hz": round(actual_hz, 3),
            "median_dt_s": round(median_dt, 5),
            "rel_error": round(float(err), 4),
        },
    )


def _check_monotonic(ts: np.ndarray, *, name: str) -> CheckResult:
    if ts.size < 2:
        return CheckResult(name, True, {"n": int(ts.size)})
    deltas = np.diff(ts)
    n_violations = int(np.sum(deltas <= 0))
    return CheckResult(
        name,
        passed=bool(n_violations == 0),
        detail={"n": int(ts.size), "n_non_increasing": n_violations},
    )


def _check_no_gaps(ts: np.ndarray, gap_limit_s: float, *, name: str) -> CheckResult:
    """No inter-sample interval should exceed 2x nominal period."""
    if ts.size < 2:
        return CheckResult(name, True, {"n": int(ts.size)})
    deltas = np.diff(ts)
    max_dt = float(np.max(deltas))
    return CheckResult(
        name,
        passed=bool(max_dt <= gap_limit_s),
        detail={"max_dt_s": round(max_dt, 5), "limit_s": round(gap_limit_s, 5)},
    )


def _check_xcorr_positive_lag(
    cmd: dict[str, np.ndarray],
    meas_ts: np.ndarray,
    meas_x: np.ndarray,
    meta: dict[str, Any],
) -> CheckResult:
    """The cmd→meas correlation peak should be at positive lag (causal).

    Resamples both to a common 50 Hz grid using nearest-neighbor; computes
    cross-correlation; peak lag in seconds = (peak_idx - n) / fs.
    """
    name = "xcorr_positive_lag"
    if meas_ts.size < 4:
        return CheckResult(name, False, {"reason": "not enough measured samples"})

    # Use vx as the comparable channel (most likely non-trivial); for wz-only
    # recipes we'd need to derive vx from yaw, but for now just use raw cmd_vx.
    fs = 50.0
    anchor_wall = float(meta["clock_anchor"]["wall"])
    anchor_mono = float(meta["clock_anchor"]["monotonic"])

    # Convert tx_mono to "wall-equivalent" time using the anchor offset, so
    # cmd and meas axes are in the same wall-clock domain.
    cmd_t = cmd["tx_mono"] - anchor_mono + anchor_wall
    cmd_v = cmd["vx"]
    meas_t = meas_ts
    meas_dx = np.gradient(meas_x, meas_t) if meas_t.size > 2 else np.zeros_like(meas_t)

    t0 = max(cmd_t.min(), meas_t.min())
    t1 = min(cmd_t.max(), meas_t.max())
    if t1 - t0 < 1.0:
        return CheckResult(name, False, {"reason": "overlap window <1s"})

    grid = np.arange(t0, t1, 1.0 / fs)
    cmd_g = np.interp(grid, cmd_t, cmd_v)
    meas_g = np.interp(grid, meas_t, meas_dx)

    cmd_g = cmd_g - cmd_g.mean()
    meas_g = meas_g - meas_g.mean()
    if np.std(cmd_g) < 1e-6 or np.std(meas_g) < 1e-6:
        return CheckResult(
            name,
            passed=True,  # constant signal, correlation undefined; don't fail
            detail={
                "reason": "low-variance signal",
                "cmd_std": float(np.std(cmd_g)),
                "meas_std": float(np.std(meas_g)),
            },
        )

    # Limit lag search to ±0.5 s — deadtime is well under that.
    max_lag_samples = int(0.5 * fs)
    n = len(cmd_g)
    xcorr = np.correlate(meas_g, cmd_g, mode="full")
    lags = np.arange(-(n - 1), n)
    mask = (lags >= -max_lag_samples) & (lags <= max_lag_samples)
    lags_lim = lags[mask]
    xcorr_lim = xcorr[mask]
    peak_lag_samples = int(lags_lim[np.argmax(xcorr_lim)])
    peak_lag_s = peak_lag_samples / fs
    peak_value = float(np.max(xcorr_lim) / (np.std(cmd_g) * np.std(meas_g) * n))

    # Decision: only fail if correlation is meaningful (> 0.3) AND lag is
    # meaningfully negative. Low-correlation runs are data-quality issues,
    # not causality issues — flag with a different reason.
    weak_corr = peak_value < 0.3
    if weak_corr:
        passed = True
        reason = "weak correlation — cmd↔meas relationship unclear (noisy run)"
    else:
        # Allow a tiny negative lag from interpolation jitter (one sample).
        passed = bool(peak_lag_s >= -1.0 / fs)
        reason = None

    detail: dict[str, Any] = {
        "peak_lag_s": round(peak_lag_s, 4),
        "peak_corr": round(peak_value, 3),
    }
    if reason is not None:
        detail["note"] = reason
    return CheckResult(name, passed=passed, detail=detail)


# ---------------------------------------------------------------------------- noise floor


def compute_noise_floor(run_dir: Path) -> dict[str, dict[str, float]]:
    """Compute sigma, mean, span on each derived measured channel during pre-roll.

    Channels:
      vx, vy: body-frame translational, derived as gradient(x or y) rotated by -yaw
      wz:     yaw rate, gradient(yaw) directly
      x, y:   raw world-frame position
      yaw:    raw yaw

    Returns the dict and writes it into ``run.json`` (additive — never
    overwrites existing fields). Pre-roll is identified from
    ``cmd_monotonic.jsonl``'s ``phase == "pre_roll"`` window.
    """
    from dimos.utils.characterization.scripts.analyze import (
        reconstruct_body_velocities,
    )

    run_dir = Path(run_dir).expanduser().resolve()
    run_json_path = run_dir / "run.json"
    meta = json.loads(run_json_path.read_text())

    pre_window = _pre_roll_wall_window(run_dir, meta)
    meas_ts, meas_x, meas_y, meas_yaw = _load_measured_for_run(run_dir, meta)

    out: dict[str, dict[str, float]] = {}
    if meas_ts.size < 3 or pre_window is None:
        out = {"_unavailable": {"reason": "insufficient measured samples"}}  # type: ignore[dict-item]
    else:
        # Slice to pre-roll window only.
        ts_lo, ts_hi = pre_window
        mask = (meas_ts >= ts_lo) & (meas_ts <= ts_hi)
        if int(mask.sum()) < 3:
            out = {
                "_unavailable": {"reason": "fewer than 3 pre-roll samples", "n": int(mask.sum())}
            }  # type: ignore[dict-item]
        else:
            t = meas_ts[mask]
            x = meas_x[mask]
            y = meas_y[mask]
            yaw = meas_yaw[mask]
            # Reconstruct body-frame velocities on the same window (small
            # window, no SavGol filtering — too few samples).
            vx, vy, wz = reconstruct_body_velocities(t, x, y, yaw, window=3, order=1)
            for name, arr in [
                ("vx", vx),
                ("vy", vy),
                ("wz", wz),
                ("x", x),
                ("y", y),
                ("yaw", yaw),
            ]:
                out[name] = {
                    "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
                    "mean": float(np.mean(arr)),
                    "span": float(np.ptp(arr)),
                    "n": int(arr.size),
                }

    # Append (don't overwrite existing fields).
    meta["noise_floor"] = out
    _atomic_write_json(run_json_path, meta)
    return out


def compute_noise_session(session_dir: Path) -> dict[str, Any]:
    """Run noise-floor computation across every run in a session."""
    session_dir = Path(session_dir).expanduser().resolve()
    rows = []
    for rd in sorted(p for p in session_dir.iterdir() if p.is_dir() and p.name[0].isdigit()):
        try:
            r = compute_noise_floor(rd)
            vx_std = (r.get("vx") or {}).get("std")
            wz_std = (r.get("wz") or {}).get("std")
            rows.append({"run_id": rd.name, "vx_std": vx_std, "wz_std": wz_std})
        except Exception as e:  # pragma: no cover
            rows.append({"run_id": rd.name, "error": str(e)})
    return {"session_dir": str(session_dir), "runs": rows}


def _pre_roll_wall_window(run_dir: Path, meta: dict[str, Any]) -> tuple[float, float] | None:
    """Wall-clock [start, end] of the pre-roll window.

    The cmd_monotonic.jsonl rows have both tx_mono and tx_wall, so we
    can directly read pre-roll wall-clock bounds without translating.
    """
    cmd_path = run_dir / "cmd_monotonic.jsonl"
    if not cmd_path.exists():
        return None
    pre_walls: list[float] = []
    with cmd_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("phase") == "pre_roll":
                pre_walls.append(float(d["tx_wall"]))
    if not pre_walls:
        return None
    return (min(pre_walls), max(pre_walls))


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str) + "\n")
    os.replace(tmp, path)


__all__ = [
    "CheckResult",
    "compute_noise_floor",
    "compute_noise_session",
    "validate_run",
    "validate_session",
]
