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

"""Session 1 - fit FOPDT to a session, aggregate, pool, report."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import re
from typing import Any

import numpy as np

from dimos.utils.characterization.modeling.fopdt import FopdtParams, fit_fopdt

# --- I/O helper ---


def atomic_write_json(path: Path, data: Any) -> None:
    """Atomic JSON write - `.tmp` then os.replace, mkdir parents as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str) + "\n")
    os.replace(tmp, path)


# --- Per-run fitting (was per_run.py) ---

_RECIPE_RE = re.compile(r"^e\d+_(vx|vy|wz)_([+-]?\d+(?:\.\d+)?)$")


@dataclass
class RunFit:
    run_id: str
    run_dir: str
    recipe: str
    channel: str | None
    amplitude: float | None
    direction: str | None
    mode: str
    split: str | None
    params: FopdtParams | None
    params_down: FopdtParams | None = None
    skip_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    extra_down: dict[str, Any] = field(default_factory=dict)

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


def parse_recipe_name(name: str) -> tuple[str, float] | None:
    """Return (channel, signed_amplitude) for parseable step recipe names."""
    m = _RECIPE_RE.match(name)
    if m is None:
        return None
    return (m.group(1), float(m.group(2)))


def _detect_channel_and_amplitude(run) -> tuple[str, float] | None:
    """Read channel and signed amplitude from cmd arrays. Used when the
    recipe name doesn't match the regex (e.g. E8 short_step recipes)."""
    from dimos.utils.characterization.plotting import _dominant_channel

    channel = _dominant_channel(run)
    cmd = {"vx": run.cmd_vx, "vy": run.cmd_vy, "wz": run.cmd_wz}[channel]
    if cmd.size == 0:
        return None
    nonzero = np.flatnonzero(np.abs(cmd) > 1e-6)
    if nonzero.size == 0:
        return None
    amp = float(cmd[nonzero[0]])
    if abs(amp) < 1e-6:
        return None
    return (channel, amp)


def _noise_std_for_channel(noise_floor: dict[str, Any] | None, channel: str) -> float | None:
    """Pull per-channel sigma from ``run.json["noise_floor"]``. Returns None when missing."""
    if not noise_floor or "_unavailable" in noise_floor:
        return None
    entry = noise_floor.get(channel)
    if not isinstance(entry, dict):
        return None
    std = entry.get("std")
    if std is None:
        return None
    try:
        v = float(std)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def fit_run(run_dir: Path, *, mode: str) -> RunFit:
    """Load one run, fit rise + fall FOPDTs, return a RunFit."""
    from dimos.utils.characterization.plotting import _channel_arrays
    from dimos.utils.characterization.scripts.analyze import load_run

    run_dir = Path(run_dir).expanduser().resolve()
    run_id = run_dir.name

    try:
        run = load_run(run_dir)
    except Exception as e:
        return RunFit(
            run_id=run_id,
            run_dir=str(run_dir),
            recipe="<unknown>",
            channel=None,
            amplitude=None,
            direction=None,
            mode=mode,
            split=None,
            params=None,
            skip_reason=f"load_run failed: {type(e).__name__}: {e}",
        )

    recipe = run.metadata["recipe"]["name"]
    test_type = run.metadata["recipe"]["test_type"]
    if test_type != "step":
        return RunFit(
            run_id=run_id,
            run_dir=str(run_dir),
            recipe=recipe,
            channel=None,
            amplitude=None,
            direction=None,
            mode=mode,
            split=None,
            params=None,
            skip_reason=f"not a step recipe (test_type={test_type})",
        )

    parsed = parse_recipe_name(recipe) or _detect_channel_and_amplitude(run)
    if parsed is None:
        return RunFit(
            run_id=run_id,
            run_dir=str(run_dir),
            recipe=recipe,
            channel=None,
            amplitude=None,
            direction=None,
            mode=mode,
            split=None,
            params=None,
            skip_reason=f"could not infer channel/amplitude for recipe {recipe!r}",
        )
    channel, amplitude = parsed
    direction = "forward" if amplitude > 0 else "reverse"
    split = "train" if direction == "forward" else "validate"

    cmd_arr, meas_arr = _channel_arrays(run, channel)
    cmd_ts_rel = run.cmd_ts_rel
    meas_ts_rel = run.meas_ts_rel

    if meas_ts_rel.size < 4:
        return RunFit(
            run_id=run_id,
            run_dir=str(run_dir),
            recipe=recipe,
            channel=channel,
            amplitude=amplitude,
            direction=direction,
            mode=mode,
            split=split,
            params=None,
            skip_reason="fewer than 4 measured samples",
        )

    nonzero = np.flatnonzero(np.abs(cmd_arr) > 1e-6)
    if nonzero.size == 0:
        return RunFit(
            run_id=run_id,
            run_dir=str(run_dir),
            recipe=recipe,
            channel=channel,
            amplitude=amplitude,
            direction=direction,
            mode=mode,
            split=split,
            params=None,
            skip_reason="no nonzero command on parsed channel",
        )
    step_t = float(cmd_ts_rel[nonzero[0]])
    duration = float(run.metadata["recipe"]["duration_s"])
    active_end_t = step_t + duration

    pre_mask = meas_ts_rel < step_t
    baseline = float(np.mean(meas_arr[pre_mask])) if pre_mask.any() else 0.0

    fit_mask = (meas_ts_rel >= step_t) & (meas_ts_rel <= active_end_t)
    if int(fit_mask.sum()) < 4:
        return RunFit(
            run_id=run_id,
            run_dir=str(run_dir),
            recipe=recipe,
            channel=channel,
            amplitude=amplitude,
            direction=direction,
            mode=mode,
            split=split,
            params=None,
            skip_reason=f"fewer than 4 samples in fit window ({int(fit_mask.sum())})",
        )

    t_fit = meas_ts_rel[fit_mask] - step_t
    y_fit = meas_arr[fit_mask] - baseline

    noise_std = _noise_std_for_channel(run.metadata.get("noise_floor"), channel)

    params = fit_fopdt(
        t_fit,
        y_fit,
        u_step=amplitude,
        noise_std=noise_std,
        fit_window_s=(0.0, float(t_fit[-1])),
    )

    # Step-down (fall) fit: cmd amplitude -> 0, baseline = last 30% of active.
    post_roll_s = float(run.metadata["recipe"].get("post_roll_s", 1.0))
    fall_end_t = active_end_t + post_roll_s

    ss_window_lo = step_t + 0.7 * duration
    ss_mask = (meas_ts_rel >= ss_window_lo) & (meas_ts_rel <= active_end_t)
    baseline_down = float(np.mean(meas_arr[ss_mask])) if ss_mask.any() else baseline
    fall_mask = (meas_ts_rel >= active_end_t) & (meas_ts_rel <= fall_end_t)

    params_down: FopdtParams | None
    extra_down: dict[str, Any]
    if int(fall_mask.sum()) < 4:
        params_down = None
        extra_down = {
            "active_end_t": active_end_t,
            "fall_end_t": fall_end_t,
            "baseline_down": baseline_down,
            "skip_reason": f"fewer than 4 samples in fall window ({int(fall_mask.sum())})",
        }
    else:
        t_fall = meas_ts_rel[fall_mask] - active_end_t
        y_fall = meas_arr[fall_mask] - baseline_down
        params_down = fit_fopdt(
            t_fall,
            y_fall,
            u_step=-amplitude,
            noise_std=noise_std,
            fit_window_s=(0.0, float(t_fall[-1])),
        )
        extra_down = {
            "active_end_t": active_end_t,
            "fall_end_t": fall_end_t,
            "baseline_down": baseline_down,
            "n_samples_fall": int(t_fall.size),
        }

    return RunFit(
        run_id=run_id,
        run_dir=str(run_dir),
        recipe=recipe,
        channel=channel,
        amplitude=amplitude,
        direction=direction,
        mode=mode,
        split=split,
        params=params,
        params_down=params_down,
        skip_reason=None,
        extra={
            "step_t": step_t,
            "active_end_t": active_end_t,
            "baseline": baseline,
            "noise_std": noise_std,
            "n_meas_total": int(meas_ts_rel.size),
        },
        extra_down=extra_down,
    )


def select_edge(run_fits: list[RunFit], edge: str) -> list[RunFit]:
    """Project RunFits onto one edge ('rise' or 'fall')."""
    if edge == "rise":
        return run_fits
    if edge != "fall":
        raise ValueError(f"edge must be 'rise' or 'fall', got {edge!r}")
    out: list[RunFit] = []
    for rf in run_fits:
        out.append(
            RunFit(
                run_id=rf.run_id,
                run_dir=rf.run_dir,
                recipe=rf.recipe,
                channel=rf.channel,
                amplitude=rf.amplitude,
                direction=rf.direction,
                mode=rf.mode,
                split=rf.split,
                params=rf.params_down,
                params_down=None,
                skip_reason=rf.skip_reason,
                extra=rf.extra_down or {},
                extra_down={},
            )
        )
    return out


# --- Per-recipe aggregation (was aggregate.py) ---


@dataclass
class _ParamStats:
    mean: float
    std: float
    ci_low: float
    ci_high: float
    n: int

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GroupFit:
    key: dict[str, Any]
    n_runs_input: int
    n_runs_kept: int
    n_runs_rejected_2sigma: int
    n_runs_failed: int
    n_runs_degenerate: int
    rejected_run_ids: list[dict[str, Any]]
    failed_run_ids: list[str]
    K: dict[str, Any] | None
    tau: dict[str, Any] | None
    L: dict[str, Any] | None
    rmse_median: float | None
    notes: list[str] = field(default_factory=list)

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


def aggregate_group(run_fits: list[RunFit], *, sigma_reject: float = 2.0) -> GroupFit:
    """Aggregate per-run FOPDT fits within one recipe group."""
    if not run_fits:
        raise ValueError("aggregate_group requires at least one RunFit")

    head = run_fits[0]
    key = {
        "channel": head.channel,
        "amplitude": head.amplitude,
        "direction": head.direction,
        "recipe": head.recipe,
        "mode": head.mode,
    }

    failed_ids: list[str] = []
    degenerate_count = 0
    usable: list[RunFit] = []
    for rf in run_fits:
        if rf.skip_reason is not None:
            failed_ids.append(rf.run_id)
            continue
        if rf.params is None or not rf.params.converged:
            failed_ids.append(rf.run_id)
            continue
        if rf.params.degenerate:
            degenerate_count += 1
            failed_ids.append(rf.run_id)
            continue
        usable.append(rf)

    notes: list[str] = []
    if degenerate_count:
        notes.append(f"{degenerate_count} run(s) had singular covariance (degenerate fit)")

    rejected: list[dict[str, Any]] = []
    if len(usable) >= 3:
        kept: list[RunFit] = []
        for param_name in ("K", "tau", "L"):
            arr = np.asarray([_p_get(rf, param_name) for rf in usable], dtype=float)
            med = float(np.median(arr))
            sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
            if sd <= 0:
                continue
            for rf, val in zip(usable, arr, strict=False):
                delta = abs(val - med) / sd
                if delta > sigma_reject and rf.run_id not in {r["run_id"] for r in rejected}:
                    rejected.append(
                        {
                            "run_id": rf.run_id,
                            "param": param_name,
                            "value": float(val),
                            "median": med,
                            "delta_sigma": float(delta),
                        }
                    )
        rejected_ids = {r["run_id"] for r in rejected}
        kept = [rf for rf in usable if rf.run_id not in rejected_ids]
    else:
        kept = list(usable)
        if len(usable) > 0 and len(usable) < 3:
            notes.append(f"only {len(usable)} usable fit(s); skipped 2-sigma rejection")

    K_stats = _inverse_variance_combine(kept, "K")
    tau_stats = _inverse_variance_combine(kept, "tau")
    L_stats = _inverse_variance_combine(kept, "L")

    rmses = [
        float(rf.params.rmse)
        for rf in kept
        if rf.params is not None and np.isfinite(rf.params.rmse)
    ]
    rmse_median = float(np.median(rmses)) if rmses else None

    return GroupFit(
        key=key,
        n_runs_input=len(run_fits),
        n_runs_kept=len(kept),
        n_runs_rejected_2sigma=len(rejected),
        n_runs_failed=len(failed_ids) - degenerate_count,
        n_runs_degenerate=degenerate_count,
        rejected_run_ids=rejected,
        failed_run_ids=failed_ids,
        K=K_stats.asdict() if K_stats else None,
        tau=tau_stats.asdict() if tau_stats else None,
        L=L_stats.asdict() if L_stats else None,
        rmse_median=rmse_median,
        notes=notes,
    )


def aggregate_session(run_fits: list[RunFit], *, sigma_reject: float = 2.0) -> list[GroupFit]:
    """Bucket per-run fits by recipe and aggregate each group."""
    by_recipe: dict[str, list[RunFit]] = {}
    for rf in run_fits:
        if rf.recipe == "<unknown>" or rf.channel is None:
            continue
        by_recipe.setdefault(rf.recipe, []).append(rf)
    return [aggregate_group(fits, sigma_reject=sigma_reject) for fits in by_recipe.values()]


def _p_get(rf: RunFit, name: str) -> float:
    assert rf.params is not None
    return float(getattr(rf.params, name))


def _ci_width(rf: RunFit, name: str) -> float:
    assert rf.params is not None
    lo, hi = getattr(rf.params, f"{name}_ci")
    return float(hi - lo)


def _inverse_variance_combine(kept: list[RunFit], param_name: str) -> _ParamStats | None:
    """Inverse-variance weighted mean + pooled CI for one parameter."""
    if not kept:
        return None
    vals = np.asarray([_p_get(rf, param_name) for rf in kept], dtype=float)
    widths = np.asarray([_ci_width(rf, param_name) for rf in kept], dtype=float)

    finite = np.isfinite(vals) & np.isfinite(widths) & (widths > 0)
    if not finite.any():
        if not np.isfinite(vals).any():
            return None
        v = vals[np.isfinite(vals)]
        mean = float(np.mean(v))
        std = float(np.std(v, ddof=1)) if v.size > 1 else 0.0
        ci_half = 1.96 * (std / np.sqrt(v.size)) if v.size > 1 else 0.0
        return _ParamStats(
            mean=mean, std=std, ci_low=mean - ci_half, ci_high=mean + ci_half, n=int(v.size)
        )

    v = vals[finite]
    sigmas = (widths[finite] / 2.0) / 1.96
    weights = 1.0 / (sigmas**2)
    mean = float(np.sum(weights * v) / np.sum(weights))
    pooled_var = 1.0 / float(np.sum(weights))
    pooled_sigma = float(np.sqrt(pooled_var))
    std = float(np.std(v, ddof=1)) if v.size > 1 else 0.0
    return _ParamStats(
        mean=mean,
        std=std,
        ci_low=mean - 1.96 * pooled_sigma,
        ci_high=mean + 1.96 * pooled_sigma,
        n=int(v.size),
    )


# --- Per-channel pooling and cross-mode comparison (was pool.py) ---


def _ci_overlap(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    """True when two CI intervals overlap (or either is missing - treated conservatively as True)."""
    if a is None or b is None:
        return True
    return not (a["ci_high"] < b["ci_low"] or b["ci_high"] < a["ci_low"])


def _combine_ivw(means: list[float], sigmas: list[float]) -> tuple[float, float, float, float]:
    """Inverse-variance combine. Returns (mean, std, ci_low, ci_high)."""
    means_a = np.asarray(means, dtype=float)
    sigmas_a = np.asarray(sigmas, dtype=float)
    finite = np.isfinite(means_a) & np.isfinite(sigmas_a) & (sigmas_a > 0)
    if not finite.any():
        if np.isfinite(means_a).any():
            v = means_a[np.isfinite(means_a)]
            m = float(np.mean(v))
            s = float(np.std(v, ddof=1)) if v.size > 1 else 0.0
            return (m, s, m - 1.96 * s, m + 1.96 * s)
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    m = means_a[finite]
    sig = sigmas_a[finite]
    w = 1.0 / (sig**2)
    mean = float(np.sum(w * m) / np.sum(w))
    pooled_sigma = float(math.sqrt(1.0 / float(np.sum(w))))
    std = float(np.std(m, ddof=1)) if m.size > 1 else 0.0
    return (mean, std, mean - 1.96 * pooled_sigma, mean + 1.96 * pooled_sigma)


def _stats_to_pair(stats: dict[str, Any] | None) -> tuple[float, float] | None:
    """Convert a stats dict to (mean, sigma) for IVW combination."""
    if stats is None:
        return None
    mean = stats.get("mean")
    lo = stats.get("ci_low")
    hi = stats.get("ci_high")
    if mean is None or lo is None or hi is None:
        return None
    width = hi - lo
    sigma = (width / 2.0) / 1.96 if width > 0 else 0.0
    return (float(mean), float(sigma))


def _linear_regression_with_ci(
    x: np.ndarray, y: np.ndarray, sigma_y: np.ndarray | None = None
) -> dict[str, Any]:
    """OLS y = a + b*x with 95% CI on slope. NaN for n < 3."""
    n = int(x.size)
    if n < 3:
        return {
            "slope": float("nan"),
            "intercept": float("nan"),
            "slope_ci": [float("nan"), float("nan")],
            "intercept_ci": [float("nan"), float("nan")],
            "n": n,
        }
    if sigma_y is not None and np.all(np.isfinite(sigma_y)) and np.all(sigma_y > 0):
        w = 1.0 / (sigma_y**2)
    else:
        w = np.ones_like(x)
    sw = float(np.sum(w))
    swx = float(np.sum(w * x))
    swy = float(np.sum(w * y))
    swxx = float(np.sum(w * x * x))
    swxy = float(np.sum(w * x * y))
    denom = sw * swxx - swx**2
    if denom <= 0:
        return {
            "slope": float("nan"),
            "intercept": float("nan"),
            "slope_ci": [float("nan"), float("nan")],
            "intercept_ci": [float("nan"), float("nan")],
            "n": n,
        }
    slope = (sw * swxy - swx * swy) / denom
    intercept = (swy - slope * swx) / sw
    resid = y - (intercept + slope * x)
    dof = max(1, n - 2)
    s2 = float(np.sum(w * resid**2)) / dof
    var_slope = s2 * sw / denom
    var_intercept = s2 * swxx / denom
    se_slope = math.sqrt(max(var_slope, 0.0))
    se_intercept = math.sqrt(max(var_intercept, 0.0))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "slope_ci": [float(slope - 1.96 * se_slope), float(slope + 1.96 * se_slope)],
        "intercept_ci": [
            float(intercept - 1.96 * se_intercept),
            float(intercept + 1.96 * se_intercept),
        ],
        "n": n,
    }


def pool_session(group_fits: list[GroupFit], *, mode: str) -> dict[str, Any]:
    """Pool a session's per-group fits across direction and amplitude."""
    by_channel: dict[str, list[GroupFit]] = {}
    for g in group_fits:
        ch = g.key.get("channel")
        if not isinstance(ch, str):
            continue
        by_channel.setdefault(ch, []).append(g)

    channels_out: dict[str, Any] = {}
    diag = {"n_groups_total": len(group_fits), "n_groups_with_fit": 0, "n_groups_without_fit": 0}
    for g in group_fits:
        if g.K is not None and g.tau is not None and g.L is not None:
            diag["n_groups_with_fit"] += 1
        else:
            diag["n_groups_without_fit"] += 1

    for channel, groups in sorted(by_channel.items()):
        channels_out[channel] = _pool_channel(groups)

    return {"mode": mode, "channels": channels_out, "diagnostics": diag}


def _pool_channel(groups: list[GroupFit]) -> dict[str, Any]:
    """Pool one channel's groups across direction + amplitude."""
    by_amp_dir: dict[tuple[float, str], GroupFit] = {}
    for g in groups:
        amp = g.key.get("amplitude")
        direction = g.key.get("direction")
        if amp is None or direction is None:
            continue
        by_amp_dir[(abs(float(amp)), direction)] = g

    amps = sorted({a for (a, _d) in by_amp_dir})
    direction_asymmetric = False
    for amp in amps:
        fwd = by_amp_dir.get((amp, "forward"))
        rev = by_amp_dir.get((amp, "reverse"))
        if fwd is None or rev is None:
            continue
        for p in ("K", "tau", "L"):
            if not _ci_overlap(getattr(fwd, p), getattr(rev, p)):
                direction_asymmetric = True
                break
        if direction_asymmetric:
            break

    per_amp_entries: list[dict[str, Any]] = []
    if direction_asymmetric:
        for (_abs_amp, direction), g in sorted(by_amp_dir.items()):
            per_amp_entries.append(
                {
                    "amplitude": float(g.key["amplitude"]),
                    "direction": direction,
                    "K": g.K,
                    "tau": g.tau,
                    "L": g.L,
                    "n_runs_kept": g.n_runs_kept,
                }
            )
        pool_input = [(abs(float(g.key["amplitude"])), g) for g in groups]
    else:
        for amp in amps:
            fwd = by_amp_dir.get((amp, "forward"))
            rev = by_amp_dir.get((amp, "reverse"))
            entry = {
                "amplitude": float(amp),
                "direction": "pooled",
                "K": _pool_two(fwd, rev, "K"),
                "tau": _pool_two(fwd, rev, "tau"),
                "L": _pool_two(fwd, rev, "L"),
                "n_runs_kept": (fwd.n_runs_kept if fwd else 0) + (rev.n_runs_kept if rev else 0),
            }
            per_amp_entries.append(entry)
        pool_input = [
            (amp, _entry_to_fakegroup(e)) for amp, e in zip(amps, per_amp_entries, strict=False)
        ]

    linear_in_amp: dict[str, bool] = {}
    gain_schedule: dict[str, Any] = {}
    pooled: dict[str, Any] = {}
    for p in ("K", "tau", "L"):
        amps_arr = np.asarray([a for a, _g in pool_input], dtype=float)
        means: list[float] = []
        sigmas: list[float] = []
        for _a, g in pool_input:
            pair = _stats_to_pair(getattr(g, p))
            if pair is None:
                means.append(float("nan"))
                sigmas.append(float("nan"))
            else:
                means.append(pair[0])
                sigmas.append(pair[1])
        means_a = np.asarray(means, dtype=float)
        sigmas_a = np.asarray(sigmas, dtype=float)
        finite = np.isfinite(means_a)
        if int(finite.sum()) >= 3:
            reg = _linear_regression_with_ci(
                amps_arr[finite],
                means_a[finite],
                sigma_y=(sigmas_a[finite] if np.all(np.isfinite(sigmas_a[finite])) else None),
            )
            slope_ci = reg["slope_ci"]
            ci_includes_zero = slope_ci[0] <= 0.0 <= slope_ci[1]
            linear_in_amp[p] = bool(ci_includes_zero)
            gain_schedule[p] = None if ci_includes_zero else reg
        else:
            linear_in_amp[p] = True
            gain_schedule[p] = None

        mean, std, lo, hi = _combine_ivw(means, sigmas)
        n_groups = int(np.isfinite(means_a).sum())
        pooled[p] = {"mean": mean, "std": std, "ci_low": lo, "ci_high": hi, "n_groups": n_groups}

    return {
        "direction_asymmetric": direction_asymmetric,
        "linear_in_amplitude": linear_in_amp,
        "pooled": pooled,
        "gain_schedule": gain_schedule,
        "per_amplitude": per_amp_entries,
    }


def _pool_two(a: GroupFit | None, b: GroupFit | None, p: str) -> dict[str, Any] | None:
    """Inverse-variance combine two GroupFit stats dicts."""
    pairs: list[tuple[float, float]] = []
    for g in (a, b):
        if g is None:
            continue
        pair = _stats_to_pair(getattr(g, p))
        if pair is not None:
            pairs.append(pair)
    if not pairs:
        return None
    means = [pp[0] for pp in pairs]
    sigmas = [pp[1] for pp in pairs]
    mean, std, lo, hi = _combine_ivw(means, sigmas)
    return {"mean": mean, "std": std, "ci_low": lo, "ci_high": hi, "n": len(pairs)}


class _FakeGroup:
    """Stand-in so amplitude pooling can treat direction-pooled and per-direction entries uniformly."""

    def __init__(self, K, tau, L) -> None:
        self.K = K
        self.tau = tau
        self.L = L


def _entry_to_fakegroup(entry: dict[str, Any]) -> _FakeGroup:
    return _FakeGroup(K=entry["K"], tau=entry["tau"], L=entry["L"])


def compare_models(default_summary: dict[str, Any], rage_summary: dict[str, Any]) -> dict[str, Any]:
    """Compare default-mode and rage-mode model_summary.json dicts."""
    out: dict[str, Any] = {
        "default_mode": default_summary.get("mode"),
        "rage_mode": rage_summary.get("mode"),
        "channels": {},
    }
    default_channels = default_summary.get("channels", {})
    rage_channels = rage_summary.get("channels", {})
    for channel in sorted(set(default_channels) | set(rage_channels)):
        d = default_channels.get(channel) or {}
        r = rage_channels.get(channel) or {}
        d_pool = d.get("pooled") or {}
        r_pool = r.get("pooled") or {}

        per_param: dict[str, Any] = {}
        worst = "identical"
        for p in ("K", "tau", "L"):
            ds = d_pool.get(p)
            rs = r_pool.get(p)
            verdict = _compare_one_param(ds, rs)
            per_param[p] = verdict
            worst = _worst_verdict(worst, verdict["verdict"])

        out["channels"][channel] = {
            "verdict": worst,
            "params": per_param,
            "default_pooled": d_pool,
            "rage_pooled": r_pool,
        }
    return out


def _compare_one_param(ds: dict[str, Any] | None, rs: dict[str, Any] | None) -> dict[str, Any]:
    if ds is None or rs is None:
        return {"verdict": "missing", "default": ds, "rage": rs, "ratio": None, "ci_overlap": None}
    d_mean = ds.get("mean")
    r_mean = rs.get("mean")
    if d_mean is None or r_mean is None or not (math.isfinite(d_mean) and math.isfinite(r_mean)):
        return {"verdict": "missing", "default": ds, "rage": rs, "ratio": None, "ci_overlap": None}
    if abs(d_mean) < 1e-9:
        ratio = float("inf") if abs(r_mean) > 1e-9 else 1.0
    else:
        ratio = r_mean / d_mean
    overlap = _ci_overlap(ds, rs)
    deviation = abs(ratio - 1.0) if math.isfinite(ratio) else float("inf")
    if deviation < 0.05:
        verdict = "identical"
    elif overlap or deviation <= 0.20:
        verdict = "equivalent"
    else:
        verdict = "differs"
    return {"verdict": verdict, "default": ds, "rage": rs, "ratio": ratio, "ci_overlap": overlap}


def _worst_verdict(a: str, b: str) -> str:
    order = {"identical": 0, "equivalent": 1, "differs": 2, "missing": 3}
    return a if order.get(a, 0) >= order.get(b, 0) else b


def compare_rise_fall(rise_summary: dict[str, Any], fall_summary: dict[str, Any]) -> dict[str, Any]:
    """Compare rise vs fall pooled FOPDT params per channel - surfaces accel-vs-decel asymmetry."""
    out: dict[str, Any] = {"channels": {}}
    rise_channels = rise_summary.get("channels", {})
    fall_channels = fall_summary.get("channels", {})
    for channel in sorted(set(rise_channels) | set(fall_channels)):
        rch = rise_channels.get(channel) or {}
        fch = fall_channels.get(channel) or {}
        rp = rch.get("pooled") or {}
        fp = fch.get("pooled") or {}

        per_param: dict[str, Any] = {}
        worst = "identical"
        for p in ("K", "tau", "L"):
            verdict = _compare_one_param(rp.get(p), fp.get(p))
            per_param[p] = {
                "verdict": verdict["verdict"],
                "rise": verdict["default"],
                "fall": verdict["rage"],
                "ratio_fall_over_rise": verdict["ratio"],
                "ci_overlap": verdict["ci_overlap"],
            }
            worst = _worst_verdict(worst, verdict["verdict"])

        out["channels"][channel] = {"verdict": worst, "params": per_param}
    return out


# --- Session orchestration (was session.py) ---


def _read_mode(session_dir: Path) -> str:
    """Read mode from session.json; defaults to 'default' when absent."""
    sj = session_dir / "session.json"
    if not sj.exists():
        return "default"
    try:
        meta = json.loads(sj.read_text())
    except Exception:
        return "default"
    return "rage" if bool(meta.get("rage")) else "default"


def _discover_runs(session_dir: Path) -> list[Path]:
    return sorted(p for p in session_dir.iterdir() if p.is_dir() and p.name and p.name[0].isdigit())


def fit_session_runs(session_dir: Path, mode: str | None = None) -> list[RunFit]:
    """Per-run fitting only; doesn't aggregate or write artifacts."""
    session_dir = Path(session_dir).expanduser().resolve()
    if not session_dir.is_dir():
        raise FileNotFoundError(f"not a directory: {session_dir}")
    if mode is None:
        mode = _read_mode(session_dir)
    out: list[RunFit] = []
    for rd in _discover_runs(session_dir):
        try:
            out.append(fit_run(rd, mode=mode))
        except Exception as e:
            out.append(
                RunFit(
                    run_id=rd.name,
                    run_dir=str(rd),
                    recipe="<unknown>",
                    channel=None,
                    amplitude=None,
                    direction=None,
                    mode=mode,
                    split=None,
                    params=None,
                    skip_reason=f"fit_run raised: {type(e).__name__}: {e}",
                )
            )
    return out


def fit_session(session_dir: Path, *, write_plots_enabled: bool = True) -> dict[str, Any]:
    """Run the full per-session FOPDT pipeline. Returns the model_summary dict."""
    from dimos.utils.characterization.modeling.fit_report import render_markdown, write_plots

    session_dir = Path(session_dir).expanduser().resolve()
    mode = _read_mode(session_dir)
    out_dir = session_dir / "modeling"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"

    run_fits = fit_session_runs(session_dir, mode=mode)
    summary = _build_summary(run_fits, mode=mode, provenance={"session_dir": str(session_dir)})

    atomic_write_json(
        out_dir / "fits_per_run.json",
        {
            "session_dir": str(session_dir),
            "mode": mode,
            "n_runs": len(run_fits),
            "runs": [rf.asdict() for rf in run_fits],
        },
    )
    atomic_write_json(
        out_dir / "fits_per_group.json",
        {
            "session_dir": str(session_dir),
            "mode": mode,
            "n_groups_rise": len(summary["_rise_groups_in_memory"]),
            "n_groups_fall": len(summary["_fall_groups_in_memory"]),
            "rise_groups": [g.asdict() for g in summary["_rise_groups_in_memory"]],
            "fall_groups": [g.asdict() for g in summary["_fall_groups_in_memory"]],
        },
    )
    rise_groups = summary.pop("_rise_groups_in_memory")
    fall_groups = summary.pop("_fall_groups_in_memory")
    atomic_write_json(out_dir / "model_summary.json", summary)

    md = render_markdown(
        session_dir=session_dir,
        mode=mode,
        summary=summary,
        group_fits=rise_groups,
        run_fits=run_fits,
        fall_groups=fall_groups,
    )
    (out_dir / "model_report.md").write_text(md)

    if write_plots_enabled:
        write_plots(
            plots_dir=plots_dir,
            summary=summary,
            group_fits=rise_groups,
            run_fits=run_fits,
            fall_groups=fall_groups,
        )

    return summary


def _build_summary(
    run_fits: list[RunFit], *, mode: str, provenance: dict[str, Any]
) -> dict[str, Any]:
    """Aggregate + pool rise and fall edges, return combined summary."""
    rise_fits = select_edge(run_fits, "rise")
    fall_fits = select_edge(run_fits, "fall")
    rise_groups = aggregate_session(rise_fits)
    fall_groups = aggregate_session(fall_fits)
    rise_summary = pool_session(rise_groups, mode=mode)
    fall_summary = pool_session(fall_groups, mode=mode)
    rise_vs_fall = compare_rise_fall(rise_summary, fall_summary)

    return {
        **provenance,
        "mode": mode,
        "channels": rise_summary["channels"],
        "diagnostics": rise_summary["diagnostics"],
        "rise": rise_summary,
        "fall": fall_summary,
        "rise_vs_fall": rise_vs_fall,
        "_rise_groups_in_memory": rise_groups,
        "_fall_groups_in_memory": fall_groups,
    }


def fit_all_sessions(
    parent_dir: Path, *, force: bool = False, write_plots_enabled: bool = True
) -> dict[str, Any]:
    """Discover session_* dirs under parent_dir and fit each."""
    parent_dir = Path(parent_dir).expanduser().resolve()
    if not parent_dir.is_dir():
        raise FileNotFoundError(f"not a directory: {parent_dir}")

    sessions = sorted(
        p for p in parent_dir.iterdir() if p.is_dir() and p.name.startswith("session_")
    )
    index_rows: list[dict[str, Any]] = []
    for s in sessions:
        ms = s / "modeling" / "model_summary.json"
        status = "ok"
        summary: dict[str, Any] | None
        if ms.exists() and not force:
            try:
                summary = json.loads(ms.read_text())
                status = "cached"
            except Exception:
                summary = None
                status = "cached_read_failed"
        else:
            try:
                summary = fit_session(s, write_plots_enabled=write_plots_enabled)
            except Exception as e:
                summary = None
                status = f"failed: {type(e).__name__}: {e}"

        n_runs = sum(1 for _ in _discover_runs(s)) if s.is_dir() else 0
        diag = (summary or {}).get("diagnostics") or {}
        n_groups_with_fit = int(diag.get("n_groups_with_fit") or 0)
        index_rows.append(
            {
                "session": s.name,
                "session_dir": str(s),
                "mode": (summary or {}).get("mode") or _read_mode(s),
                "status": status,
                "n_runs": n_runs,
                "n_groups_with_fit": n_groups_with_fit,
                "model_summary": str(ms) if ms.exists() else None,
            }
        )

    index = {"parent_dir": str(parent_dir), "n_sessions": len(sessions), "sessions": index_rows}
    atomic_write_json(parent_dir / "models_index.json", index)
    return index


def pool_runs_across_sessions(
    session_dirs: list[Path], *, mode_label: str
) -> tuple[dict[str, Any], list[RunFit]]:
    """Concatenate RunFits from multiple sessions, then aggregate + pool."""
    all_runs: list[RunFit] = []
    for s in session_dirs:
        all_runs.extend(fit_session_runs(Path(s), mode=mode_label))
    summary = _build_summary(
        all_runs,
        mode=mode_label,
        provenance={
            "session_dirs": [str(Path(s).expanduser().resolve()) for s in session_dirs],
            "n_sessions": len(session_dirs),
        },
    )
    summary.pop("_rise_groups_in_memory", None)
    summary.pop("_fall_groups_in_memory", None)
    return summary, all_runs


def compare_pooled(
    default_sessions: list[Path],
    rage_sessions: list[Path],
    *,
    out_path: Path,
) -> dict[str, Any]:
    """Pool RunFits across all default-mode + rage-mode sessions, then compare."""
    from dimos.utils.characterization.modeling.fit_report import render_compare_markdown

    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    default_pooled, _d_runs = pool_runs_across_sessions(default_sessions, mode_label="default")
    rage_pooled, _r_runs = pool_runs_across_sessions(rage_sessions, mode_label="rage")

    verdict = compare_models(default_pooled, rage_pooled)

    md_lines = [render_compare_markdown(verdict).rstrip("\n"), ""]
    md_lines.append("## Pooled provenance")
    md_lines.append("")
    md_lines.append(
        f"- default mode: {len(default_sessions)} session(s), "
        f"{default_pooled.get('diagnostics', {}).get('n_groups_total', 0)} groups"
    )
    for s in default_sessions:
        md_lines.append(f"  - `{s}`")
    md_lines.append(
        f"- rage mode: {len(rage_sessions)} session(s), "
        f"{rage_pooled.get('diagnostics', {}).get('n_groups_total', 0)} groups"
    )
    for s in rage_sessions:
        md_lines.append(f"  - `{s}`")
    md_lines.append("")
    md_lines.append("## Pooled K/tau/L per channel")
    md_lines.append("")
    md_lines.append("| mode | channel | K (95% CI) | tau (95% CI) | L (95% CI) |")
    md_lines.append("|---|---|---|---|---|")
    for label, summary in (("default", default_pooled), ("rage", rage_pooled)):
        for channel, ch in sorted((summary.get("channels") or {}).items()):
            pooled = ch.get("pooled") or {}
            row = [label, channel]
            for p in ("K", "tau", "L"):
                stats = pooled.get(p) or {}
                row.append(_fmt_stats(stats))
            md_lines.append("| " + " | ".join(row) + " |")
    md_lines.append("")

    out_path.write_text("\n".join(md_lines) + "\n")
    pooled_path = out_path.with_name(out_path.stem + "_pooled.json")
    atomic_write_json(
        pooled_path,
        {
            "default_pooled": default_pooled,
            "rage_pooled": rage_pooled,
            "verdict": verdict,
        },
    )
    return verdict


def _fmt_stats(stats: dict[str, Any]) -> str:
    mean = stats.get("mean")
    lo = stats.get("ci_low")
    hi = stats.get("ci_high")

    def f(v):
        try:
            return f"{float(v):.4g}"
        except Exception:
            return "-"

    return f"{f(mean)} [{f(lo)}, {f(hi)}]"


def compare_two_sessions(
    default_session: Path, rage_session: Path, *, out_path: Path
) -> dict[str, Any]:
    """Compare two pre-fit sessions via their model_summary.json. Legacy two-session entry point."""
    from dimos.utils.characterization.modeling.fit_report import render_compare_markdown

    default_session = Path(default_session).expanduser().resolve()
    rage_session = Path(rage_session).expanduser().resolve()

    def _load(s: Path) -> dict[str, Any]:
        ms = s / "modeling" / "model_summary.json"
        if not ms.exists():
            raise FileNotFoundError(f"{ms} not found - run `process_session fit {s}` first.")
        return json.loads(ms.read_text())

    d_summary = _load(default_session)
    r_summary = _load(rage_session)
    verdict = compare_models(d_summary, r_summary)
    verdict["default_session"] = str(default_session)
    verdict["rage_session"] = str(rage_session)

    out_path = Path(out_path).expanduser().resolve()
    md = render_compare_markdown(verdict)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    pooled_path = out_path.with_name(out_path.stem + "_pooled.json")
    atomic_write_json(pooled_path, verdict)
    return verdict


__all__ = [
    "GroupFit",
    "RunFit",
    "aggregate_group",
    "aggregate_session",
    "atomic_write_json",
    "compare_models",
    "compare_pooled",
    "compare_rise_fall",
    "compare_two_sessions",
    "fit_all_sessions",
    "fit_run",
    "fit_session",
    "fit_session_runs",
    "parse_recipe_name",
    "pool_runs_across_sessions",
    "pool_session",
    "select_edge",
]
