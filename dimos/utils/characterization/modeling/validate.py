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

"""Session 2 - validate the fitted FOPDT against held-out runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dimos.utils.characterization.modeling.fit import (
    RunFit,
    _read_mode,
    aggregate_session,
    atomic_write_json,
    fit_session_runs,
    pool_session,
    select_edge,
)
from dimos.utils.characterization.modeling.fopdt import FopdtParams
from dimos.utils.characterization.modeling.simulate import simulate_fopdt

# --- Per-run validation thresholds ---
#
# Pass/marginal/fail thresholds on smoothed normalized RMSE = SmoothedRMSE / |amp|.
# Smoothing is Sav-Gol over the residual to suppress leg-jitter the FOPDT
# (deterministic) model cannot predict. Raw nRMSE is still computed and
# surfaced for traceability - it just isn't the verdict driver.
_RISE_PASS = 0.10
_RISE_MARGINAL = 0.20
_FALL_PASS = 0.15
_FALL_MARGINAL = 0.25

# Sav-Gol residual smoothing (window in samples, polynomial order). At
# ~50 Hz this gives ~220 ms window - suppresses gait-cadence noise
# (~2-5 Hz on the Go2 trot) without distorting the FOPDT-scale dynamics
# (~tau ~ 0.25 s).
_RESID_SMOOTH_WINDOW = 11
_RESID_SMOOTH_POLY = 3

# Channel pass-rate thresholds (validation-set, not per-run).
_PASS_RATE_PASS = 0.80
_PASS_RATE_MARGINAL = 0.60


# --- Per-run validation result + entry point (was validate_run.py) ---


@dataclass
class ValidationResult:
    run_id: str
    run_dir: str
    recipe: str
    channel: str | None
    amplitude: float | None
    direction: str | None
    mode: str
    split: str | None

    used_gain: float
    used_tau: float
    used_deadtime: float
    used_gain_fall: float | None = None
    used_tau_fall: float | None = None
    used_deadtime_fall: float | None = None

    rise_metrics: dict[str, Any] | None = None
    fall_metrics: dict[str, Any] | None = None
    overall_metrics: dict[str, Any] | None = None

    verdict: str = "skip"
    skip_reason: str | None = None

    extra: dict[str, Any] = field(default_factory=dict)

    # In-memory only - used by report's plot stage. Excluded from JSON.
    _t_meas: np.ndarray | None = field(default=None, repr=False, compare=False)
    _y_meas: np.ndarray | None = field(default=None, repr=False, compare=False)
    _y_pred: np.ndarray | None = field(default=None, repr=False, compare=False)
    _cmd_at_meas: np.ndarray | None = field(default=None, repr=False, compare=False)

    def asdict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("_t_meas", "_y_meas", "_y_pred", "_cmd_at_meas"):
            d.pop(k, None)
        return d


def _eval_param(channel_summary: dict[str, Any], param: str, abs_amp: float) -> float:
    """Evaluate K/tau/L at this run's |amplitude|, honoring the gain schedule."""
    if not channel_summary:
        return float("nan")
    linear_map = channel_summary.get("linear_in_amplitude") or {}
    is_linear = bool(linear_map.get(param, True))
    if not is_linear:
        sched = (channel_summary.get("gain_schedule") or {}).get(param)
        if sched and sched.get("slope") is not None and sched.get("intercept") is not None:
            slope = float(sched["slope"])
            intercept = float(sched["intercept"])
            if np.isfinite(slope) and np.isfinite(intercept):
                return float(intercept + slope * abs_amp)
    pooled = (channel_summary.get("pooled") or {}).get(param) or {}
    mean = pooled.get("mean")
    return float(mean) if mean is not None else float("nan")


def _fopdt_from_summary(
    summary: dict[str, Any] | None, channel: str, abs_amp: float
) -> FopdtParams | None:
    """Build a FopdtParams from a pooled summary at the given amplitude."""
    if summary is None:
        return None
    ch = (summary.get("channels") or {}).get(channel)
    if not ch:
        return None
    K = _eval_param(ch, "K", abs_amp)
    tau = _eval_param(ch, "tau", abs_amp)
    L = _eval_param(ch, "L", abs_amp)
    if not (np.isfinite(K) and np.isfinite(tau) and np.isfinite(L)):
        return None
    return FopdtParams(
        K=K,
        tau=tau,
        L=L,
        K_ci=(K, K),
        tau_ci=(tau, tau),
        L_ci=(L, L),
        rmse=0.0,
        r_squared=1.0,
        n_samples=0,
        fit_window_s=(0.0, 0.0),
        degenerate=False,
        converged=True,
    )


def _zoh_resample(
    t_query: np.ndarray, t_grid: np.ndarray, vals: np.ndarray, *, left: float
) -> np.ndarray:
    """ZOH lookup at t_query against (t_grid, vals)."""
    idx = np.searchsorted(t_grid, t_query, side="right") - 1
    out = np.empty_like(t_query, dtype=float)
    pre = idx < 0
    valid = ~pre
    out[pre] = left
    if valid.any():
        out[valid] = vals[idx[valid]]
    return out


def _smooth_residual(resid: np.ndarray) -> np.ndarray:
    """Sav-Gol smooth a residual; falls back to a tiny window when n is small."""
    n = int(resid.size)
    if n < 4:
        return resid.astype(float, copy=True)
    if n >= _RESID_SMOOTH_WINDOW and _RESID_SMOOTH_POLY < _RESID_SMOOTH_WINDOW:
        from scipy.signal import savgol_filter

        return savgol_filter(resid, _RESID_SMOOTH_WINDOW, _RESID_SMOOTH_POLY, mode="nearest")
    w = max(3, n if n % 2 == 1 else n - 1)
    p = min(2, w - 1)
    from scipy.signal import savgol_filter

    return savgol_filter(resid, w, p, mode="nearest")


def _compute_metrics(
    y_meas: np.ndarray,
    y_pred: np.ndarray,
    *,
    amp: float,
    noise_std: float | None = None,
) -> dict[str, Any] | None:
    """Residual metrics, raw and smoothed."""
    if y_meas.size == 0 or y_pred.size == 0 or y_meas.size != y_pred.size:
        return None
    resid = y_meas - y_pred
    rmse = float(np.sqrt(np.mean(resid**2)))
    max_abs = float(np.max(np.abs(resid)))
    norm_rmse = rmse / abs(amp) if abs(amp) > 1e-9 else float("nan")

    resid_s = _smooth_residual(resid)
    rmse_s = float(np.sqrt(np.mean(resid_s**2)))
    norm_rmse_s = rmse_s / abs(amp) if abs(amp) > 1e-9 else float("nan")

    residual_over_noise: float | None
    if noise_std is not None and noise_std > 0:
        residual_over_noise = float(rmse_s / noise_std)
    else:
        residual_over_noise = None

    y_mean = float(np.mean(y_meas))
    ss_tot = float(np.sum((y_meas - y_mean) ** 2))
    ss_res = float(np.sum(resid**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return {
        "rmse": rmse,
        "max_abs": max_abs,
        "norm_rmse": norm_rmse,
        "rmse_smoothed": rmse_s,
        "norm_rmse_smoothed": norm_rmse_s,
        "residual_over_noise": residual_over_noise,
        "noise_std": (float(noise_std) if noise_std is not None else None),
        "n_samples": int(y_meas.size),
        "r2_oos": r2,
    }


def _verdict_per_run(rise: dict[str, Any] | None, fall: dict[str, Any] | None) -> str:
    """Pass/marginal/fail per run, driven by smoothed nRMSE."""
    if rise is None:
        return "skip"
    rise_n = rise.get("norm_rmse_smoothed", float("nan"))
    if not np.isfinite(rise_n):
        return "skip"
    fall_n = (fall or {}).get("norm_rmse_smoothed")
    fall_finite = fall_n is not None and np.isfinite(fall_n)

    rise_pass = rise_n <= _RISE_PASS
    rise_marginal = rise_n <= _RISE_MARGINAL
    fall_pass = (not fall_finite) or fall_n <= _FALL_PASS
    fall_marginal = (not fall_finite) or fall_n <= _FALL_MARGINAL

    if rise_pass and fall_pass:
        return "pass"
    if rise_marginal and fall_marginal:
        return "marginal"
    return "fail"


def validate_run(
    run_dir: Path,
    *,
    rise_summary: dict[str, Any],
    fall_summary: dict[str, Any] | None = None,
    mode: str = "default",
    keep_traces: bool = True,
) -> ValidationResult:
    """Validate one run against a fitted model. Returns a ValidationResult."""
    from dimos.utils.characterization.modeling.fit import (
        _detect_channel_and_amplitude,
        _noise_std_for_channel,
        parse_recipe_name,
    )
    from dimos.utils.characterization.plotting import _channel_arrays
    from dimos.utils.characterization.scripts.analyze import load_run

    run_dir = Path(run_dir).expanduser().resolve()
    run_id = run_dir.name

    try:
        run = load_run(run_dir)
    except Exception as e:
        return ValidationResult(
            run_id=run_id,
            run_dir=str(run_dir),
            recipe="<unknown>",
            channel=None,
            amplitude=None,
            direction=None,
            mode=mode,
            split=None,
            used_gain=float("nan"),
            used_tau=float("nan"),
            used_deadtime=float("nan"),
            skip_reason=f"load_run failed: {type(e).__name__}: {e}",
        )

    recipe = run.metadata["recipe"]["name"]
    test_type = run.metadata["recipe"]["test_type"]
    if test_type != "step":
        return ValidationResult(
            run_id=run_id,
            run_dir=str(run_dir),
            recipe=recipe,
            channel=None,
            amplitude=None,
            direction=None,
            mode=mode,
            split=None,
            used_gain=float("nan"),
            used_tau=float("nan"),
            used_deadtime=float("nan"),
            skip_reason=f"not a step recipe (test_type={test_type})",
        )

    parsed = parse_recipe_name(recipe) or _detect_channel_and_amplitude(run)
    if parsed is None:
        return ValidationResult(
            run_id=run_id,
            run_dir=str(run_dir),
            recipe=recipe,
            channel=None,
            amplitude=None,
            direction=None,
            mode=mode,
            split=None,
            used_gain=float("nan"),
            used_tau=float("nan"),
            used_deadtime=float("nan"),
            skip_reason=f"could not infer channel/amplitude for recipe {recipe!r}",
        )
    channel, amplitude = parsed
    direction = "forward" if amplitude > 0 else "reverse"
    split = "train" if direction == "forward" else "validate"
    abs_amp = abs(amplitude)

    rise_params = _fopdt_from_summary(rise_summary, channel, abs_amp)
    if rise_params is None:
        return ValidationResult(
            run_id=run_id,
            run_dir=str(run_dir),
            recipe=recipe,
            channel=channel,
            amplitude=amplitude,
            direction=direction,
            mode=mode,
            split=split,
            used_gain=float("nan"),
            used_tau=float("nan"),
            used_deadtime=float("nan"),
            skip_reason=f"no rise model for channel {channel}",
        )
    fall_params = _fopdt_from_summary(fall_summary, channel, abs_amp) if fall_summary else None

    cmd_arr, meas_arr = _channel_arrays(run, channel)
    cmd_ts = run.cmd_ts_rel
    meas_ts = run.meas_ts_rel

    if meas_ts.size < 4:
        return ValidationResult(
            run_id=run_id,
            run_dir=str(run_dir),
            recipe=recipe,
            channel=channel,
            amplitude=amplitude,
            direction=direction,
            mode=mode,
            split=split,
            used_gain=rise_params.K,
            used_tau=rise_params.tau,
            used_deadtime=rise_params.L,
            skip_reason="fewer than 4 measured samples",
        )

    nonzero = np.flatnonzero(np.abs(cmd_arr) > 1e-6)
    if nonzero.size == 0:
        return ValidationResult(
            run_id=run_id,
            run_dir=str(run_dir),
            recipe=recipe,
            channel=channel,
            amplitude=amplitude,
            direction=direction,
            mode=mode,
            split=split,
            used_gain=rise_params.K,
            used_tau=rise_params.tau,
            used_deadtime=rise_params.L,
            skip_reason="no nonzero command on parsed channel",
        )
    step_t = float(cmd_ts[nonzero[0]])
    duration = float(run.metadata["recipe"]["duration_s"])
    active_end_t = step_t + duration
    post_roll_s = float(run.metadata["recipe"].get("post_roll_s", 1.0))
    fall_end_t = active_end_t + post_roll_s

    pre_mask = meas_ts < step_t
    baseline = float(np.mean(meas_arr[pre_mask])) if pre_mask.any() else 0.0
    y_meas = meas_arr - baseline

    cmd_at_meas = _zoh_resample(meas_ts, cmd_ts, cmd_arr, left=0.0)

    y_pred = simulate_fopdt(
        meas_ts,
        cmd_at_meas,
        rise_params,
        fall_params=fall_params,
        initial=0.0,
        pre_cmd=0.0,
    )

    rise_mask = (meas_ts >= step_t) & (meas_ts <= active_end_t)
    fall_mask = (meas_ts > active_end_t) & (meas_ts <= fall_end_t)
    overall_mask = (meas_ts >= step_t) & (meas_ts <= fall_end_t)

    noise_std = _noise_std_for_channel(run.metadata.get("noise_floor"), channel)

    rise_metrics = _compute_metrics(
        y_meas[rise_mask],
        y_pred[rise_mask],
        amp=amplitude,
        noise_std=noise_std,
    )
    fall_metrics = (
        _compute_metrics(
            y_meas[fall_mask],
            y_pred[fall_mask],
            amp=amplitude,
            noise_std=noise_std,
        )
        if int(fall_mask.sum()) >= 4
        else None
    )
    overall_metrics = _compute_metrics(
        y_meas[overall_mask],
        y_pred[overall_mask],
        amp=amplitude,
        noise_std=noise_std,
    )

    verdict = _verdict_per_run(rise_metrics, fall_metrics)

    return ValidationResult(
        run_id=run_id,
        run_dir=str(run_dir),
        recipe=recipe,
        channel=channel,
        amplitude=amplitude,
        direction=direction,
        mode=mode,
        split=split,
        used_gain=rise_params.K,
        used_tau=rise_params.tau,
        used_deadtime=rise_params.L,
        used_gain_fall=(fall_params.K if fall_params else None),
        used_tau_fall=(fall_params.tau if fall_params else None),
        used_deadtime_fall=(fall_params.L if fall_params else None),
        rise_metrics=rise_metrics,
        fall_metrics=fall_metrics,
        overall_metrics=overall_metrics,
        verdict=verdict,
        skip_reason=None,
        extra={
            "step_t": step_t,
            "active_end_t": active_end_t,
            "fall_end_t": fall_end_t,
            "baseline": baseline,
            "noise_std": (float(noise_std) if noise_std is not None else None),
            "n_meas_total": int(meas_ts.size),
            "thresholds": {
                "rise_pass": _RISE_PASS,
                "rise_marginal": _RISE_MARGINAL,
                "fall_pass": _FALL_PASS,
                "fall_marginal": _FALL_MARGINAL,
                "metric": "norm_rmse_smoothed",
                "smoothing_window": _RESID_SMOOTH_WINDOW,
                "smoothing_polyorder": _RESID_SMOOTH_POLY,
            },
        },
        _t_meas=(meas_ts.copy() if keep_traces else None),
        _y_meas=(y_meas.copy() if keep_traces else None),
        _y_pred=(y_pred.copy() if keep_traces else None),
        _cmd_at_meas=(cmd_at_meas.copy() if keep_traces else None),
    )


# --- Aggregation across runs (was validate_aggregate.py) ---


@dataclass
class GroupSummary:
    key: dict[str, Any]
    n_total: int
    n_pass: int
    n_marginal: int
    n_fail: int
    n_skip: int
    rise_norm_rmse: dict[str, Any]
    fall_norm_rmse: dict[str, Any] | None
    rise_norm_rmse_raw: dict[str, Any] | None = None
    fall_norm_rmse_raw: dict[str, Any] | None = None
    rise_residual_over_noise: dict[str, Any] | None = None
    fall_residual_over_noise: dict[str, Any] | None = None
    worst_run_ids: list[str] = field(default_factory=list)

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChannelSummary:
    channel: str
    mode: str
    n_total: int
    n_pass: int
    n_marginal: int
    n_fail: int
    n_skip: int
    pass_rate: float
    verdict: str
    rise_norm_rmse: dict[str, Any]
    fall_norm_rmse: dict[str, Any] | None
    rise_norm_rmse_raw: dict[str, Any] | None = None
    fall_norm_rmse_raw: dict[str, Any] | None = None
    rise_residual_over_noise: dict[str, Any] | None = None
    fall_residual_over_noise: dict[str, Any] | None = None
    by_amp_direction: list[GroupSummary] = field(default_factory=list)
    worst_run_ids: list[str] = field(default_factory=list)

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


def _quantiles(values: list[float]) -> dict[str, Any]:
    finite = [v for v in values if v is not None and np.isfinite(v)]
    if not finite:
        return {
            "median": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "max": float("nan"),
            "n": 0,
        }
    arr = np.asarray(finite, dtype=float)
    return {
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
        "n": int(arr.size),
    }


def _channel_verdict(pass_rate: float) -> str:
    if pass_rate >= _PASS_RATE_PASS:
        return "pass"
    if pass_rate >= _PASS_RATE_MARGINAL:
        return "marginal"
    return "fail"


def _metric_value(metrics: dict[str, Any] | None, key: str) -> float | None:
    if metrics is None:
        return None
    v = metrics.get(key)
    return v if (v is not None and np.isfinite(v)) else None


def _norm_rmse_of(metrics: dict[str, Any] | None) -> float | None:
    """Smoothed nRMSE - the verdict-driving metric."""
    return _metric_value(metrics, "norm_rmse_smoothed")


def aggregate_validation(
    results: list[ValidationResult],
    *,
    mode: str = "default",
    worst_n: int = 4,
) -> dict[str, Any]:
    """Aggregate ValidationResults into validation_summary.json shape."""
    by_channel: dict[str, list[ValidationResult]] = {}
    for r in results:
        if r.channel is None:
            continue
        by_channel.setdefault(r.channel, []).append(r)

    n_pass = sum(1 for r in results if r.verdict == "pass")
    n_marg = sum(1 for r in results if r.verdict == "marginal")
    n_fail = sum(1 for r in results if r.verdict == "fail")
    n_skip = sum(1 for r in results if r.verdict == "skip")

    channels_out: dict[str, Any] = {}
    for channel in sorted(by_channel):
        ch_summary = _aggregate_channel(
            by_channel[channel],
            channel=channel,
            mode=mode,
            worst_n=worst_n,
        )
        channels_out[channel] = ch_summary.asdict()

    return {
        "mode": mode,
        "n_runs_total": len(results),
        "n_runs_pass": n_pass,
        "n_runs_marginal": n_marg,
        "n_runs_fail": n_fail,
        "n_runs_skip": n_skip,
        "channels": channels_out,
        "thresholds": {
            "metric": "norm_rmse_smoothed",
            "rise_pass_norm_rmse": _RISE_PASS,
            "rise_marginal_norm_rmse": _RISE_MARGINAL,
            "fall_pass_norm_rmse": _FALL_PASS,
            "fall_marginal_norm_rmse": _FALL_MARGINAL,
            "channel_pass_rate_pass": _PASS_RATE_PASS,
            "channel_pass_rate_marginal": _PASS_RATE_MARGINAL,
        },
    }


def _aggregate_channel(
    results: list[ValidationResult], *, channel: str, mode: str, worst_n: int
) -> ChannelSummary:
    by_amp_dir: dict[tuple[float, str], list[ValidationResult]] = {}
    for r in results:
        if r.amplitude is None or r.direction is None:
            continue
        by_amp_dir.setdefault((float(r.amplitude), r.direction), []).append(r)

    groups: list[GroupSummary] = []
    for (amp, direction), rs in sorted(by_amp_dir.items()):
        groups.append(
            _aggregate_group(
                rs,
                key={"channel": channel, "amplitude": amp, "direction": direction},
                worst_n=worst_n,
            )
        )

    rise_vals = [v for v in (_norm_rmse_of(r.rise_metrics) for r in results) if v is not None]
    fall_vals = [v for v in (_norm_rmse_of(r.fall_metrics) for r in results) if v is not None]
    rise_vals_raw = [
        v for v in (_metric_value(r.rise_metrics, "norm_rmse") for r in results) if v is not None
    ]
    fall_vals_raw = [
        v for v in (_metric_value(r.fall_metrics, "norm_rmse") for r in results) if v is not None
    ]
    rise_ron = [
        v
        for v in (_metric_value(r.rise_metrics, "residual_over_noise") for r in results)
        if v is not None
    ]
    fall_ron = [
        v
        for v in (_metric_value(r.fall_metrics, "residual_over_noise") for r in results)
        if v is not None
    ]

    n_pass = sum(1 for r in results if r.verdict == "pass")
    n_marg = sum(1 for r in results if r.verdict == "marginal")
    n_fail = sum(1 for r in results if r.verdict == "fail")
    n_skip = sum(1 for r in results if r.verdict == "skip")
    n_scored = n_pass + n_marg + n_fail
    pass_rate = (n_pass / n_scored) if n_scored > 0 else float("nan")

    scored = [(r, _norm_rmse_of(r.rise_metrics)) for r in results]
    scored = [(r, v) for r, v in scored if v is not None]
    scored.sort(key=lambda rv: rv[1], reverse=True)
    worst_run_ids = [r.run_id for r, _ in scored[:worst_n]]

    return ChannelSummary(
        channel=channel,
        mode=mode,
        n_total=len(results),
        n_pass=n_pass,
        n_marginal=n_marg,
        n_fail=n_fail,
        n_skip=n_skip,
        pass_rate=pass_rate,
        verdict=(_channel_verdict(pass_rate) if np.isfinite(pass_rate) else "skip"),
        rise_norm_rmse=_quantiles(rise_vals),
        fall_norm_rmse=(_quantiles(fall_vals) if fall_vals else None),
        rise_norm_rmse_raw=(_quantiles(rise_vals_raw) if rise_vals_raw else None),
        fall_norm_rmse_raw=(_quantiles(fall_vals_raw) if fall_vals_raw else None),
        rise_residual_over_noise=(_quantiles(rise_ron) if rise_ron else None),
        fall_residual_over_noise=(_quantiles(fall_ron) if fall_ron else None),
        by_amp_direction=groups,
        worst_run_ids=worst_run_ids,
    )


def _aggregate_group(
    results: list[ValidationResult], *, key: dict[str, Any], worst_n: int
) -> GroupSummary:
    rise_vals = [v for v in (_norm_rmse_of(r.rise_metrics) for r in results) if v is not None]
    fall_vals = [v for v in (_norm_rmse_of(r.fall_metrics) for r in results) if v is not None]
    rise_vals_raw = [
        v for v in (_metric_value(r.rise_metrics, "norm_rmse") for r in results) if v is not None
    ]
    fall_vals_raw = [
        v for v in (_metric_value(r.fall_metrics, "norm_rmse") for r in results) if v is not None
    ]
    rise_ron = [
        v
        for v in (_metric_value(r.rise_metrics, "residual_over_noise") for r in results)
        if v is not None
    ]
    fall_ron = [
        v
        for v in (_metric_value(r.fall_metrics, "residual_over_noise") for r in results)
        if v is not None
    ]
    n_pass = sum(1 for r in results if r.verdict == "pass")
    n_marg = sum(1 for r in results if r.verdict == "marginal")
    n_fail = sum(1 for r in results if r.verdict == "fail")
    n_skip = sum(1 for r in results if r.verdict == "skip")

    scored = [(r, _norm_rmse_of(r.rise_metrics)) for r in results]
    scored = [(r, v) for r, v in scored if v is not None]
    scored.sort(key=lambda rv: rv[1], reverse=True)
    worst_run_ids = [r.run_id for r, _ in scored[:worst_n]]

    return GroupSummary(
        key=key,
        n_total=len(results),
        n_pass=n_pass,
        n_marginal=n_marg,
        n_fail=n_fail,
        n_skip=n_skip,
        rise_norm_rmse=_quantiles(rise_vals),
        fall_norm_rmse=(_quantiles(fall_vals) if fall_vals else None),
        rise_norm_rmse_raw=(_quantiles(rise_vals_raw) if rise_vals_raw else None),
        fall_norm_rmse_raw=(_quantiles(fall_vals_raw) if fall_vals_raw else None),
        rise_residual_over_noise=(_quantiles(rise_ron) if rise_ron else None),
        fall_residual_over_noise=(_quantiles(fall_ron) if fall_ron else None),
        worst_run_ids=worst_run_ids,
    )


# --- Diagnosis (was diagnose.py) ---
#
# Pattern detectors run only when a channel is marginal/failing.
# Patterns -> recommended upgrades:
#   DC-offset residuals             -> bias / wrong K
#   Residual scales with |amp|      -> nonlinearity in amplitude
#   Persistent fwd/rev difference   -> directional asymmetry
#   Spike at step edge              -> wrong L (deadtime mis-estimated)
#   Strong residual autocorrelation -> unmodeled higher-order dynamics

_DC_OFFSET_FRACTION = 0.05
_AMP_TREND_R = 0.6
_DIR_DIFF_FRACTION = 0.30
_EDGE_SPIKE_RATIO = 1.8
_AUTOCORR_LAG1 = 0.5


@dataclass
class Finding:
    pattern: str
    severity: str
    evidence: dict[str, Any]
    recommendation: str

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Diagnosis:
    channel: str
    findings: list[Finding] = field(default_factory=list)

    def asdict(self) -> dict[str, Any]:
        return {"channel": self.channel, "findings": [f.asdict() for f in self.findings]}


def _residuals_for(r: ValidationResult) -> np.ndarray | None:
    if r._t_meas is None or r._y_meas is None or r._y_pred is None:
        return None
    step_t = (r.extra or {}).get("step_t")
    active_end_t = (r.extra or {}).get("active_end_t")
    if step_t is None or active_end_t is None:
        return None
    mask = (r._t_meas >= float(step_t)) & (r._t_meas <= float(active_end_t))
    if int(mask.sum()) < 8:
        return None
    return (r._y_meas - r._y_pred)[mask]


def _check_dc_offset(results: list[ValidationResult]) -> Finding | None:
    means: list[float] = []
    for r in results:
        resid = _residuals_for(r)
        if resid is None or r.amplitude is None or abs(r.amplitude) < 1e-9:
            continue
        means.append(float(np.mean(resid) / r.amplitude))
    if len(means) < 3:
        return None
    arr = np.asarray(means)
    avg_offset = float(np.mean(arr))
    consistent = float(np.std(arr)) < 0.5 * abs(avg_offset) if abs(avg_offset) > 1e-6 else False
    if abs(avg_offset) > _DC_OFFSET_FRACTION and consistent:
        sev = "high" if abs(avg_offset) > 0.10 else "medium"
        return Finding(
            pattern="dc_offset",
            severity=sev,
            evidence={
                "mean_normalized_residual": avg_offset,
                "std_across_runs": float(np.std(arr)),
                "n_runs": len(means),
            },
            recommendation=(
                "Residual mean is consistently nonzero across runs - K (steady-state "
                f"gain) is mis-estimated by ~{abs(avg_offset):.1%}. Consider re-fitting K with a "
                "longer steady-state window or check for biased baseline."
            ),
        )
    return None


def _check_amplitude_trend(results: list[ValidationResult]) -> Finding | None:
    pairs: list[tuple[float, float]] = []
    for r in results:
        if r.amplitude is None:
            continue
        nr = (r.rise_metrics or {}).get("norm_rmse")
        if nr is None or not np.isfinite(nr):
            continue
        pairs.append((abs(float(r.amplitude)), float(nr)))
    if len(pairs) < 6:
        return None
    amps = np.asarray([a for a, _ in pairs])
    nrm = np.asarray([n for _, n in pairs])
    if np.std(amps) < 1e-9 or np.std(nrm) < 1e-9:
        return None
    r_pearson = float(np.corrcoef(amps, nrm)[0, 1])
    if abs(r_pearson) > _AMP_TREND_R:
        sev = "high" if abs(r_pearson) > 0.85 else "medium"
        return Finding(
            pattern="amplitude_nonlinearity",
            severity=sev,
            evidence={
                "pearson_r_amp_vs_norm_rmse": r_pearson,
                "n_runs": len(pairs),
                "amp_range": (float(amps.min()), float(amps.max())),
                "norm_rmse_range": (float(nrm.min()), float(nrm.max())),
            },
            recommendation=(
                f"Normalized RMSE correlates with |amplitude| (Pearson r={r_pearson:.2f}); "
                "the plant is nonlinear in amplitude. Consider gain-scheduling "
                "(if not already enabled), saturation modelling, or refitting "
                "per amplitude band."
            ),
        )
    return None


def _check_direction_asymmetry(results: list[ValidationResult]) -> Finding | None:
    fwd: list[float] = []
    rev: list[float] = []
    for r in results:
        nr = (r.rise_metrics or {}).get("norm_rmse")
        if nr is None or not np.isfinite(nr):
            continue
        if r.direction == "forward":
            fwd.append(float(nr))
        elif r.direction == "reverse":
            rev.append(float(nr))
    if len(fwd) < 3 or len(rev) < 3:
        return None
    f_med, r_med = float(np.median(fwd)), float(np.median(rev))
    denom = max(f_med, r_med, 1e-9)
    diff_frac = abs(f_med - r_med) / denom
    if diff_frac > _DIR_DIFF_FRACTION:
        sev = "high" if diff_frac > 0.6 else "medium"
        return Finding(
            pattern="direction_asymmetry",
            severity=sev,
            evidence={
                "forward_median_norm_rmse": f_med,
                "reverse_median_norm_rmse": r_med,
                "diff_fraction": diff_frac,
                "n_forward": len(fwd),
                "n_reverse": len(rev),
            },
            recommendation=(
                "Forward and reverse runs show systematically different residuals "
                f"(forward median nRMSE={f_med:.2%}, reverse median nRMSE={r_med:.2%}). "
                "The plant is direction-asymmetric - controller may need direction-"
                "aware feedforward or per-direction gain schedule."
            ),
        )
    return None


def _check_edge_spike(results: list[ValidationResult]) -> Finding | None:
    ratios: list[float] = []
    for r in results:
        resid = _residuals_for(r)
        if resid is None or resid.size < 16:
            continue
        n_edge = max(4, int(0.2 * resid.size))
        near_edge = float(np.max(np.abs(resid[:n_edge])))
        median_abs = float(np.median(np.abs(resid)))
        if median_abs < 1e-9:
            continue
        ratios.append(near_edge / median_abs)
    if len(ratios) < 3:
        return None
    median_ratio = float(np.median(ratios))
    if median_ratio > _EDGE_SPIKE_RATIO:
        sev = "high" if median_ratio > 3.0 else "medium"
        return Finding(
            pattern="edge_spike",
            severity=sev,
            evidence={"median_ratio_edge_to_median": median_ratio, "n_runs": len(ratios)},
            recommendation=(
                "Residual is concentrated near the step edge (median ratio "
                f"{median_ratio:.2f}x the rest-of-window median). L (deadtime) is likely "
                "mis-estimated. Re-check deadtime extraction or use a finer "
                "L grid in the fitter."
            ),
        )
    return None


def _check_oscillation(results: list[ValidationResult]) -> Finding | None:
    autocorrs: list[float] = []
    for r in results:
        resid = _residuals_for(r)
        if resid is None or resid.size < 16:
            continue
        a = resid - float(np.mean(resid))
        denom = float(np.sum(a * a))
        if denom < 1e-12:
            continue
        ac1 = float(np.sum(a[1:] * a[:-1]) / denom)
        autocorrs.append(ac1)
    if len(autocorrs) < 3:
        return None
    median_ac = float(np.median(autocorrs))
    if abs(median_ac) > _AUTOCORR_LAG1:
        sev = "high" if abs(median_ac) > 0.75 else "medium"
        return Finding(
            pattern="oscillation",
            severity=sev,
            evidence={"median_lag1_autocorr": median_ac, "n_runs": len(autocorrs)},
            recommendation=(
                f"Residuals show strong lag-1 autocorrelation (median {median_ac:.2f}); "
                "the plant has dynamics beyond a single first-order pole. "
                "Consider upgrading to a second-order model (FOPDT2 or "
                "underdamped second-order)."
            ),
        )
    return None


def diagnose_channel(results: list[ValidationResult], *, channel: str) -> Diagnosis:
    diag = Diagnosis(channel=channel)
    for check in (
        _check_dc_offset,
        _check_amplitude_trend,
        _check_direction_asymmetry,
        _check_edge_spike,
        _check_oscillation,
    ):
        f = check(results)
        if f is not None:
            diag.findings.append(f)
    sev_rank = {"high": 0, "medium": 1, "low": 2}
    diag.findings.sort(key=lambda f: sev_rank.get(f.severity, 3))
    return diag


def diagnose_validation(results: list[ValidationResult], summary: dict[str, Any]) -> dict[str, Any]:
    """Run per-channel checks for any channel that's marginal/failing."""
    channels = summary.get("channels") or {}
    per_channel: dict[str, list[ValidationResult]] = {}
    for r in results:
        if r.channel:
            per_channel.setdefault(r.channel, []).append(r)
    out: dict[str, Any] = {"channels": {}}
    for ch, ch_results in sorted(per_channel.items()):
        verdict = (channels.get(ch) or {}).get("verdict")
        if verdict not in ("marginal", "fail"):
            continue
        d = diagnose_channel(ch_results, channel=ch)
        out["channels"][ch] = d.asdict()
    return out


# --- Markdown + plots (was validate_report.py) ---


def render_validation_markdown(
    *,
    session_dir: Path | None,
    mode: str,
    summary: dict[str, Any],
    diagnosis: dict[str, Any] | None,
    n_train: int,
    n_validate: int,
) -> str:
    """Human-readable validation report."""
    lines: list[str] = []
    lines.append("# Validation report")
    lines.append("")
    if session_dir is not None:
        lines.append(f"- Session: `{session_dir}`")
    lines.append(f"- Mode: `{mode}`")
    lines.append(f"- Training runs: {n_train} (forward direction)")
    lines.append(f"- Validation runs: {n_validate} (held out - reverse direction)")
    lines.append(f"- Total runs validated: {summary.get('n_runs_total', 0)}")
    lines.append(f"  - pass: {summary.get('n_runs_pass', 0)}")
    lines.append(f"  - marginal: {summary.get('n_runs_marginal', 0)}")
    lines.append(f"  - fail: {summary.get('n_runs_fail', 0)}")
    lines.append(f"  - skip: {summary.get('n_runs_skip', 0)}")
    lines.append("")

    # Headline.
    lines.append("## Verdict")
    lines.append("")
    channels = summary.get("channels") or {}
    overall = _overall_verdict(channels)
    lines.append(f"**Overall: {overall.upper()}**")
    lines.append("")
    for ch, cs in sorted(channels.items()):
        verdict = cs.get("verdict", "skip")
        n = cs.get("n_total", 0)
        pr = cs.get("pass_rate", float("nan"))
        rise_med = (cs.get("rise_norm_rmse") or {}).get("median", float("nan"))
        lines.append(
            f"- **{ch}**: {verdict.upper()} - {cs.get('n_pass', 0)}/{n} pass "
            f"({_pct(pr)}), median rise nRMSE = {_pct(rise_med)}"
        )
    lines.append("")

    # Summary table - verdict-driving metric (smoothed nRMSE) headlines,
    # raw nRMSE shown alongside for traceability, residual-over-noise
    # tells you whether the model is at the channel's noise ceiling.
    lines.append("## Summary table")
    lines.append("")
    lines.append(
        "Verdict driver: **smoothed** nRMSE = "
        "RMSE(Sav-Gol(meas-pred)) / |amp|. "
        "`raw` is the un-smoothed equivalent. "
        "`r/sigma` is rmse_smoothed / noise_floor - values <= 2 mean "
        "the model is at the noise ceiling for that channel."
    )
    lines.append("")
    lines.append(
        "| channel | verdict | pass | marginal | fail | skip | "
        "rise nRMSE (smooth, med) | rise nRMSE (raw, med) | "
        "fall nRMSE (smooth, med) | rise r/sigma (med) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for ch, cs in sorted(channels.items()):
        rs = cs.get("rise_norm_rmse") or {}
        fs = cs.get("fall_norm_rmse") or {}
        rs_raw = cs.get("rise_norm_rmse_raw") or {}
        ron = cs.get("rise_residual_over_noise") or {}
        lines.append(
            "| {ch} | {v} | {p} | {m} | {f} | {s} | {rm} | {rmr} | {fm} | {ron} |".format(
                ch=ch,
                v=cs.get("verdict", "skip"),
                p=cs.get("n_pass", 0),
                m=cs.get("n_marginal", 0),
                f=cs.get("n_fail", 0),
                s=cs.get("n_skip", 0),
                rm=_pct(rs.get("median")),
                rmr=_pct(rs_raw.get("median")) if rs_raw else "-",
                fm=_pct(fs.get("median")) if fs else "-",
                ron=_num(ron.get("median")) if ron else "-",
            )
        )
    lines.append("")

    # Per (channel, amp, direction).
    lines.append("## Per (channel, amplitude, direction)")
    lines.append("")
    for ch, cs in sorted(channels.items()):
        lines.append(f"### {ch}")
        lines.append("")
        lines.append(
            "| amplitude | direction | n | pass | rise nRMSE (med) | fall nRMSE (med) | worst |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for g in cs.get("by_amp_direction") or []:
            key = g.get("key") or {}
            rn = g.get("rise_norm_rmse") or {}
            fn = g.get("fall_norm_rmse") or {}
            worst = ", ".join(g.get("worst_run_ids") or [])
            lines.append(
                "| {amp:+.2f} | {dir} | {n} | {p}/{n} | {rm} | {fm} | {w} |".format(
                    amp=float(key.get("amplitude", 0.0)),
                    dir=key.get("direction", ""),
                    n=g.get("n_total", 0),
                    p=g.get("n_pass", 0),
                    rm=_pct(rn.get("median")),
                    fm=_pct(fn.get("median")) if fn else "-",
                    w=worst,
                )
            )
        lines.append("")

    # Diagnosis - only render the section if at least one channel
    # produced at least one finding.
    diag_channels = (diagnosis or {}).get("channels") or {}
    has_any_finding = any((d.get("findings") or []) for d in diag_channels.values())
    if has_any_finding:
        lines.append("## Diagnosis (marginal/failing channels only)")
        lines.append("")
        for ch, d in sorted(diag_channels.items()):
            findings = d.get("findings") or []
            if not findings:
                continue
            lines.append(f"### {ch}")
            lines.append("")
            for f in findings:
                lines.append(f"- **{f.get('pattern')}** ({f.get('severity')})")
                lines.append(f"  - {f.get('recommendation', '')}")
                ev = f.get("evidence") or {}
                if ev:
                    parts = [f"{k}={_fmt(v)}" for k, v in ev.items()]
                    lines.append(f"  - Evidence: {', '.join(parts)}")
            lines.append("")

    # Recommendation.
    lines.append("## Ready for Session 3?")
    lines.append("")
    lines.append(_recommendation(overall, channels))
    lines.append("")
    lines.append("## Thresholds used")
    lines.append("")
    th = summary.get("thresholds") or {}
    for k, v in th.items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")

    return "\n".join(lines) + "\n"


def _overall_verdict(channels: dict[str, Any]) -> str:
    """Worst per-channel verdict wins."""
    order = {"pass": 0, "marginal": 1, "fail": 2, "skip": 3}
    worst = "pass"
    for cs in channels.values():
        v = cs.get("verdict", "skip")
        if order.get(v, 3) > order.get(worst, 0):
            worst = v
    return worst


def _recommendation(overall: str, channels: dict[str, Any]) -> str:
    if overall == "pass":
        return (
            "FOPDT generalizes within thresholds across direction. The "
            "fitted model is ready to feed Session 3 (closed-loop "
            "simulator + lambda-tuning)."
        )
    if overall == "marginal":
        return (
            "FOPDT works but with caveats - see the diagnosis section "
            "for which channels need attention. Either narrow the "
            "operating envelope (e.g. cap amplitude) or upgrade the "
            "model along the highest-severity finding before Session 3."
        )
    return (
        "FOPDT does not generalize within thresholds. See the diagnosis "
        "section for the upgrade path. Do not proceed to Session 3 with "
        "the current model."
    )


def _pct(v: Any) -> str:
    try:
        f = float(v)
        if not np.isfinite(f):
            return "-"
        return f"{f:.1%}"
    except (TypeError, ValueError):
        return "-"


def _num(v: Any) -> str:
    try:
        f = float(v)
        if not np.isfinite(f):
            return "-"
        return f"{f:.2f}"
    except (TypeError, ValueError):
        return "-"


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        if not np.isfinite(v):
            return "nan"
        return f"{v:.4g}"
    if isinstance(v, (tuple, list)):
        return "(" + ", ".join(_fmt(x) for x in v) + ")"
    return str(v)


# --------------------------------------------------------------------------- plots


def write_validation_plots(
    *,
    plots_dir: Path,
    results: list[ValidationResult],
    summary: dict[str, Any],
    n_per_channel: int = 6,
) -> list[Path]:
    """Per-channel best/median/worst overlay plots + nRMSE distribution."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return written

    by_channel: dict[str, list[ValidationResult]] = {}
    for r in results:
        if r.channel and r._t_meas is not None:
            by_channel.setdefault(r.channel, []).append(r)

    for channel, ch_results in by_channel.items():
        # Pick best, median, worst by rise nRMSE.
        scored: list[tuple[float, ValidationResult]] = []
        for r in ch_results:
            nr = (r.rise_metrics or {}).get("norm_rmse")
            if nr is None or not np.isfinite(nr):
                continue
            scored.append((float(nr), r))
        if not scored:
            continue
        scored.sort(key=lambda x: x[0])
        picks = _pick_best_median_worst(scored, n_per_channel)
        for label, (_nrmse, r) in picks:
            try:
                p = plots_dir / f"{channel}__{label}__{r.run_id}__overlay.svg"
                _plot_overlay(p, r, label=label, plt=plt)
                written.append(p)
            except Exception:
                continue

        # Distribution plot.
        try:
            p = plots_dir / f"{channel}__norm_rmse_distribution.svg"
            _plot_distribution(p, channel, ch_results, plt=plt)
            written.append(p)
        except Exception:
            pass

    return written


def _pick_best_median_worst(
    scored: list[tuple[float, ValidationResult]], n: int
) -> list[tuple[str, tuple[float, ValidationResult]]]:
    """Best, median, and worst-N from a sorted-ascending list."""
    out: list[tuple[str, tuple[float, ValidationResult]]] = []
    if not scored:
        return out
    out.append(("best", scored[0]))
    mid_idx = len(scored) // 2
    out.append(("median", scored[mid_idx]))
    # Take up to (n-2) worst, but skip duplicates of best/median.
    remaining = max(0, n - 2)
    for nrmse, r in reversed(scored):
        if (nrmse, r) in (out[0][1], out[1][1]):
            continue
        out.append((f"worst_{r.run_id}", (nrmse, r)))
        remaining -= 1
        if remaining <= 0:
            break
    return out


def _plot_overlay(path: Path, r: ValidationResult, *, label: str, plt) -> None:
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(9, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 2.5, 1]},
    )
    ax_cmd, ax_y, ax_res = axes
    t = r._t_meas
    cmd = r._cmd_at_meas if r._cmd_at_meas is not None else np.zeros_like(t)
    y_meas = r._y_meas
    y_pred = r._y_pred

    ax_cmd.plot(t, cmd, "k-", lw=1.0)
    ax_cmd.set_ylabel("cmd")
    ax_cmd.grid(alpha=0.3)
    ax_cmd.set_title(f"{r.recipe} - {r.run_id} ({label}) - verdict: {r.verdict}")

    ax_y.plot(t, y_meas, "b.", ms=2.5, alpha=0.55, label="measured")
    ax_y.plot(t, y_pred, "r-", lw=1.5, label="predicted (FOPDT)")
    ax_y.set_ylabel("y (baseline-subtracted)")
    ax_y.legend(loc="best", fontsize=8)
    ax_y.grid(alpha=0.3)

    res = y_meas - y_pred
    ax_res.plot(t, res, "g-", lw=0.8)
    ax_res.axhline(0, color="k", lw=0.5)
    ax_res.set_ylabel("residual")
    ax_res.set_xlabel("t (s)")
    ax_res.grid(alpha=0.3)

    rise = r.rise_metrics or {}
    fall = r.fall_metrics or {}
    info = []
    if rise:
        info.append(f"rise nRMSE = {rise.get('norm_rmse', float('nan')):.2%}")
    if fall:
        info.append(f"fall nRMSE = {fall.get('norm_rmse', float('nan')):.2%}")
    info.append(f"K={r.used_gain:.3g}, tau={r.used_tau:.3g}, L={r.used_deadtime:.3g}")
    ax_res.text(
        0.02,
        0.85,
        "  ".join(info),
        transform=ax_res.transAxes,
        fontsize=9,
        family="monospace",
        bbox={"facecolor": "white", "alpha": 0.7, "lw": 0},
    )

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_distribution(path: Path, channel: str, results: list[ValidationResult], *, plt) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    by_dir: dict[str, list[float]] = {"forward": [], "reverse": []}
    for r in results:
        nr = (r.rise_metrics or {}).get("norm_rmse")
        if nr is None or not np.isfinite(nr) or r.direction is None:
            continue
        by_dir.setdefault(r.direction, []).append(float(nr))

    bins = np.linspace(0, max(0.3, max((max(v) if v else 0) for v in by_dir.values()) * 1.05), 25)
    for direction, color in (("forward", "tab:blue"), ("reverse", "tab:orange")):
        v = by_dir.get(direction) or []
        if not v:
            continue
        ax.hist(v, bins=bins, alpha=0.55, label=f"{direction} (n={len(v)})", color=color)

    ax.axvline(0.10, color="green", linestyle="--", lw=1, label="pass (10%)")
    ax.axvline(0.20, color="orange", linestyle="--", lw=1, label="marginal (20%)")
    ax.set_xlabel("rise normalized RMSE = RMSE / |amplitude|")
    ax.set_ylabel("count")
    ax.set_title(f"{channel}: validation rise-nRMSE distribution")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --- Session orchestration (was validate_session.py) ---


def _pool_forward_only(
    run_fits: list[RunFit], *, mode: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build (rise_summary, fall_summary) from the forward-direction subset."""
    fwd = [r for r in run_fits if r.direction == "forward" and r.skip_reason is None]
    rise_groups = aggregate_session(select_edge(fwd, "rise"))
    fall_groups = aggregate_session(select_edge(fwd, "fall"))
    rise_summary = pool_session(rise_groups, mode=mode)
    fall_summary = pool_session(fall_groups, mode=mode)
    return rise_summary, fall_summary


def validate_session_direction_holdout(
    session_dir: Path,
    *,
    write_plots_enabled: bool = True,
) -> dict[str, Any]:
    """Run direction-holdout validation on one session, write artifacts."""
    from dimos.utils.characterization.modeling.validate_report import (
        render_validation_markdown,
        write_validation_plots,
    )

    session_dir = Path(session_dir).expanduser().resolve()
    if not session_dir.is_dir():
        raise FileNotFoundError(f"not a directory: {session_dir}")
    mode = _read_mode(session_dir)

    out_dir = session_dir / "modeling" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_fits = fit_session_runs(session_dir, mode=mode)
    rise_summary, fall_summary = _pool_forward_only(run_fits, mode=mode)

    reverse_runs = [r for r in run_fits if r.direction == "reverse" and r.skip_reason is None]
    n_train = sum(1 for r in run_fits if r.direction == "forward" and r.skip_reason is None)
    n_validate_runs = len(reverse_runs)

    results: list[ValidationResult] = []
    for rf in reverse_runs:
        try:
            v = validate_run(
                Path(rf.run_dir),
                rise_summary=rise_summary,
                fall_summary=fall_summary,
                mode=mode,
                keep_traces=True,
            )
        except Exception as e:
            v = ValidationResult(
                run_id=rf.run_id,
                run_dir=rf.run_dir,
                recipe=rf.recipe,
                channel=rf.channel,
                amplitude=rf.amplitude,
                direction=rf.direction,
                mode=mode,
                split=rf.split,
                used_gain=float("nan"),
                used_tau=float("nan"),
                used_deadtime=float("nan"),
                skip_reason=f"validate_run raised: {type(e).__name__}: {e}",
            )
        results.append(v)

    summary = aggregate_validation(results, mode=mode)
    summary["session_dir"] = str(session_dir)
    summary["n_train_runs_forward"] = n_train
    summary["n_validate_runs_reverse"] = n_validate_runs

    diagnosis: dict[str, Any] | None = diagnose_validation(results, summary)
    if not (diagnosis.get("channels") or {}):
        diagnosis = None

    atomic_write_json(
        out_dir / "validation_per_run.json",
        {
            "session_dir": str(session_dir),
            "mode": mode,
            "n_runs": len(results),
            "results": [r.asdict() for r in results],
        },
    )
    atomic_write_json(out_dir / "validation_summary.json", summary)
    if diagnosis is not None:
        atomic_write_json(out_dir / "diagnosis.json", diagnosis)

    md = render_validation_markdown(
        session_dir=session_dir,
        mode=mode,
        summary=summary,
        diagnosis=diagnosis,
        n_train=n_train,
        n_validate=n_validate_runs,
    )
    (out_dir / "validation_report.md").write_text(md)

    if write_plots_enabled:
        plots_dir = out_dir / "plots"
        write_validation_plots(plots_dir=plots_dir, results=results, summary=summary)

    return summary


__all__ = [
    "ChannelSummary",
    "Diagnosis",
    "Finding",
    "GroupSummary",
    "ValidationResult",
    "aggregate_validation",
    "diagnose_channel",
    "diagnose_validation",
    "render_validation_markdown",
    "validate_run",
    "validate_session_direction_holdout",
    "write_validation_plots",
]
