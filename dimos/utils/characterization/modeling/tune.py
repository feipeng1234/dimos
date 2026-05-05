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

"""Session 3 - lambda-tune PI gains from the validated FOPDT model.

End-to-end:
    load model_summary.json (rise pool)
    pool nominal K, tau, L per channel
    sweep lambda-multiplier over [1.0, 3.0]; score each candidate against
        a battery of reference signals via the closed-loop simulator
    pick winner; run robustness sweeps (K +/- 15%, tau / L +/- 20%)
    write tuning_summary.json + tuning_report.md + plots

This file used to be six modules (tune, tune_session, closed_loop,
robustness, references, plus parts of fopdt's surface). Same one-file
consolidation as fit.py and validate.py. ``simulate`` (the FOPDT
forward propagator from Session 2) is the one external dependency we
keep since the closed-loop sim wraps it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from dimos.utils.characterization.modeling.fit import atomic_write_json
from dimos.utils.characterization.modeling.fopdt import FopdtParams
from dimos.utils.characterization.modeling.simulate import simulate_fopdt

# =============================================================================
# Reference-signal builders (was references.py)
# =============================================================================


def step(amplitude: float, t_start: float = 0.5) -> Callable[[float], float]:
    """Single step from 0 to ``amplitude`` at ``t_start`` and hold."""

    def r(t: float) -> float:
        return amplitude if t >= t_start else 0.0

    return r


def staircase(
    amplitudes: Sequence[float],
    dwell_s: float,
    t_start: float = 0.5,
) -> Callable[[float], float]:
    """Step-and-hold sequence. ``amplitudes[0]`` becomes active at t_start."""
    levels = list(amplitudes)
    edges = [t_start + i * dwell_s for i in range(len(levels))]

    def r(t: float) -> float:
        if t < t_start:
            return 0.0
        for i in range(len(levels) - 1, -1, -1):
            if t >= edges[i]:
                return float(levels[i])
        return 0.0

    return r


def ramp(
    slope: float,
    duration: float,
    t_start: float = 0.5,
    final_hold: bool = True,
) -> Callable[[float], float]:
    """Linear ramp at ``slope`` for ``duration`` seconds starting at t_start."""
    final_value = slope * duration

    def r(t: float) -> float:
        if t < t_start:
            return 0.0
        elapsed = t - t_start
        if elapsed >= duration:
            return float(final_value) if final_hold else 0.0
        return float(slope * elapsed)

    return r


def sinusoid(
    amplitude: float,
    freq_hz: float,
    offset: float = 0.0,
    t_start: float = 0.0,
    phase_rad: float = 0.0,
) -> Callable[[float], float]:
    """``offset + amplitude * sin(2*pi*f*(t-t_start) + phase)`` for t >= t_start."""
    omega = 2.0 * math.pi * freq_hz

    def r(t: float) -> float:
        if t < t_start:
            return float(offset)
        return float(offset + amplitude * math.sin(omega * (t - t_start) + phase_rad))

    return r


def from_array(t_array: np.ndarray, r_array: np.ndarray) -> Callable[[float], float]:
    """ZOH lookup against (t_array, r_array). Used to replay recorded velocity profiles."""
    t_array = np.asarray(t_array, dtype=float)
    r_array = np.asarray(r_array, dtype=float)
    if t_array.shape != r_array.shape:
        raise ValueError("t_array and r_array must have the same shape")
    if t_array.size == 0:
        raise ValueError("t_array must be non-empty")

    def r(t: float) -> float:
        if t < t_array[0]:
            return float(r_array[0])
        if t >= t_array[-1]:
            return float(r_array[-1])
        idx = int(np.searchsorted(t_array, t, side="right") - 1)
        idx = max(0, min(idx, t_array.size - 1))
        return float(r_array[idx])

    return r


def realistic_path_velocity(profile_path: Path) -> Callable[[float], float]:
    """Load a recorded path-following velocity profile (npy or csv with (t, v) columns)."""
    profile_path = Path(profile_path).expanduser()
    if not profile_path.exists():
        raise FileNotFoundError(f"velocity profile not found: {profile_path}")
    if profile_path.suffix == ".npy":
        arr = np.load(profile_path)
    else:
        arr = np.loadtxt(profile_path, delimiter=",")
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"expected (N, 2) array; got shape {arr.shape}")
    return from_array(arr[:, 0], arr[:, 1])


# =============================================================================
# Closed-loop simulator (was closed_loop.py)
# =============================================================================
#
# The plant model is Session 2's simulate_fopdt - we wrap it. At each
# control tick the controller computes u from r-y, the plant simulator
# extends its output one step based on the entire u history (deadtime
# handled by the plant). Anti-windup is back-calculation: when u_sat
# differs from u_raw, the integrator is bled by (u_sat - u_raw) * Kt * dt.
# With Kt ~ 1/tau, saturation excursions unwind on the same timescale as
# the plant.


@dataclass
class PIController:
    Kp: float
    Ki: float
    Kt: float = 0.0
    u_min: float = -np.inf
    u_max: float = np.inf
    integrator: float = field(default=0.0)

    def reset(self) -> None:
        self.integrator = 0.0

    def step(self, ref: float, meas: float, dt: float) -> tuple[float, float]:
        """Advance one control tick. Returns (u_raw, u_sat)."""
        e = ref - meas
        u_raw = self.Kp * e + self.Ki * self.integrator
        u_sat = float(np.clip(u_raw, self.u_min, self.u_max))
        self.integrator += (e + self.Kt * (u_sat - u_raw)) * dt
        return float(u_raw), u_sat


@dataclass
class SimResult:
    t: np.ndarray
    r: np.ndarray
    u: np.ndarray
    u_raw: np.ndarray
    y: np.ndarray
    metrics: dict[str, float] = field(default_factory=dict)
    plant: dict[str, float] = field(default_factory=dict)
    controller: dict[str, float] = field(default_factory=dict)

    def asdict(self) -> dict[str, Any]:
        return {
            "t": self.t.tolist(),
            "r": self.r.tolist(),
            "u": self.u.tolist(),
            "u_raw": self.u_raw.tolist(),
            "y": self.y.tolist(),
            "metrics": self.metrics,
            "plant": self.plant,
            "controller": self.controller,
        }


def _compute_step_metrics(
    t: np.ndarray,
    r: np.ndarray,
    u: np.ndarray,
    u_raw: np.ndarray,
    y: np.ndarray,
    *,
    settle_band: float = 0.02,
) -> dict[str, float]:
    """Standard step-response metrics + closed-loop control effort."""
    if t.size < 2:
        return {}
    e = r - y
    dt = float(np.mean(np.diff(t)))
    iae = float(np.sum(np.abs(e)) * dt)
    itae = float(np.sum(t * np.abs(e)) * dt)
    rmse = float(np.sqrt(np.mean(e**2)))
    max_u = float(np.max(np.abs(u)))
    saturation_fraction = float(np.mean(np.abs(u_raw - u) > 1e-9))

    r_final = float(r[-1])
    if abs(r_final) > 1e-9 and r_final > 0:
        overshoot = float(max(0.0, np.max(y) - r_final) / abs(r_final))
    elif abs(r_final) > 1e-9 and r_final < 0:
        overshoot = float(max(0.0, abs(np.min(y)) - abs(r_final)) / abs(r_final))
    else:
        overshoot = float("nan")

    if abs(r_final) > 1e-9:
        outside = np.abs(y - r_final) > settle_band * abs(r_final)
        if outside.any():
            last_outside = int(np.where(outside)[0][-1])
            settle_time = float(t[last_outside]) if last_outside + 1 < t.size else float("inf")
        else:
            settle_time = 0.0
    else:
        settle_time = float("nan")

    return {
        "iae": iae,
        "itae": itae,
        "rmse": rmse,
        "overshoot": overshoot,
        "settle_time_s": settle_time,
        "max_abs_u": max_u,
        "saturation_fraction": saturation_fraction,
    }


def simulate_closed_loop(
    plant: FopdtParams,
    controller: PIController,
    reference_fn: Callable[[float], float],
    *,
    duration_s: float,
    control_dt_s: float = 0.02,
    initial_y: float = 0.0,
    settle_band: float = 0.02,
) -> SimResult:
    """Simulate a closed-loop response."""
    if duration_s <= 0 or control_dt_s <= 0:
        raise ValueError("duration_s and control_dt_s must be positive")

    n = int(np.floor(duration_s / control_dt_s)) + 1
    t = np.arange(n, dtype=float) * control_dt_s
    r = np.array([float(reference_fn(float(ti))) for ti in t])
    u = np.zeros(n)
    u_raw = np.zeros(n)
    y = np.zeros(n)
    y[0] = initial_y

    controller.reset()

    # Step k uses y[k-1] (latest measurement available at t[k-1]) to compute u[k].
    # The plant simulator advances y[k] using the full u[:k+1] history (FOPDT
    # deadtime handled internally).
    for k in range(1, n):
        ur, us = controller.step(float(r[k - 1]), float(y[k - 1]), control_dt_s)
        u_raw[k] = ur
        u[k] = us
        y_traj = simulate_fopdt(t[: k + 1], u[: k + 1], plant, initial=initial_y)
        y[k] = float(y_traj[k])

    metrics = _compute_step_metrics(t, r, u, u_raw, y, settle_band=settle_band)
    return SimResult(
        t=t,
        r=r,
        u=u,
        u_raw=u_raw,
        y=y,
        metrics=metrics,
        plant={"K": plant.K, "tau": plant.tau, "L": plant.L},
        controller={
            "Kp": controller.Kp,
            "Ki": controller.Ki,
            "Kt": controller.Kt,
            "u_min": controller.u_min,
            "u_max": controller.u_max,
        },
    )


# =============================================================================
# Lambda-tuning + multiplier sweep (was tune.py)
# =============================================================================
#
# Lambda tuning (SIMC / IMC for FOPDT plants) gives PI gains as a closed
# form of (K, tau, L) plus a designer choice lambda (closed-loop time
# constant).
#
#   lambda = max(tau, L) * multiplier      # multiplier in [1, 3]
#   Kp = tau / (K * (lambda + L))
#   Ki = Kp / tau
#   Kt = 1 / tau                            # back-calculation anti-windup gain


@dataclass
class Gains:
    Kp: float
    Ki: float
    Kt: float
    multiplier: float
    lambda_s: float

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TuningCandidate:
    gains: Gains
    per_reference: list[dict[str, Any]]
    cost: float
    cost_breakdown: dict[str, float]

    def asdict(self) -> dict[str, Any]:
        return {
            "gains": self.gains.asdict(),
            "per_reference": self.per_reference,
            "cost": self.cost,
            "cost_breakdown": self.cost_breakdown,
        }


@dataclass
class TuningResult:
    plant: dict[str, float]
    saturation_limits: tuple[float, float]
    multipliers: list[float]
    candidates: list[TuningCandidate]
    best_index: int

    @property
    def best(self) -> TuningCandidate:
        return self.candidates[self.best_index]

    def asdict(self) -> dict[str, Any]:
        return {
            "plant": self.plant,
            "saturation_limits": list(self.saturation_limits),
            "multipliers": self.multipliers,
            "best_index": self.best_index,
            "best_gains": self.best.gains.asdict(),
            "candidates": [c.asdict() for c in self.candidates],
        }


def lambda_tune(K: float, tau: float, L: float, multiplier: float = 1.0) -> Gains:
    """Standard lambda-tuning for FOPDT -> PI."""
    lam = max(tau, L) * multiplier
    Kp = tau / (K * (lam + L))
    Ki = Kp / tau
    Kt = 1.0 / tau
    return Gains(Kp=Kp, Ki=Ki, Kt=Kt, multiplier=multiplier, lambda_s=lam)


# Cost weights - tuned so each term contributes roughly the same
# magnitude on a typical step response.
_W_IAE = 1.0
_W_OVERSHOOT = 2.0
_W_SATURATION = 0.5
_W_SETTLE = 0.2


def _scenario_cost(metrics: dict[str, float]) -> float:
    iae = float(metrics.get("iae", 0.0))
    over = float(metrics.get("overshoot", 0.0))
    if not np.isfinite(over):
        over = 0.0
    sat = float(metrics.get("saturation_fraction", 0.0))
    settle = float(metrics.get("settle_time_s", 0.0))
    if not np.isfinite(settle):
        settle = 5.0  # ran out of horizon without settling - heavy penalty
    return _W_IAE * iae + _W_OVERSHOOT * over + _W_SATURATION * sat + _W_SETTLE * settle


def default_scenarios(
    saturation_limits: tuple[float, float],
) -> list[tuple[str, Callable[[float], float], float]]:
    """Battery of reference signals to score each candidate against.

    Returns (label, reference_fn, duration_s). Amplitudes scale to the
    actuator range so the same battery works for vx (~1.0 m/s) and wz
    (~1.5 rad/s) without manual tweaks.
    """
    u_max = float(saturation_limits[1])
    return [
        ("step_50pct", step(amplitude=0.5 * u_max, t_start=0.5), 4.0),
        ("step_90pct", step(amplitude=0.9 * u_max, t_start=0.5), 4.0),
        ("step_neg_70pct", step(amplitude=-0.7 * u_max, t_start=0.5), 4.0),
        (
            "staircase",
            staircase(
                amplitudes=[0.3 * u_max, 0.6 * u_max, 0.9 * u_max, 0.3 * u_max, 0.0],
                dwell_s=1.5,
                t_start=0.5,
            ),
            8.5,
        ),
        ("ramp", ramp(slope=0.5 * u_max, duration=1.5, t_start=0.5), 4.0),
    ]


def tune_channel(
    plant: FopdtParams,
    *,
    saturation_limits: tuple[float, float],
    multiplier_range: tuple[float, float] = (1.0, 3.0),
    n_multipliers: int = 5,
    control_dt_s: float = 0.02,
    scenarios: list[tuple[str, Callable[[float], float], float]] | None = None,
) -> TuningResult:
    """Sweep lambda-multiplier; score each against the scenario battery; return winner."""
    if scenarios is None:
        scenarios = default_scenarios(saturation_limits)

    multipliers = list(np.linspace(multiplier_range[0], multiplier_range[1], n_multipliers))

    candidates: list[TuningCandidate] = []
    for m in multipliers:
        gains = lambda_tune(plant.K, plant.tau, plant.L, multiplier=float(m))
        per_ref: list[dict[str, Any]] = []
        cost_sum = 0.0
        breakdown_sum = {
            "iae": 0.0,
            "overshoot": 0.0,
            "saturation_fraction": 0.0,
            "settle_time_s": 0.0,
        }
        for label, ref_fn, dur in scenarios:
            ctrl = PIController(
                Kp=gains.Kp,
                Ki=gains.Ki,
                Kt=gains.Kt,
                u_min=saturation_limits[0],
                u_max=saturation_limits[1],
            )
            sim = simulate_closed_loop(
                plant,
                ctrl,
                ref_fn,
                duration_s=dur,
                control_dt_s=control_dt_s,
            )
            sc_cost = _scenario_cost(sim.metrics)
            per_ref.append(
                {
                    "scenario": label,
                    "duration_s": dur,
                    "metrics": sim.metrics,
                    "cost": sc_cost,
                }
            )
            cost_sum += sc_cost
            for k in breakdown_sum:
                v = float(sim.metrics.get(k, 0.0))
                if np.isfinite(v):
                    breakdown_sum[k] += v
        n_scenarios = max(1, len(scenarios))
        breakdown = {k: v / n_scenarios for k, v in breakdown_sum.items()}
        candidates.append(
            TuningCandidate(
                gains=gains,
                per_reference=per_ref,
                cost=cost_sum,
                cost_breakdown=breakdown,
            )
        )

    best_index = int(np.argmin([c.cost for c in candidates]))
    return TuningResult(
        plant={"K": plant.K, "tau": plant.tau, "L": plant.L},
        saturation_limits=saturation_limits,
        multipliers=multipliers,
        candidates=candidates,
        best_index=best_index,
    )


# =============================================================================
# Robustness sweeps (was robustness.py)
# =============================================================================
#
# Two checks before deploy:
#   gain_sweep:  hold controller fixed, sweep K across run-to-run variance band
#                (default +/-15%). Confirm tracking error stays bounded.
#   param_sweep: same shape but for tau or L (model uncertainty).
# A controller passes if every point in the sweep settles within the horizon,
# doesn't go unstable, and stays inside the saturation budget.


@dataclass
class SweepPoint:
    label: str
    plant: dict[str, float]
    metrics: dict[str, float]
    stable: bool
    saturation_fraction: float

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SweepResult:
    swept: str
    nominal: dict[str, float]
    points: list[SweepPoint] = field(default_factory=list)
    all_stable: bool = True
    worst_iae: float = 0.0

    def asdict(self) -> dict[str, Any]:
        return {
            "swept": self.swept,
            "nominal": self.nominal,
            "all_stable": self.all_stable,
            "worst_iae": self.worst_iae,
            "points": [p.asdict() for p in self.points],
        }


def _is_stable(y: np.ndarray, max_y: float) -> bool:
    if not np.all(np.isfinite(y)):
        return False
    bound = max(10.0 * max_y, 1.0)
    return bool(np.max(np.abs(y)) < bound)


def _scaled_plant(nominal: FopdtParams, *, which: str, factor: float) -> FopdtParams:
    K, tau, L = nominal.K, nominal.tau, nominal.L
    if which == "K":
        K = K * factor
    elif which == "tau":
        tau = tau * factor
    elif which == "L":
        L = L * factor
    else:
        raise ValueError(f"unknown parameter to scale: {which!r}")
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


def gain_sweep(
    plant: FopdtParams,
    controller: PIController,
    reference_fn: Callable[[float], float],
    *,
    k_range: tuple[float, float] = (0.85, 1.15),
    n: int = 7,
    duration_s: float = 4.0,
    control_dt_s: float = 0.02,
) -> SweepResult:
    """Sweep K across [k_range[0]*K_nom, k_range[1]*K_nom]."""
    factors = list(np.linspace(k_range[0], k_range[1], n))
    sat_max = max(abs(controller.u_min), abs(controller.u_max))
    nominal = {"K": plant.K, "tau": plant.tau, "L": plant.L}
    result = SweepResult(swept="K", nominal=nominal)
    for f in factors:
        scaled = _scaled_plant(plant, which="K", factor=float(f))
        controller.reset()
        sim = simulate_closed_loop(
            scaled,
            controller,
            reference_fn,
            duration_s=duration_s,
            control_dt_s=control_dt_s,
        )
        stable = _is_stable(sim.y, max_y=sat_max)
        sat_frac = float(sim.metrics.get("saturation_fraction", 0.0))
        result.points.append(
            SweepPoint(
                label=f"K={f:.2f}*nom",
                plant={"K": scaled.K, "tau": scaled.tau, "L": scaled.L},
                metrics=sim.metrics,
                stable=stable,
                saturation_fraction=sat_frac,
            )
        )
        if not stable:
            result.all_stable = False
        iae = float(sim.metrics.get("iae", 0.0))
        if np.isfinite(iae) and iae > result.worst_iae:
            result.worst_iae = iae
    return result


def param_sweep(
    plant: FopdtParams,
    controller: PIController,
    reference_fn: Callable[[float], float],
    *,
    which: str,
    factor_range: tuple[float, float] = (0.80, 1.20),
    n: int = 7,
    duration_s: float = 4.0,
    control_dt_s: float = 0.02,
) -> SweepResult:
    """Sweep tau or L over factor_range to cover model uncertainty."""
    if which not in ("tau", "L"):
        raise ValueError(f"param_sweep accepts 'tau' or 'L', got {which!r}")
    factors = list(np.linspace(factor_range[0], factor_range[1], n))
    sat_max = max(abs(controller.u_min), abs(controller.u_max))
    nominal = {"K": plant.K, "tau": plant.tau, "L": plant.L}
    result = SweepResult(swept=which, nominal=nominal)
    for f in factors:
        scaled = _scaled_plant(plant, which=which, factor=float(f))
        controller.reset()
        sim = simulate_closed_loop(
            scaled,
            controller,
            reference_fn,
            duration_s=duration_s,
            control_dt_s=control_dt_s,
        )
        stable = _is_stable(sim.y, max_y=sat_max)
        result.points.append(
            SweepPoint(
                label=f"{which}={f:.2f}*nom",
                plant={"K": scaled.K, "tau": scaled.tau, "L": scaled.L},
                metrics=sim.metrics,
                stable=stable,
                saturation_fraction=float(sim.metrics.get("saturation_fraction", 0.0)),
            )
        )
        if not stable:
            result.all_stable = False
        iae = float(sim.metrics.get("iae", 0.0))
        if np.isfinite(iae) and iae > result.worst_iae:
            result.worst_iae = iae
    return result


# =============================================================================
# Session orchestration (was tune_session.py)
# =============================================================================

# Default saturation limits (cmd-space) per channel. wz is firmware-limited
# at the actual rate output, but the cmd we send is bounded by the cmd
# range used in characterization (E2 collected up to +/-1.5).
_DEFAULT_SAT_LIMITS = {
    "vx": (-1.0, 1.0),
    "vy": (-1.0, 1.0),
    "wz": (-1.5, 1.5),
}

# Default control rate (50 Hz matches the harness scheduler).
_DEFAULT_CONTROL_DT_S = 0.02

# K-sweep range - Session 2's measured run-to-run variance.
_DEFAULT_K_SWEEP = (0.85, 1.15)

# tau / L uncertainty band for the model-uncertainty sweep.
_DEFAULT_PARAM_SWEEP = (0.80, 1.20)


def _plant_from_summary(summary: dict[str, Any], channel: str) -> FopdtParams | None:
    """Build a FopdtParams from the pooled rise model for one channel."""
    rise = summary.get("rise") or summary
    ch = (rise.get("channels") or {}).get(channel)
    if not ch:
        return None
    pooled = ch.get("pooled") or {}
    K = (pooled.get("K") or {}).get("mean")
    tau = (pooled.get("tau") or {}).get("mean")
    L = (pooled.get("L") or {}).get("mean")
    if not all(v is not None and np.isfinite(float(v)) for v in (K, tau, L)):
        return None
    return FopdtParams(
        K=float(K),
        tau=float(tau),
        L=float(L),
        K_ci=(float(K), float(K)),
        tau_ci=(float(tau), float(tau)),
        L_ci=(float(L), float(L)),
        rmse=0.0,
        r_squared=1.0,
        n_samples=0,
        fit_window_s=(0.0, 0.0),
        degenerate=False,
        converged=True,
    )


def _validation_reference(plant: FopdtParams, sat_limits: tuple[float, float]):
    """One canonical reference signal used for the robustness sweeps."""
    return step(amplitude=0.7 * sat_limits[1], t_start=0.5)


def tune_session(
    session_dir: Path,
    *,
    write_plots_enabled: bool = True,
    saturation_limits: dict[str, tuple[float, float]] | None = None,
    control_dt_s: float = _DEFAULT_CONTROL_DT_S,
) -> dict[str, Any]:
    """End-to-end per-channel tuning. Writes JSON + markdown + plots.

    Returns the in-memory tuning summary.
    """
    session_dir = Path(session_dir).expanduser().resolve()
    summary_path = session_dir / "modeling" / "model_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"{summary_path} not found - run `process_session fit` first.")
    summary = json.loads(summary_path.read_text())

    sat_limits = dict(_DEFAULT_SAT_LIMITS)
    if saturation_limits:
        sat_limits.update(saturation_limits)

    out_dir = session_dir / "modeling" / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"

    rise_summary = summary.get("rise") or summary
    channels_summary = rise_summary.get("channels") or {}

    channel_results: dict[str, Any] = {}
    for channel in sorted(channels_summary):
        plant = _plant_from_summary(summary, channel)
        if plant is None:
            channel_results[channel] = {"skip_reason": "no pooled FOPDT params"}
            continue
        sat = sat_limits.get(channel, (-1.0, 1.0))

        tr = tune_channel(plant, saturation_limits=sat, control_dt_s=control_dt_s)
        winner = tr.best

        ctrl = PIController(
            Kp=winner.gains.Kp,
            Ki=winner.gains.Ki,
            Kt=winner.gains.Kt,
            u_min=sat[0],
            u_max=sat[1],
        )
        ref_fn = _validation_reference(plant, sat)
        gs = gain_sweep(
            plant, ctrl, ref_fn, k_range=_DEFAULT_K_SWEEP, duration_s=4.0, control_dt_s=control_dt_s
        )
        ctrl.reset()
        ts = param_sweep(
            plant,
            ctrl,
            ref_fn,
            which="tau",
            factor_range=_DEFAULT_PARAM_SWEEP,
            duration_s=4.0,
            control_dt_s=control_dt_s,
        )
        ctrl.reset()
        ls = param_sweep(
            plant,
            ctrl,
            ref_fn,
            which="L",
            factor_range=_DEFAULT_PARAM_SWEEP,
            duration_s=4.0,
            control_dt_s=control_dt_s,
        )

        channel_results[channel] = {
            "plant": {"K": plant.K, "tau": plant.tau, "L": plant.L},
            "saturation_limits": list(sat),
            "tuning": tr.asdict(),
            "robustness": {
                "K_sweep": gs.asdict(),
                "tau_sweep": ts.asdict(),
                "L_sweep": ls.asdict(),
            },
            "verdict": _channel_tune_verdict(winner.cost_breakdown, gs, ts, ls),
        }

        if write_plots_enabled:
            try:
                _plot_channel(plots_dir, channel, plant, winner, sat, gs, ts, ls, control_dt_s)
            except Exception as e:
                channel_results[channel]["plot_error"] = f"{type(e).__name__}: {e}"

    out = {
        "session_dir": str(session_dir),
        "control_dt_s": control_dt_s,
        "channels": channel_results,
        "thresholds": {
            "k_sweep_range": list(_DEFAULT_K_SWEEP),
            "tau_sweep_range": list(_DEFAULT_PARAM_SWEEP),
            "L_sweep_range": list(_DEFAULT_PARAM_SWEEP),
        },
    }

    atomic_write_json(out_dir / "tuning_summary.json", out)
    (out_dir / "tuning_report.md").write_text(_render_tuning_markdown(out))
    return out


def _channel_tune_verdict(cost_breakdown, gs, ts, ls) -> str:
    """Pass/marginal/fail per channel based on robustness sweeps."""
    if not (gs.all_stable and ts.all_stable and ls.all_stable):
        return "fail"
    overshoot = float(cost_breakdown.get("overshoot", 0.0))
    saturated = float(cost_breakdown.get("saturation_fraction", 0.0))
    if not np.isfinite(overshoot):
        overshoot = 0.0
    if overshoot <= 0.05 and saturated <= 0.50:
        return "pass"
    return "marginal"


# ---------------------------------------------------------------------- markdown


def _render_tuning_markdown(out: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Tuning report")
    lines.append("")
    lines.append(f"- Session: `{out['session_dir']}`")
    lines.append(
        f"- Control rate: {1.0 / out['control_dt_s']:.0f} Hz (dt = {out['control_dt_s']} s)"
    )
    lines.append("")
    lines.append("## Per channel")
    lines.append("")
    for channel, ch in sorted((out.get("channels") or {}).items()):
        lines.append(f"### {channel}")
        lines.append("")
        if "skip_reason" in ch:
            lines.append(f"- **skipped**: {ch['skip_reason']}")
            lines.append("")
            continue
        plant = ch["plant"]
        lines.append(f"- Plant: K={plant['K']:.4g}, tau={plant['tau']:.4g} s, L={plant['L']:.4g} s")
        lines.append(
            f"- Saturation: u in [{ch['saturation_limits'][0]}, {ch['saturation_limits'][1]}]"
        )
        winner = (ch.get("tuning") or {}).get("best_gains") or {}
        lines.append(
            f"- **Chosen gains**: Kp={winner.get('Kp', 0.0):.4f}, "
            f"Ki={winner.get('Ki', 0.0):.4f}, "
            f"Kt={winner.get('Kt', 0.0):.4f} (lambda-multiplier "
            f"= {winner.get('multiplier', 0.0):.2f})"
        )
        lines.append(f"- Robustness verdict: **{ch.get('verdict', '?').upper()}**")
        lines.append("")
        lines.append("#### Multiplier sweep")
        lines.append("")
        lines.append(
            "| multiplier | lambda (s) | Kp | Ki | total cost | mean IAE | mean overshoot | mean settle |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for c in (ch.get("tuning") or {}).get("candidates") or []:
            g = c.get("gains") or {}
            br = c.get("cost_breakdown") or {}
            lines.append(
                "| {m:.2f} | {lam:.3f} | {Kp:.3f} | {Ki:.3f} | {cost:.3f} | "
                "{iae:.3f} | {ovr:.1%} | {set:.2f} s |".format(
                    m=g.get("multiplier", 0.0),
                    lam=g.get("lambda_s", 0.0),
                    Kp=g.get("Kp", 0.0),
                    Ki=g.get("Ki", 0.0),
                    cost=c.get("cost", 0.0),
                    iae=br.get("iae", 0.0),
                    ovr=br.get("overshoot", 0.0),
                    **{"set": br.get("settle_time_s", 0.0)},
                )
            )
        lines.append("")
        lines.append("#### Robustness - K sweep (run-to-run variance)")
        lines.append("")
        gs = (ch.get("robustness") or {}).get("K_sweep") or {}
        lines.append(f"- All stable: **{gs.get('all_stable')}**")
        lines.append(f"- Worst IAE: {gs.get('worst_iae', 0.0):.3f}")
        lines.append("")
        lines.append("| label | K | overshoot | settle | sat fraction | stable |")
        lines.append("|---|---|---|---|---|---|")
        for p in gs.get("points") or []:
            m = p.get("metrics") or {}
            lines.append(
                "| {l} | {K:.3f} | {ovr:.1%} | {st:.2f} s | {sat:.0%} | {stab} |".format(
                    l=p.get("label", ""),
                    K=(p.get("plant") or {}).get("K", 0.0),
                    ovr=m.get("overshoot", 0.0),
                    st=m.get("settle_time_s", 0.0),
                    sat=m.get("saturation_fraction", 0.0),
                    stab=p.get("stable", "?"),
                )
            )
        lines.append("")
        for which, label in (("tau_sweep", "tau"), ("L_sweep", "L")):
            sw = (ch.get("robustness") or {}).get(which) or {}
            lines.append(f"#### Robustness - {label} sweep (model uncertainty)")
            lines.append("")
            lines.append(f"- All stable: **{sw.get('all_stable')}**")
            lines.append(f"- Worst IAE: {sw.get('worst_iae', 0.0):.3f}")
            lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------- plots


def _plot_channel(
    plots_dir: Path,
    channel: str,
    plant: FopdtParams,
    winner,
    sat: tuple[float, float],
    gs,
    ts,
    ls,
    control_dt_s: float,
) -> None:
    """Step + staircase + K-sweep plots for one channel."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenarios = default_scenarios(sat)
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(12, 2.6 * len(scenarios)), sharex=False)
    if len(scenarios) == 1:
        axes = [axes]
    for ax, (label, ref_fn, dur) in zip(axes, scenarios, strict=False):
        ctrl = PIController(
            Kp=winner.gains.Kp,
            Ki=winner.gains.Ki,
            Kt=winner.gains.Kt,
            u_min=sat[0],
            u_max=sat[1],
        )
        sim = simulate_closed_loop(
            plant,
            ctrl,
            ref_fn,
            duration_s=dur,
            control_dt_s=control_dt_s,
        )
        ax.plot(sim.t, sim.r, "k-", lw=1.0, alpha=0.55, label="reference")
        ax.plot(sim.t, sim.y, color="tab:blue", lw=1.5, label="output y")
        ax.plot(sim.t, sim.u, color="tab:red", lw=1.0, alpha=0.7, label="control u")
        ax.axhline(sat[0], color="grey", lw=0.5, ls=":")
        ax.axhline(sat[1], color="grey", lw=0.5, ls=":")
        m = sim.metrics
        ax.set_title(
            f"{channel}  -  {label}  -  IAE={m.get('iae', 0):.3f}, "
            f"overshoot={m.get('overshoot', 0):.1%}, "
            f"settle={m.get('settle_time_s', 0):.2f}s",
            fontsize=10,
        )
        ax.set_xlabel("t (s)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(
        f"{channel}: closed-loop response with chosen gains "
        f"(Kp={winner.gains.Kp:.3f}, Ki={winner.gains.Ki:.3f}, "
        f"lambda-mult={winner.gains.multiplier:.2f})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(plots_dir / f"{channel}__step_responses.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    Ks = [p.plant["K"] for p in gs.points]
    iaes = [p.metrics.get("iae", 0.0) for p in gs.points]
    overs = [p.metrics.get("overshoot", 0.0) for p in gs.points]
    axes[0].plot(Ks, iaes, "o-", color="tab:blue")
    axes[0].set_xlabel("plant K")
    axes[0].set_ylabel("IAE")
    axes[0].set_title(f"{channel}: K-sweep IAE (controller fixed)")
    axes[0].grid(alpha=0.3)
    axes[1].plot(Ks, overs, "o-", color="tab:orange")
    axes[1].axhline(0.05, color="red", lw=1, ls="--", label="5% target")
    axes[1].set_xlabel("plant K")
    axes[1].set_ylabel("overshoot")
    axes[1].yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    axes[1].set_title(f"{channel}: K-sweep overshoot")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / f"{channel}__K_sweep.svg")
    plt.close(fig)


__all__ = [
    "Gains",
    "PIController",
    "SimResult",
    "SweepPoint",
    "SweepResult",
    "TuningCandidate",
    "TuningResult",
    "default_scenarios",
    "from_array",
    "gain_sweep",
    "lambda_tune",
    "param_sweep",
    "ramp",
    "realistic_path_velocity",
    "simulate_closed_loop",
    "sinusoid",
    "staircase",
    "step",
    "tune_channel",
    "tune_session",
]
