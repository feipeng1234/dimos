# Copyright 2025-2026 Dimensional Inc.
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

"""Render reachability heatmaps from ``eval_pointing_reachability`` JSONs.

Two render modes:

* **Single arm** — pass only ``--left``.  Produces a single equirectangular
  azimuth x elevation heatmap with per-cell mode annotations and per-axis
  success-rate bars.

* **Dual arm** — pass both ``--left`` and ``--right``.  Three stacked
  heatmaps (left-only, right-only, combined "either arm") plus per-axis
  comparison bars.  Combined panel is the headline view: it shows how
  much azimuth coverage you actually get when ``point_at`` can route to
  the appropriate arm.

Typical session::

    uv run python -m dimos.manipulation.eval_pointing_reachability \\
        --arm left_arm --out /tmp/reach_left.json
    uv run python -m dimos.manipulation.eval_pointing_reachability \\
        --arm right_arm --out /tmp/reach_right.json
    uv run python -m dimos.manipulation.visualize_pointing_reachability \\
        --left /tmp/reach_left.json --right /tmp/reach_right.json \\
        --out /tmp/dual_arm_heatmap.png
"""

from __future__ import annotations

import argparse
import json
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

# Color tokens shared across panels.
_ARM_COLORS = {
    "left": "#2c7fb8",  # blue — left arm only
    "right": "#d7301f",  # red — right arm only
    "both": "#762a83",  # purple — either arm reaches this cell
    "wedged": "#bababa",  # gray — at least one arm got wedged
    "neither": "#f0f0f0",  # near-white — neither arm
}

_OUTCOME_COLORS_SINGLE = {
    "FAIL": "#d73027",
    "WEDGED": "#bababa",
    "SUCCESS": "#1a9850",
}


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _build_grid(
    rows: list[list[dict]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_el, n_az = len(rows), len(rows[0])
    az = np.array([c["az"] for c in rows[0]], dtype=float)
    el = np.array([r[0]["el"] for r in rows], dtype=float)
    success = np.zeros((n_el, n_az), dtype=bool)
    wedged = np.zeros((n_el, n_az), dtype=bool)
    for i, row in enumerate(rows):
        for j, c in enumerate(row):
            success[i, j] = c["ok"]
            wedged[i, j] = bool(c.get("wedged_before"))
    return az, el, success, wedged


def _panel_single(
    ax,
    az: np.ndarray,
    el: np.ndarray,
    success: np.ndarray,
    wedged: np.ndarray,
    title: str,
    success_color: str,
    rows: list[list[dict]] | None = None,
) -> None:
    """One arm: equirectangular sphere unfolding, color-coded outcome."""
    code = np.zeros_like(success, dtype=int)  # 0 fail, 1 wedged, 2 success
    code[wedged] = 1
    code[success] = 2
    cmap = ListedColormap(
        [_OUTCOME_COLORS_SINGLE["FAIL"], _OUTCOME_COLORS_SINGLE["WEDGED"], success_color]
    )
    ax.imshow(
        code,
        extent=[az[0] - 15, az[-1] + 15, el[0] - 7.5, el[-1] + 7.5],
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=2,
        interpolation="nearest",
    )
    if rows is not None:
        for i, row in enumerate(rows):
            for j, c in enumerate(row):
                if c["ok"]:
                    init = c.get("mode", "?")[:1].upper() if c.get("mode", "-") != "-" else "✓"
                    ax.text(
                        az[j],
                        el[i],
                        init,
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white",
                        fontweight="bold",
                    )
    ax.axvline(0, color="white", lw=1.5, alpha=0.8)
    ax.text(
        0,
        el[-1] + 5,
        "↑ facing",
        ha="center",
        va="bottom",
        fontsize=8,
        color="white",
        fontweight="bold",
    )
    ax.axhline(0, color="white", lw=0.6, alpha=0.5, ls=":")
    ax.set_xticks(az)
    ax.set_yticks(el)
    ax.set_xlabel("azimuth (°)")
    ax.set_ylabel("elevation (°)")
    n_ok = int(success.sum())
    n_total = success.size
    n_w = int(wedged.sum())
    ax.set_title(
        f"{title} — {n_ok}/{n_total} ({100 * n_ok / n_total:.0f}%)"
        + (f", wedged={n_w}" if n_w else "")
    )


def _panel_combined(
    ax,
    az: np.ndarray,
    el: np.ndarray,
    left_succ: np.ndarray,
    right_succ: np.ndarray,
    left_w: np.ndarray,
    right_w: np.ndarray,
) -> None:
    """Both arms: left-only blue, right-only red, both purple, gray wedged."""
    code = np.zeros_like(left_succ, dtype=int)
    # 0 neither, 1 wedged, 2 left-only, 3 right-only, 4 both
    code[left_w | right_w] = 1
    code[left_succ & ~right_succ] = 2
    code[right_succ & ~left_succ] = 3
    code[left_succ & right_succ] = 4
    cmap = ListedColormap(
        [
            _ARM_COLORS["neither"],
            _ARM_COLORS["wedged"],
            _ARM_COLORS["left"],
            _ARM_COLORS["right"],
            _ARM_COLORS["both"],
        ]
    )
    ax.imshow(
        code,
        extent=[az[0] - 15, az[-1] + 15, el[0] - 7.5, el[-1] + 7.5],
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=4,
        interpolation="nearest",
    )
    n_el, n_az = left_succ.shape
    for i in range(n_el):
        for j in range(n_az):
            if left_succ[i, j] and right_succ[i, j]:
                tag = "L+R"
            elif left_succ[i, j]:
                tag = "L"
            elif right_succ[i, j]:
                tag = "R"
            else:
                continue
            ax.text(
                az[j],
                el[i],
                tag,
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )
    ax.axvline(0, color="white", lw=1.5, alpha=0.8)
    ax.text(
        0,
        el[-1] + 5,
        "↑ facing",
        ha="center",
        va="bottom",
        fontsize=8,
        color="white",
        fontweight="bold",
    )
    ax.axhline(0, color="white", lw=0.6, alpha=0.5, ls=":")
    ax.set_xticks(az)
    ax.set_yticks(el)
    ax.set_xlabel("azimuth from robot facing (°)")
    ax.set_ylabel("elevation (°)")
    either = left_succ | right_succ
    n_either = int(either.sum())
    n_total = either.size
    ax.set_title(
        f"Combined coverage (either arm) — {n_either}/{n_total} "
        f"({100 * n_either / n_total:.0f}%)\n"
        f"L={int(left_succ.sum())}  R={int(right_succ.sum())}  "
        f"both={int((left_succ & right_succ).sum())}  "
        f"only-L={int((left_succ & ~right_succ).sum())}  "
        f"only-R={int((right_succ & ~left_succ).sum())}"
    )


def _panel_per_el(ax, el: np.ndarray, left_succ: np.ndarray, right_succ: np.ndarray | None) -> None:
    rates_L = left_succ.mean(axis=1) * 100
    if right_succ is not None:
        rates_R = right_succ.mean(axis=1) * 100
        rates_E = (left_succ | right_succ).mean(axis=1) * 100
        width = 4
        ax.barh(el - width, rates_L, height=width, color=_ARM_COLORS["left"], label="left")
        ax.barh(el, rates_R, height=width, color=_ARM_COLORS["right"], label="right")
        ax.barh(el + width, rates_E, height=width, color=_ARM_COLORS["both"], label="either")
        ax.legend(loc="lower right", fontsize=8)
    else:
        ax.barh(el, rates_L, height=10, color=_ARM_COLORS["left"], label="success")
        for b, r in zip(ax.patches, rates_L, strict=False):
            ax.text(r + 1, b.get_y() + b.get_height() / 2, f"{r:.0f}%", va="center", fontsize=8)
    ax.set_xlim(0, 110)
    ax.set_xlabel("success rate (%)")
    ax.set_ylabel("elevation (°)")
    ax.set_yticks(el)
    ax.set_title("Per elevation")
    ax.grid(axis="x", alpha=0.3)


def _panel_per_az(ax, az: np.ndarray, left_succ: np.ndarray, right_succ: np.ndarray | None) -> None:
    rates_L = left_succ.mean(axis=0) * 100
    if right_succ is not None:
        rates_R = right_succ.mean(axis=0) * 100
        rates_E = (left_succ | right_succ).mean(axis=0) * 100
        width = 8
        ax.bar(az - width, rates_L, width=width, color=_ARM_COLORS["left"], label="left")
        ax.bar(az, rates_R, width=width, color=_ARM_COLORS["right"], label="right")
        ax.bar(az + width, rates_E, width=width, color=_ARM_COLORS["both"], label="either")
        ax.legend(loc="upper right", fontsize=8)
    else:
        ax.bar(az, rates_L, width=24, color=_ARM_COLORS["left"])
    ax.axvline(0, color="dimgray", lw=1, alpha=0.6)
    ax.text(0, 105, "facing", ha="center", fontsize=8, color="dimgray")
    ax.set_xlim(az[0] - 25, az[-1] + 25)
    ax.set_ylim(0, 110)
    ax.set_xlabel("azimuth (°)")
    ax.set_ylabel("success rate (%)")
    ax.set_xticks(az)
    ax.set_title("Per azimuth")
    ax.grid(axis="y", alpha=0.3)


def _render_dual(left_data: dict, right_data: dict, out_path: str) -> None:
    az, el, left_s, left_w = _build_grid(left_data["rows"])
    _, _, right_s, right_w = _build_grid(right_data["rows"])

    fig = plt.figure(figsize=(20, 13))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.2, 1], hspace=0.45, wspace=0.20)

    _panel_single(
        fig.add_subplot(gs[0, 0]),
        az,
        el,
        left_s,
        left_w,
        "LEFT arm",
        _ARM_COLORS["left"],
        rows=left_data["rows"],
    )
    _panel_single(
        fig.add_subplot(gs[0, 1]),
        az,
        el,
        right_s,
        right_w,
        "RIGHT arm",
        _ARM_COLORS["right"],
        rows=right_data["rows"],
    )
    _panel_combined(fig.add_subplot(gs[1, :]), az, el, left_s, right_s, left_w, right_w)
    _panel_per_el(fig.add_subplot(gs[2, 0]), el, left_s, right_s)
    _panel_per_az(fig.add_subplot(gs[2, 1]), az, left_s, right_s)

    args = left_data.get("args", {})
    fig.suptitle(
        f"Dual-arm pointing reachability — distance={args.get('distance', '?')}m  "
        f"reach={args.get('reach', '?')}m\n"
        f"Each arm sent home before every cell • sibling-arm collisions filtered",
        fontsize=14,
        y=0.995,
    )
    plt.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path}")


def _render_single(data: dict, out_path: str) -> None:
    az, el, succ, wedged = _build_grid(data["rows"])
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1], hspace=0.35, wspace=0.30)

    _panel_single(
        fig.add_subplot(gs[0, :]),
        az,
        el,
        succ,
        wedged,
        data.get("args", {}).get("arm", "arm"),
        _ARM_COLORS["left"],
        rows=data["rows"],
    )
    _panel_per_el(fig.add_subplot(gs[1, 0]), el, succ, None)
    _panel_per_az(fig.add_subplot(gs[1, 1]), az, succ, None)

    args = data.get("args", {})
    fig.suptitle(
        f"Pointing reachability — {args.get('arm', '?')}  "
        f"distance={args.get('distance', '?')}m  reach={args.get('reach', '?')}m",
        fontsize=14,
        y=0.995,
    )
    plt.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0] if __doc__ else "",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--left",
        required=True,
        help="JSON from eval_pointing_reachability (single-arm or 'left' role)",
    )
    ap.add_argument(
        "--right", default=None, help="optional second JSON; if given, renders dual-arm comparison"
    )
    ap.add_argument("--out", default="/tmp/pointing_reachability_heatmap.png")
    args = ap.parse_args()

    left_data = _load(args.left)
    if args.right:
        right_data = _load(args.right)
        _render_dual(left_data, right_data, args.out)
    else:
        _render_single(left_data, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
