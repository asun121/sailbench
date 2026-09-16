"""
Generate a polar diagram for a boat configuration.

For each commanded True Wind Angle (TWA), sweeps sail trim to find maximum
steady-state boat speed. Prints a polar table and saves a plot to disk.

Heading is held by a PI+rate controller and every point is reported against
the TWA the boat actually achieved, not the one commanded. A proportional-only
controller cannot null the sail's constant yaw moment, so it settles with a
steady-state heading offset -- around 15 degrees on flingo_floty -- and filing
those points under the commanded TWA shifts the whole polar. Points where the
boat could not hold heading at all are dropped: that is the no-go zone.

Usage:
    python scripts/polar_diagram.py
    python scripts/polar_diagram.py --config flingo_floty.yaml
    python scripts/polar_diagram.py --twa-step 2 --sail-step 5
    python scripts/polar_diagram.py --output my_polar.png
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sailbench.models.model import State
from sailbench.sim.sailboat_hub import SailboatHub
from sailbench.solvers.rk4 import rk4_step

_CONFIG = "basic_sailbot.yaml"
_DT = 0.02
_WARMUP_STEPS = 600   # 12 s: let sail/rudder servos settle and speed converge
_MEASURE_STEPS = 400  # 8 s: average speed over this window
_KP = 40.0            # heading-hold P-gain (deg/rad); negated because positive rudder = starboard turn
_KI = 12.0            # heading-hold I-gain; nulls the sail's standing yaw moment
_KD = 3.0             # yaw-rate damping term
_INTEGRAL_CLAMP = 1.5    # rad*s: anti-windup bound on the integral term
_MAX_RUDDER_DEG = 35.0
_SPEED_FRACTION = 0.08   # of true wind speed: below this, the boat is drifting, not sailing
_SPEED_FLOOR = 0.10      # m/s: absolute floor for the same test
_MAX_HEADING_ERR_DEG = 10.0  # above this the boat never settled on the commanded heading


def _heading_error_rad(psi: tuple[float, float], target_rad: float) -> float:
    """Signed angle from current heading to target, in [-π, π]."""
    c, s = psi
    sin_err = c * math.sin(target_rad) - s * math.cos(target_rad)
    cos_err = c * math.cos(target_rad) + s * math.sin(target_rad)
    return math.atan2(sin_err, cos_err)


class SteadyState(NamedTuple):
    """What one fixed-heading, fixed-trim run settled to."""

    speed: float          # [m/s] mean speed over the measure window
    achieved_twa: float   # [deg] mean true wind angle the boat actually sailed
    heading_err: float    # [deg] mean |heading error|; large means it never settled
    leeway: float         # [deg] mean angle between heading and track


class PolarResult(NamedTuple):
    """One sweep's worth of polar data, one entry per commanded TWA."""

    twa_cmd: np.ndarray       # [deg] angle asked for
    twa_achieved: np.ndarray  # [deg] angle actually sailed -- plot against this
    speed: np.ndarray         # [m/s]
    best_sail: np.ndarray     # [deg] sheet limit that produced `speed`
    leeway: np.ndarray        # [deg]
    sailing: np.ndarray       # bool: held heading and made way; False = no-go


def _steady_state(
    heading_rad: float,
    sail_limit_rad: float,
    upwind_rad: float,
    warmup: int = _WARMUP_STEPS,
    measure: int = _MEASURE_STEPS,
) -> SteadyState:
    """
    Simulate at fixed heading and sail trim; return what it settled to.

    A fresh SailboatHub is created so servo state doesn't bleed between runs.

    Heading is held by a PI controller with yaw-rate damping. The integral term
    matters: the sail applies a constant yaw moment, and proportional control
    alone leaves a standing heading offset that silently mislabels the TWA of
    every point in the polar.
    """
    hub = SailboatHub(config_file=_CONFIG)
    state = State(
        x=0.0, y=0.0,
        psi=(math.cos(heading_rad), math.sin(heading_rad)),
        u=0.3, v=0.0, r=0.0,
    )
    integral = 0.0
    speeds: list[float] = []
    twas: list[float] = []
    errs: list[float] = []
    leeways: list[float] = []

    for i in range(warmup + measure):
        err = _heading_error_rad(state.psi, heading_rad)
        integral = float(np.clip(integral + err * _DT, -_INTEGRAL_CLAMP, _INTEGRAL_CLAMP))
        # Negated KP/KI: positive error (need to turn left) → negative rudder (turn left)
        drive = -_KP * err - _KI * integral + _KD * state.r
        rudder = float(np.clip(drive, -_MAX_RUDDER_DEG, _MAX_RUDDER_DEG))
        state = hub.step(state=state, dt=_DT, solver=rk4_step,
                         sail_angle=sail_limit_rad, rudder_angle=rudder)
        if i >= warmup:
            speeds.append(math.hypot(state.u, state.v))
            errs.append(abs(math.degrees(err)))
            twas.append(abs(math.degrees(_heading_error_rad(state.psi, upwind_rad))))
            leeways.append(math.degrees(math.atan2(state.v, state.u)))

    return SteadyState(
        speed=float(np.mean(speeds)),
        achieved_twa=float(np.mean(twas)),
        heading_err=float(np.mean(errs)),
        leeway=float(np.mean(leeways)),
    )


def compute_polar(
    twa_deg: np.ndarray,
    sail_deg: np.ndarray,
) -> PolarResult:
    """
    Sweep sail trim at each commanded TWA and keep the best run.

    For each TWA, sweeps all sail trim angles and keeps the maximum speed.
    Wind direction is read directly from the simulator config.

    ``achieved_twa_deg`` is the wind angle the boat actually sailed, which is
    what the polar must be plotted against. ``held_heading`` is False where the
    boat never settled on the commanded heading -- those points are not polar
    data, they are the no-go zone.
    """
    hub_ref = SailboatHub(config_file=_CONFIG)
    wind_dir_rad = math.radians(float(hub_ref.sail_cfg.get("wind_dir_deg", 90.0)))
    # upwind_rad: direction toward the wind source (opposite of wind-blows-to)
    upwind_rad = wind_dir_rad + math.pi

    n_twa = len(twa_deg)
    n_sail = len(sail_deg)
    total = n_twa * n_sail
    print(f"  {n_twa} TWA angles × {n_sail} sail angles = {total} simulations")

    speeds = np.zeros(n_twa)
    best_sail = np.zeros(n_twa)
    achieved = np.zeros(n_twa)
    leeway = np.zeros(n_twa)
    held = np.zeros(n_twa, dtype=bool)
    done = 0

    for i, twa in enumerate(twa_deg):
        heading = upwind_rad - math.radians(twa)  # port-tack convention (wind on left)
        best: SteadyState | None = None
        best_s = float(sail_deg[0])

        for s in sail_deg:
            res = _steady_state(heading, math.radians(float(s)), upwind_rad)
            done += 1
            # Only a run that held its heading is a candidate; fall back to the
            # fastest run so a no-go angle still reports something meaningful.
            better = (
                best is None
                or (res.heading_err <= _MAX_HEADING_ERR_DEG < best.heading_err)
                or (
                    (res.heading_err <= _MAX_HEADING_ERR_DEG)
                    == (best.heading_err <= _MAX_HEADING_ERR_DEG)
                    and res.speed > best.speed
                )
            )
            if better or best is None:
                best = res
                best_s = float(s)
            best_so_far = best.speed
            pct = 100.0 * done / total
            print(
                f"\r  [{pct:5.1f}%]  TWA={twa:5.1f}°  sail={s:5.1f}°  "
                f"spd={res.speed:.3f} m/s  err={res.heading_err:5.1f}°  "
                f"best={best_so_far:.3f} m/s",
                end="", flush=True,
            )

        if best is None:  # sail_deg is never empty, but keep the type honest
            msg = "sail trim sweep produced no runs"
            raise RuntimeError(msg)
        speeds[i] = best.speed
        best_sail[i] = best_s
        achieved[i] = best.achieved_twa
        leeway[i] = best.leeway
        held[i] = best.heading_err <= _MAX_HEADING_ERR_DEG

    print()
    return PolarResult(
        twa_cmd=np.asarray(twa_deg, dtype=float),
        twa_achieved=achieved,
        speed=speeds,
        best_sail=best_sail,
        leeway=leeway,
        sailing=held,
    )


def _sailing_threshold(wind_speed: float) -> float:
    """Speed below which the boat is drifting rather than sailing."""
    return max(_SPEED_FRACTION * wind_speed, _SPEED_FLOOR)


def find_no_go_deg(result: PolarResult, wind_speed: float) -> float:
    """
    Return the closest wind angle the boat can actually hold and sail.

    Two things must both be true. The heading controller has to settle -- a boat
    that falls off to a beam reach is not sailing the angle it was asked to
    sail. And the boat has to make real way: head to wind it will sit there
    quite stably at a tenth of a knot, which is not sailing either.
    """
    valid = result.sailing & (result.speed > _sailing_threshold(wind_speed))
    if not valid.any():
        return float(result.twa_achieved[-1])
    return float(np.min(result.twa_achieved[valid]))


def print_table(result: PolarResult) -> None:
    """Print speed and VMG against the wind angle the boat actually achieved."""
    twa_deg, achieved_twa = result.twa_cmd, result.twa_achieved
    speeds, best_sail, leeway, held = (
        result.speed, result.best_sail, result.leeway, result.sailing,
    )
    vmg = speeds * np.cos(np.radians(achieved_twa))
    masked = np.where(held, vmg, -np.inf)
    best_idx = int(np.argmax(masked)) if held.any() else int(np.argmax(vmg))

    print(
        f"\n  {'CmdTWA':>7}  {'AchTWA':>7}  {'Speed':>7}  {'VMG':>7}  "
        f"{'BestSail':>9}  {'Leeway':>7}  {'Sailing':>7}",
    )
    print("  " + "-" * 66)
    rows = zip(twa_deg, achieved_twa, speeds, best_sail, leeway)
    for i, (twa, ach, spd, sail, lee) in enumerate(rows):
        tag = "  <-- best VMG" if i == best_idx else ""
        flag = "yes" if held[i] else "NO"
        print(
            f"  {twa:7.1f}  {ach:7.1f}  {spd:7.3f}  {vmg[i]:7.3f}  "
            f"{sail:9.1f}  {lee:7.2f}  {flag:>7}{tag}",
        )
    if not held.all():
        print("\n  'Sailing' = NO: the boat either never settled on the commanded")
        print("  heading, or settled but made no real way. Those rows are the no-go")
        print("  zone, not polar points, and are not plotted.")


def plot_polar(result: PolarResult, no_go_deg: float, output: Path) -> None:
    """Plot the polar against achieved TWA, dropping angles the boat could not hold."""
    # Only points where the heading controller settled are polar data.
    held = result.sailing
    twa_deg = result.twa_achieved[held]
    speeds = result.speed[held]
    order = np.argsort(twa_deg)
    twa_deg, speeds = twa_deg[order], speeds[order]

    vmg = speeds * np.cos(np.radians(twa_deg))
    best_vmg_idx = int(np.argmax(vmg))
    best_vmg_twa = float(twa_deg[best_vmg_idx])
    best_vmg_spd = float(speeds[best_vmg_idx])
    wind_speed = SailboatHub(config_file=_CONFIG).sail_cfg.get("wind_speed", "?")

    # Mirror for port tack (symmetric hull → same speeds)
    twa_port = twa_deg[::-1][:-1]
    twa_full = np.concatenate([-twa_port, twa_deg])
    spd_full = np.concatenate([speeds[::-1][:-1], speeds])

    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(
        f"Polar Diagram — {_CONFIG}  |  Wind {wind_speed} m/s  |  "
        f"No-go ≈ ±{no_go_deg:.0f}°  |  Best upwind VMG at TWA ≈ {best_vmg_twa:.0f}°",
        fontsize=11,
    )

    # ── Polar (circular) subplot ──────────────────────────────────────────────
    ax1 = fig.add_subplot(1, 2, 1, polar=True)
    ax1.set_theta_zero_location("N")   # 0° (upwind) at top
    ax1.set_theta_direction(-1)        # clockwise, matching nautical convention

    ax1.plot(np.radians(twa_full), spd_full, "b-", lw=2, label="Boat speed")

    # VMG tangent line from origin to optimal upwind point
    ax1.plot(
        [0, math.radians(best_vmg_twa)],
        [0, best_vmg_spd],
        "g--", lw=1.5, alpha=0.8,
        label=f"VMG tangent (TWA={best_vmg_twa:.0f}°)",
    )
    ax1.plot([0, -math.radians(best_vmg_twa)], [0, best_vmg_spd], "g--", lw=1.5, alpha=0.8)

    # No-go zone shading
    r_ceil = float(np.max(spd_full)) * 1.15
    theta_ng = np.linspace(-math.radians(no_go_deg), math.radians(no_go_deg), 80)
    ax1.fill_between(theta_ng, 0, r_ceil, alpha=0.2, color="red",
                     label=f"No-go (±{no_go_deg:.0f}°)")

    ax1.set_title("Polar  (N = upwind, achieved TWA)", pad=15)
    ax1.legend(loc="lower right", fontsize=8)

    # ── Cartesian subplot ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(twa_deg, speeds, "b-o", ms=4, lw=2, label="Boat speed (m/s)")
    ax2.plot(twa_deg, vmg, "g--o", ms=4, lw=1.5, label="Upwind VMG (m/s)")
    ax2.axvline(no_go_deg, color="red", ls="--",
                label=f"No-go boundary ≈ {no_go_deg:.0f}°")
    ax2.axvspan(0, no_go_deg, alpha=0.12, color="red")
    ax2.axvline(best_vmg_twa, color="green", ls=":", lw=1.5,
                label=f"Best VMG angle ≈ {best_vmg_twa:.0f}°")
    ax2.set_xlabel("Achieved True Wind Angle (°)")
    ax2.set_ylabel("Speed (m/s)")
    ax2.set_title("Speed and VMG vs. True Wind Angle")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 180)
    ax2.set_ylim(bottom=0)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {output}")


def main() -> None:
    global _CONFIG  # noqa: PLW0603 - the sweep helpers read the active config

    p = argparse.ArgumentParser(description="Generate a sailing polar diagram")
    p.add_argument("--config", type=str, default=_CONFIG, metavar="YAML",
                   help=f"boat config under configs/ (default {_CONFIG})")
    p.add_argument("--twa-step", type=float, default=5.0, metavar="DEG",
                   help="TWA resolution in degrees (default 5)")
    p.add_argument("--sail-step", type=float, default=10.0, metavar="DEG",
                   help="Sail trim sweep step in degrees (default 10)")
    p.add_argument("--output", type=str, default="scripts/polar_diagram.png",
                   help="Output PNG path (default: scripts/polar_diagram.png)")
    args = p.parse_args()
    _CONFIG = args.config

    twa = np.arange(0.0, 181.0, args.twa_step)
    # Start sail at first non-zero step: sheet_limit=0 returns no sail force,
    # which produces an anomalous residual speed from hull/keel dynamics.
    sail = np.arange(args.sail_step, 86.0, args.sail_step)

    hub_ref = SailboatHub(config_file=_CONFIG)
    wind_speed = hub_ref.sail_cfg.get("wind_speed", "?")
    wind_dir = hub_ref.sail_cfg.get("wind_dir_deg", "?")
    print(f"Config  : {_CONFIG}")
    print(f"Wind    : {wind_speed} m/s toward {wind_dir}°")
    print(f"TWA     : 0° – 180°, step {args.twa_step:.0f}°")
    print(f"Sail    : 0° – 85°, step {args.sail_step:.0f}°")
    print()

    result = compute_polar(twa, sail)
    tws = float(wind_speed) if isinstance(wind_speed, (int, float)) else 0.0
    # A point is polar data only if the boat held heading AND made way.
    result = result._replace(
        sailing=result.sailing & (result.speed > _sailing_threshold(tws)),
    )
    no_go = find_no_go_deg(result, tws)

    vmg = np.where(
        result.sailing,
        result.speed * np.cos(np.radians(result.twa_achieved)),
        -np.inf,
    )
    best_vmg_twa = float(result.twa_achieved[int(np.argmax(vmg))])

    print_table(result)
    print(f"\nNo-go zone half-angle : ~{no_go:.0f}° (closest angle actually held)")
    print(f"Best upwind VMG angle : ~{best_vmg_twa:.0f}° achieved TWA")

    if not result.sailing.any():
        print("\nThe boat sailed no commanded heading; nothing to plot.")
        return
    plot_polar(result, no_go, Path(args.output))


if __name__ == "__main__":
    main()
