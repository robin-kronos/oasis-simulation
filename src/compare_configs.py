"""Compare cooling-shelter configurations against an untreated baseline.

Six configurations on the same 80 x 80 m site, same Phoenix heatwave forcing,
measured over the same occupied zone - so the only thing that differs is what
is built on the ground.

Produces, for a presentation:
  compare_daily_max.png    daily maximum air temperature, run-long
  compare_day_night.png    daily maximum and daily minimum, stacked
  compare_maps.png         air temperature map per configuration at peak heat
  compare_cost.png         cooling achieved against water and power used
"""
import os
import pickle

import numpy as np

from components import (
    Canopy,
    Mister,
    Park,
    ReflectivePavement,
    WaterFeature,
)
from config import Parameters
from simulation import Simulation
from simulation.weather import PhoenixHeatwaveWeather

DAYS = 31
HOURS = DAYS * 24.0

# The zone people actually occupy. The same patch of ground in every
# configuration, including the baseline, so the comparison is like for like.
ZONE = dict(x_m=42.0, y_m=38.0, radius_m=14.0)

# Misting rate per unit area. 4e-6 (the hotspot.py default) turned out to
# add only 0.1 K over shade alone - too small to see. 5e-5 is a system that
# is actually sized to do work, and makes the water cost visible, which is
# the COL-1 vs NAT-3 trade the report has to argue.
MIST_RATE = 5e-5           # kg/(m2 s)

# Validated categorical palette (dataviz reference instance, light mode).
# Baseline is deliberately NOT a palette hue: it is the reference, drawn in
# neutral ink so the eye reads it as the line to beat.
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#dcdbd6"
SURFACE = "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def configurations():
    """Name -> component list. Ordered from doing nothing to doing everything."""
    return [
        ("Baseline: bare asphalt", []),
        ("Reflective pavement", [
            ReflectivePavement(x_m=40, y_m=40, radius_m=40, strength=0.55),
        ]),
        ("Trees and shade", [
            Canopy(x_m=16, y_m=20, radius_m=9, strength=0.80),
            Canopy(x_m=64, y_m=58, radius_m=9, strength=0.80),
            Canopy(x_m=20, y_m=42, radius_m=8, strength=0.80),
            Canopy(x_m=42, y_m=38, radius_m=14, strength=0.92),
        ]),
        ("Green and water", [
            Park(x_m=62, y_m=20, radius_m=11),
            Park(x_m=42, y_m=38, radius_m=14),
            WaterFeature(x_m=18, y_m=62, radius_m=7),
        ]),
        ("Shade and misting", [
            Canopy(x_m=16, y_m=20, radius_m=9, strength=0.80),
            Canopy(x_m=64, y_m=58, radius_m=9, strength=0.80),
            Canopy(x_m=20, y_m=42, radius_m=8, strength=0.80),
            Canopy(x_m=42, y_m=38, radius_m=14, strength=0.92),
            Mister(x_m=42, y_m=38, radius_m=11, strength=MIST_RATE),
        ]),
        ("Full cooling shelter", [
            ReflectivePavement(x_m=40, y_m=40, radius_m=40, strength=0.55),
            Park(x_m=62, y_m=20, radius_m=11),
            WaterFeature(x_m=18, y_m=62, radius_m=7),
            Canopy(x_m=16, y_m=20, radius_m=9, strength=0.80),
            Canopy(x_m=64, y_m=58, radius_m=9, strength=0.80),
            Canopy(x_m=20, y_m=42, radius_m=8, strength=0.80),
            Canopy(x_m=42, y_m=38, radius_m=14, strength=0.92),
            Mister(x_m=42, y_m=38, radius_m=11, strength=MIST_RATE),
        ]),
    ]


def zone_mask(params):
    n, dx = params.n, params.dx
    rows, cols = np.ogrid[:n, :n]
    y = (rows + 0.5) * dx
    x = (cols + 0.5) * dx
    return ((x - ZONE["x_m"]) ** 2 + (y - ZONE["y_m"]) ** 2
            <= ZONE["radius_m"] ** 2)


def daily_extremes(hours, values):
    hours = np.asarray(hours)
    day = ((hours - hours[0]) // 24).astype(int)
    highs, lows = [], []
    for d in range(day.max() + 1):
        sel = day == d
        if sel.any():
            highs.append(values[sel].max())
            lows.append(values[sel].min())
    return np.array(highs), np.array(lows)


def run_all(params):
    mask = zone_mask(params)
    results = []
    for name, components in configurations():
        sim = Simulation(
            params=params, components=components,
            weather=PhoenixHeatwaveWeather(),
        )
        record = sim.run(HOURS)
        air = np.array([s.air_c[mask].mean() for s in record.snapshots])
        surface = np.array([s.surface_c[mask].mean() for s in record.snapshots])
        highs, lows = daily_extremes(record.hour, air)
        s_highs, _ = daily_extremes(record.hour, surface)
        peak_snap = max(record.snapshots, key=lambda s: s.air_c[mask].mean())
        results.append(dict(
            name=name,
            daily_max=highs,
            daily_min=lows,
            surface_max=s_highs,
            peak_map=peak_snap.air_c,
            peak_hour=peak_snap.hour,
            water_l_year=sim.total_water_kg_s() * 86400.0 * 365.0,
            power_w=sim.total_power_w(),
            n_components=len(components),
        ))
        print("  %-26s daily max %5.2f C   daily min %5.2f C" % (
            name, highs.mean(), lows.mean()))
    return results


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9)


def colour_for(index):
    # Baseline is neutral ink; interventions take palette slots in fixed order.
    return INK_SOFT if index == 0 else SERIES[index - 1]


def _spread(values, minimum_gap):
    """Nudge labels apart so none overlaps its neighbour.

    svg has no collision avoidance: without this the lower four end-labels
    land on top of each other whenever the configurations converge.
    """
    order = np.argsort(values)
    placed = np.array(values, dtype=float)
    for a, b in zip(order[:-1], order[1:]):
        if placed[b] - placed[a] < minimum_gap:
            placed[b] = placed[a] + minimum_gap
    return placed


def plot_series(ax, results, key, title, ylabel):
    days = np.arange(1, len(results[0][key]) + 1)
    finals = [res[key][-1] for res in results]
    # Space labels against the PANEL's full range, not the spread of the final
    # values: at night several configurations land on the same number and a
    # gap derived from their spread would be nearly zero.
    everything = np.concatenate([res[key] for res in results])
    label_gap = 0.045 * (everything.max() - everything.min())
    label_y = _spread(finals, label_gap)

    for i, res in enumerate(results):
        is_baseline = i == 0
        ax.plot(
            days, res[key],
            color=colour_for(i),
            linewidth=2.4 if is_baseline else 2.0,
            linestyle="--" if is_baseline else "-",
            label=res["name"],
            zorder=3 if is_baseline else 4,
        )
        # Direct end-labels: the relief the palette's contrast WARN requires.
        ax.annotate(
            " {0:.1f}".format(res[key][-1]),
            xy=(days[-1], finals[i]),
            xytext=(days[-1] + 0.6, label_y[i]),
            color=INK if is_baseline else colour_for(i),
            fontsize=9, va="center", ha="left", fontweight="600",
            annotation_clip=False,
        )
    style_axes(ax)
    if title:
        ax.set_title(title, color=INK, fontsize=12, fontweight="600",
                     loc="left", pad=10)
    ax.set_xlabel("Day of the heatwave", color=INK_SOFT, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_SOFT, fontsize=10)
    ax.set_xlim(1, days[-1] + max(1.6, 0.09 * len(days)))


def figure_daily_max(results, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    best = min(results[1:], key=lambda r: r["daily_max"].mean())
    gap = results[0]["daily_max"].mean() - best["daily_max"].mean()
    plot_series(
        ax, results, "daily_max", None,
        "Daily maximum air temperature in the occupied zone (C)",
    )
    ax.legend(
        frameon=False, fontsize=9.5, labelcolor=INK_SOFT,
        loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=3,
    )
    fig.suptitle(
        "{0} holds the daily peak {1:.1f} K below bare asphalt".format(
            best["name"], gap),
        color=INK, fontsize=14, fontweight="600", x=0.006, ha="left",
    )
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def figure_day_night(results, path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True,
                             sharex=True)
    fig.patch.set_facecolor(SURFACE)
    plot_series(axes[0], results, "daily_max", None,
                "Daily maximum air (C)")
    axes[0].set_xlabel("")
    plot_series(axes[1], results, "daily_min", "Overnight minimum",
                "Daily minimum air (C)")
    axes[0].legend(frameon=False, fontsize=9.5, labelcolor=INK_SOFT,
                   loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=3)
    day_gap = max(results[0]["daily_max"].mean() - r["daily_max"].mean()
                  for r in results[1:])
    night_gap = max(results[0]["daily_min"].mean() - r["daily_min"].mean()
                    for r in results[1:])
    fig.suptitle(
        "Best case cuts the daytime peak by {0:.1f} K but the night by only "
        "{1:.1f} K".format(day_gap, night_gap),
        color=INK, fontsize=14, fontweight="600", x=0.006, ha="left",
    )
    axes[1].annotate(
        "Baseline, reflective pavement and trees coincide: none of them"
        " changes the overnight minimum",
        xy=(0.015, 0.93), xycoords="axes fraction",
        color=INK_SOFT, fontsize=9.5, va="top",
    )
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def figure_maps(results, params, path):
    import matplotlib.pyplot as plt

    extent = [0, params.n * params.dx, 0, params.n * params.dx]
    vmin = min(r["peak_map"].min() for r in results)
    vmax = max(r["peak_map"].max() for r in results)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, res in zip(axes.flat, results):
        im = ax.imshow(res["peak_map"], origin="lower", extent=extent,
                       cmap="inferno", vmin=vmin, vmax=vmax)
        ax.set_title(res["name"], color=INK, fontsize=10.5, fontweight="600")
        ax.set_xlabel("x (m)", color=INK_SOFT, fontsize=9)
        ax.set_ylabel("y (m)", color=INK_SOFT, fontsize=9)
        ax.tick_params(colors=INK_SOFT, labelsize=8)
    bar = fig.colorbar(im, ax=axes, shrink=0.85)
    bar.set_label("Air temperature (C)", color=INK_SOFT, fontsize=10)
    bar.ax.tick_params(colors=INK_SOFT, labelsize=8)
    fig.suptitle(
        "Air temperature at peak heat - one colour scale across all six",
        color=INK, fontsize=13, fontweight="600", x=0.01, ha="left",
    )
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def figure_cost(results, path):
    import matplotlib.pyplot as plt

    baseline = results[0]["daily_max"].mean()
    names = [r["name"] for r in results[1:]]
    cooling = [baseline - r["daily_max"].mean() for r in results[1:]]
    water = [r["water_l_year"] / 1000.0 for r in results[1:]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    y = np.arange(len(names))

    axes[0].barh(y, cooling, color=SERIES[:len(names)], height=0.6)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, color=INK, fontsize=9.5)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Reduction in daily maximum air (K)", color=INK_SOFT,
                       fontsize=10)
    axes[0].set_title("Cooling achieved", color=INK, fontsize=12,
                      fontweight="600", loc="left")
    for i, value in enumerate(cooling):
        axes[0].annotate(" {0:.2f} K".format(value), xy=(value, i),
                         va="center", ha="left", fontsize=9, color=INK)
    style_axes(axes[0])

    axes[1].barh(y, water, color=SERIES[:len(names)], height=0.6)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].set_xlabel("Water use (m3/year, continuous operation)",
                       color=INK_SOFT, fontsize=10)
    axes[1].set_title("Water cost", color=INK, fontsize=12,
                      fontweight="600", loc="left")
    for i, value in enumerate(water):
        axes[1].annotate(" {0:.0f}".format(value), xy=(value, i),
                         va="center", ha="left", fontsize=9, color=INK)
    style_axes(axes[1])

    fig.suptitle("Only misting costs water; the passive measures cost none",
                 color=INK, fontsize=13, fontweight="600", x=0.01, ha="left")
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def print_table(results):
    baseline = results[0]
    print("")
    print("%-26s %9s %9s %9s %12s %8s" % (
        "Configuration", "day max", "night min", "surf max", "water m3/yr", "power W"))
    print("-" * 80)
    for res in results:
        print("%-26s %8.2f %9.2f %9.2f %12.0f %8.0f" % (
            res["name"], res["daily_max"].mean(), res["daily_min"].mean(),
            res["surface_max"].mean(), res["water_l_year"] / 1000.0,
            res["power_w"]))
    print("-" * 80)
    print("%-26s %8s %9s %9s" % ("Reduction vs baseline", "", "", ""))
    for res in results[1:]:
        print("%-26s %+8.2f %+9.2f %+9.2f" % (
            res["name"],
            res["daily_max"].mean() - baseline["daily_max"].mean(),
            res["daily_min"].mean() - baseline["daily_min"].mean(),
            res["surface_max"].mean() - baseline["surface_max"].mean()))


CACHE = "compare_results.pkl"


if __name__ == "__main__":
    params = Parameters(n=40, dx=2.0, dt=60.0, output_dt=1800.0,
                        diffusion_dt=30.0)
    # Simulating six months of weather takes a quarter of an hour; redrawing
    # the figures should not. Delete the cache file to force a fresh run.
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as fh:
            results = pickle.load(fh)
        print("Loaded cached results from %s (delete it to re-simulate)" % CACHE)
    else:
        print("Running %d configurations over %d days..." % (
            len(configurations()), DAYS))
        results = run_all(params)
        with open(CACHE, "wb") as fh:
            pickle.dump(results, fh)
    print_table(results)

    figure_daily_max(results, "compare_daily_max.png")
    figure_day_night(results, "compare_day_night.png")
    figure_maps(results, params, "compare_maps.png")
    figure_cost(results, "compare_cost.png")
    print("")
    print("Wrote compare_daily_max.png, compare_day_night.png, "
          "compare_maps.png, compare_cost.png")
