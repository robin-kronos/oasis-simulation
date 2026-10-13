"""Plotting of simulation output.

matplotlib is imported lazily so the package can be used headlessly (tests,
batch runs, CI) without the dependency installed.
"""
from __future__ import annotations

from typing import Optional

from .config import Parameters
from .metrics import History


def plot_results(
    params: Parameters,
    history: History,
    save_path: Optional[str] = "oasis_results.png",
    snapshot_hour: float = 14.0,
    show: bool = False,
):
    """Four-panel summary: air, RH and surface maps plus daily time series."""
    import matplotlib.pyplot as plt

    snap = history.snapshot_nearest(snapshot_hour)
    extent = [0, params.domain_size_m, 0, params.domain_size_m]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    panels = [
        (axes[0, 0], snap.air_c, "inferno",
         "Outdoor air temperature at {0:.1f} h (C)".format(snap.hour), None, None),
        (axes[0, 1], snap.rh_pct, "YlGnBu", "Relative humidity (%)", 0, 100),
        (axes[1, 0], snap.surface_c, "magma", "Surface temperature (C)", None, None),
    ]
    for ax, data, cmap, title, vmin, vmax in panels:
        im = ax.imshow(
            data, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax
        )
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        fig.colorbar(im, ax=ax, shrink=0.85)

    ts = axes[1, 1]
    ts.plot(history.hour, history.air_mean, label="Mean air temperature")
    ts.plot(history.hour, history.surface_mean, label="Mean surface temperature")
    ts.fill_between(
        history.hour, history.air_min, history.air_max, alpha=0.18,
        label="Air min-max across domain",
    )
    ts.set_xlabel("Hour of simulated day")
    ts.set_ylabel("Temperature (C)")
    ts.grid(True, alpha=0.3)
    ts.legend(fontsize=8)

    fig.suptitle("Project Oasis - prototype output (synthetic weather)", fontsize=14)
    if save_path:
        fig.savefig(save_path, dpi=160)
    if show:
        plt.show()
    return fig
