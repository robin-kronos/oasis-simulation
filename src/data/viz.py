"""Plotting of simulation output.

matplotlib is imported lazily so the package can be used headlessly (tests,
batch runs, CI) without the dependency installed.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from config import Parameters

from .history import Record, Snapshot


def _extent(params: Parameters) -> List[float]:
    """Axis extent in metres for an n x n grid of dx-sized cells."""
    size = params.n * params.dx
    return [0.0, size, 0.0, size]


def _value_range(snapshots: List[Snapshot], attr: str) -> Tuple[float, float]:
    """Min and max of one field across every frame.

    Animations need a colour scale fixed across frames; letting each frame
    autoscale makes the colours mean something different every tick, which
    hides exactly the change you are trying to see.
    """
    values = [getattr(s, attr) for s in snapshots]
    return min(v.min() for v in values), max(v.max() for v in values)


def plot_results(
    params: Parameters,
    record: Record,
    save_path: Optional[str] = "oasis_results.png",
    snapshot_hour: float = 14.0,
    show: bool = False,
    create_gif: bool = False,
    gif_path: str = "oasis_run.gif",
    gif_interval: int = 1,
):
    """Air and surface maps at one hour, plus the daily time series.

    With create_gif, also writes an animation of the whole run.
    """
    import matplotlib.pyplot as plt

    snap = record.snapshot_nearest(snapshot_hour)
    extent = _extent(params)

    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    ax_air = fig.add_subplot(grid[0, 0])
    ax_surface = fig.add_subplot(grid[0, 1])
    ax_series = fig.add_subplot(grid[1, :])

    panels = [
        (ax_air, snap.air_c, "inferno",
         "Outdoor air temperature at {0:.1f} h (C)".format(snap.hour)),
        (ax_surface, snap.surface_c, "magma",
         "Surface temperature at {0:.1f} h (C)".format(snap.hour)),
    ]
    for ax, data, cmap, title in panels:
        im = ax.imshow(data, origin="lower", extent=extent, cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        fig.colorbar(im, ax=ax, shrink=0.85)

    ax_series.plot(record.hour, record.air_mean, label="Mean air temperature")
    ax_series.plot(record.hour, record.surface_mean, label="Mean surface temperature")
    ax_series.fill_between(
        record.hour, record.air_min, record.air_max, alpha=0.18,
        label="Air min-max across domain",
    )
    ax_series.set_xlabel("Hour of simulated day")
    ax_series.set_ylabel("Temperature (C)")
    ax_series.grid(True, alpha=0.3)
    ax_series.legend(fontsize=8)

    fig.suptitle("Project Oasis - prototype output (synthetic weather)", fontsize=14)
    if save_path:
        fig.savefig(save_path, dpi=160)
    if create_gif:
        make_gif(params, record, save_path=gif_path, interval=gif_interval)
    if show:
        plt.show()
    return fig


def make_gif(
    params: Parameters,
    record: Record,
    save_path: str = "oasis_run.gif",
    interval: int = 1,
    fps: int = 12,
):
    """Animate the air and surface fields over the whole run.

    `interval` takes every Nth snapshot. A full run records thousands of
    frames, so raise it rather than writing one frame per time step.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    frames = record.snapshots[::max(1, interval)]
    if not frames:
        raise ValueError("no snapshots were recorded; nothing to animate")

    extent = _extent(params)
    air_lo, air_hi = _value_range(frames, "air_c")
    surf_lo, surf_hi = _value_range(frames, "surface_c")

    fig, (ax_air, ax_surface) = plt.subplots(1, 2, figsize=(12, 5),
                                             constrained_layout=True)
    im_air = ax_air.imshow(frames[0].air_c, origin="lower", extent=extent,
                           cmap="inferno", vmin=air_lo, vmax=air_hi)
    im_surface = ax_surface.imshow(frames[0].surface_c, origin="lower",
                                   extent=extent, cmap="magma",
                                   vmin=surf_lo, vmax=surf_hi)
    for ax, im, label in ((ax_air, im_air, "Air temperature (C)"),
                          (ax_surface, im_surface, "Surface temperature (C)")):
        ax.set_title(label)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        fig.colorbar(im, ax=ax, shrink=0.85)
    title = fig.suptitle("")

    def draw(frame: Snapshot):
        im_air.set_data(frame.air_c)
        im_surface.set_data(frame.surface_c)
        title.set_text("Project Oasis - hour {0:.2f}".format(frame.hour))
        return im_air, im_surface, title

    animation = FuncAnimation(fig, draw, frames=frames, blit=False)
    animation.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return save_path
