"""Grid operators and explicit-scheme stability helpers."""
from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np


def laplacian(a: np.ndarray, dx: float) -> np.ndarray:
    """5-point Laplacian with zero-normal-gradient (edge-replicated) boundaries."""
    p = np.pad(a, 1, mode="edge")
    return (p[1:-1, 2:] + p[1:-1, :-2] + p[2:, 1:-1] + p[:-2, 1:-1] - 4 * a) / dx ** 2


def cell_centres(n: int, dx: float) -> np.ndarray:
    """Cell-centre coordinates (m) along one axis."""
    return (np.arange(n) + 0.5) * dx


def radial_distance(n: int, dx: float, x_m: float, y_m: float) -> np.ndarray:
    """Distance (m) from (x_m, y_m) to every cell centre."""
    xs = cell_centres(n, dx)
    xx, yy = np.meshgrid(xs, xs)
    return np.sqrt((xx - x_m) ** 2 + (yy - y_m) ** 2)


def gaussian_kernel(n: int, dx: float, x_m: float, y_m: float, radius_m: float) -> np.ndarray:
    """Dimensionless influence field peaking at the centre, zero past the radius.

    Sigma is half the stated radius, so the footprint is practically zero by
    2.5 radii, where it is truncated to keep components local.
    """
    r = radial_distance(n, dx, x_m, y_m)
    sigma = max(radius_m / 2.0, dx / 2.0)
    k = np.exp(-0.5 * (r / sigma) ** 2)
    k[r > 2.5 * radius_m] = 0.0
    return k


def diffusion_stability_limit(dx: float, diffusivity: float) -> float:
    """Largest stable explicit time step (s) for 2-D diffusion: dx^2 / (4 D)."""
    if diffusivity <= 0.0:
        return float("inf")
    return dx ** 2 / (4.0 * diffusivity)


def check_explicit_stability(
    dt: float, dx: float, diffusivities: Iterable[Tuple[str, float]]
) -> List[str]:
    """Return a warning per term whose explicit diffusion step is unstable.

    An empty list means every supplied term satisfies dt <= dx^2 / (4 D).
    """
    warnings: List[str] = []
    for name, d in diffusivities:
        limit = diffusion_stability_limit(dx, d)
        if dt > limit:
            warnings.append(
                "{0}: dt={1:.1f} s exceeds the stability limit {2:.1f} s "
                "(D={3:g} m2/s, dx={4:g} m)".format(name, dt, limit, d, dx)
            )
    return warnings
