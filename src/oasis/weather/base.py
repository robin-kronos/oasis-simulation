"""Background weather forcing interface.

The simulator only needs one thing from weather: given an hour, return the
background state. Anything satisfying `WeatherSource` can drive a run, so a
measured time series can replace the synthetic day without touching physics.
"""
from __future__ import annotations

from dataclasses import dataclass

try:  # pragma: no cover - typing only
    from typing import Protocol
except ImportError:  # pragma: no cover - Python < 3.8
    Protocol = object  # type: ignore


@dataclass(frozen=True)
class WeatherState:
    """Background conditions above the modelled surface at one instant."""

    air_c: float
    solar_w_m2: float
    rh_pct: float
    wind_m_s: float


class WeatherSource(Protocol):
    """Anything that can report background weather at a simulated hour."""

    def at(self, hour: float) -> WeatherState:
        ...
