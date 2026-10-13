"""Prognostic state of the domain."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..components.base import StaticFields
from ..config import Parameters
from ..psychrometrics import humidity_ratio_from_rh, rh_from_humidity_ratio
from ..weather.base import WeatherState


@dataclass
class SimulationState:
    """The three evolving fields plus the static maps they interact with.

    ta: near-ground air temperature (C)
    ts: surface temperature (C)
    w:  humidity ratio (kg water / kg dry air)
    """

    ta: np.ndarray
    ts: np.ndarray
    w: np.ndarray
    static: StaticFields

    @classmethod
    def initial(
        cls, params: Parameters, static: StaticFields, weather: WeatherState
    ) -> "SimulationState":
        """Start the domain at the background air state, surface slightly warmer."""
        n = params.n
        ta = np.full((n, n), weather.air_c, dtype=float)
        return cls(
            ta=ta,
            ts=ta + 3.0,
            w=np.full(
                (n, n),
                float(humidity_ratio_from_rh(weather.air_c, weather.rh_pct)),
                dtype=float,
            ),
            static=static,
        )

    def relative_humidity(self) -> np.ndarray:
        """Relative humidity field (%) implied by the current air state."""
        return rh_from_humidity_ratio(self.ta, self.w)

    def copy(self) -> "SimulationState":
        return SimulationState(
            ta=self.ta.copy(), ts=self.ts.copy(), w=self.w.copy(), static=self.static
        )
