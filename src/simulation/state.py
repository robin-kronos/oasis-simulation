"""Prognostic state of the domain."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from config import Parameters


@dataclass
class SimulationState:
    # Three n x n grids, taken from parameters that describe near-surface air and surface temperature
    # and humidity ratio per spatial point.
    ta: np.ndarray # air Tempratrue
    ts: np.ndarray # surface temprature
    w: np.ndarray  # Humidity

    @classmethod
    def initial(cls, params: Parameters) -> "SimulationState":
        n = params.n
        ta = np.full((n, n), 30.0, dtype=float)
        return cls(
            ta=ta,
            ts=ta + 3.0,                              # surface starts warmer
            w=np.full((n, n), 0.008, dtype=float),    # ~22 % RH at 30 C
        )

    # def relative_humidity(self) -> np.ndarray:
    #     """Relative humidity field (%) implied by the current air state."""
    #     return rh_from_humidity_ratio(self.ta, self.w)



    def copy(self) -> "SimulationState":
        return SimulationState(
            ta=self.ta.copy(), ts=self.ts.copy(), w=self.w.copy()
        )

## update the state Using euler
    def makestep(self, dta, dts, dw,dt):
        self.ta += dta * dt
        self.ts += dts * dt
        self.w += dw * dt
