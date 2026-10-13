"""Time integration of the coupled surface / air / moisture system."""
from __future__ import annotations

import warnings
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..components.base import Component, StaticFields
from ..config import Parameters
from ..metrics import History, ResourceAccountant, ResourceUse
from ..weather.base import WeatherSource, WeatherState
from ..weather.synthetic import SyntheticDiurnalWeather
from . import physics
from .state import SimulationState


class Simulator:
    """Runs one scenario: parameters plus a set of placed components."""

    def __init__(
        self,
        params: Optional[Parameters] = None,
        components: Optional[Sequence[Component]] = None,
        weather: Optional[WeatherSource] = None,
        start_hour: float = 0.0,
    ):
        self.params = params or Parameters()
        self.params.validate()
        self.components: List[Component] = list(components or [])
        self.weather = weather or SyntheticDiurnalWeather()
        self.start_hour = start_hour

        for issue in self.params.stability_warnings():
            warnings.warn("Numerical stability - " + issue, RuntimeWarning)

        self.static = self._build_static_fields()
        self.state = SimulationState.initial(
            self.params, self.static, self.weather.at(start_hour)
        )
        self.accountant = ResourceAccountant(
            self.params, self.static.mist, self.total_power_w()
        )

    # --- setup ---

    def _build_static_fields(self) -> StaticFields:
        fields = StaticFields.blank(self.params.n, self.params.base_albedo)
        for component in self.components:
            component.validate(self.params.n, self.params.dx)
            component.apply(fields, self.params.n, self.params.dx)
        fields.shade = np.clip(fields.shade, 0.0, 1.0)
        fields.albedo = np.clip(fields.albedo, 0.0, 1.0)
        return fields

    def total_power_w(self) -> float:
        return float(sum(c.power_w(self.params) for c in self.components))

    # --- integration ---

    def step(self, hour: float) -> WeatherState:
        """Advance the state by one time step at the given simulated hour."""
        p = self.params
        bg = self.weather.at(hour)
        fluxes = physics.compute_fluxes(self.state, p, bg)

        # All tendencies are evaluated on the same old state, then applied
        # together, so the update does not depend on field ordering.
        dts = physics.surface_tendency(self.state, p, fluxes)
        dta = physics.air_tendency(self.state, p, bg, fluxes)
        dw = physics.moisture_tendency(self.state, p, bg, fluxes)

        self.state.ts = np.clip(self.state.ts + p.dt * dts, *p.surface_temp_bounds)
        self.state.ta = np.clip(self.state.ta + p.dt * dta, *p.air_temp_bounds)
        self.state.w = np.clip(self.state.w + p.dt * dw, *p.humidity_ratio_bounds)
        self.accountant.advance(p.dt)
        return bg

    def run(self, progress: bool = False) -> Tuple[History, ResourceUse]:
        """Integrate the full duration, recording history as configured."""
        p = self.params
        steps = p.steps
        record_every = max(1, int(round(p.record_interval_s / p.dt)))
        history = History()
        for i in range(steps):
            hour = self.start_hour + i * p.dt / 3600.0
            self.step(hour)
            if i % record_every == 0 or i == steps - 1:
                history.record(
                    hour + p.dt / 3600.0,
                    self.state,
                    self.accountant.water_litres,
                    self.accountant.energy_kwh,
                )
                if progress:
                    print(
                        "  t={0:5.2f} h  air {1:5.2f} C  surface {2:5.2f} C".format(
                            history.hour[-1], history.air_mean[-1],
                            history.surface_mean[-1],
                        )
                    )
        return history, self.accountant.totals()

    def describe(self) -> List[str]:
        p = self.params
        lines = [
            "Domain:   {0:.0f} m x {0:.0f} m at {1:.1f} m cells ({2} x {2})".format(
                p.domain_size_m, p.dx, p.n
            ),
            "Duration: {0:.1f} h at dt={1:.0f} s ({2} steps)".format(
                p.duration_hours, p.dt, p.steps
            ),
            "Components ({0}):".format(len(self.components)),
        ]
        lines += ["  - " + c.describe() for c in self.components] or ["  (none)"]
        return lines
