"""Time integration of the coupled surface / air / moisture system."""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from config import Parameters
from data.history import Record

from . import Physics
from .Physics import diffuse_air
from .map.citymap import CityMap
from .state import SimulationState
from .weather import SyntheticDiurnalWeather


class Simulation:
    """Runs one scenario: parameters, a city map, and a weather source.

    Components are placed on the map at construction. Pass either a ready
    `map`, or `components` and let the simulation build the map from them.
    """

    def __init__(
        self,
        params: Optional[Parameters] = None,
        components: Optional[Sequence[object]] = None,
        map: Optional[CityMap] = None,
        weather: Optional[object] = None,
    ):
        self.params = params or Parameters()
        self.components = list(components or [])
        self.weather = weather or SyntheticDiurnalWeather()
        self.record = Record()

        if map is None:
            self.map = CityMap.initial(self.params, self.components)
        else:
            # A map was supplied: place any components on top of it.
            self.map = map.place(self.components, self.params)

        self.state = SimulationState.initial(self.params)
        self._check_diffusion_stability()

    # --- setup ---

    def _check_diffusion_stability(self):
        p = self.params
        limit = 0.25 * p.dx ** 2 / p.K_air if p.K_air > 0 else float("inf")
        if p.diffusion_dt > limit:
            raise ValueError(
                "diffusion_dt={0:g} s exceeds the stability limit {1:.1f} s "
                "for K_air={2:g} m2/s at dx={3:g} m".format(
                    p.diffusion_dt, limit, p.K_air, p.dx
                )
            )

    def total_power_w(self) -> float:
        return float(sum(c.power_w(self.params) for c in self.components))

    def total_water_kg_s(self) -> float:
        return float(sum(c.water_kg_s(self.params) for c in self.components))

    @property
    def steps(self) -> int:
        return int(self.params.duration_hours * 3600.0 / self.params.dt)

    def hour_at(self, index: int) -> float:
        return self.params.start_time + index * self.params.dt / 3600.0

    # --- integration ---

    def step(self, hour: float) -> None:
        """Advance the state by one time step, by operator splitting.

        Order: surface and air tendencies, then misting, then horizontal
        diffusion (sub-stepped for stability), then ventilation toward the
        background. Each operator sees the state the previous one left.
        """
        p = self.params
        physics_dt = p.dt
        diffusion_dt = p.diffusion_dt
        n_diffusion = max(1, int(round(physics_dt / diffusion_dt)))

        background = self.weather.at(hour)

        dta, dts, dw = Physics.calcDelta(
            background.temperature_c,
            background.solar_w_m2,
            self.state,
            self.map,
            windspeed=background.wind_m_s,
        )
        self.state.makestep(dta, dts, dw, physics_dt)

        Physics.apply_misting(
            state=self.state,
            city_map=self.map,
            dt=physics_dt,
            h=Physics.h,
        )

        # Air temperature and humidity both diffuse, over the same sub-steps.
        for _ in range(n_diffusion):
            self.state.ta = diffuse_air(
                ta=self.state.ta, dx=p.dx, dt=diffusion_dt, K=p.K_air
            )
            self.state.w = diffuse_air(
                ta=self.state.w, dx=p.dx, dt=diffusion_dt, K=p.K_air
            )

        Physics.apply_ventilation(
            state=self.state,
            background_temp=background.temperature_c,
            background_w=p.background_w,
            dt=physics_dt,
            tau=p.ventilation_tau,
        )

    def run(self, amountofhours: Optional[float] = None) -> Record:
        """Integrate and return the record. Defaults to params.duration_hours."""
        p = self.params
        hours = p.duration_hours if amountofhours is None else amountofhours
        self.record = Record()

        n_steps = int(round(hours * 3600.0 / p.dt))
        steps_per_output = int(round(p.output_dt / p.dt))

        if not np.isclose(n_steps * p.dt, hours * 3600.0):
            raise ValueError("Duration must be divisible by dt")
        if not np.isclose(steps_per_output * p.dt, p.output_dt):
            raise ValueError("Output interval must be divisible by dt")

        for i in range(n_steps):
            hour = self.hour_at(i)
            self.step(hour)
            # Record the hour AFTER the step: that is the state being saved.
            if (i + 1) % steps_per_output == 0:
                self.record.save(self.state, hour + p.dt / 3600.0)

        return self.record

    def describe(self) -> List[str]:
        p = self.params
        lines = [
            "Domain:   {0:.0f} m x {0:.0f} m at {1:.1f} m cells ({2} x {2})".format(
                p.n * p.dx, p.dx, p.n
            ),
            "Weather:  {0}".format(type(self.weather).__name__),
            "Surfaces: {0}".format(
                ", ".join(
                    "{0} {1}".format(v, k)
                    for k, v in sorted(self.map.surface_counts(p).items())
                )
            ),
            "Components ({0}):".format(len(self.components)),
        ]
        lines += ["  - " + c.describe() for c in self.components] or ["  (none)"]
        lines.append(
            "Draw:     {0:.0f} W, {1:.4f} kg/s water".format(
                self.total_power_w(), self.total_water_kg_s()
            )
        )
        return lines
