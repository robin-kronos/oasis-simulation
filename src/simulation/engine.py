from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np
from config import Parameters

from . import Physics
from .Physics import diffuse_air
from .state import SimulationState
from data.history import Record

from .weather import Weather

# TODO: replace with the real Component type once components/ exists.
Component = Any


class Simulation:
    def __init__(
        self,
        params: Optional[Parameters] = None,
        components: Optional[Sequence[Component]] = None,
        map: Optional[np.ndarray] = None,
        weather: Optional[Weather] = None,
    ):
        self.params = params or Parameters()
        self.components = list(components or [])
        self.record = Record()
        self.state = SimulationState.initial(self.params)
        self.map = map
        self.weather = weather


    @property
    def steps(self) -> int:
        return int(self.params.duration_hours * 3600.0 / self.params.dt)

    def hour_at(self, index: int) -> float:
        return self.params.start_time + index * self.params.dt / 3600.0

    def step(self, hour: float,) -> None:
        """Advance the state by one time step.

        Not implemented yet. This will eventually:
          1. ask the weather source for background conditions at `hour`
          2. compute fluxes from the current state
          3. turn those into tendencies (dts, dta, dw)
          4. apply field += dt * tendency to all three together


          ToDo: steps every minute/second for beter accuracy
        """
        physics_dt = self.params.dt
        diffusion_dt = self.params.diffusion_dt

        n_diffusion = round(physics_dt / diffusion_dt)

        T_city, I = Weather.at_time(hour) #I is sun radation;Todo: implement Weather

        dta, dts, dw = Physics.calcDelta(T_city, I, self.state, self.map)
        self.state.makestep(dta,dts,dw, self.params.dt);

        Physics.apply_misting(
            state=self.state,
            city_map=self.map,
            dt=physics_dt,
            h=10.0)


        for _ in range(n_diffusion):
            self.state.ta = diffuse_air(
                ta=self.state.ta,
                dx=self.params.dx,
                dt=diffusion_dt,
                K=self.params.K_air
            )
        self.state.w = diffuse_air(
            self.state.w,
            dx=self.params.dx,
            dt=diffusion_dt,
            K=self.params.K_air
        )

        Physics.apply_ventilation(
            state=self.state,
            background_temp=T_city,
            background_w=self.params.background_w,
            dt=physics_dt,
            tau=self.params.ventilation_tau
        )

        pass

    def run(self,amountofhours) -> Record:
        """Integrate the full duration and return the final state."""
        # n_steps = int(
        #     amountofhours * 3600.0 / self.params.dt
        # )
        #
        # for i in range(n_steps):
        #     hour = self.hour_at(i)
        #
        #     self.step(hour)
        #
        #     self.record.save(
        #         self.state,
        #         hour + self.params.dt / 3600.0
        #     )
        #
        # return self.record
        self.record = Record()

        dt = self.params.dt
        output_dt = self.params.output_dt

        n_steps = round(amountofhours * 3600.0 / dt)
        steps_per_output = round(output_dt / dt)

        if not np.isclose(n_steps * dt, amountofhours * 3600.0):
            raise ValueError("Duration must be divisible by dt")

        if not np.isclose(steps_per_output * dt, output_dt):
            raise ValueError("Output interval must be divisible by dt")

        for i in range(n_steps):

            hour = self.hour_at(i)

            # Calculate physics and diffusion
            self.step(hour)

            # Save only at the end of each hour
            if (i + 1) % steps_per_output == 0:
                save_hour = (
                        hour + dt / 3600.0
                )

                self.record.save(
                    self.state,
                    save_hour
                )

        return self.record
