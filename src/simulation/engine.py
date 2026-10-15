from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np
from config import Parameters

from . import Physics
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
        T_city, I = Weather.at_time(hour) #I is sun radation;Todo: implement Weather

        dta, dts, dw = Physics.calcDelta(T_city, I, self.state, self.map)
        self.state.makestep(dta,dts,dw, self.params.dt);

        pass

    def run(self,amountofhours) -> Record:
        """Integrate the full duration and return the final state."""
        n_steps = int(
            amountofhours * 3600.0 / self.params.dt
        )

        for i in range(n_steps):
            hour = self.hour_at(i)

            self.step(hour)

            self.record.save(
                self.state,
                hour + self.params.dt / 3600.0
            )

        return self.record
