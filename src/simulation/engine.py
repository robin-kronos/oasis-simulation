from __future__ import annotations

from typing import Any, List, Optional, Sequence

from .config import Parameters
from .state import SimulationState

# TODO: replace with the real Component type once components/ exists.
Component = Any


class Simulation:
    def __init__(
        self,
        params: Optional[Parameters] = None,
        components: Optional[Sequence[Component]] = None,
    ):
        self.params = params or Parameters()
        self.components = list(components or [])
        self.state = SimulationState.initial(self.params)

    @property
    def steps(self) -> int:
        return int(self.params.duration_hours * 3600.0 / self.params.dt)

    def hour_at(self, index: int) -> float:
        return self.params.start_time + index * self.params.dt / 3600.0

    def step(self, hour: float) -> None:
        """Advance the state by one time step.

        Not implemented yet. This will eventually:
          1. ask the weather source for background conditions at `hour`
          2. compute fluxes from the current state
          3. turn those into tendencies (dts, dta, dw)
          4. apply field += dt * tendency to all three together
        """
        pass

    def run(self) -> SimulationState:
        """Integrate the full duration and return the final state."""
        for i in range(self.steps):
            self.step(self.hour_at(i))
        return self.state # TODO Return Historical data
