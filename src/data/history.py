from dataclasses import dataclass, field
from typing import List
import numpy as np
from simulation.state import SimulationState

@dataclass
class Snapshot:
    hour: float
    air_c: np.ndarray
    surface_c: np.ndarray
    # rh_pct: np.ndarray    # restore once state.relative_humidity() exists


@dataclass
class Record: 
    hour: List[float] = field(default_factory=list)
    air_mean: List[float] = field(default_factory=list)
    air_max: List[float] = field(default_factory=list)
    air_min: List[float] = field(default_factory=list)
    surface_mean: List[float] = field(default_factory=list)
    surface_max: List[float] = field(default_factory=list)
    # rh_mean: List[float] = field(default_factory=list)
    # rh_max: List[float] = field(default_factory=list)
    # water_l_cumulative: List[float] = field(default_factory=list)
    # energy_kwh_cumulative: List[float] = field(default_factory=list)
    snapshots: List[Snapshot] = field(default_factory=list)

    def save(
            self, 
            state: SimulationState,
            hour: float
        ) -> None:
            # rh = state.relative_humidity()
            self.hour.append(hour)
            self.air_mean.append(float(np.mean(state.ta)))
            self.air_max.append(float(np.max(state.ta)))
            self.air_min.append(float(np.min(state.ta)))
            self.surface_mean.append(float(np.mean(state.ts)))
            self.surface_max.append(float(np.max(state.ts)))
            # self.rh_mean.append(float(np.mean(rh)))
            # self.rh_max.append(float(np.max(rh)))
            # self.water_l_cumulative.append(water_l)
            # self.energy_kwh_cumulative.append(energy_kwh)
            self.snapshots.append(
                Snapshot(hour, state.ta.copy(), state.ts.copy())
            )


    #TODO relative humidity 
    def __str__(self) -> str:
        """Human-readable summary of everything recorded so far."""
        if not self.hour:
            return "Record: empty (nothing recorded yet)"
        return "\n".join([
            "Record: {0} samples, {1:.2f} h -> {2:.2f} h".format(
                len(self.hour), self.hour[0], self.hour[-1]
            ),
            "  air temperature C      min {0:6.2f}  mean {1:6.2f}  max {2:6.2f}".format(
                min(self.air_min), float(np.mean(self.air_mean)), max(self.air_max)
            ),
            "  surface temperature C  mean {0:6.2f}  peak {1:6.2f}".format(
                float(np.mean(self.surface_mean)), max(self.surface_max)
            ),
            "  snapshots: {0}".format(len(self.snapshots)),
        ])




