"""History recording and resource accounting."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List

import numpy as np

from .config import Parameters

if TYPE_CHECKING:  # avoids a cycle: simulation.engine imports this module
    from .simulation.state import SimulationState


@dataclass
class Snapshot:
    """Full field maps at one recorded instant."""

    hour: float
    air_c: np.ndarray
    rh_pct: np.ndarray
    surface_c: np.ndarray


@dataclass
class History:
    """Scalar time series plus periodic field snapshots."""

    hour: List[float] = field(default_factory=list)
    air_mean: List[float] = field(default_factory=list)
    air_max: List[float] = field(default_factory=list)
    air_min: List[float] = field(default_factory=list)
    surface_mean: List[float] = field(default_factory=list)
    surface_max: List[float] = field(default_factory=list)
    rh_mean: List[float] = field(default_factory=list)
    rh_max: List[float] = field(default_factory=list)
    water_l_cumulative: List[float] = field(default_factory=list)
    energy_kwh_cumulative: List[float] = field(default_factory=list)
    snapshots: List[Snapshot] = field(default_factory=list)

    def record(
        self,
        hour: float,
        state: "SimulationState",
        water_l: float,
        energy_kwh: float,
        snapshot: bool = True,
    ) -> None:
        rh = state.relative_humidity()
        self.hour.append(hour)
        self.air_mean.append(float(np.mean(state.ta)))
        self.air_max.append(float(np.max(state.ta)))
        self.air_min.append(float(np.min(state.ta)))
        self.surface_mean.append(float(np.mean(state.ts)))
        self.surface_max.append(float(np.max(state.ts)))
        self.rh_mean.append(float(np.mean(rh)))
        self.rh_max.append(float(np.max(rh)))
        self.water_l_cumulative.append(water_l)
        self.energy_kwh_cumulative.append(energy_kwh)
        if snapshot:
            self.snapshots.append(
                Snapshot(hour, state.ta.copy(), rh.copy(), state.ts.copy())
            )

    def snapshot_nearest(self, hour: float) -> Snapshot:
        """The recorded snapshot closest to the requested hour."""
        if not self.snapshots:
            raise ValueError("no snapshots were recorded")
        return min(self.snapshots, key=lambda s: abs(s.hour - hour))

    def to_dict(self) -> Dict[str, List[float]]:
        """Scalar series only, suitable for CSV or JSON export."""
        return {
            k: list(v)
            for k, v in vars(self).items()
            if k != "snapshots"
        }


@dataclass
class ResourceUse:
    """Cumulative consumption over a run."""

    water_litres: float
    energy_kwh: float
    peak_power_w: float

    def summary_lines(self) -> List[str]:
        return [
            "Water use:       {0:.1f} L".format(self.water_litres),
            "Electricity use: {0:.2f} kWh".format(self.energy_kwh),
            "Peak draw:       {0:.0f} W".format(self.peak_power_w),
        ]


class ResourceAccountant:
    """Integrates water and electricity use as the run proceeds."""

    def __init__(self, params: Parameters, mist_field: np.ndarray, power_w: float):
        self.params = params
        # Total evaporated water across the domain, kg/s (1 kg water ~ 1 L).
        self.water_kg_s = float(
            np.sum(mist_field * params.mist_water_kg_m2_s) * params.cell_area_m2
        )
        self.power_w = power_w
        self.elapsed_s = 0.0

    def advance(self, dt: float) -> None:
        self.elapsed_s += dt

    @property
    def water_litres(self) -> float:
        return self.water_kg_s * self.elapsed_s

    @property
    def energy_kwh(self) -> float:
        return self.power_w * self.elapsed_s / 3.6e6

    def totals(self) -> ResourceUse:
        return ResourceUse(self.water_litres, self.energy_kwh, self.power_w)
