"""Simulation parameters and scenario configuration.

Values are *effective* parameterizations for a prototype, not calibrated
street-scale physics. See README for the standing caveat.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Tuple

from .numerics import check_explicit_stability


class ConfigError(ValueError):
    """Raised when a configuration cannot produce a meaningful run."""


@dataclass
class Parameters:
    """Physical and numerical parameters for one run."""

    # --- Grid and time ---
    n: int = 40                               # cells per side (40 x 40)
    dx: float = 2.0                           # metres per cell
    dt: float = 30.0                          # seconds, explicit time step
    duration_hours: float = 24.0
    record_interval_s: float = 600.0          # history sampling interval

    # --- Surface energy balance ---
    surface_heat_capacity: float = 150_000.0  # J/(m2 K), effective slab
    surface_air_h: float = 10.0               # W/(m2 K), convective exchange
    surface_radiative_h: float = 5.5          # W/(m2 K), linearized longwave
    surface_diffusivity: float = 0.0005       # m2/s, lateral spreading
    base_albedo: float = 0.20
    solar_absorptivity: float = 0.80

    # --- Near-ground air layer ---
    air_diffusivity: float = 0.01             # m2/s, horizontal mixing
    moisture_diffusivity: float = 0.005       # m2/s, horizontal mixing
    air_density: float = 1.15                 # kg/m3
    air_cp: float = 1005.0                    # J/(kg K)
    air_mixing_height: float = 2.0            # m, represented layer depth
    ventilation_tau: float = 60.0             # s, relaxation to background

    # --- Misting ---
    mist_water_kg_m2_s: float = 1.0e-5        # evaporated water per unit mist
    latent_heat: float = 2.43e6               # J/kg
    mister_power_w: float = 250.0             # electrical draw per mister

    # --- Numerical safety clamps ---
    surface_temp_bounds: Tuple[float, float] = (-20.0, 90.0)
    air_temp_bounds: Tuple[float, float] = (-20.0, 65.0)
    humidity_ratio_bounds: Tuple[float, float] = (0.0, 0.05)

    @property
    def domain_size_m(self) -> float:
        return self.n * self.dx

    @property
    def cell_area_m2(self) -> float:
        return self.dx ** 2

    @property
    def steps(self) -> int:
        return int(self.duration_hours * 3600.0 / self.dt)

    @property
    def air_capacity_per_area(self) -> float:
        """Heat capacity of the represented air column, J/(m2 K)."""
        return self.air_density * self.air_cp * self.air_mixing_height

    @property
    def dry_air_mass_per_area(self) -> float:
        """Dry-air mass in the represented column, kg/m2."""
        return self.air_density * self.air_mixing_height

    def stability_warnings(self) -> List[str]:
        """Explicit-diffusion stability warnings, empty when the step is safe."""
        warnings = check_explicit_stability(
            self.dt,
            self.dx,
            [
                ("air temperature", self.air_diffusivity),
                ("moisture", self.moisture_diffusivity),
                ("surface temperature", self.surface_diffusivity),
            ],
        )
        if self.ventilation_tau > 0 and self.dt > 2.0 * self.ventilation_tau:
            warnings.append(
                "ventilation: dt={0:.1f} s exceeds 2*tau={1:.1f} s; the relaxation "
                "term will oscillate".format(self.dt, 2.0 * self.ventilation_tau)
            )
        return warnings

    def validate(self) -> None:
        """Raise ConfigError on values that cannot produce a meaningful run."""
        if self.n < 3:
            raise ConfigError("n must be at least 3 cells per side, got {0}".format(self.n))
        if self.dx <= 0:
            raise ConfigError("dx must be positive, got {0}".format(self.dx))
        if self.dt <= 0:
            raise ConfigError("dt must be positive, got {0}".format(self.dt))
        if self.duration_hours <= 0:
            raise ConfigError(
                "duration_hours must be positive, got {0}".format(self.duration_hours)
            )
        if self.steps < 1:
            raise ConfigError(
                "duration_hours={0} with dt={1} s yields no steps".format(
                    self.duration_hours, self.dt
                )
            )
        if not 0.0 <= self.base_albedo <= 1.0:
            raise ConfigError("base_albedo must be in [0, 1], got {0}".format(self.base_albedo))
        if self.air_mixing_height <= 0:
            raise ConfigError("air_mixing_height must be positive")
        if self.ventilation_tau <= 0:
            raise ConfigError("ventilation_tau must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Parameters":
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ConfigError(
                "unknown parameter(s): {0}".format(", ".join(sorted(unknown)))
            )
        return cls(**data)
