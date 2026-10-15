"""The static map the physics runs over.

Four grids describing what is physically on the ground. They are written once
at setup, by the components, and never change during a run - which is what
separates them from the three evolving fields in SimulationState.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class CityMap:
    surfacetype: np.ndarray     # surface code per cell: asphalt, vegetation, ...
    albedo: np.ndarray          # 0-1 reflectance
    shade_fraction: np.ndarray  # 0-1 fraction of solar intercepted above
    mist_rate: np.ndarray       # potential evaporation, kg/(m2 s)

    def __post_init__(self):
        shape = self.surfacetype.shape
        for name in ("albedo", "shade_fraction", "mist_rate"):
            if getattr(self, name).shape != shape:
                raise ValueError(
                    "{0} has shape {1}, expected {2}".format(
                        name, getattr(self, name).shape, shape
                    )
                )

    @classmethod
    def initial(cls, params, components=None):
        """A bare asphalt domain, with any components imprinted on it."""
        n = params.n
        city_map = cls(
            surfacetype=np.full((n, n), params.Asphalt, dtype=int),
            albedo=np.full((n, n), 0.12, dtype=float),
            shade_fraction=np.zeros((n, n), dtype=float),
            mist_rate=np.zeros((n, n), dtype=float),
        )
        city_map.place(components or [], params)
        return city_map

    def place(self, components, params):
        """Imprint components onto the map, in the order given.

        Later components win where they overwrite - a park laid over paving
        replaces it - so order is meaningful for surface type and albedo.
        """
        for component in components:
            component.validate(params)
            component.apply(self, params)
        np.clip(self.shade_fraction, 0.0, 1.0, out=self.shade_fraction)
        np.clip(self.albedo, 0.0, 1.0, out=self.albedo)
        return self

    def surface_counts(self, params):
        """Cell count per surface type, for a run summary."""
        names = {
            params.Asphalt: "asphalt",
            params.Vegetation: "vegetation",
            params.Wet_Surface: "wet surface",
            params.Roof: "roof",
        }
        counts = {}
        for code, name in names.items():
            cells = int((self.surfacetype == code).sum())
            if cells:
                counts[name] = cells
        return counts
