"""The intervention types that can be placed on a city map.

Each one writes into exactly the city-map fields it affects, and says in its
docstring how `strength` is read. Composition rules live here rather than in
the physics: shade takes the maximum across overlaps, misting accumulates.
"""
from dataclasses import dataclass

import numpy as np

from .base import Component, register


@register
@dataclass
class Canopy(Component):
    """Tree canopy or shade sail: intercepts a fraction of incoming solar.

    `strength` is the shade fraction, 0-1. Overlapping canopies take the
    maximum rather than summing, since nothing can block more than all the sun.
    """

    kind = "canopy"

    def apply(self, city_map, params):
        area = self.mask(params)
        shade = np.clip(self.strength, 0.0, 1.0)
        city_map.shade_fraction[area] = np.maximum(
            city_map.shade_fraction[area], shade
        )


@register
@dataclass
class Mister(Component):
    """Evaporative misting station.

    `strength` is the potential evaporation rate in kg/(m2 s). Overlapping
    misters add, because two stations really do emit more water than one.
    How much of it actually evaporates is decided by the physics, which stops
    at saturation.
    """

    kind = "mister"
    power_per_unit_w = 250.0

    def apply(self, city_map, params):
        city_map.mist_rate[self.mask(params)] += max(self.strength, 0.0)

    def power_w(self, params):
        return self.power_per_unit_w

    def water_kg_s(self, params):
        return max(self.strength, 0.0) * self.area_m2(params)


@register
@dataclass
class ReflectivePavement(Component):
    """Cool pavement: a high-albedo coating over the existing surface.

    `strength` is the replacement albedo, capped at 0.9 since no real surface
    reflects everything. The surface type is unchanged - it is still paving,
    it just reflects more.
    """

    kind = "reflective_pavement"

    def apply(self, city_map, params):
        area = self.mask(params)
        city_map.albedo[area] = min(max(self.strength, 0.0), 0.9)


@register
@dataclass
class Park(Component):
    """Planted ground: converts the surface to vegetation.

    This is the one intervention that changes the surface *type*, which
    switches the physics onto the Penman-Monteith evapotranspiration path.
    `strength` is the albedo of the planting, 0.23 for typical grass.
    """

    kind = "park"
    strength: float = 0.23

    def apply(self, city_map, params):
        area = self.mask(params)
        city_map.surfacetype[area] = params.Vegetation
        city_map.albedo[area] = np.clip(self.strength, 0.0, 0.9)


@register
@dataclass
class WaterFeature(Component):
    """Pond, fountain basin or deliberately wet paving.

    Becomes a wet surface, which evaporates with no stomatal resistance -
    the strongest evaporative cooling available here. `strength` is the
    albedo; open water is dark, around 0.06.
    """

    kind = "water_feature"
    strength: float = 0.06

    def apply(self, city_map, params):
        area = self.mask(params)
        city_map.surfacetype[area] = params.Wet_Surface
        city_map.albedo[area] = np.clip(self.strength, 0.0, 0.9)


@register
@dataclass
class GreenRoof(Component):
    """Roof-level planting.

    Shares the roof storage coefficient but evaporates like vegetation, so it
    is registered as vegetation with a roof-like albedo. `strength` is albedo.
    """

    kind = "green_roof"
    strength: float = 0.20

    def apply(self, city_map, params):
        area = self.mask(params)
        city_map.surfacetype[area] = params.Vegetation
        city_map.albedo[area] = np.clip(self.strength, 0.0, 0.9)
