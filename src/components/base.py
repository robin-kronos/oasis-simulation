"""Component interface and registry.

A component is an intervention placed on the city map. Each one imprints
itself on the map's static fields once at setup; the physics never sees the
component again, only the fields it wrote. New intervention types subclass
`Component` and register a kind name, and nothing else needs to change.

Positions and radii are in metres from the lower-left corner of the domain.
"""
from dataclasses import dataclass

import numpy as np


class ComponentError(ValueError):
    """Raised for an unknown or misconfigured component."""


@dataclass
class Component:
    """Base intervention: a circular footprint centred on (x_m, y_m).

    `strength` is interpreted per subclass - a shade fraction, a replacement
    albedo, or a water rate.
    """

    x_m: float
    y_m: float
    radius_m: float
    strength: float = 1.0

    kind = "component"

    def validate(self, params):
        if self.radius_m <= 0:
            raise ComponentError(
                "{0} at ({1}, {2}): radius_m must be positive".format(
                    self.kind, self.x_m, self.y_m
                )
            )
        extent = params.n * params.dx
        if not (0.0 <= self.x_m <= extent and 0.0 <= self.y_m <= extent):
            raise ComponentError(
                "{0} at ({1}, {2}) lies outside the {3:.0f} m domain".format(
                    self.kind, self.x_m, self.y_m, extent
                )
            )

    def mask(self, params):
        """Boolean grid, True inside the footprint.

        Cell centres sit at (i + 0.5) * dx, so a component on a cell boundary
        is not silently snapped to one side.
        """
        n, dx = params.n, params.dx
        rows, cols = np.ogrid[:n, :n]
        y = (rows + 0.5) * dx
        x = (cols + 0.5) * dx
        return (x - self.x_m) ** 2 + (y - self.y_m) ** 2 <= self.radius_m ** 2

    def apply(self, city_map, params):
        """Imprint this component onto the city map."""
        raise NotImplementedError

    def power_w(self, params):
        """Steady electrical draw in watts."""
        return 0.0

    def water_kg_s(self, params):
        """Steady water draw in kg/s (1 kg of water is 1 litre)."""
        return 0.0

    def area_m2(self, params):
        return float(self.mask(params).sum()) * params.dx ** 2

    def describe(self):
        return "{0} at ({1:.0f}, {2:.0f}) m, r={3:.0f} m, strength={4:g}".format(
            self.kind, self.x_m, self.y_m, self.radius_m, self.strength
        )


_REGISTRY = {}


def register(cls):
    """Class decorator adding a component subclass to the registry."""
    _REGISTRY[cls.kind] = cls
    return cls


def available_kinds():
    return sorted(_REGISTRY)


def build(kind, **kwargs):
    """Construct a registered component by kind name."""
    try:
        cls = _REGISTRY[kind]
    except KeyError:
        raise ComponentError(
            "unknown component kind {0!r}; available: {1}".format(
                kind, ", ".join(available_kinds())
            )
        )
    try:
        return cls(**kwargs)
    except TypeError as exc:
        # Missing or misspelled fields in a scenario file land here.
        raise ComponentError(
            "could not build component {0!r}: {1}".format(kind, exc)
        )


def from_dict(data):
    """Build a component from a plain dict carrying a 'kind' key."""
    data = dict(data)
    try:
        kind = data.pop("kind")
    except KeyError:
        raise ComponentError("component entry is missing the 'kind' key")
    return build(kind, **data)
