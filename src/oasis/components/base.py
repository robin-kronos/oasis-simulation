"""Component interface and registry.

A component is an intervention placed in the domain. Each one imprints itself
on the static field maps (shade, albedo, mist) once at setup, and may report a
steady electrical draw. New intervention types subclass `Component` and
register a kind name; nothing else in the package needs to change.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Type

import numpy as np

from ..numerics import gaussian_kernel


class ComponentError(ValueError):
    """Raised for an unknown or misconfigured component."""


@dataclass
class Component:
    """Base intervention placed at (x_m, y_m) with a Gaussian footprint.

    `strength` is interpreted per subclass: a shade coverage fraction, a
    misting intensity multiplier, or a replacement albedo.
    """

    x_m: float
    y_m: float
    radius_m: float
    strength: float = 1.0

    #: Registry key; set by subclasses.
    kind: str = "component"

    def validate(self, n: int, dx: float) -> None:
        if self.radius_m <= 0:
            raise ComponentError(
                "{0} at ({1}, {2}): radius_m must be positive".format(
                    self.kind, self.x_m, self.y_m
                )
            )
        extent = n * dx
        if not (0.0 <= self.x_m <= extent and 0.0 <= self.y_m <= extent):
            raise ComponentError(
                "{0} at ({1}, {2}) lies outside the {3:.0f} m domain".format(
                    self.kind, self.x_m, self.y_m, extent
                )
            )

    def footprint(self, n: int, dx: float) -> np.ndarray:
        """Dimensionless 0-1 influence field over the grid."""
        return gaussian_kernel(n, dx, self.x_m, self.y_m, self.radius_m)

    def apply(self, fields: "StaticFields", n: int, dx: float) -> None:
        """Imprint this component onto the static field maps."""
        raise NotImplementedError

    def power_w(self, params) -> float:
        """Steady electrical draw in watts."""
        return 0.0

    def describe(self) -> str:
        return "{0} at ({1:.0f}, {2:.0f}) m, r={3:.0f} m, strength={4:.2f}".format(
            self.kind, self.x_m, self.y_m, self.radius_m, self.strength
        )


@dataclass
class StaticFields:
    """Time-invariant maps the components write into."""

    shade: np.ndarray       # 0-1 fraction of solar intercepted
    albedo: np.ndarray      # 0-1 surface reflectance
    mist: np.ndarray        # dimensionless misting intensity multiplier

    @classmethod
    def blank(cls, n: int, base_albedo: float) -> "StaticFields":
        return cls(
            shade=np.zeros((n, n), dtype=float),
            albedo=np.full((n, n), base_albedo, dtype=float),
            mist=np.zeros((n, n), dtype=float),
        )


_REGISTRY: Dict[str, Type[Component]] = {}


def register(cls: Type[Component]) -> Type[Component]:
    """Class decorator adding a component subclass to the registry."""
    _REGISTRY[cls.kind] = cls
    return cls


def available_kinds():
    return sorted(_REGISTRY)


def build(kind: str, **kwargs) -> Component:
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
        # Missing or misspelled fields in a scenario file reach us here.
        raise ComponentError(
            "could not build component {0!r}: {1}".format(kind, exc)
        )


def from_dict(data: Dict[str, Any]) -> Component:
    """Build a component from a plain dict with a 'kind' key."""
    data = dict(data)
    try:
        kind = data.pop("kind")
    except KeyError:
        raise ComponentError("component entry is missing the 'kind' key")
    return build(kind, **data)
