"""Shade structures: tree canopy or built shade sail."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import Component, StaticFields, register


@dataclass
class _ShadeBase(Component):
    """Intercepts a fraction of incoming solar over its footprint.

    `strength` is the peak coverage fraction (0-1). Overlapping shade takes the
    maximum rather than summing, so two canopies cannot block more than all of
    the sun.
    """

    def apply(self, fields: StaticFields, n: int, dx: float) -> None:
        k = self.footprint(n, dx)
        fields.shade = np.maximum(fields.shade, np.clip(self.strength * k, 0.0, 1.0))


@register
@dataclass
class Canopy(_ShadeBase):
    kind: str = "canopy"


@register
@dataclass
class Tree(_ShadeBase):
    """Alias kind retained for scenario files written against the prototype."""

    kind: str = "tree"
