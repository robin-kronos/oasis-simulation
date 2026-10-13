"""Evaporative misting station."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import Component, StaticFields, register


@register
@dataclass
class Mister(Component):
    """Adds evaporated water over its footprint, cooling air and surface.

    `strength` is an intensity multiplier (0-2) on the per-area water rate in
    `Parameters.mist_water_kg_m2_s`. Overlapping misters add, since two
    stations really do put out more water than one.
    """

    kind: str = "mister"

    def apply(self, fields: StaticFields, n: int, dx: float) -> None:
        k = self.footprint(n, dx)
        fields.mist += np.clip(self.strength, 0.0, 2.0) * k

    def power_w(self, params) -> float:
        return params.mister_power_w
