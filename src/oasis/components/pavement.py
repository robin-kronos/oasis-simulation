"""High-albedo (cool) pavement treatment."""
from __future__ import annotations

from dataclasses import dataclass

from .base import Component, StaticFields, register


@register
@dataclass
class ReflectivePavement(Component):
    """Blends the surface toward a higher albedo over its footprint.

    `strength` is the replacement albedo at full footprint weight, capped at
    0.9 since no real surface reflects everything.
    """

    kind: str = "reflective_pavement"

    def apply(self, fields: StaticFields, n: int, dx: float) -> None:
        k = self.footprint(n, dx)
        target = min(max(self.strength, 0.0), 0.9)
        fields.albedo = fields.albedo * (1.0 - k) + target * k
