"""Named example scenarios and JSON scenario loading."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .components import Component, Canopy, Mister, ReflectivePavement, from_dict
from .config import ConfigError, Parameters


@dataclass
class Scenario:
    """A named parameter set plus the components placed in the domain."""

    name: str
    params: Parameters = field(default_factory=Parameters)
    components: List[Component] = field(default_factory=list)
    description: str = ""

    @classmethod
    def from_json(cls, path: str) -> "Scenario":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Scenario":
        if not isinstance(data, dict):
            raise ConfigError("scenario file must contain a JSON object")
        params = Parameters.from_dict(data.get("parameters", {}))
        entries = data.get("components", [])
        if not isinstance(entries, list):
            raise ConfigError("'components' must be a list")
        return cls(
            name=data.get("name", "unnamed"),
            params=params,
            components=[from_dict(e) for e in entries],
            description=data.get("description", ""),
        )


def baseline() -> Scenario:
    """Bare asphalt-like surface: the do-nothing reference case."""
    return Scenario(
        name="baseline",
        description="No interventions; reference for measuring any cooling effect.",
        components=[],
    )


def shade_only() -> Scenario:
    return Scenario(
        name="shade-only",
        description="A single canopy, no water or electricity use.",
        components=[Canopy(x_m=40, y_m=40, radius_m=12, strength=0.85)],
    )


def mist_only() -> Scenario:
    return Scenario(
        name="mist-only",
        description="A misting station with no shade above it.",
        components=[Mister(x_m=40, y_m=40, radius_m=8, strength=1.0)],
    )


def combined() -> Scenario:
    """The prototype's original demonstration configuration."""
    return Scenario(
        name="combined",
        description="Canopy and mister together, plus reflective pavement nearby.",
        components=[
            Canopy(x_m=40, y_m=40, radius_m=12, strength=0.85),
            Mister(x_m=40, y_m=40, radius_m=8, strength=1.0),
            ReflectivePavement(x_m=20, y_m=20, radius_m=10, strength=0.55),
        ],
    )


BUILTIN = {
    "baseline": baseline,
    "shade-only": shade_only,
    "mist-only": mist_only,
    "combined": combined,
}


def load(name: str) -> Scenario:
    try:
        return BUILTIN[name]()
    except KeyError:
        raise ConfigError(
            "unknown scenario {0!r}; available: {1}".format(
                name, ", ".join(sorted(BUILTIN))
            )
        )
