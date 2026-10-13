"""Project Oasis: a prototype 2-D urban cooling simulator.

Models how shade, misting and reflective pavement alter near-ground air
temperature, surface temperature and humidity over a small urban plot.

Prototype only: synthetic weather forcing and effective parameterizations.
Not a validated street-scale weather or thermal-comfort model.
"""
from .config import ConfigError, Parameters
from .components import Canopy, Component, Mister, ReflectivePavement, Tree
from .metrics import History, ResourceUse, Snapshot
from .scenarios import Scenario
from .scenarios import load as load_scenario
from .simulation.engine import Simulator
from .simulation.state import SimulationState
from .weather import SyntheticDiurnalWeather, WeatherState

__version__ = "0.2.0"

__all__ = [
    "Parameters", "ConfigError",
    "Component", "Canopy", "Tree", "Mister", "ReflectivePavement",
    "Simulator", "SimulationState",
    "History", "Snapshot", "ResourceUse",
    "Scenario", "load_scenario",
    "SyntheticDiurnalWeather", "WeatherState",
    "__version__",
]
