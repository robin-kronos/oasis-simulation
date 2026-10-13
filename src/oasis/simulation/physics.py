"""Tendency terms for the surface, air and moisture budgets.

Each function returns a rate of change per second; the engine does the time
integration. Splitting them out this way keeps every flux individually
inspectable and testable.

Caveat: these are effective bulk parameterizations over a single 2 m air
layer, not a resolved boundary-layer scheme.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import Parameters
from ..numerics import laplacian
from ..psychrometrics import humidity_ratio_from_rh
from ..weather.base import WeatherState
from .state import SimulationState


@dataclass
class FluxDiagnostics:
    """Per-cell fluxes for one step, in W/m2 (water_flux in kg/(m2 s))."""

    q_solar: np.ndarray
    q_surface_air: np.ndarray
    q_longwave: np.ndarray
    q_evap: np.ndarray
    water_flux: np.ndarray


def compute_fluxes(
    state: SimulationState, params: Parameters, weather: WeatherState
) -> FluxDiagnostics:
    """Instantaneous energy and water fluxes given the current state."""
    static = state.static
    # Shortwave actually absorbed by the surface, after shade and reflection.
    # NOTE: (1 - albedo) already is the absorbed fraction for an opaque
    # surface, so params.solar_absorptivity is deliberately NOT applied here;
    # using both would double-count. See README "Known issues".
    q_solar = (1.0 - static.albedo) * weather.solar_w_m2 * (1.0 - static.shade)
    # Convective exchange; positive warms the surface.
    q_surface_air = params.surface_air_h * (state.ta - state.ts)
    # Linearized net longwave loss to the background.
    q_longwave = params.surface_radiative_h * (weather.air_c - state.ts)
    # Evaporative cooling, limited by the configured mist water rate.
    water_flux = static.mist * params.mist_water_kg_m2_s
    q_evap = water_flux * params.latent_heat
    return FluxDiagnostics(q_solar, q_surface_air, q_longwave, q_evap, water_flux)


def surface_tendency(
    state: SimulationState, params: Parameters, fluxes: FluxDiagnostics
) -> np.ndarray:
    """dTs/dt (K/s): energy balance over the effective slab plus lateral spread."""
    net = fluxes.q_solar + fluxes.q_surface_air + fluxes.q_longwave
    return net / params.surface_heat_capacity + params.surface_diffusivity * laplacian(
        state.ts, params.dx
    )


def air_tendency(
    state: SimulationState,
    params: Parameters,
    weather: WeatherState,
    fluxes: FluxDiagnostics,
) -> np.ndarray:
    """dTa/dt (K/s): surface exchange, ventilation, evaporation, mixing."""
    capacity = params.air_capacity_per_area
    q_from_surface = -fluxes.q_surface_air
    q_vent = capacity * (weather.air_c - state.ta) / params.ventilation_tau
    net = q_from_surface + q_vent - fluxes.q_evap
    return net / capacity + params.air_diffusivity * laplacian(state.ta, params.dx)


def moisture_tendency(
    state: SimulationState,
    params: Parameters,
    weather: WeatherState,
    fluxes: FluxDiagnostics,
) -> np.ndarray:
    """dw/dt (1/s): evaporated water in, ventilation toward background, mixing."""
    w_bg = float(humidity_ratio_from_rh(weather.air_c, weather.rh_pct))
    source = fluxes.water_flux / params.dry_air_mass_per_area
    vent = (w_bg - state.w) / params.ventilation_tau
    return source + vent + params.moisture_diffusivity * laplacian(state.w, params.dx)
