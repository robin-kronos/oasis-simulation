"""Moist-air property conversions.

All functions accept scalars or arrays; array inputs return arrays. Pressure
defaults to standard sea-level pressure because the prototype does not model
elevation.
"""
from __future__ import annotations

import numpy as np

STANDARD_PRESSURE_PA = 101_325.0
#: Ratio of the molar masses of water vapour and dry air.
MOLAR_MASS_RATIO = 0.622


def saturation_vapor_pressure_pa(temp_c):
    """Magnus approximation of saturation vapour pressure over liquid water."""
    return 610.94 * np.exp(17.625 * temp_c / (temp_c + 243.04))


def humidity_ratio_from_rh(temp_c, rh_pct, pressure_pa: float = STANDARD_PRESSURE_PA):
    """Humidity ratio (kg water / kg dry air) from temperature and RH."""
    e = (np.asarray(rh_pct) / 100.0) * saturation_vapor_pressure_pa(np.asarray(temp_c))
    return MOLAR_MASS_RATIO * e / (pressure_pa - e)


def rh_from_humidity_ratio(temp_c, w, pressure_pa: float = STANDARD_PRESSURE_PA):
    """Relative humidity (%) from temperature and humidity ratio, clipped to 0-100."""
    w = np.asarray(w)
    e = pressure_pa * w / (MOLAR_MASS_RATIO + w)
    rh = 100.0 * e / saturation_vapor_pressure_pa(np.asarray(temp_c))
    return np.clip(rh, 0.0, 100.0)


def vapor_pressure_pa(w, pressure_pa: float = STANDARD_PRESSURE_PA):
    """Partial pressure of water vapour (Pa) from humidity ratio."""
    w = np.asarray(w)
    return pressure_pa * w / (MOLAR_MASS_RATIO + w)


def dewpoint_c(temp_c, rh_pct):
    """Dew-point temperature, inverting the Magnus form used above."""
    rh = np.clip(np.asarray(rh_pct), 1e-3, 100.0)
    gamma = np.log(rh / 100.0) + 17.625 * np.asarray(temp_c) / (np.asarray(temp_c) + 243.04)
    return 243.04 * gamma / (17.625 - gamma)
