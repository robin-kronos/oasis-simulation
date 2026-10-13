"""Synthetic diurnal forcing.

Illustrative hot-arid summer day for exercising the model. This is a smooth
analytic cycle, not measured data and not a forecast for any real location.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import WeatherState


@dataclass
class SyntheticDiurnalWeather:
    """Smooth daily cycle: warm overnight, hot afternoon, solar noon-ish peak."""

    mean_air_c: float = 36.0
    air_swing_c: float = 7.0          # +/- about the mean => ~29 to 43 C
    peak_solar_w_m2: float = 900.0
    mean_rh_pct: float = 22.0
    rh_swing_pct: float = 7.0         # anti-correlated with temperature
    rh_bounds: tuple = (8.0, 45.0)
    wind_m_s: float = 2.0
    sunrise_hour: float = 6.0
    sunset_hour: float = 19.0
    phase_offset_hour: float = 9.0    # coolest shortly after sunrise

    def at(self, hour: float) -> WeatherState:
        phase = 2.0 * np.pi * (hour - self.phase_offset_hour) / 24.0
        air_c = self.mean_air_c + self.air_swing_c * np.sin(phase)
        rh = self.mean_rh_pct - self.rh_swing_pct * np.sin(phase)
        rh = float(np.clip(rh, *self.rh_bounds))
        return WeatherState(
            air_c=float(air_c),
            solar_w_m2=self._solar(hour),
            rh_pct=rh,
            wind_m_s=self.wind_m_s,
        )

    def _solar(self, hour: float) -> float:
        """Half-sine between sunrise and sunset, zero at night."""
        hour_of_day = hour % 24.0
        if not self.sunrise_hour <= hour_of_day <= self.sunset_hour:
            return 0.0
        day_length = self.sunset_hour - self.sunrise_hour
        daylight = np.sin(np.pi * (hour_of_day - self.sunrise_hour) / day_length)
        return float(self.peak_solar_w_m2 * max(0.0, daylight))
