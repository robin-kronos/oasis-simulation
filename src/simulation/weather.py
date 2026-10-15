"""Background weather forcing.

The engine needs one thing from weather: given an hour, the background state
above the domain. Anything with an `at(hour)` method can drive a run, so a
measured time series can replace the synthetic day without touching physics.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WeatherState:
    """Background conditions above the modelled surface at one instant."""

    temperature_c: float   # background air temperature
    solar_w_m2: float      # downward shortwave reaching the surface
    rh_pct: float          # background relative humidity, 0-100
    wind_m_s: float        # wind speed


@dataclass
class SyntheticDiurnalWeather:
    """Smooth analytic day: cool night, hot afternoon, solar between sunrise
    and sunset.

    Illustrative only - not measured data and not a forecast for anywhere.
    Hours beyond 24 wrap, so multi-day runs repeat the same cycle.
    """

    mean_temp_c: float = 30.0
    temp_swing_c: float = 7.0      # +/- about the mean
    peak_solar_w_m2: float = 700.0
    mean_rh_pct: float = 60.0
    rh_swing_pct: float = 20.0     # anti-correlated with temperature
    wind_m_s: float = 5.0
    sunrise_hour: float = 6.0
    sunset_hour: float = 18.0
    peak_temp_hour: float = 15.0

    def at(self, hour):
        """Background state at `hour` (hours since the start of the run)."""
        phase = 2.0 * np.pi * (hour - self.peak_temp_hour) / 24.0
        temperature = self.mean_temp_c + self.temp_swing_c * np.cos(phase)
        rh = self.mean_rh_pct - self.rh_swing_pct * np.cos(phase)
        return WeatherState(
            temperature_c=float(temperature),
            solar_w_m2=self._solar(hour),
            rh_pct=float(np.clip(rh, 0.0, 100.0)),
            wind_m_s=self.wind_m_s,
        )

    def _solar(self, hour):
        """Half-sine between sunrise and sunset, zero at night.

        Wraps on 24 h so day 2 of a multi-day run still gets a sunrise.
        """
        hour_of_day = hour % 24.0
        if not self.sunrise_hour <= hour_of_day <= self.sunset_hour:
            return 0.0
        day_length = self.sunset_hour - self.sunrise_hour
        daylight = np.sin(np.pi * (hour_of_day - self.sunrise_hour) / day_length)
        return float(self.peak_solar_w_m2 * max(0.0, daylight))


@dataclass
class ConstantWeather:
    """Fixed conditions, for isolating one mechanism in a test."""

    temperature_c: float = 30.0
    solar_w_m2: float = 700.0
    rh_pct: float = 40.0
    wind_m_s: float = 5.0

    def at(self, hour):
        return WeatherState(
            self.temperature_c, self.solar_w_m2, self.rh_pct, self.wind_m_s
        )


@dataclass
class HeatwaveWeather:
    """A synthetic day that gets hotter each day, for multi-day runs.

    Without a trend, identical forcing every day means nothing accumulates
    after the first day of spin-up.
    """

    base: SyntheticDiurnalWeather = None
    warming_per_day_c: float = 1.5

    def __post_init__(self):
        if self.base is None:
            self.base = SyntheticDiurnalWeather()

    def at(self, hour):
        state = self.base.at(hour)
        offset = self.warming_per_day_c * (hour // 24.0)
        return WeatherState(
            temperature_c=state.temperature_c + offset,
            solar_w_m2=state.solar_w_m2,
            rh_pct=state.rh_pct,
            wind_m_s=state.wind_m_s,
        )


def rh_from_dewpoint(temp_c, dewpoint_c):
    """Relative humidity (%) implied by a temperature and a dew point.

    Magnus form, the same approximation the physics uses for saturation.
    """
    def es(t):
        return 610.94 * np.exp(17.625 * t / (t + 243.04))
    return float(np.clip(100.0 * es(dewpoint_c) / es(temp_c), 0.0, 100.0))


@dataclass
class PhoenixSummerWeather:
    """Peak-summer Phoenix, built from NWS Sky Harbor July normals.

    Defaults are the 1991-2020 July normals: average high 106.5 F (41.4 C),
    average low 84.5 F (29.2 C). Humidity is carried as a dew point, which is
    roughly flat across a day while RH swings inversely with temperature -
    about 20 % in the afternoon and 40 % overnight, as Phoenix actually runs.

    `day_to_day_c` adds seeded day-to-day variation so a month-long run is not
    thirty identical days. The seed makes it reproducible.
    """

    mean_high_c: float = 41.4
    mean_low_c: float = 29.2
    dewpoint_c: float = 14.1        # July 2024 mean dew point, 57.3 F
    peak_solar_w_m2: float = 1000.0  # clear-sky GHI at solar noon
    sunrise_hour: float = 5.5
    sunset_hour: float = 19.5
    peak_temp_hour: float = 16.0
    wind_m_s: float = 3.0
    day_to_day_c: float = 1.8
    seed: int = 0
    peak_excess_high_c: float = 0.0
    peak_excess_low_c: float = 0.0

    ANOMALY_BLOCK_DAYS = 31  # anomalies sum to zero over each block
    _ANOMALY_HORIZON = 403   # 13 blocks

    def _day_offset(self, day):
        """Deterministic per-day warm/cool anomaly, zero-mean by construction.

        Drawing one sample per day leaves the month-mean off by whatever the
        draw happened to be, which would bias the run against the published
        normals. Drawing the whole series and centring it removes that.
        """
        if self.day_to_day_c <= 0:
            return 0.0
        if getattr(self, "_anomalies", None) is None:
            rng = np.random.RandomState(self.seed)
            block = self.ANOMALY_BLOCK_DAYS
            blocks = self._ANOMALY_HORIZON // block
            raw = rng.normal(0.0, self.day_to_day_c, (blocks, block))
            # Centre each month-long block, so any whole-month run reproduces
            # the published monthly normals rather than one draw's luck.
            self._anomalies = (raw - raw.mean(axis=1, keepdims=True)).ravel()
        return float(self._anomalies[int(day) % self._ANOMALY_HORIZON])

    def daily_extremes(self, day):
        """(high, low) in Celsius for a whole day, including any anomaly."""
        anomaly = self._day_offset(day)
        envelope = self.heat_envelope(day)
        return (
            self.mean_high_c + anomaly + self.peak_excess_high_c * envelope,
            self.mean_low_c + anomaly + self.peak_excess_low_c * envelope,
        )

    def heat_envelope(self, day):
        """0 to 1 strength of a heat event. Flat zero for a normal summer."""
        return 0.0

    def at(self, hour):
        day = int(hour // 24)
        high, low = self.daily_extremes(day)
        temperature = self._temperature(hour % 24.0, high, low, day)
        return WeatherState(
            temperature_c=temperature,
            solar_w_m2=self._solar(hour),
            rh_pct=rh_from_dewpoint(temperature, self.dewpoint_c),
            wind_m_s=self.wind_m_s,
        )

    def _temperature(self, hour_of_day, high, low, day):
        """Two-piece cosine: minimum at sunrise, maximum in late afternoon.

        A single cosine would put the minimum 12 h before the peak, which is
        around 04:00 - too early. Splitting the rise and the fall puts both
        turning points where they actually occur.
        """
        rise_start, peak = self.sunrise_hour, self.peak_temp_hour
        span = high - low
        if rise_start <= hour_of_day <= peak:
            frac = (hour_of_day - rise_start) / (peak - rise_start)
            return low + span * 0.5 * (1.0 - np.cos(np.pi * frac))
        # Falling limb, wrapping past midnight to the next sunrise.
        if hour_of_day > peak:
            elapsed = hour_of_day - peak
            next_low = self.daily_extremes(day + 1)[1]
        else:
            elapsed = hour_of_day + 24.0 - peak
            next_low = low
        total = 24.0 - (peak - rise_start)
        frac = elapsed / total
        return high - (high - next_low) * 0.5 * (1.0 - np.cos(np.pi * frac))

    def _solar(self, hour):
        hour_of_day = hour % 24.0
        if not self.sunrise_hour <= hour_of_day <= self.sunset_hour:
            return 0.0
        day_length = self.sunset_hour - self.sunrise_hour
        daylight = np.sin(np.pi * (hour_of_day - self.sunrise_hour) / day_length)
        return float(self.peak_solar_w_m2 * max(0.0, daylight))


@dataclass
class PhoenixHeatwaveWeather(PhoenixSummerWeather):
    """A Phoenix heatwave on the scale of July 2023.

    July 2023 averaged 114.7 F (45.9 C) high and 90.8 F (32.7 C) low, with 31
    consecutive days at or above 110 F (43.3 C) and 16 consecutive nights that
    never fell below 90 F (32.2 C).

    Rather than holding that level flat, the event builds over `onset_days`,
    holds for `peak_days`, then breaks - which is how a real ridge behaves.
    """

    # July 2023 ran 4.5 K above the normal high and 3.5 K above the normal low:
    # the nights warmed less than the days, which is what the record shows.
    peak_excess_high_c: float = 4.5
    peak_excess_low_c: float = 3.5
    onset_days: float = 3.0
    peak_days: float = 25.0       # 2023 held every one of 31 days over 110 F
    decay_days: float = 3.0
    dewpoint_c: float = 15.5      # monsoon moisture rising into the event
    day_to_day_c: float = 1.2     # a locked-in ridge varies less day to day

    def heat_envelope(self, day):
        """Builds, holds, then breaks - how a real ridge behaves."""
        if day < self.onset_days:
            frac = (day + 1.0) / self.onset_days
            return 0.5 * (1.0 - np.cos(np.pi * frac))
        if day < self.onset_days + self.peak_days:
            return 1.0
        past = day - (self.onset_days + self.peak_days)
        if past >= self.decay_days:
            return 0.0
        return 0.5 * (1.0 + np.cos(np.pi * past / self.decay_days))
