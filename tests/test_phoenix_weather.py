"""The Phoenix forcing against the published climate record.

Reference figures are NWS Phoenix (Sky Harbor):
  - July normals, 1991-2020: average high 106.5 F (41.4 C),
    average low 84.5 F (84.5 F = 29.2 C), monthly mean 95.5 F (35.3 C)
  - July 2023, the warmest on record: average high 114.7 F (45.9 C),
    average low 90.8 F (32.7 C), monthly mean 102.7 F (39.3 C), with 31
    consecutive days at or above 110 F and 16 consecutive nights at or
    above 90 F.

These tests pin the forcing to those numbers. They say nothing about whether
the surface physics is right - only that the weather driving it is faithful.
"""
import numpy as np
import pytest

from simulation.weather import (
    PhoenixHeatwaveWeather,
    PhoenixSummerWeather,
    rh_from_dewpoint,
)

DAYS = 31


def daily_extremes(weather, days=DAYS):
    highs, lows = [], []
    for day in range(days):
        series = [
            weather.at(day * 24 + minute / 60.0).temperature_c
            for minute in range(0, 24 * 60, 10)
        ]
        highs.append(max(series))
        lows.append(min(series))
    return np.array(highs), np.array(lows)


# --- normal peak summer ---

def test_july_normal_high_matches_the_published_normal():
    highs, _ = daily_extremes(PhoenixSummerWeather())
    assert highs.mean() == pytest.approx(41.4, abs=0.1)


def test_july_normal_low_matches_the_published_normal():
    _, lows = daily_extremes(PhoenixSummerWeather())
    assert lows.mean() == pytest.approx(29.2, abs=0.1)


def test_july_normal_monthly_mean_matches():
    highs, lows = daily_extremes(PhoenixSummerWeather())
    assert ((highs + lows) / 2).mean() == pytest.approx(35.3, abs=0.1)


def test_day_to_day_anomalies_cancel_over_a_month():
    # Each month-long block is centred, so a 31-day run reproduces the
    # normals rather than whatever one random draw happened to be.
    w = PhoenixSummerWeather()
    offsets = [w._day_offset(d) for d in range(DAYS)]
    assert np.mean(offsets) == pytest.approx(0.0, abs=1e-9)
    assert np.std(offsets) > 0.5   # but days still differ from one another


def test_anomalies_are_reproducible_from_the_seed():
    a = daily_extremes(PhoenixSummerWeather(seed=3), days=5)[0]
    b = daily_extremes(PhoenixSummerWeather(seed=3), days=5)[0]
    c = daily_extremes(PhoenixSummerWeather(seed=4), days=5)[0]
    assert np.allclose(a, b)
    assert not np.allclose(a, c)


# --- heatwave ---

def test_heatwave_high_matches_july_2023():
    highs, _ = daily_extremes(PhoenixHeatwaveWeather())
    assert highs.mean() == pytest.approx(45.9, abs=0.5)


def test_heatwave_low_matches_july_2023():
    _, lows = daily_extremes(PhoenixHeatwaveWeather())
    assert lows.mean() == pytest.approx(32.7, abs=0.5)


def test_heatwave_nights_warm_less_than_days():
    # 2023 ran 4.5 K above the normal high but only 3.5 K above the low.
    normal_high, normal_low = daily_extremes(PhoenixSummerWeather())
    wave_high, wave_low = daily_extremes(PhoenixHeatwaveWeather())
    day_excess = wave_high.mean() - normal_high.mean()
    night_excess = wave_low.mean() - normal_low.mean()
    assert day_excess > night_excess > 0


def test_heatwave_holds_most_of_the_month_above_110f():
    highs, _ = daily_extremes(PhoenixHeatwaveWeather())
    # 2023 managed all 31; the modelled onset and break cost a couple of days.
    assert (highs >= 43.3).sum() >= 28


def test_heatwave_builds_and_breaks():
    w = PhoenixHeatwaveWeather()
    assert w.heat_envelope(0) < w.heat_envelope(10)
    assert w.heat_envelope(10) == pytest.approx(1.0)
    assert w.heat_envelope(40) == 0.0


# --- diurnal shape ---

def test_minimum_falls_at_sunrise_and_maximum_in_late_afternoon():
    w = PhoenixSummerWeather()
    hours = np.arange(0, 24, 1 / 6.0)
    temps = np.array([w.at(h).temperature_c for h in hours])
    assert hours[temps.argmin()] == pytest.approx(w.sunrise_hour, abs=0.5)
    assert hours[temps.argmax()] == pytest.approx(w.peak_temp_hour, abs=0.5)


def test_sun_is_up_only_between_sunrise_and_sunset():
    w = PhoenixSummerWeather()
    assert w.at(4.0).solar_w_m2 == 0.0
    assert w.at(21.0).solar_w_m2 == 0.0
    assert w.at(12.5).solar_w_m2 > 900.0


def test_solar_repeats_every_day_of_the_month():
    w = PhoenixSummerWeather()
    for day in range(DAYS):
        assert w.at(day * 24 + 12.5).solar_w_m2 == pytest.approx(
            w.at(12.5).solar_w_m2
        )


# --- humidity ---

def test_humidity_is_low_by_day_and_higher_at_night():
    # Phoenix runs about 20 % in the afternoon and 40 % overnight at a
    # mid-July dew point; RH swings because temperature does, not moisture.
    w = PhoenixSummerWeather()
    afternoon = w.at(16.0).rh_pct
    overnight = w.at(5.5).rh_pct
    assert 12.0 < afternoon < 28.0
    assert 30.0 < overnight < 50.0
    assert overnight > afternoon


def test_dewpoint_conversion_is_consistent():
    assert rh_from_dewpoint(25.0, 25.0) == pytest.approx(100.0, abs=0.5)
    assert rh_from_dewpoint(40.0, 14.1) < 25.0
    assert 0.0 <= rh_from_dewpoint(45.0, -10.0) <= 100.0


def test_heatwave_is_more_humid_than_a_normal_july():
    # Monsoon moisture rides along with the 2023-scale event.
    assert PhoenixHeatwaveWeather().dewpoint_c > PhoenixSummerWeather().dewpoint_c
