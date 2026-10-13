import numpy as np
import pytest

from oasis.components import Canopy, Mister, ReflectivePavement
from oasis.config import ConfigError, Parameters
from oasis.simulation.engine import Simulator
from oasis.weather import SyntheticDiurnalWeather, WeatherState


def short_params(**kwargs):
    base = dict(n=12, dx=2.0, dt=30.0, duration_hours=1.0, record_interval_s=600.0)
    base.update(kwargs)
    return Parameters(**base)


class ConstantWeather:
    """Fixed background conditions, for isolating a single mechanism."""

    def __init__(self, air_c=35.0, solar=800.0, rh=20.0, wind=2.0):
        self.state = WeatherState(air_c, solar, rh, wind)

    def at(self, hour):
        return self.state


# --- configuration ---

def test_invalid_parameters_are_rejected_at_construction():
    with pytest.raises(ConfigError):
        Simulator(short_params(n=1), [])


def test_unstable_time_step_warns_but_still_runs():
    with pytest.warns(RuntimeWarning, match="stability"):
        Simulator(short_params(dt=400.0, duration_hours=1.0), [])


# --- state and conservation ---

def test_state_starts_at_background_conditions():
    weather = ConstantWeather(air_c=35.0)
    sim = Simulator(short_params(), [], weather=weather)
    assert np.allclose(sim.state.ta, 35.0)
    assert np.allclose(sim.state.ts, 38.0)  # surface starts 3 K warmer


def test_fields_stay_finite_and_within_clamps_over_a_run():
    sim = Simulator(short_params(duration_hours=6.0), [
        Mister(x_m=12.0, y_m=12.0, radius_m=6.0, strength=2.0),
    ])
    sim.run()
    for field in (sim.state.ta, sim.state.ts, sim.state.w):
        assert np.all(np.isfinite(field))
    lo, hi = sim.params.air_temp_bounds
    assert np.all((sim.state.ta >= lo) & (sim.state.ta <= hi))
    assert np.all(sim.state.w >= 0.0)


def test_uniform_domain_with_no_components_stays_uniform():
    # Nothing breaks symmetry, so every cell must follow the same trajectory.
    sim = Simulator(short_params(duration_hours=3.0), [], weather=ConstantWeather())
    sim.run()
    assert sim.state.ta.std() == pytest.approx(0.0, abs=1e-12)
    assert sim.state.ts.std() == pytest.approx(0.0, abs=1e-12)


def test_run_without_components_uses_no_water_or_power():
    sim = Simulator(short_params(), [])
    _, resources = sim.run()
    assert resources.water_litres == 0.0
    assert resources.energy_kwh == 0.0


# --- physical behaviour ---

def test_shade_lowers_surface_temperature_under_the_canopy():
    weather = ConstantWeather(solar=900.0)
    shaded = Simulator(
        short_params(duration_hours=4.0),
        [Canopy(x_m=13.0, y_m=13.0, radius_m=6.0, strength=0.9)],
        weather=weather,
    )
    bare = Simulator(short_params(duration_hours=4.0), [], weather=weather)
    shaded.run()
    bare.run()
    centre = (6, 6)
    assert shaded.state.ts[centre] < bare.state.ts[centre]


def test_reflective_pavement_lowers_surface_temperature():
    weather = ConstantWeather(solar=900.0)
    cool = Simulator(
        short_params(duration_hours=4.0),
        [ReflectivePavement(x_m=13.0, y_m=13.0, radius_m=6.0, strength=0.7)],
        weather=weather,
    )
    bare = Simulator(short_params(duration_hours=4.0), [], weather=weather)
    cool.run()
    bare.run()
    assert cool.state.ts[6, 6] < bare.state.ts[6, 6]


def test_misting_cools_air_and_raises_humidity():
    weather = ConstantWeather(solar=900.0)
    misted = Simulator(
        short_params(duration_hours=4.0),
        [Mister(x_m=13.0, y_m=13.0, radius_m=6.0, strength=1.0)],
        weather=weather,
    )
    dry = Simulator(short_params(duration_hours=4.0), [], weather=weather)
    misted.run()
    dry.run()
    centre = (6, 6)
    assert misted.state.ta[centre] < dry.state.ta[centre]
    assert misted.state.w[centre] > dry.state.w[centre]


def test_no_solar_at_night_cools_the_surface():
    night = Simulator(
        short_params(duration_hours=4.0), [], weather=ConstantWeather(solar=0.0)
    )
    start = night.state.ts.mean()
    night.run()
    assert night.state.ts.mean() < start


# --- resources ---

def test_water_and_energy_scale_with_duration():
    one = Simulator(short_params(duration_hours=1.0),
                    [Mister(x_m=12.0, y_m=12.0, radius_m=6.0)])
    two = Simulator(short_params(duration_hours=2.0),
                    [Mister(x_m=12.0, y_m=12.0, radius_m=6.0)])
    _, r1 = one.run()
    _, r2 = two.run()
    assert r2.water_litres == pytest.approx(2.0 * r1.water_litres, rel=1e-9)
    assert r2.energy_kwh == pytest.approx(2.0 * r1.energy_kwh, rel=1e-9)


def test_two_misters_draw_twice_the_power():
    one = Simulator(short_params(), [Mister(x_m=10.0, y_m=10.0, radius_m=4.0)])
    two = Simulator(short_params(), [
        Mister(x_m=6.0, y_m=6.0, radius_m=4.0),
        Mister(x_m=18.0, y_m=18.0, radius_m=4.0),
    ])
    assert two.total_power_w() == pytest.approx(2.0 * one.total_power_w())


# --- history ---

def test_history_records_the_expected_number_of_samples():
    params = short_params(duration_hours=1.0, dt=30.0, record_interval_s=600.0)
    history, _ = Simulator(params, []).run()
    # 120 steps, sampled every 20 steps, plus the forced final sample.
    assert len(history.hour) == 7
    assert len(history.snapshots) == len(history.hour)
    assert history.hour[-1] == pytest.approx(params.duration_hours, abs=1e-6)


def test_snapshot_nearest_picks_the_closest_hour():
    history, _ = Simulator(short_params(duration_hours=2.0), []).run()
    snap = history.snapshot_nearest(1.0)
    assert abs(snap.hour - 1.0) <= 600.0 / 3600.0


def test_history_exports_only_scalar_series():
    history, _ = Simulator(short_params(), []).run()
    exported = history.to_dict()
    assert "snapshots" not in exported
    assert len(set(len(v) for v in exported.values())) == 1


# --- weather ---

def test_synthetic_weather_is_dark_at_night_and_bright_at_noon():
    weather = SyntheticDiurnalWeather()
    assert weather.at(2.0).solar_w_m2 == 0.0
    assert weather.at(23.0).solar_w_m2 == 0.0
    assert weather.at(12.5).solar_w_m2 > 800.0


def test_synthetic_weather_repeats_across_days():
    weather = SyntheticDiurnalWeather()
    for hour in (3.0, 9.0, 14.0, 20.0):
        today = weather.at(hour)
        tomorrow = weather.at(hour + 24.0)
        assert today.air_c == pytest.approx(tomorrow.air_c)
        assert today.solar_w_m2 == pytest.approx(tomorrow.solar_w_m2)
