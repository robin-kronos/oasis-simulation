"""Tests for the city-map simulation (src/simulation, src/components)."""
import numpy as np
import pytest

import simulation.Physics as Physics
from components import (
    Canopy,
    ComponentError,
    GreenRoof,
    Mister,
    Park,
    ReflectivePavement,
    WaterFeature,
    available_kinds,
    build,
    from_dict,
)
from config import Parameters
from simulation import Simulation
from simulation.map.citymap import CityMap
from simulation.weather import (
    ConstantWeather,
    HeatwaveWeather,
    SyntheticDiurnalWeather,
)


def short(**kwargs):
    base = dict(n=12, dx=5.0, dt=60.0, duration_hours=2.0, output_dt=3600.0)
    base.update(kwargs)
    return Parameters(**base)


CENTRE = 30.0  # 12 cells x 5 m -> a 60 m domain


# --- components ---

def test_registry_lists_every_intervention():
    assert set(available_kinds()) == {
        "canopy", "mister", "reflective_pavement",
        "park", "water_feature", "green_roof",
    }


def test_unknown_kind_is_rejected():
    with pytest.raises(ComponentError):
        build("teleporter", x_m=1, y_m=1, radius_m=1)


def test_missing_field_is_a_component_error_not_a_typeerror():
    with pytest.raises(ComponentError):
        build("mister")


def test_from_dict_requires_a_kind():
    with pytest.raises(ComponentError):
        from_dict({"x_m": 1, "y_m": 1, "radius_m": 1})


def test_component_outside_the_domain_is_rejected():
    with pytest.raises(ComponentError):
        Canopy(x_m=500.0, y_m=30.0, radius_m=5.0).validate(short())


def test_non_positive_radius_is_rejected():
    with pytest.raises(ComponentError):
        Canopy(x_m=30.0, y_m=30.0, radius_m=0.0).validate(short())


def test_canopy_writes_shade_only_inside_its_footprint():
    p = short()
    m = CityMap.initial(p, [Canopy(x_m=CENTRE, y_m=CENTRE, radius_m=10, strength=0.8)])
    assert m.shade_fraction.max() == pytest.approx(0.8)
    assert m.shade_fraction[0, 0] == 0.0


def test_overlapping_canopies_take_the_maximum():
    p = short()
    m = CityMap.initial(p, [
        Canopy(x_m=CENTRE, y_m=CENTRE, radius_m=10, strength=0.6),
        Canopy(x_m=CENTRE, y_m=CENTRE, radius_m=10, strength=0.6),
    ])
    # Two canopies cannot block 120 percent of the sun.
    assert m.shade_fraction.max() == pytest.approx(0.6)


def test_overlapping_misters_accumulate():
    p = short()
    one = CityMap.initial(p, [Mister(x_m=CENTRE, y_m=CENTRE, radius_m=10, strength=1e-5)])
    two = CityMap.initial(p, [
        Mister(x_m=CENTRE, y_m=CENTRE, radius_m=10, strength=1e-5),
        Mister(x_m=CENTRE, y_m=CENTRE, radius_m=10, strength=1e-5),
    ])
    assert two.mist_rate.max() == pytest.approx(2 * one.mist_rate.max())


def test_park_and_water_change_the_surface_type():
    p = short()
    m = CityMap.initial(p, [
        Park(x_m=15.0, y_m=15.0, radius_m=8),
        WaterFeature(x_m=45.0, y_m=45.0, radius_m=8),
    ])
    counts = m.surface_counts(p)
    assert counts["vegetation"] > 0
    assert counts["wet surface"] > 0
    assert counts["asphalt"] > 0


def test_reflective_pavement_raises_albedo_but_keeps_the_surface_type():
    p = short()
    m = CityMap.initial(p, [
        ReflectivePavement(x_m=CENTRE, y_m=CENTRE, radius_m=10, strength=0.55)
    ])
    assert m.albedo.max() == pytest.approx(0.55)
    assert m.surface_counts(p) == {"asphalt": p.n * p.n}


def test_albedo_is_capped_below_one():
    p = short()
    m = CityMap.initial(p, [
        ReflectivePavement(x_m=CENTRE, y_m=CENTRE, radius_m=10, strength=5.0)
    ])
    assert m.albedo.max() <= 0.9


def test_later_components_win_where_they_overlap():
    p = short()
    m = CityMap.initial(p, [
        ReflectivePavement(x_m=CENTRE, y_m=CENTRE, radius_m=12, strength=0.55),
        Park(x_m=CENTRE, y_m=CENTRE, radius_m=12),
    ])
    i = p.n // 2
    assert m.surfacetype[i, i] == p.Vegetation


def test_only_misters_draw_power_and_water():
    p = short()
    mister = Mister(x_m=CENTRE, y_m=CENTRE, radius_m=10, strength=1e-5)
    assert mister.power_w(p) > 0
    assert mister.water_kg_s(p) > 0
    assert Canopy(x_m=CENTRE, y_m=CENTRE, radius_m=10).power_w(p) == 0.0
    assert Park(x_m=CENTRE, y_m=CENTRE, radius_m=10).water_kg_s(p) == 0.0


def test_city_map_rejects_mismatched_field_shapes():
    with pytest.raises(ValueError):
        CityMap(
            surfacetype=np.zeros((4, 4), dtype=int),
            albedo=np.zeros((3, 3)),
            shade_fraction=np.zeros((4, 4)),
            mist_rate=np.zeros((4, 4)),
        )


# --- weather ---

def test_synthetic_weather_is_dark_at_night_and_bright_at_noon():
    w = SyntheticDiurnalWeather()
    assert w.at(2.0).solar_w_m2 == 0.0
    assert w.at(23.0).solar_w_m2 == 0.0
    assert w.at(12.0).solar_w_m2 > 600.0


def test_synthetic_weather_repeats_across_days():
    w = SyntheticDiurnalWeather()
    for hour in (3.0, 9.0, 14.0, 20.0):
        assert w.at(hour).solar_w_m2 == pytest.approx(w.at(hour + 24.0).solar_w_m2)
        assert w.at(hour).temperature_c == pytest.approx(
            w.at(hour + 24.0).temperature_c
        )


def test_weather_reports_all_four_quantities():
    state = SyntheticDiurnalWeather().at(12.0)
    assert state.temperature_c > 0
    assert state.solar_w_m2 > 0
    assert 0.0 <= state.rh_pct <= 100.0
    assert state.wind_m_s > 0


def test_heatwave_weather_warms_day_on_day():
    hw = HeatwaveWeather(warming_per_day_c=2.0)
    assert hw.at(15.0 + 24.0).temperature_c == pytest.approx(
        hw.at(15.0).temperature_c + 2.0
    )


def test_wind_speed_reaches_the_physics():
    # Faster wind lowers aerodynamic resistance over vegetation, changing the
    # sensible heat flux. If wind were ignored these two would match.
    p = short()
    m = CityMap.initial(p, [Park(x_m=CENTRE, y_m=CENTRE, radius_m=25)])
    sim = Simulation(params=p, map=m, weather=ConstantWeather())
    slow = Physics.calcDelta(30.0, 700.0, sim.state, m, windspeed=1.0)
    fast = Physics.calcDelta(30.0, 700.0, sim.state, m, windspeed=10.0)
    assert not np.allclose(slow[0], fast[0])


# --- physics ---

def test_vectorised_physics_matches_the_cell_by_cell_reference():
    rng = np.random.RandomState(3)
    n = 10
    m = CityMap(
        surfacetype=rng.randint(0, 4, (n, n)),
        albedo=rng.uniform(0.05, 0.6, (n, n)),
        shade_fraction=rng.uniform(0.0, 1.0, (n, n)),
        mist_rate=np.zeros((n, n)),
    )

    class State(object):
        pass

    s = State()
    s.ta = rng.uniform(15.0, 45.0, (n, n))
    s.ts = rng.uniform(15.0, 70.0, (n, n))
    s.w = rng.uniform(0.004, 0.02, (n, n))
    fast = Physics.calcDelta(28.0, 650.0, s, m, windspeed=4.0)
    slow = Physics.calcDelta_reference(28.0, 650.0, s, m, windspeed=4.0)
    for a, b in zip(fast, slow):
        assert np.allclose(a, b, rtol=0, atol=1e-18)


def test_diffusion_rejects_an_unstable_timestep():
    with pytest.raises(ValueError, match="stability"):
        Physics.diffuse_air(np.zeros((5, 5)), dx=1.0, dt=100.0, K=1.0)


def test_diffusion_conserves_total_heat():
    rng = np.random.RandomState(1)
    field = rng.rand(8, 8)
    out = Physics.diffuse_air(field, dx=2.0, dt=2.0, K=0.01)
    assert out.sum() == pytest.approx(field.sum())


def test_misting_stops_at_saturation():
    # An absurd mist rate must not drive humidity past saturation.
    p = short()
    m = CityMap.initial(p, [Mister(x_m=CENTRE, y_m=CENTRE, radius_m=25, strength=1.0)])
    sim = Simulation(params=p, map=m, weather=ConstantWeather())
    Physics.apply_misting(sim.state, m, dt=600.0, h=Physics.h)
    saturated = Physics.saturation_humidity_ratio(sim.state.ta)
    assert np.all(sim.state.w <= saturated + 1e-9)


# --- engine ---

def test_simulation_runs_and_records_the_expected_samples():
    sim = Simulation(params=short(duration_hours=3.0), weather=ConstantWeather())
    record = sim.run(3.0)
    assert len(record.hour) == 3           # output_dt is one hour
    assert len(record.snapshots) == 3
    assert record.hour[-1] == pytest.approx(3.0)


def test_run_defaults_to_the_configured_duration():
    sim = Simulation(params=short(duration_hours=2.0), weather=ConstantWeather())
    assert len(sim.run().hour) == 2


def test_fields_stay_finite_over_a_run():
    sim = Simulation(
        params=short(duration_hours=6.0),
        components=[Mister(x_m=CENTRE, y_m=CENTRE, radius_m=15, strength=1e-4)],
    )
    sim.run(6.0)
    for field in (sim.state.ta, sim.state.ts, sim.state.w):
        assert np.all(np.isfinite(field))
    assert np.all(sim.state.w >= 0.0)


def test_uniform_domain_stays_uniform():
    # Nothing breaks symmetry, so every cell must follow one trajectory.
    sim = Simulation(params=short(duration_hours=4.0), weather=ConstantWeather())
    sim.run(4.0)
    assert sim.state.ta.std() == pytest.approx(0.0, abs=1e-9)
    assert sim.state.ts.std() == pytest.approx(0.0, abs=1e-9)


def test_each_intervention_cools_the_centre():
    p = short(duration_hours=12.0)
    i = p.n // 2

    def peak_air(components):
        sim = Simulation(params=p, components=components)
        record = sim.run(12.0)
        return max(s.air_c[i, i] for s in record.snapshots)

    base = peak_air([])
    for component in (
        Canopy(x_m=CENTRE, y_m=CENTRE, radius_m=20, strength=0.85),
        Mister(x_m=CENTRE, y_m=CENTRE, radius_m=20, strength=2e-4),
        ReflectivePavement(x_m=CENTRE, y_m=CENTRE, radius_m=20, strength=0.55),
        Park(x_m=CENTRE, y_m=CENTRE, radius_m=20),
        WaterFeature(x_m=CENTRE, y_m=CENTRE, radius_m=20),
        GreenRoof(x_m=CENTRE, y_m=CENTRE, radius_m=20),
    ):
        assert peak_air([component]) < base, component.kind


def test_unstable_diffusion_settings_are_rejected_at_construction():
    with pytest.raises(ValueError, match="stability"):
        Simulation(params=short(diffusion_dt=1000.0, K_air=1.0))


def test_duration_must_divide_evenly():
    sim = Simulation(params=short(dt=60.0), weather=ConstantWeather())
    with pytest.raises(ValueError, match="divisible"):
        sim.run(0.001)
