import numpy as np
import pytest

from oasis import psychrometrics as psy


def test_saturation_pressure_matches_known_values():
    # Magnus form at 0 C is the triple-point value used as the constant.
    assert psy.saturation_vapor_pressure_pa(0.0) == pytest.approx(610.94, rel=1e-6)
    # Saturation pressure rises steeply with temperature.
    assert psy.saturation_vapor_pressure_pa(40.0) > 7000.0


def test_humidity_ratio_round_trips_through_rh():
    for temp in (10.0, 25.0, 45.0):
        for rh in (5.0, 22.0, 60.0, 99.0):
            w = psy.humidity_ratio_from_rh(temp, rh)
            assert psy.rh_from_humidity_ratio(temp, w) == pytest.approx(rh, rel=1e-9)


def test_rh_is_clipped_to_physical_range():
    # A humidity ratio far above saturation still reports at most 100 %.
    assert psy.rh_from_humidity_ratio(20.0, 0.05) == pytest.approx(100.0)
    assert psy.rh_from_humidity_ratio(20.0, 0.0) == pytest.approx(0.0)


def test_humidity_ratio_increases_with_rh():
    low = psy.humidity_ratio_from_rh(35.0, 10.0)
    high = psy.humidity_ratio_from_rh(35.0, 80.0)
    assert high > low


def test_dewpoint_equals_temperature_at_saturation():
    assert psy.dewpoint_c(25.0, 100.0) == pytest.approx(25.0, abs=1e-9)
    assert psy.dewpoint_c(35.0, 20.0) < 35.0


def test_functions_accept_arrays():
    temps = np.array([20.0, 30.0, 40.0])
    w = psy.humidity_ratio_from_rh(temps, 50.0)
    assert w.shape == temps.shape
    assert np.all(np.diff(w) > 0)
