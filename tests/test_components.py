import numpy as np
import pytest

from oasis.components import (
    Canopy,
    ComponentError,
    Mister,
    ReflectivePavement,
    StaticFields,
    available_kinds,
    build,
    from_dict,
)
from oasis.config import Parameters

N, DX = 40, 2.0


def _fields():
    return StaticFields.blank(N, 0.20)


def test_registry_lists_every_builtin_kind():
    assert set(available_kinds()) == {
        "canopy", "tree", "mister", "reflective_pavement"
    }


def test_build_rejects_unknown_kind():
    with pytest.raises(ComponentError) as exc:
        build("solar_panel", x_m=10, y_m=10, radius_m=5)
    assert "solar_panel" in str(exc.value)


def test_from_dict_requires_a_kind_key():
    with pytest.raises(ComponentError):
        from_dict({"x_m": 10, "y_m": 10, "radius_m": 5})


def test_component_outside_the_domain_is_rejected():
    with pytest.raises(ComponentError):
        Canopy(x_m=500.0, y_m=40.0, radius_m=5.0).validate(N, DX)


def test_non_positive_radius_is_rejected():
    with pytest.raises(ComponentError):
        Canopy(x_m=40.0, y_m=40.0, radius_m=0.0).validate(N, DX)


def test_canopy_shades_only_near_itself():
    fields = _fields()
    Canopy(x_m=41.0, y_m=41.0, radius_m=8.0, strength=0.85).apply(fields, N, DX)
    assert fields.shade.max() == pytest.approx(0.85)
    # A far corner is untouched.
    assert fields.shade[0, 0] == 0.0


def test_overlapping_shade_takes_the_maximum_not_the_sum():
    fields = _fields()
    for _ in range(3):
        Canopy(x_m=41.0, y_m=41.0, radius_m=8.0, strength=0.6).apply(fields, N, DX)
    # Three overlapping canopies must not block 180 % of the sun.
    assert fields.shade.max() == pytest.approx(0.6)
    assert fields.shade.max() <= 1.0


def test_overlapping_misters_accumulate():
    one, two = _fields(), _fields()
    Mister(x_m=41.0, y_m=41.0, radius_m=8.0, strength=1.0).apply(one, N, DX)
    for _ in range(2):
        Mister(x_m=41.0, y_m=41.0, radius_m=8.0, strength=1.0).apply(two, N, DX)
    assert two.mist.max() == pytest.approx(2.0 * one.mist.max())


def test_reflective_pavement_raises_albedo_within_its_footprint():
    fields = _fields()
    ReflectivePavement(x_m=41.0, y_m=41.0, radius_m=10.0, strength=0.55).apply(
        fields, N, DX
    )
    assert fields.albedo.max() == pytest.approx(0.55)
    assert fields.albedo[0, 0] == pytest.approx(0.20)
    assert np.all(fields.albedo <= 1.0)


def test_pavement_albedo_is_capped_below_one():
    fields = _fields()
    ReflectivePavement(x_m=41.0, y_m=41.0, radius_m=10.0, strength=5.0).apply(
        fields, N, DX
    )
    assert fields.albedo.max() <= 0.9


def test_only_misters_draw_power():
    params = Parameters()
    assert Mister(x_m=1, y_m=1, radius_m=1).power_w(params) == params.mister_power_w
    assert Canopy(x_m=1, y_m=1, radius_m=1).power_w(params) == 0.0
    assert ReflectivePavement(x_m=1, y_m=1, radius_m=1).power_w(params) == 0.0


def test_build_reports_missing_fields_as_component_error():
    # Scenario files routinely omit fields; that must not raise TypeError.
    with pytest.raises(ComponentError) as exc:
        build("mister")
    assert "x_m" in str(exc.value)


def test_build_reports_misspelled_fields_as_component_error():
    with pytest.raises(ComponentError) as exc:
        build("mister", x_m=1, y_m=1, radius_m=1, strenght=2)
    assert "strenght" in str(exc.value)
