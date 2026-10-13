import numpy as np
import pytest

from oasis import numerics as nm


def _coordinate_grids(n=8, dx=2.0):
    xs = nm.cell_centres(n, dx)
    return np.meshgrid(xs, xs)


def test_laplacian_of_a_linear_field_is_zero():
    xx, yy = _coordinate_grids()
    lap = nm.laplacian(3.0 * xx + 2.0 * yy, 2.0)
    # Edges use replicated boundaries, so only the interior is exactly zero.
    assert np.allclose(lap[1:-1, 1:-1], 0.0)


def test_laplacian_of_a_paraboloid_is_four():
    xx, yy = _coordinate_grids()
    lap = nm.laplacian(xx ** 2 + yy ** 2, 2.0)
    assert np.allclose(lap[1:-1, 1:-1], 4.0)


def test_laplacian_conserves_total_under_replicated_boundaries():
    # Zero-normal-gradient edges mean diffusion moves heat without creating it.
    rng = np.random.RandomState(0)
    field = rng.rand(10, 10)
    assert nm.laplacian(field, 1.0).sum() == pytest.approx(0.0, abs=1e-10)


def test_gaussian_kernel_is_bounded_and_local():
    k = nm.gaussian_kernel(40, 2.0, 40.0, 40.0, 8.0)
    assert k.max() <= 1.0
    assert k.min() >= 0.0
    # Nothing beyond 2.5 radii of the centre.
    r = nm.radial_distance(40, 2.0, 40.0, 40.0)
    assert np.all(k[r > 2.5 * 8.0] == 0.0)


def test_gaussian_kernel_peaks_at_the_component_centre():
    # Centre placed on a cell centre so the peak is exactly 1.
    k = nm.gaussian_kernel(40, 2.0, 41.0, 41.0, 8.0)
    assert k.max() == pytest.approx(1.0)
    assert np.unravel_index(np.argmax(k), k.shape) == (20, 20)


def test_diffusion_stability_limit():
    assert nm.diffusion_stability_limit(2.0, 0.01) == pytest.approx(100.0)
    assert nm.diffusion_stability_limit(2.0, 0.0) == float("inf")


def test_stability_check_flags_only_unstable_terms():
    assert nm.check_explicit_stability(30.0, 2.0, [("air", 0.01)]) == []
    warnings = nm.check_explicit_stability(300.0, 2.0, [("air", 0.01), ("slow", 1e-6)])
    assert len(warnings) == 1
    assert "air" in warnings[0]
