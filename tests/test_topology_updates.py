import numpy as np
import pytest

from beamz.optimization.topology import apply_density_update


def test_apply_density_update_changes_masked_region():
    density = np.zeros((3, 3), dtype=float)
    mask = np.zeros_like(density, dtype=bool)
    mask[1, 1] = True
    gradient = np.ones_like(density)

    new_density, applied_gradient, delta, change_norm, max_update = apply_density_update(
        density, gradient, mask, learning_rate=0.2
    )

    assert pytest.approx(new_density[1, 1]) == 0.2
    assert change_norm > 0.0
    assert pytest.approx(max_update) == change_norm
    assert np.all(new_density[~mask] == 0.0)
    assert np.all(delta[~mask] == 0.0)
    assert applied_gradient[1, 1] == pytest.approx(1.0)


def test_apply_density_update_obeys_clip_and_mask():
    density = np.full((3, 3), 0.95)
    mask = np.zeros_like(density, dtype=bool)
    mask[1, 1] = True
    gradient = np.ones_like(density)

    new_density, _, delta, change_norm, max_update = apply_density_update(
        density, gradient, mask, learning_rate=0.5
    )

    # Value is clipped to 1.0 and change stays inside the mask.
    assert pytest.approx(new_density[1, 1]) == 1.0
    assert np.all(new_density[~mask] == density[~mask])
    assert np.all(delta[~mask] == 0.0)
    assert change_norm == pytest.approx(max_update)


def test_apply_density_update_zero_gradient_returns_no_change():
    density = np.random.rand(4, 4)
    mask = np.ones_like(density, dtype=bool)
    gradient = np.zeros_like(density)

    new_density, applied_gradient, delta, change_norm, max_update = apply_density_update(
        density, gradient, mask, learning_rate=0.3
    )

    assert np.allclose(new_density, density)
    assert np.allclose(applied_gradient, 0.0)
    assert np.allclose(delta, 0.0)
    assert change_norm == 0.0
    assert max_update == 0.0
