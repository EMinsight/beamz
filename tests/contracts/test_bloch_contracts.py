"""Bloch public configuration and compilation contracts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import beamz as bz
from beamz.devices.boundaries import (
    bloch_storage_phases,
    validate_boundary_compatibility,
)


@pytest.mark.contract
def test_bloch_validation_and_physical_axes():
    with pytest.raises(ValueError, match="three finite"):
        bz.Bloch(wavevector=(np.nan, 0, 0))
    with pytest.raises(ValueError, match="outside"):
        bz.Bloch(axes="x", wavevector=(0, 1, 0))
    with pytest.raises(ValueError, match="Conflicting"):
        validate_boundary_compatibility(
            [bz.Periodic(axes="x"), bz.Bloch(axes="x", wavevector=(1, 0, 0))],
            is_3d=True,
        )
    with pytest.raises(ValueError, match="conflicting edges"):
        validate_boundary_compatibility(
            [bz.Bloch(axes="x"), bz.PML(edges="left")], is_3d=True
        )
    grid = bz.RectilinearGrid.from_spacing((2, 3, 4), 0.1)
    phases = bloch_storage_phases(
        [bz.Bloch(axes=("x", "z"), wavevector=(2, 0, -3))],
        grid,
        is_3d=False,
        plane_2d="xz",
    )
    np.testing.assert_allclose(phases, np.exp(1j * np.array([-3 * 0.4, 2 * 0.2])))


@pytest.mark.contract
def test_zero_bloch_matches_periodic_and_cache_distinguishes_wavevectors():
    kwargs = dict(size=(160e-9, 160e-9, 0), resolution=40e-9, run_time=1e-15)
    ordinary = bz.Simulation(**kwargs, boundaries=[bz.Periodic()])
    zero = bz.Simulation(**kwargs, boundaries=[bz.Bloch()])
    a, b = ordinary.initial_state(), zero.initial_state()
    assert a.ez.dtype == b.ez.dtype == jnp.float32
    for x, y in zip(
        jax.tree.leaves(ordinary.step(a)), jax.tree.leaves(zero.step(b)), strict=True
    ):
        np.testing.assert_array_equal(x, y)
    first = bz.Simulation(
        **kwargs, boundaries=[bz.Bloch(wavevector=(1e6, 0, 0))]
    ).compile()
    second = bz.Simulation(
        **kwargs, boundaries=[bz.Bloch(wavevector=(-1e6, 0, 0))]
    ).compile()
    assert first is not second
    assert first.boundary.periodic_phases != second.boundary.periodic_phases
