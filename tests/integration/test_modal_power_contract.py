"""Optimization and ordinary analysis must use the same modal power basis."""

import numpy as np
import pytest

from beamz.optimization import ModePower
from tests.topology_3d_case import make_3d_design


@pytest.mark.parametrize("source_power", [0.4, 2.5])
def test_bound_mode_power_matches_public_mixed_polarization(source_power):
    design = make_3d_design(polarization=None)
    source = design.simulation.sources[0].updated_copy(power=source_power)
    design = design.updated_copy(
        simulation=design.simulation.updated_copy(sources=[source])
    )
    params = design.design_region.initial_parameters
    bound_power = design.make_objective_fn(ModePower("output"))(params)
    result = design.to_simulation(params).run(backend="jax", progress=False)
    public_power = result.mode("output").power.sel(direction="+", mode_index=0)
    np.testing.assert_allclose(
        bound_power, public_power.item() / source_power, rtol=3e-5, atol=1e-8
    )
