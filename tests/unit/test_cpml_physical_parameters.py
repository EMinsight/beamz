"""Regression contracts for physical CPML defaults (issue #316)."""

from types import SimpleNamespace

import numpy as np
import pytest

from beamz import EPS_0, LIGHT_SPEED, PEC, PML, Design, Material, Simulation
from beamz.devices._boundary_compile import _AbsorberCompiler


@pytest.mark.parametrize("dimensions", [2, 3])
@pytest.mark.parametrize("alpha", [None, 0.0, 125.0])
def test_cpml_parameters_are_invariant_under_mesh_and_timestep_refinement(
    dimensions, alpha
):
    fields = SimpleNamespace(permittivity=np.ones((2,) * dimensions))
    boundary = PML(thickness=0.4e-6, formulation="cpml", alpha_max=alpha)
    compiler = _AbsorberCompiler(boundary)
    dx = 1e-6 / 320
    dt = 0.5 * dx / LIGHT_SPEED
    reference = compiler._resolved_profile_params(fields, dx, dt)
    for spacing, step in ((dx, dt / 2), (dx / 2, dt / 2), (dx / 2, dt / 4)):
        assert compiler._resolved_profile_params(fields, spacing, step) == reference
    expected_alpha = (
        0.1 * EPS_0 * LIGHT_SPEED / boundary.thickness if alpha is None else alpha
    )
    assert reference[1] == pytest.approx(expected_alpha)
    assert boundary.alpha_max == alpha


def test_default_cell_thickness_resolves_before_physical_alpha():
    fields = SimpleNamespace(permittivity=np.ones((2, 2)))
    compiler = _AbsorberCompiler(PML(formulation="cpml"))
    profiles = [
        compiler._resolved_profile_boundary(fields, dx, 1e-17).spec
        for dx in (1e-8, 2e-8)
    ]
    for profile, dx in zip(profiles, (1e-8, 2e-8), strict=True):
        assert profile.thickness == 12 * dx
        assert profile.alpha_max == pytest.approx(0.1 * EPS_0 * LIGHT_SPEED / (12 * dx))
    assert profiles[0].alpha_max == 2 * profiles[1].alpha_max


@pytest.mark.parametrize("dimensions,polarization", [(2, "te"), (2, "tm"), (3, "tm")])
def test_sampled_cpml_profiles_do_not_change_with_timestep(dimensions, polarization):
    dx = 0.1e-6
    dt = 0.4 * dx / LIGHT_SPEED
    sim = Simulation(
        design=Design(
            width=2e-6,
            height=2e-6,
            depth=2e-6 if dimensions == 3 else 0.0,
            material=Material(),
        ),
        polarization=polarization,
        sources=[],
        boundaries=[PML(thickness=0.4e-6, formulation="cpml")],
        resolution=dx,
        time=np.array([0.0, dt]),
    )
    reference = sim.pml_data
    refined = sim.updated_copy(time=np.array([0.0, dt / 2])).pml_data

    def compare(left, right):
        for key in left:
            if isinstance(left[key], dict):
                compare(left[key], right[key])
            elif key in {"formulation", "resolved_parameters"}:
                assert left[key] == right[key]
            else:
                np.testing.assert_array_equal(left[key], right[key])

    compare(reference, refined)


def test_diagnostics_preserve_each_boundary_and_explicit_overrides():
    dx = 0.1e-6
    sim = Simulation(
        design=Design(width=3e-6, height=3e-6, material=Material()),
        sources=[],
        boundaries=[
            PEC(edges="bottom"),
            PML(edges="left", thickness=0.4e-6, formulation="cpml"),
            PML(
                edges="right",
                thickness=0.6e-6,
                formulation="cpml",
                sigma_max=321.0,
                alpha_max=0.0,
                kappa_max=3.0,
            ),
        ],
        resolution=dx,
        time=np.array([0.0, 0.4 * dx / LIGHT_SPEED]),
    )
    first, second = sim.pml_data["resolved_parameters"]
    assert first["edges"] == ("left",)
    assert first["alpha_max"] == pytest.approx(0.1 * EPS_0 * LIGHT_SPEED / 0.4e-6)
    assert second == dict(
        edges=("right",),
        formulation="cpml",
        thickness=0.6e-6,
        sigma_max=321.0,
        alpha_max=0.0,
        kappa_max=3.0,
        m=3,
        target_reflection=1e-6,
    )
    with pytest.raises(TypeError):
        first["alpha_max"] = 0.0
