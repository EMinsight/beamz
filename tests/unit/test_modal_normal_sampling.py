"""Modal extraction must invert the actual Yee monitor measurement."""

import numpy as np
import pytest

from beamz import RectilinearGrid
from beamz.analysis.modal_projection.colocation import _sample_modal_basis_normal_3d
from beamz.analysis.mode_projection import _project_modal_coefficients_3d_group
from beamz.lattice import (
    _component_plane_plan_3d,
    component_coordinates_rectilinear,
    component_shapes,
)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("grid_kind", ["uniform", "translated", "stretched"])
@pytest.mark.parametrize("position", [0.4, 0.45, 0.427])
@pytest.mark.parametrize("amplitudes", [(1, 0), (0, 1), (0.7 + 0.2j, -0.3 + 0.4j)])
def test_sampled_modes_recover_directional_mixtures(
    axis, grid_kind, position, amplitudes
):
    """Gather analytic waves on native lattices, then recover their amplitudes.

    The oracle is a volume of traveling waves sampled by the production monitor
    gather, not fields synthesized from the projection matrix under test.
    """
    edges = np.linspace(0, 1, 11)
    normal = (
        np.array([0, 0.08, 0.19, 0.31, 0.44, 0.59, 0.72, 0.84, 0.93, 1])
        if grid_kind == "stretched"
        else edges
    )
    if grid_kind == "translated":
        normal = normal + 0.13
        position += 0.13
    axes = {key: normal if key == axis else edges for key in ("x", "y", "z")}
    grid = RectilinearGrid(axes["x"], axes["y"], axes["z"])
    shape = tuple(len(axes[key]) - 1 for key in ("z", "y", "x"))
    shapes = component_shapes(shape)
    e, h = {"x": ("Ey", "Hz"), "y": ("Ez", "Hx"), "z": ("Ex", "Hy")}[axis]
    components = {
        "x": ("Ey", "Ez", "Hy", "Hz"),
        "y": ("Ez", "Ex", "Hx", "Hz"),
        "z": ("Ex", "Ey", "Hx", "Hy"),
    }[axis]
    coords = (np.array([0.35, 0.55]), np.array([0.35, 0.55]))
    beta = 5.8
    basis = {
        name: np.full(4, np.sqrt(2) if name in (e, h) else 0, dtype=complex)
        for name in components
    }
    backward = {
        name: values * (-1 if name.startswith("H") else 1)
        for name, values in basis.items()
    }
    sampling = dict(
        axis=axis,
        plane_position=position,
        wave_number=beta,
        grid_shape=shape,
        resolution=0.1,
        grid=None if grid_kind == "uniform" else grid,
    )
    forward_measured = _sample_modal_basis_normal_3d(basis, direction=1, **sampling)
    backward_measured = _sample_modal_basis_normal_3d(
        backward, direction=-1, **sampling
    )
    measured = {}
    for name in components:
        native = component_coordinates_rectilinear(name, grid)
        x = native[axis]
        normal_shape = [1, 1, 1]
        normal_shape[{"z": 0, "y": 1, "x": 2}[axis]] = len(x)
        values = amplitudes[0] * basis[name][0] * np.exp(
            1j * beta * (x - position)
        ) + amplitudes[1] * backward[name][0] * np.exp(-1j * beta * (x - position))
        volume = np.broadcast_to(values.reshape(normal_shape), shapes[name])
        indices, weights = _component_plane_plan_3d(
            name,
            normal_axis=axis,
            plane_position=position,
            coordinates=coords,
            resolution=0.1,
            grid_shape=shape,
            field_shape=shapes[name],
            grid=None if grid_kind == "uniform" else grid,
        )
        measured[name] = np.sum(volume.ravel()[indices] * weights, axis=1)
    projection = dict(
        components=components,
        axis=axis,
        d_area=0.25,
        direction_sign=1,
        mode_components=forward_measured,
        mode_components_bwd=backward_measured,
    )
    coefficients, residual, _, _ = _project_modal_coefficients_3d_group(
        measured, [projection]
    )
    np.testing.assert_allclose(coefficients[0], amplitudes, atol=2e-7, rtol=2e-7)
    assert residual < 2e-7
