from __future__ import annotations

import numpy as np

from beamz.simulation.fields import Fields
from beamz.simulation.yee import (
    component_coordinates_2d_um,
    component_coordinates_3d_um,
    component_shape_2d,
    component_shape_3d,
    tm_xy_full_component_coordinates_2d_um,
    tm_xy_full_component_shape_2d,
)


def test_component_shape_3d_matches_expected_beamz_storage():
    grid_shape = (24, 24, 24)
    assert component_shape_3d("Ex", grid_shape) == (24, 24, 23)
    assert component_shape_3d("Ey", grid_shape) == (24, 23, 24)
    assert component_shape_3d("Ez", grid_shape) == (23, 24, 24)


def test_component_coordinates_3d_follow_standard_yee_offsets():
    grid_shape = (24, 24, 24)
    dx_um = 0.125

    ex = component_coordinates_3d_um("Ex", grid_shape, dx_um)
    ey = component_coordinates_3d_um("Ey", grid_shape, dx_um)
    ez = component_coordinates_3d_um("Ez", grid_shape, dx_um)
    hx = component_coordinates_3d_um("Hx", grid_shape, dx_um)

    np.testing.assert_allclose(ex["x"][0], 0.0625)
    np.testing.assert_allclose(ex["x"][-1], 2.8125)
    np.testing.assert_allclose(ey["y"][0], 0.0625)
    np.testing.assert_allclose(ey["y"][-1], 2.8125)
    np.testing.assert_allclose(ez["z"][0], 0.0625)
    np.testing.assert_allclose(ez["z"][-1], 2.8125)

    np.testing.assert_allclose(ex["z"][0], 0.0)
    np.testing.assert_allclose(ex["y"][0], 0.0)
    np.testing.assert_allclose(ey["x"][0], 0.0)
    np.testing.assert_allclose(ey["z"][0], 0.0)
    np.testing.assert_allclose(ez["x"][0], 0.0)
    np.testing.assert_allclose(ez["y"][0], 0.0)

    np.testing.assert_allclose(hx["x"][0], 0.0)
    np.testing.assert_allclose(hx["y"][0], 0.0625)
    np.testing.assert_allclose(hx["z"][0], 0.0625)


def test_component_coordinates_2d_follow_standard_xy_offsets():
    grid_shape = (24, 24)
    dx_um = 0.125

    assert component_shape_2d("Ez", grid_shape, "xy") == (25, 25)

    ez = component_coordinates_2d_um("Ez", grid_shape, dx_um, "xy")

    np.testing.assert_allclose(ez["y"][0], 0.0)
    np.testing.assert_allclose(ez["x"][0], 0.0)


def test_tm_xy_full_state_coordinates_follow_physical_tmz_offsets():
    grid_shape = (24, 24)
    dx_um = 0.125

    assert tm_xy_full_component_shape_2d("Ez", grid_shape) == (25, 25)
    assert tm_xy_full_component_shape_2d("Hx", grid_shape) == (24, 25)
    assert tm_xy_full_component_shape_2d("Hy", grid_shape) == (25, 24)

    ez = tm_xy_full_component_coordinates_2d_um("Ez", grid_shape, dx_um)
    hx = tm_xy_full_component_coordinates_2d_um("Hx", grid_shape, dx_um)
    hy = tm_xy_full_component_coordinates_2d_um("Hy", grid_shape, dx_um)

    np.testing.assert_allclose(ez["y"][0], 0.0)
    np.testing.assert_allclose(ez["x"][0], 0.0)
    np.testing.assert_allclose(hx["y"][0], 0.0625)
    np.testing.assert_allclose(hx["x"][0], 0.0)
    np.testing.assert_allclose(hy["y"][0], 0.0)
    np.testing.assert_allclose(hy["x"][0], 0.0625)


def test_xy_generic_h_coordinates_follow_native_tmz_offsets():
    assert component_shape_2d("Hx", (24, 24), "xy") == (24, 25)
    hy = component_coordinates_2d_um("Hy", (24, 24), 0.125, "xy")
    np.testing.assert_allclose(hy["y"][0], 0.0)
    np.testing.assert_allclose(hy["x"][0], 0.0625)


def test_fields_expose_component_material_arrays_for_3d():
    grid = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5) + 1.0
    fields = Fields(
        permittivity=grid,
        conductivity=np.zeros_like(grid),
        permeability=2.0 * grid,
        resolution=0.1,
    )

    for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        material = fields.material_for_component(component)
        assert material.shape == getattr(fields, component).shape

    assert fields.material_for_component("Ex") is fields.eps_x
    assert fields.material_for_component("Ey") is fields.eps_y
    assert fields.material_for_component("Ez") is fields.eps_z
    assert fields.material_for_component("Hx") is fields.mu_hx
    assert fields.material_for_component("Hy") is fields.mu_hy
    assert fields.material_for_component("Hz") is fields.mu_hz


def test_fields_expose_component_material_arrays_for_xy_2d():
    grid = np.arange(4 * 5, dtype=np.float32).reshape(4, 5) + 1.0
    fields = Fields(
        permittivity=grid,
        conductivity=np.zeros_like(grid),
        permeability=2.0 * grid,
        resolution=0.1,
        plane_2d="xy",
    )

    for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        material = fields.material_for_component(component)
        assert material.shape == getattr(fields, component).shape

    np.testing.assert_array_equal(fields.material_for_component("Ez"), fields.eps_z)
    np.testing.assert_array_equal(fields.material_for_component("Hx"), fields.mu_hx)
    np.testing.assert_array_equal(fields.material_for_component("Hy"), fields.mu_hy)
