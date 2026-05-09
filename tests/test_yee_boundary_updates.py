from __future__ import annotations

import numpy as np

from beamz.simulation.boundaries import (
    PEC,
    build_h_boundary_views_for_e_3d,
    create_metallic_boundary_masks,
    full_pec_curl_e_to_h_2d_xy,
    full_pec_curl_e_to_h_3d,
    full_pec_curl_h_to_e_2d_xy,
    full_pec_curl_h_to_e_3d,
    has_full_pec_2d_xy,
    initialize_full_pec_2d_xy_state,
    initialize_full_pec_3d_state,
    normalize_boundaries,
    pec_curl_e_to_h_3d,
    pec_curl_h_to_e_3d,
)
from beamz.simulation.fields import Fields
from beamz.simulation.ops import curl_h_to_e_3d
from beamz.simulation.yee import (
    sample_voxel_grid_at_compact_component_2d,
    sample_voxel_grid_at_component_2d,
    sample_voxel_grid_at_tm_xy_full_component_2d,
)


def _uniform_3d_fields(n: int = 4) -> Fields:
    permittivity = np.ones((n, n, n), dtype=np.float32)
    conductivity = np.zeros((n, n, n), dtype=np.float32)
    permeability = np.ones((n, n, n), dtype=np.float32)
    fields = Fields(
        permittivity=permittivity,
        conductivity=conductivity,
        permeability=permeability,
        resolution=1.0,
    )
    fields.boundaries = normalize_boundaries([], is_3d=True)
    fields.set_metallic_masks(
        create_metallic_boundary_masks(fields, boundaries=[], is_3d=True)
    )
    return fields


def test_3d_pec_update_keeps_constrained_ex_planes_zero():
    fields = _uniform_3d_fields()
    dt = 1e-3

    # Create nonzero curl contributions that touch Ex boundary planes via both
    # dHz/dy and dHy/dz terms.
    hz = np.zeros(fields.Hz.shape, dtype=np.float32)
    hz[:, 1:, :] = 1.0
    hy = np.zeros(fields.Hy.shape, dtype=np.float32)
    hy[1:, :, :] = 1.0

    fields.Hz = hz
    fields.Hy = hy

    fields.update_e(dt)
    ex = np.asarray(fields.Ex)

    np.testing.assert_allclose(ex[0, :, :], 0.0)
    np.testing.assert_allclose(ex[:, 0, :], 0.0)
    assert np.max(np.abs(ex[-1, 1:, :])) > 0.0
    assert np.max(np.abs(ex[1:, -1, :])) > 0.0
    assert np.max(np.abs(ex[1:-1, 1:-1, :])) > 0.0


def test_3d_default_pec_masks_zero_expected_stored_planes():
    fields = _uniform_3d_fields()
    masks = create_metallic_boundary_masks(fields, boundaries=[], is_3d=True)

    for name, mask in masks.items():
        assert mask.shape == getattr(fields, name).shape

    assert np.all(np.asarray(masks["Ex"])[:, 0, :])
    assert np.all(np.asarray(masks["Ex"])[0, :, :])
    assert not np.any(np.asarray(masks["Ex"])[1:, -1, :])
    assert not np.any(np.asarray(masks["Ex"])[-1, 1:, :])

    assert np.all(np.asarray(masks["Ey"])[:, :, 0])
    assert np.all(np.asarray(masks["Ey"])[0, :, :])
    assert not np.any(np.asarray(masks["Ey"])[1:, :, -1])
    assert not np.any(np.asarray(masks["Ey"])[-1, 1:, 1:])

    assert np.all(np.asarray(masks["Ez"])[:, 0, :])
    assert np.all(np.asarray(masks["Ez"])[:, :, 0])
    assert not np.any(np.asarray(masks["Ez"])[1:, -1, 1:])
    assert not np.any(np.asarray(masks["Ez"])[1:, 1:, -1])


def test_empty_boundary_list_resolves_to_explicit_pec():
    resolved = normalize_boundaries([], is_3d=True)
    assert len(resolved) == 1
    assert isinstance(resolved[0], PEC)
    assert resolved[0]._get_edges_for_dimensionality(True) == [
        "left",
        "right",
        "top",
        "bottom",
        "front",
        "back",
    ]


def test_pec_curl_h_to_e_3d_keeps_boundary_planes_zero():
    fields = _uniform_3d_fields()

    hz = np.zeros(fields.Hz.shape, dtype=np.float32)
    hz[:, 1:, :] = 1.0
    hy = np.zeros(fields.Hy.shape, dtype=np.float32)
    hy[1:, :, :] = 0.25
    hx = np.zeros(fields.Hx.shape, dtype=np.float32)

    curl_hx, curl_hy, curl_hz = pec_curl_h_to_e_3d(
        hx,
        hy,
        hz,
        resolution=1.0,
        ex_shape=fields.Ex.shape,
        ey_shape=fields.Ey.shape,
        ez_shape=fields.Ez.shape,
    )

    assert np.max(np.abs(np.asarray(curl_hx)[:, 0, :])) > 0.0
    assert np.max(np.abs(np.asarray(curl_hx)[:, -1, :])) > 0.0
    assert np.max(np.abs(np.asarray(curl_hx)[1:-1, 1:-1, :])) > 0.0


def test_pec_curl_e_to_h_3d_matches_h_shapes():
    fields = _uniform_3d_fields()

    curl_ex, curl_ey, curl_ez = pec_curl_e_to_h_3d(
        fields.Ex,
        fields.Ey,
        fields.Ez,
        resolution=1.0,
        hx_shape=fields.Hx.shape,
        hy_shape=fields.Hy.shape,
        hz_shape=fields.Hz.shape,
    )

    assert curl_ex.shape == fields.Hx.shape
    assert curl_ey.shape == fields.Hy.shape
    assert curl_ez.shape == fields.Hz.shape


def test_initialize_full_pec_3d_state_adds_missing_high_planes():
    fields = _uniform_3d_fields()
    state = initialize_full_pec_3d_state(fields)

    assert state.Ex.shape == tuple(v + 1 for v in fields.Ex.shape)
    assert state.Ey.shape == tuple(v + 1 for v in fields.Ey.shape)
    assert state.Ez.shape == tuple(v + 1 for v in fields.Ez.shape)
    assert state.Hx.shape == tuple(v + 1 for v in fields.Hx.shape)
    assert state.Hy.shape == tuple(v + 1 for v in fields.Hy.shape)
    assert state.Hz.shape == tuple(v + 1 for v in fields.Hz.shape)

    np.testing.assert_allclose(np.asarray(state.Ex)[-1, :, :], 0.0)
    np.testing.assert_allclose(np.asarray(state.Ex)[:, -1, :], 0.0)
    np.testing.assert_allclose(np.asarray(state.Ey)[-1, :, :], 0.0)
    np.testing.assert_allclose(np.asarray(state.Ey)[:, :, -1], 0.0)
    np.testing.assert_allclose(np.asarray(state.Ez)[:, -1, :], 0.0)
    np.testing.assert_allclose(np.asarray(state.Ez)[:, :, -1], 0.0)


def test_full_pec_material_regions_sample_true_yee_positions():
    permittivity = np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)
    conductivity = np.zeros_like(permittivity)
    permeability = np.ones_like(permittivity)
    fields = Fields(
        permittivity=permittivity,
        conductivity=conductivity,
        permeability=permeability,
        resolution=1.0,
    )
    state = initialize_full_pec_3d_state(fields)

    expected_ex = permittivity[np.ix_([1, 2, 3], [1, 2, 3], [0, 1, 2, 3])]
    expected_ey = permittivity[np.ix_([1, 2, 3], [0, 1, 2, 3], [1, 2, 3])]
    expected_ez = permittivity[np.ix_([0, 1, 2, 3], [1, 2, 3], [1, 2, 3])]

    np.testing.assert_array_equal(np.asarray(state.eps_x_region), expected_ex)
    np.testing.assert_array_equal(np.asarray(state.eps_y_region), expected_ey)
    np.testing.assert_array_equal(np.asarray(state.eps_z_region), expected_ez)


def test_xy_2d_component_sampling_matches_expected_owned_voxels():
    grid = np.arange(4 * 5, dtype=np.float32).reshape(4, 5)

    ex = sample_voxel_grid_at_component_2d(grid, "Ex", "xy")
    ey = sample_voxel_grid_at_component_2d(grid, "Ey", "xy")
    ez = sample_voxel_grid_at_component_2d(grid, "Ez", "xy")
    hx = sample_voxel_grid_at_compact_component_2d(grid, "Hx", "xy")
    hy = sample_voxel_grid_at_compact_component_2d(grid, "Hy", "xy")
    hz = sample_voxel_grid_at_component_2d(grid, "Hz", "xy")

    np.testing.assert_array_equal(np.asarray(ex), grid[:, :-1])
    np.testing.assert_array_equal(np.asarray(ey), grid[:-1, :])
    np.testing.assert_array_equal(np.asarray(ez), grid)
    np.testing.assert_array_equal(np.asarray(hx), grid[:, :-1])
    np.testing.assert_array_equal(np.asarray(hy), grid[:-1, :])
    np.testing.assert_array_equal(np.asarray(hz), grid[:-1, :-1])


def test_xy_2d_full_pec_state_adds_missing_h_boundary_edges():
    permittivity = np.ones((4, 5), dtype=np.float32)
    conductivity = np.zeros_like(permittivity)
    permeability = np.ones_like(permittivity)
    fields = Fields(
        permittivity=permittivity,
        conductivity=conductivity,
        permeability=permeability,
        resolution=1.0,
        plane_2d="xy",
    )
    fields.boundaries = normalize_boundaries([], is_3d=False)

    state = initialize_full_pec_2d_xy_state(fields)
    assert state.Ez.shape == (fields.Ez.shape[0] + 1, fields.Ez.shape[1] + 1)
    assert state.Hx.shape == (fields.Ez.shape[0], fields.Ez.shape[1] + 1)
    assert state.Hy.shape == (fields.Ez.shape[0] + 1, fields.Ez.shape[1])
    assert has_full_pec_2d_xy(fields.boundaries, "xy")
    np.testing.assert_allclose(np.asarray(state.Hx)[:, 0], 0.0)
    np.testing.assert_allclose(np.asarray(state.Hx)[:, -1], 0.0)
    np.testing.assert_allclose(np.asarray(state.Hy)[0, :], 0.0)
    np.testing.assert_allclose(np.asarray(state.Hy)[-1, :], 0.0)


def test_2d_tm_pec_update_keeps_constrained_h_edges_zero():
    permittivity = np.ones((4, 5), dtype=np.float32)
    conductivity = np.zeros_like(permittivity)
    permeability = np.ones_like(permittivity)
    fields = Fields(
        permittivity=permittivity,
        conductivity=conductivity,
        permeability=permeability,
        resolution=1.0,
        plane_2d="xy",
    )
    fields.boundaries = normalize_boundaries([], is_3d=False)

    ez = np.zeros((fields.Ez.shape[0] + 1, fields.Ez.shape[1] + 1), dtype=np.float32)
    ez[1:4, 1:5] = 1.0
    fields.full_tm_2d_xy_state = initialize_full_pec_2d_xy_state(fields)
    fields.full_tm_2d_xy_state.Ez = ez

    fields.update_h(dt=1e-3)
    state = fields.full_tm_2d_xy_state

    np.testing.assert_allclose(np.asarray(state.Hx)[:, 0], 0.0)
    np.testing.assert_allclose(np.asarray(state.Hx)[:, -1], 0.0)
    np.testing.assert_allclose(np.asarray(state.Hy)[0, :], 0.0)
    np.testing.assert_allclose(np.asarray(state.Hy)[-1, :], 0.0)
    assert np.max(np.abs(np.asarray(state.Hx)[:, 1:-1])) > 0.0
    assert np.max(np.abs(np.asarray(state.Hy)[1:-1, :])) > 0.0


def test_xy_physical_tm_state_syncs_from_compact_injections():
    permittivity = np.ones((4, 5), dtype=np.float32)
    conductivity = np.zeros_like(permittivity)
    permeability = np.ones_like(permittivity)
    fields = Fields(
        permittivity=permittivity,
        conductivity=conductivity,
        permeability=permeability,
        resolution=1.0,
        plane_2d="xy",
    )
    fields.boundaries = normalize_boundaries([], is_3d=False)
    fields.full_tm_2d_xy_state = initialize_full_pec_2d_xy_state(fields)

    fields.Ez = fields.Ez.at[1, 2].set(3.0)
    fields.Hx = fields.Hx.at[2, 1].set(4.0)
    fields.Hy = fields.Hy.at[1, 3].set(-5.0)

    fields.sync_physical_tm_xy_from_compact()

    state = fields.full_tm_2d_xy_state
    assert float(state.Ez[1, 2]) > 0.0
    assert float(state.Ez[2, 3]) > 0.0
    assert float(state.Hx[2, 1]) == 4.0
    assert float(state.Hy[1, 3]) == -5.0


def test_xy_2d_full_state_sampling_uses_physical_tmz_h_locations():
    grid = np.arange(4 * 5, dtype=np.float32).reshape(4, 5)

    ez = sample_voxel_grid_at_tm_xy_full_component_2d(grid, "Ez")
    hx = sample_voxel_grid_at_tm_xy_full_component_2d(grid, "Hx")
    hy = sample_voxel_grid_at_tm_xy_full_component_2d(grid, "Hy")

    np.testing.assert_array_equal(
        np.asarray(ez), grid[[0, 1, 2, 3, 3]][:, [0, 1, 2, 3, 4, 4]]
    )
    np.testing.assert_array_equal(np.asarray(hx), grid[:, [0, 1, 2, 3, 4, 4]])
    np.testing.assert_array_equal(np.asarray(hy), grid[[0, 1, 2, 3, 3], :])


def test_xy_2d_full_pec_curls_match_full_shapes():
    permittivity = np.ones((4, 5), dtype=np.float32)
    conductivity = np.zeros_like(permittivity)
    permeability = np.ones_like(permittivity)
    fields = Fields(
        permittivity=permittivity,
        conductivity=conductivity,
        permeability=permeability,
        resolution=1.0,
        plane_2d="xy",
    )
    state = initialize_full_pec_2d_xy_state(fields)

    curl_hx, curl_hy = full_pec_curl_e_to_h_2d_xy(
        state.Ez,
        1.0,
        state.Hx.shape,
        state.Hy.shape,
    )
    assert curl_hx.shape == state.Hx.shape
    assert curl_hy.shape == state.Hy.shape

    curl_hz = full_pec_curl_h_to_e_2d_xy(
        state.Hx,
        state.Hy,
        1.0,
        state.Ez.shape,
    )
    assert curl_hz.shape == state.Ez.shape


def test_full_pec_curls_match_full_shapes():
    fields = _uniform_3d_fields()
    state = initialize_full_pec_3d_state(fields)

    curl_ex, curl_ey, curl_ez = full_pec_curl_e_to_h_3d(
        state.Ex,
        state.Ey,
        state.Ez,
        1.0,
        state.Hx.shape,
        state.Hy.shape,
        state.Hz.shape,
    )
    assert curl_ex.shape == state.Hx.shape
    assert curl_ey.shape == state.Hy.shape
    assert curl_ez.shape == state.Hz.shape

    curl_hx, curl_hy, curl_hz = full_pec_curl_h_to_e_3d(
        state.Hx,
        state.Hy,
        state.Hz,
        1.0,
        state.Ex.shape,
        state.Ey.shape,
        state.Ez.shape,
    )
    assert curl_hx.shape == state.Ex.shape
    assert curl_hy.shape == state.Ey.shape
    assert curl_hz.shape == state.Ez.shape
