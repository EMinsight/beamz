import numpy as np

from beamz import Design, Material, ModeSource, Monitor, Rectangle, um
from beamz.devices._placement import mirror_lock_plane_pair_regions


def _build_2d_waveguide():
    design = Design(
        width=4.0 * um,
        height=2.0 * um,
        material=Material(1.44**2),
    )
    design += Rectangle(
        position=(0.0, 0.75 * um),
        width=4.0 * um,
        height=0.5 * um,
        material=Material(3.47**2),
        depth=0.0,
    )
    return design


def test_mode_source_snapped_region_uses_physical_bounds():
    design = _build_2d_waveguide()
    dx = 0.2 * um
    grid = design.rasterize(resolution=dx)
    source = ModeSource(
        grid=grid,
        center=(1.0 * um, 1.0 * um),
        width=0.8 * um,
        wavelength=1.55 * um,
        pol="tm",
        signal=np.zeros(8, dtype=float),
        direction="+x",
    )

    source.initialize(grid.permittivity, dx)
    snapped = source.get_snapped_region()

    assert snapped is not None
    assert snapped.normal_axis == "x"
    assert snapped.plane_index == 4
    assert snapped.plane_coord == 0.9 * um
    assert snapped.axis_bounds("y") == (0.6 * um, 1.4 * um)


def test_mode_source_and_monitor_snap_to_same_2d_cross_section():
    design = _build_2d_waveguide()
    dx = 0.2 * um
    grid = design.rasterize(resolution=dx)
    source = ModeSource(
        grid=grid,
        center=(1.0 * um, 1.0 * um),
        width=0.8 * um,
        wavelength=1.55 * um,
        pol="tm",
        signal=np.zeros(8, dtype=float),
        direction="+x",
    )
    source.initialize(grid.permittivity, dx)

    monitor = Monitor(
        design=design,
        start=(1.0 * um, 0.6 * um),
        end=(1.0 * um, 1.4 * um),
        name="m",
    )
    points = monitor.get_grid_points_2d(dx, dx)
    snapped_source = source.get_snapped_region()
    snapped_monitor = monitor.get_snapped_region(dx=dx, dy=dx)

    assert snapped_source is not None
    assert snapped_monitor is not None
    assert snapped_monitor.normal_axis == "x"
    assert snapped_monitor.plane_index == snapped_source.plane_index
    assert snapped_monitor.axis_bounds("y") == snapped_source.axis_bounds("y")
    assert points == [(snapped_source.plane_index, idx) for idx in range(3, 7)]


def test_monitor_plane_3d_snap_uses_same_centered_plane_convention():
    design = Design(
        width=2.0 * um,
        height=1.5 * um,
        depth=1.0 * um,
        material=Material(1.44**2),
    )
    monitor = Monitor(
        design=design,
        start=(1.0 * um, 0.3 * um, 0.2 * um),
        end=(1.0 * um, 1.1 * um, 0.8 * um),
        name="m3d",
    )

    snapped = monitor.get_snapped_region(
        dx=0.2 * um, dy=0.2 * um, dz=0.2 * um, field_shape=(5, 8, 10)
    )
    z_idx, y_idx, x_idx = monitor.get_grid_slice_3d(
        0.2 * um, 0.2 * um, 0.2 * um, (5, 8, 10)
    )

    assert snapped is not None
    assert snapped.normal_axis == "x"
    assert snapped.plane_index == 4
    assert snapped.plane_coord == 0.9 * um
    assert (z_idx, y_idx, x_idx) == (slice(1, 4), slice(1, 6), 4)
    assert monitor.position == snapped.center


def test_mirror_lock_plane_pair_regions_uses_reflected_normal_indices():
    dx = 0.2 * um
    shape = (5, 8, 10)
    top, bottom = mirror_lock_plane_pair_regions(
        start_a=(0.5 * um, 1.1 * um, 0.2 * um),
        end_a=(1.5 * um, 1.1 * um, 0.8 * um),
        start_b=(0.5 * um, 0.3 * um, 0.2 * um),
        end_b=(1.5 * um, 0.3 * um, 0.8 * um),
        plane_normal="y",
        size_a=None,
        size_b=None,
        dx=dx,
        dy=dx,
        dz=dx,
        shape=shape,
    )

    assert top.normal_axis == "y"
    assert bottom.normal_axis == "y"
    assert top.plane_index + bottom.plane_index == shape[1] - 1
    assert top.axis_bounds("x") == bottom.axis_bounds("x")
    assert top.axis_bounds("z") == bottom.axis_bounds("z")


def test_monitor_record_fields_3d_interpolates_to_requested_plane_position():
    from beamz.simulation.yee import component_coordinates_3d_um

    dx = 0.2 * um
    design = Design(
        width=1.6 * um,
        height=1.4 * um,
        depth=1.2 * um,
        material=Material(1.44**2),
    )
    monitor = Monitor(
        design=design,
        start=(0.25 * um, 0.37 * um, 0.15 * um),
        end=(1.15 * um, 0.37 * um, 0.85 * um),
        name="arb_plane",
        dft_enabled=False,
    )
    base_shape = (6, 7, 8)

    def linear_component(component: str) -> np.ndarray:
        coords = component_coordinates_3d_um(component, base_shape, float(dx / um))
        z = np.asarray(coords["z"], dtype=float)[:, None, None] * float(um)
        y = np.asarray(coords["y"], dtype=float)[None, :, None] * float(um)
        x = np.asarray(coords["x"], dtype=float)[None, None, :] * float(um)
        return (100.0 * z) + (10.0 * y) + x

    Ex = linear_component("Ex")
    Ey = linear_component("Ey")
    Ez = linear_component("Ez")
    Hx = linear_component("Hx")
    Hy = linear_component("Hy")
    Hz = linear_component("Hz")

    monitor.record_fields_3d(Ex, Ey, Ez, Hx, Hy, Hz, t=0.0, dx=dx, dy=dx, dz=dx, step=0)
    z_coords, x_coords = monitor.get_analysis_plane_coords_3d(
        dx=dx,
        dy=dx,
        dz=dx,
        field_shape=base_shape,
    )
    expected = (
        100.0 * z_coords[:, None]
        + 10.0 * float(monitor.plane_position)
        + x_coords[None, :]
    )

    np.testing.assert_allclose(
        monitor.fields["Ex"][0], expected, atol=1e-12, rtol=1e-12
    )
    np.testing.assert_allclose(
        monitor.fields["Hy"][0], expected, atol=1e-12, rtol=1e-12
    )
