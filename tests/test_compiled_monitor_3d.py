from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from beamz import Design, Material, Monitor, um
from beamz.devices.monitors.compiler import (
    compile_monitor_specs,
    sample_compiled_monitor_plane_component_3d,
)
from beamz.simulation.yee import (
    component_coordinates_2d_um,
    component_coordinates_3d_um,
)
from tests.test_3d_constitutive_sampling import (
    _build_centered_straight_guide_sim_steps,
    _build_step_driven_test_source,
)


def _linear_component(
    component: str, base_shape: tuple[int, int, int], dx: float
) -> np.ndarray:
    coords = component_coordinates_3d_um(component, base_shape, float(dx / um))
    z = np.asarray(coords["z"], dtype=float)[:, None, None] * float(um)
    y = np.asarray(coords["y"], dtype=float)[None, :, None] * float(um)
    x = np.asarray(coords["x"], dtype=float)[None, None, :] * float(um)
    return (100.0 * z) + (10.0 * y) + x


def _linear_component_2d(component: str, grid_shape: tuple[int, int], dx: float):
    coords = component_coordinates_2d_um(component, grid_shape, float(dx), "xy")
    y = np.asarray(coords["y"], dtype=float)[:, None]
    x = np.asarray(coords["x"], dtype=float)[None, :]
    return 10.0 * x + y


def _sample_compiled_2d(field, flat_idx, weights):
    flat = np.asarray(field, dtype=np.complex128).reshape(-1)
    idx = np.asarray(flat_idx, dtype=np.int32)
    w = np.asarray(weights, dtype=np.float32)
    return np.sum(flat[idx] * w, axis=1)


def _make_arbitrary_monitor(normal: str) -> tuple[Design, Monitor]:
    design = Design(
        width=1.6 * um,
        height=1.4 * um,
        depth=1.2 * um,
        material=Material(1.44**2),
    )
    if normal == "x":
        start = (0.51 * um, 0.21 * um, 0.17 * um)
        end = (0.51 * um, 1.09 * um, 0.93 * um)
    elif normal == "y":
        start = (0.23 * um, 0.47 * um, 0.18 * um)
        end = (1.19 * um, 0.47 * um, 0.86 * um)
    else:
        start = (0.22 * um, 0.19 * um, 0.41 * um)
        end = (1.18 * um, 1.01 * um, 0.41 * um)
    monitor = Monitor(
        design=design,
        start=start,
        end=end,
        name=f"m_{normal}",
        dft_enabled=False,
    )
    return design, monitor


def test_2d_monitor_samples_te_components_at_component_locations():
    dx = 1.0
    grid_shape = (4, 5)
    design = Design(width=5.0, height=4.0, material=Material(1.0))
    monitor = Monitor(
        design=design,
        start=(1.5, 0.0),
        end=(1.5, 2.0),
        name="m2d",
        dft_enabled=False,
    )
    fields = {
        "Ex": _linear_component_2d("Ex", grid_shape, dx),
        "Ey": _linear_component_2d("Ey", grid_shape, dx),
        "Hz": _linear_component_2d("Hz", grid_shape, dx),
    }
    monitor.record_fields_2d(
        np.zeros((5, 6), dtype=np.float32),
        np.zeros((4, 6), dtype=np.float32),
        np.zeros((5, 5), dtype=np.float32),
        t=0.0,
        dx=dx,
        dy=dx,
        step=0,
        Ex=fields["Ex"],
        Ey=fields["Ey"],
        Hz=fields["Hz"],
    )

    x_targets, y_targets = monitor._line_sample_coords_2d(dx, dx)
    expected = 10.0 * np.asarray(x_targets) + np.asarray(y_targets)
    for comp in ("Ex", "Ey", "Hz"):
        np.testing.assert_allclose(monitor.fields[comp][0], expected)


def test_compiled_2d_monitor_interpolation_matches_imperative_record_fields():
    dx = 1.0
    grid_shape = (4, 5)
    design = Design(width=5.0, height=4.0, material=Material(1.0))
    monitor = Monitor(
        design=design,
        start=(1.5, 0.0),
        end=(1.5, 2.0),
        name="m2d_compiled",
        dft_enabled=False,
    )
    field_ns = SimpleNamespace(
        permittivity=np.ones(grid_shape, dtype=np.float32),
        plane_2d="xy",
        Ex=_linear_component_2d("Ex", grid_shape, dx),
        Ey=_linear_component_2d("Ey", grid_shape, dx),
        Ez=np.zeros((5, 6), dtype=np.float32),
        Hx=np.zeros((4, 6), dtype=np.float32),
        Hy=np.zeros((5, 5), dtype=np.float32),
        Hz=_linear_component_2d("Hz", grid_shape, dx),
    )
    monitor.record_fields_2d(
        field_ns.Ez,
        field_ns.Hx,
        field_ns.Hy,
        t=0.0,
        dx=dx,
        dy=dx,
        step=0,
        Ex=field_ns.Ex,
        Ey=field_ns.Ey,
        Hz=field_ns.Hz,
    )
    specs, _ = compile_monitor_specs(
        [monitor],
        field_ns,
        resolution=dx,
        num_steps=1,
        dt=1.0,
    )
    spec = specs[0]

    for comp in ("Ex", "Ey", "Hz"):
        sampled = _sample_compiled_2d(
            getattr(field_ns, comp),
            getattr(spec, f"{comp.lower()}_interp_flat_idx"),
            getattr(spec, f"{comp.lower()}_interp_weights"),
        )
        np.testing.assert_allclose(sampled, monitor.fields[comp][0])


@pytest.mark.parametrize("normal", ["x", "y", "z"])
def test_compiled_monitor_3d_plane_sampling_matches_offline_record_fields(normal: str):
    dx = 0.2 * um
    design, monitor = _make_arbitrary_monitor(normal)
    base_shape = (6, 7, 8)

    fields = {
        comp: _linear_component(comp, base_shape, dx)
        for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    }
    monitor.record_fields_3d(
        fields["Ex"],
        fields["Ey"],
        fields["Ez"],
        fields["Hx"],
        fields["Hy"],
        fields["Hz"],
        t=0.0,
        dx=dx,
        dy=dx,
        dz=dx,
        step=0,
    )

    field_ns = SimpleNamespace(**fields)
    specs, _ = compile_monitor_specs(
        [monitor],
        field_ns,
        resolution=float(dx),
        num_steps=1,
        dt=1.0,
    )
    spec = specs[0]

    for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        sampled = sample_compiled_monitor_plane_component_3d(
            fields[comp],
            getattr(spec, f"{comp.lower()}_interp_flat_idx"),
            getattr(spec, f"{comp.lower()}_interp_weights"),
            spec.min_dim0,
            spec.min_dim1,
        )
        np.testing.assert_allclose(
            sampled,
            np.asarray(monitor.fields[comp][0], dtype=np.complex128),
            atol=1e-10,
            rtol=1e-7,
        )


def test_compiled_monitor_3d_uses_full_canonical_analysis_plane():
    sim = _build_centered_straight_guide_sim_steps(
        ppw=6,
        axis="x",
        num_steps=1,
    )
    dx = float(sim.resolution)
    monitor = Monitor(
        design=sim.design,
        start=(0.5 * sim.design.width, 0.5 * sim.design.height - 6.0 * dx, 0.0),
        end=(
            0.5 * sim.design.width,
            0.5 * sim.design.height + 6.0 * dx,
            sim.design.depth,
        ),
        name="staggered",
        dft_enabled=True,
        dft_frequencies=[3.0e14],
    )

    specs, _ = compile_monitor_specs(
        [monitor],
        sim.fields,
        resolution=dx,
        num_steps=1,
        dt=float(sim.dt),
    )
    spec = specs[0]
    expected_points = spec.min_dim0 * spec.min_dim1
    coord0, coord1 = sim._monitor_analysis_plane_3d(monitor, "x")
    comp_coord0, comp_coord1 = sim._monitor_component_plane_coords_3d(
        monitor,
        "Ex",
        "x",
    )

    assert coord0.size == spec.min_dim0
    assert coord1.size == spec.min_dim1
    assert comp_coord0.size == spec.min_dim0
    assert comp_coord1.size == spec.min_dim1

    for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        flat_idx = getattr(spec, f"{comp.lower()}_interp_flat_idx")
        weights = getattr(spec, f"{comp.lower()}_interp_weights")
        assert tuple(flat_idx.shape) == (expected_points, 8)
        assert tuple(weights.shape) == (expected_points, 8)
        sampled = sample_compiled_monitor_plane_component_3d(
            np.asarray(getattr(sim.fields, comp)),
            flat_idx,
            weights,
            spec.min_dim0,
            spec.min_dim1,
        )
        assert sampled.shape == (spec.min_dim0, spec.min_dim1)


@pytest.mark.parametrize("direction,pol", [("+x", "te"), ("+y", "tm"), ("+z", "te")])
def test_compiled_monitor_3d_raw_plane_matches_offline_on_straight_guide(
    direction: str, pol: str
):
    sim = _build_centered_straight_guide_sim_steps(
        ppw=6,
        axis=direction[1],
        num_steps=40,
    )
    source, dx = _build_step_driven_test_source(sim, direction=direction, pol=pol)
    sim.sources = [source]
    spans = (12.0 * dx, 10.0 * dx)

    monitor_center = {
        "+x": (source.center[0] + 10.0 * dx, source.center[1], source.center[2]),
        "-x": (source.center[0] - 10.0 * dx, source.center[1], source.center[2]),
        "+y": (source.center[0], source.center[1] + 10.0 * dx, source.center[2]),
        "-y": (source.center[0], source.center[1] - 10.0 * dx, source.center[2]),
        "+z": (source.center[0], source.center[1], source.center[2] + 10.0 * dx),
        "-z": (source.center[0], source.center[1], source.center[2] - 10.0 * dx),
    }[direction]
    if direction[1] == "x":
        start = (
            monitor_center[0],
            monitor_center[1] - spans[0] / 2,
            monitor_center[2] - spans[1] / 2,
        )
        end = (
            monitor_center[0],
            monitor_center[1] + spans[0] / 2,
            monitor_center[2] + spans[1] / 2,
        )
    elif direction[1] == "y":
        start = (
            monitor_center[0] - spans[0] / 2,
            monitor_center[1],
            monitor_center[2] - spans[1] / 2,
        )
        end = (
            monitor_center[0] + spans[0] / 2,
            monitor_center[1],
            monitor_center[2] + spans[1] / 2,
        )
    else:
        start = (
            monitor_center[0] - spans[1] / 2,
            monitor_center[1] - spans[0] / 2,
            monitor_center[2],
        )
        end = (
            monitor_center[0] + spans[1] / 2,
            monitor_center[1] + spans[0] / 2,
            monitor_center[2],
        )
    monitor = Monitor(
        design=sim.design,
        start=start,
        end=end,
        name="cmp",
        dft_enabled=False,
    )

    for _ in range(40):
        sim.step()

    monitor.record_fields_3d(
        np.asarray(sim.fields.Ex),
        np.asarray(sim.fields.Ey),
        np.asarray(sim.fields.Ez),
        np.asarray(sim.fields.Hx),
        np.asarray(sim.fields.Hy),
        np.asarray(sim.fields.Hz),
        t=float(sim.t),
        dx=dx,
        dy=dx,
        dz=dx,
        step=int(sim.current_step),
    )
    specs, _ = compile_monitor_specs(
        [monitor],
        sim.fields,
        resolution=float(dx),
        num_steps=1,
        dt=float(sim.dt),
    )
    spec = specs[0]

    for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        sampled = sample_compiled_monitor_plane_component_3d(
            np.asarray(getattr(sim.fields, comp)),
            getattr(spec, f"{comp.lower()}_interp_flat_idx"),
            getattr(spec, f"{comp.lower()}_interp_weights"),
            spec.min_dim0,
            spec.min_dim1,
        )
        np.testing.assert_allclose(
            sampled,
            np.asarray(monitor.fields[comp][0], dtype=np.complex128),
            atol=1e-6,
            rtol=1e-6,
        )
