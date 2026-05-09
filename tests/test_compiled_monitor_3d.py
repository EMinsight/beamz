from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from beamz import PML, Design, Material, ModeSource, Monitor, Simulation, um
from beamz.devices.monitors.compiler import (
    compile_monitor_specs,
    sample_compiled_monitor_plane_component_3d,
)
from beamz.simulation.yee import component_coordinates_3d_um
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
