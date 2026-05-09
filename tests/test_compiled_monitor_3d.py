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


def _build_step_vs_compiled_dft_sims():
    wavelength = 1.55 * um
    freq = 299792458.0 / wavelength
    sim_a = _build_centered_straight_guide_sim_steps(ppw=6, axis="x", num_steps=40)
    source_a, dx = _build_step_driven_test_source(sim_a, direction="+x", pol="te")
    monitor_a = Monitor(
        design=sim_a.design,
        start=(
            source_a.center[0] + 10.0 * dx,
            source_a.center[1] - 6.0 * dx,
            source_a.center[2] - 5.0 * dx,
        ),
        end=(
            source_a.center[0] + 10.0 * dx,
            source_a.center[1] + 6.0 * dx,
            source_a.center[2] + 5.0 * dx,
        ),
        name="o1",
        record_fields=False,
        dft_enabled=True,
        dft_frequencies=np.array([freq], dtype=float),
        dft_components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        dft_window="none",
        dft_record_every_step=True,
    )
    sim_a.sources = [source_a]
    sim_a.monitors = []
    sim_a.devices = tuple(list(sim_a.sources))

    sim_b = _build_centered_straight_guide_sim_steps(ppw=6, axis="x", num_steps=40)
    source_b, _ = _build_step_driven_test_source(sim_b, direction="+x", pol="te")
    monitor_b = Monitor(
        design=sim_b.design,
        start=monitor_a.start,
        end=monitor_a.end,
        name="o1",
        record_fields=False,
        dft_enabled=True,
        dft_frequencies=np.array([freq], dtype=float),
        dft_components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        dft_window="none",
        dft_record_every_step=True,
    )
    sim_b.sources = [source_b]
    sim_b.monitors = [monitor_b]
    sim_b.devices = tuple(list(sim_b.sources) + list(sim_b.monitors))
    return sim_a, monitor_a, sim_b, monitor_b


def test_compiled_run_matches_step_fields_for_full_pec_split_modesource_3d():
    sim_step = _build_centered_straight_guide_sim_steps(ppw=6, axis="x", num_steps=40)
    source_step, _ = _build_step_driven_test_source(sim_step, direction="+x", pol="te")
    sim_step.sources = [source_step]
    sim_step.devices = tuple(list(sim_step.sources))

    sim_compiled = _build_centered_straight_guide_sim_steps(
        ppw=6, axis="x", num_steps=40
    )
    source_compiled, _ = _build_step_driven_test_source(
        sim_compiled, direction="+x", pol="te"
    )
    sim_compiled.sources = [source_compiled]
    sim_compiled.devices = tuple(list(sim_compiled.sources))

    for _ in range(40):
        sim_step.step()
    sim_compiled.run_compiled(num_steps=40, progress=False)

    for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        step_arr = np.asarray(getattr(sim_step.fields, comp), dtype=np.float64)
        compiled_arr = np.asarray(getattr(sim_compiled.fields, comp), dtype=np.float64)
        denom = max(1e-30, float(np.linalg.norm(compiled_arr)))
        rel = float(np.linalg.norm(step_arr - compiled_arr) / denom)
        assert rel < 1e-5, f"{comp} rel mismatch too large: {rel:.3e}"


def test_compiled_monitor_3d_dft_matches_step_monitor_dft_on_straight_guide():
    sim_step, monitor_step, sim_compiled, monitor_compiled = (
        _build_step_vs_compiled_dft_sims()
    )

    for _ in range(40):
        sim_step.step()
        monitor_step._dft_base_dt = float(sim_step.dt)
        monitor_step.record_fields_3d(
            np.asarray(sim_step.fields.Ex),
            np.asarray(sim_step.fields.Ey),
            np.asarray(sim_step.fields.Ez),
            np.asarray(sim_step.fields.Hx),
            np.asarray(sim_step.fields.Hy),
            np.asarray(sim_step.fields.Hz),
            t=float(sim_step.t),
            dx=float(sim_step.resolution),
            dy=float(sim_step.resolution),
            dz=float(sim_step.resolution),
            step=int(sim_step.current_step - 1),
        )

    sim_compiled.run_compiled(num_steps=40, progress=False)

    np.testing.assert_allclose(
        np.asarray(monitor_compiled._dft_weight_sum, dtype=np.float64),
        np.asarray(monitor_step._dft_weight_sum, dtype=np.float64),
        atol=1e-6,
        rtol=1e-6,
    )
    for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        compiled_accum = np.asarray(
            monitor_compiled._dft_accum[comp], dtype=np.complex128
        )
        step_accum = np.asarray(monitor_step._dft_accum[comp], dtype=np.complex128)
        denom = max(1e-30, float(np.linalg.norm(step_accum)))
        rel = float(np.linalg.norm(compiled_accum - step_accum) / denom)
        assert rel < 1e-5, f"{comp} DFT rel mismatch too large: {rel:.3e}"
