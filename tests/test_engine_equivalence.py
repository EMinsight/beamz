from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from beamz import (
    LIGHT_SPEED,
    PML,
    Design,
    Material,
    ModeSource,
    Rectangle,
    Simulation,
    calc_optimal_fdtd_params,
    ramped_cosine,
    um,
)
from beamz.const import EPS_0
from beamz.devices.sources.compiler import _as_slab_spec, _sample_waveform

FIELD_COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
TEST_WAVELENGTH = 1.55 * um

pytestmark = [pytest.mark.compiled, pytest.mark.component]


def _make_2d_sim(
    *,
    plane_2d: str,
    steps: int,
    sources=None,
    boundaries=None,
) -> tuple[Simulation, float]:
    wl = TEST_WAVELENGTH
    dx, dt = calc_optimal_fdtd_params(
        wl, 1.0, dims=2, safety_factor=0.9, points_per_wavelength=6
    )
    time = np.arange(steps, dtype=float) * dt
    design = Design(
        width=2.4 * wl,
        height=2.0 * wl,
        material=Material(permittivity=1.0),
    )
    sim = Simulation(
        design=design,
        sources=list(sources or ()),
        boundaries=list(boundaries or ()),
        time=time,
        resolution=dx,
        plane_2d=plane_2d,
    )
    return sim, wl


def _make_3d_sim(*, steps: int, sources=None, boundaries=None) -> tuple[Simulation, float]:
    wl = TEST_WAVELENGTH
    dx, dt = calc_optimal_fdtd_params(
        wl, 1.0, dims=3, safety_factor=0.9, points_per_wavelength=5
    )
    time = np.arange(steps, dtype=float) * dt
    design = Design(
        width=1.8 * wl,
        height=1.8 * wl,
        depth=1.8 * wl,
        material=Material(permittivity=1.0),
    )
    sim = Simulation(
        design=design,
        sources=list(sources or ()),
        boundaries=list(boundaries or ()),
        time=time,
        resolution=dx,
    )
    return sim, wl


def _make_2d_tm_mode_source_sim(*, steps: int) -> tuple[Simulation, float]:
    wl = TEST_WAVELENGTH
    n_core = 2.0
    n_clad = 1.45
    dx, dt = calc_optimal_fdtd_params(
        wl, n_core, dims=2, safety_factor=0.95, points_per_wavelength=10
    )

    width = 8 * wl
    height = 5 * wl
    wg_w = 0.8 * wl
    design = Design(
        width=width, height=height, material=Material(permittivity=n_clad**2)
    )
    design += Rectangle(
        position=(width / 2, height / 2),
        width=width,
        height=wg_w,
        material=Material(permittivity=n_core**2),
    )

    time = np.arange(steps, dtype=float) * dt
    freq = LIGHT_SPEED / wl
    signal = ramped_cosine(
        time,
        amplitude=0.1,
        frequency=freq,
        ramp_duration=2 / freq,
        t_max=max(time[-1], dt) * 0.5,
    )
    source = ModeSource(
        grid=design.rasterize(resolution=dx),
        center=(2 * wl, height / 2),
        width=2.0 * wg_w,
        wavelength=wl,
        pol="tm",
        signal=signal,
        direction="+x",
    )
    sim = Simulation(
        design=design,
        sources=[source],
        boundaries=[PML(thickness=1.2 * wl)],
        time=time,
        resolution=dx,
    )
    return sim, wl


def _seed_field_payload(sim: Simulation, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    payload = {}
    for idx, name in enumerate(FIELD_COMPONENTS, start=1):
        field = np.asarray(getattr(sim.fields, name))
        if field.size == 0:
            payload[name] = field.astype(np.float32, copy=True)
            continue
        payload[name] = rng.normal(
            loc=0.0,
            scale=0.015 / idx,
            size=field.shape,
        ).astype(np.float32)
    return payload


def _apply_field_payload(sim: Simulation, payload: dict[str, np.ndarray]) -> None:
    for name, values in payload.items():
        setattr(sim.fields, name, jnp.asarray(values))


def _seed_interior_field_payload(sim: Simulation, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    payload = {}
    for idx, name in enumerate(FIELD_COMPONENTS, start=1):
        field = np.zeros_like(np.asarray(getattr(sim.fields, name)), dtype=np.float32)
        if field.ndim == 3 and min(field.shape) > 2:
            interior = (slice(1, -1), slice(1, -1), slice(1, -1))
            field[interior] = rng.normal(
                loc=0.0,
                scale=0.015 / idx,
                size=field[interior].shape,
            ).astype(np.float32)
        payload[name] = field
    return payload


def _run_reference(sim: Simulation, steps: int) -> None:
    for _ in range(steps):
        assert sim.step() is True


def _assert_fields_close(
    reference: Simulation,
    compiled: Simulation,
    *,
    atol: float,
    rtol: float,
) -> None:
    assert reference.current_step == compiled.current_step
    assert reference.current_step == reference.num_steps
    assert compiled.current_step == compiled.num_steps
    for name in FIELD_COMPONENTS:
        np.testing.assert_allclose(
            np.asarray(getattr(reference.fields, name)),
            np.asarray(getattr(compiled.fields, name)),
            atol=atol,
            rtol=rtol,
            err_msg=f"field mismatch for {name}",
        )


class _PointElectricCurrentSource:
    def __init__(self, component: str, index: tuple[int, ...], *, frequency_scale: float):
        self.component = str(component)
        self.index = tuple(int(v) for v in index)
        self.frequency_scale = float(frequency_scale)
        self._advanced_index = tuple(
            np.asarray([v], dtype=np.int32) for v in self.index
        )

    def _signal(self, t_sample: float) -> float:
        return float(
            0.6 * np.sin(self.frequency_scale * float(t_sample))
            + 0.25 * np.cos(0.5 * self.frequency_scale * float(t_sample))
        )

    def get_source_terms(self, fields, t, dt, current_step, resolution, design):
        del fields, current_step, resolution, design
        signal_value = self._signal(float(t) + 0.5 * float(dt))
        values = np.asarray([-signal_value], dtype=np.float32)
        return {self.component: (values, self._advanced_index)}, {}

    def compile_source_specs(
        self,
        *,
        fields,
        dt: float,
        num_steps: int,
        t0: float,
        resolution: float,
        total_steps: int | None = None,
    ):
        del resolution
        axis = self.component[-1].lower()
        eps_region = np.asarray(
            getattr(fields, f"eps_{axis}")[self.index],
            dtype=np.float32,
        )
        sig_region = np.asarray(
            getattr(fields, f"sig_{axis}")[self.index],
            dtype=np.float32,
        )
        denom = 1.0 + sig_region * (float(dt) / (2.0 * EPS_0 * eps_region))
        source_coeff = (float(dt) / (EPS_0 * eps_region)) / denom
        coeff = np.asarray([-source_coeff], dtype=np.float32)
        waveform = _sample_waveform(
            lambda t_sample, _dt: self._signal(float(t_sample)),
            t0=t0,
            dt=dt,
            num_steps=num_steps,
            offset_fn=lambda t_sample, dt_sample: t_sample + 0.5 * dt_sample,
            total_steps=total_steps,
        )
        return (
            _as_slab_spec(
                component=self.component,
                timing="e",
                index=self._advanced_index,
                coeff=coeff,
                waveform=waveform,
                target_shape=tuple(getattr(fields, self.component).shape),
            ),
        )


def _center_index(shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(int(axis // 2) for axis in shape)


@pytest.mark.parametrize("plane_2d", ["xy", "yz", "xz"])
def test_step_and_compiled_match_seeded_fields_without_sources_2d(plane_2d):
    steps = 7
    reference, _ = _make_2d_sim(plane_2d=plane_2d, steps=steps)
    compiled, _ = _make_2d_sim(plane_2d=plane_2d, steps=steps)

    seed = {"xy": 1101, "yz": 2202, "xz": 3303}[plane_2d]
    initial_fields = _seed_field_payload(reference, seed=seed)
    _apply_field_payload(reference, initial_fields)
    _apply_field_payload(compiled, initial_fields)

    _run_reference(reference, steps)
    compiled.run_compiled(num_steps=steps, progress=False)

    _assert_fields_close(reference, compiled, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("plane_2d", ["xy", "yz", "xz"])
def test_step_and_compiled_match_point_source_without_pml_2d(plane_2d):
    steps = 9
    probe, _ = _make_2d_sim(plane_2d=plane_2d, steps=steps)
    source_index = _center_index(tuple(np.asarray(probe.fields.Ez).shape))
    source_a = _PointElectricCurrentSource("Ez", source_index, frequency_scale=3.5e14)
    source_b = _PointElectricCurrentSource("Ez", source_index, frequency_scale=3.5e14)
    reference, _ = _make_2d_sim(plane_2d=plane_2d, steps=steps, sources=[source_a])
    compiled, _ = _make_2d_sim(plane_2d=plane_2d, steps=steps, sources=[source_b])

    _run_reference(reference, steps)
    compiled.run_compiled(num_steps=steps, progress=False)

    _assert_fields_close(reference, compiled, atol=2e-6, rtol=2e-6)


def test_step_and_compiled_match_seeded_fields_with_cpml_2d_xy():
    steps = 6
    pml_thickness = 0.45 * TEST_WAVELENGTH
    reference, _ = _make_2d_sim(
        plane_2d="xy",
        steps=steps,
        boundaries=[PML(thickness=pml_thickness, formulation="cpml")],
    )
    compiled, _ = _make_2d_sim(
        plane_2d="xy",
        steps=steps,
        boundaries=[PML(thickness=pml_thickness, formulation="cpml")],
    )

    initial_fields = _seed_field_payload(reference, seed=2405)
    _apply_field_payload(reference, initial_fields)
    _apply_field_payload(compiled, initial_fields)

    _run_reference(reference, steps)
    compiled.run_compiled(num_steps=steps, progress=False)

    _assert_fields_close(reference, compiled, atol=3e-5, rtol=2e-4)


@pytest.mark.parametrize("plane_2d", ["yz", "xz"])
def test_step_and_compiled_match_seeded_fields_with_sigma_pml_2d(plane_2d):
    steps = 6
    pml_thickness = 0.45 * TEST_WAVELENGTH
    reference, _ = _make_2d_sim(
        plane_2d=plane_2d,
        steps=steps,
        boundaries=[PML(thickness=pml_thickness)],
    )
    compiled, _ = _make_2d_sim(
        plane_2d=plane_2d,
        steps=steps,
        boundaries=[PML(thickness=pml_thickness)],
    )

    initial_fields = _seed_field_payload(reference, seed=2607 if plane_2d == "yz" else 2709)
    _apply_field_payload(reference, initial_fields)
    _apply_field_payload(compiled, initial_fields)

    _run_reference(reference, steps)
    compiled.run_compiled(num_steps=steps, progress=False)

    _assert_fields_close(reference, compiled, atol=3e-5, rtol=2e-4)


def test_step_and_compiled_match_point_source_with_cpml_2d_xy():
    steps = 9
    pml_thickness = 0.45 * TEST_WAVELENGTH
    probe, _ = _make_2d_sim(
        plane_2d="xy",
        steps=steps,
        boundaries=[PML(thickness=pml_thickness, formulation="cpml")],
    )
    source_index = _center_index(tuple(np.asarray(probe.fields.Ez).shape))
    source_a = _PointElectricCurrentSource("Ez", source_index, frequency_scale=3.5e14)
    source_b = _PointElectricCurrentSource("Ez", source_index, frequency_scale=3.5e14)
    reference, _ = _make_2d_sim(
        plane_2d="xy",
        steps=steps,
        sources=[source_a],
        boundaries=[PML(thickness=pml_thickness, formulation="cpml")],
    )
    compiled, _ = _make_2d_sim(
        plane_2d="xy",
        steps=steps,
        sources=[source_b],
        boundaries=[PML(thickness=pml_thickness, formulation="cpml")],
    )

    _run_reference(reference, steps)
    compiled.run_compiled(num_steps=steps, progress=False)

    _assert_fields_close(reference, compiled, atol=5e-5, rtol=5e-4)


def test_step_and_compiled_match_seeded_fields_without_sources_3d():
    steps = 5
    reference, _ = _make_3d_sim(steps=steps)
    compiled, _ = _make_3d_sim(steps=steps)

    # Keep the deterministic seed away from the compact 3D high-wall omission so
    # this test validates the shared bulk Yee update, not ghost-boundary choices.
    initial_fields = _seed_interior_field_payload(reference, seed=3301)
    _apply_field_payload(reference, initial_fields)
    _apply_field_payload(compiled, initial_fields)

    _run_reference(reference, steps)
    compiled.run_compiled(num_steps=steps, progress=False)

    _assert_fields_close(reference, compiled, atol=2e-6, rtol=2e-6)


def test_step_and_compiled_match_point_source_without_pml_3d():
    steps = 7
    probe, _ = _make_3d_sim(steps=steps)
    source_index = _center_index(tuple(np.asarray(probe.fields.Ez).shape))
    source_a = _PointElectricCurrentSource("Ez", source_index, frequency_scale=3.0e14)
    source_b = _PointElectricCurrentSource("Ez", source_index, frequency_scale=3.0e14)
    reference, _ = _make_3d_sim(steps=steps, sources=[source_a])
    compiled, _ = _make_3d_sim(steps=steps, sources=[source_b])

    _run_reference(reference, steps)
    compiled.run_compiled(num_steps=steps, progress=False)

    _assert_fields_close(reference, compiled, atol=2e-6, rtol=2e-6)


def test_step_and_compiled_match_seeded_fields_with_cpml_3d():
    steps = 4
    pml_thickness = 0.45 * TEST_WAVELENGTH
    reference, _ = _make_3d_sim(
        steps=steps,
        boundaries=[PML(thickness=pml_thickness, formulation="cpml")],
    )
    compiled, _ = _make_3d_sim(
        steps=steps,
        boundaries=[PML(thickness=pml_thickness, formulation="cpml")],
    )

    initial_fields = _seed_field_payload(reference, seed=4407)
    _apply_field_payload(reference, initial_fields)
    _apply_field_payload(compiled, initial_fields)

    _run_reference(reference, steps)
    compiled.run_compiled(num_steps=steps, progress=False)

    _assert_fields_close(reference, compiled, atol=4e-5, rtol=3e-4)


def test_step_and_compiled_match_point_source_with_cpml_3d():
    steps = 6
    pml_thickness = 0.45 * TEST_WAVELENGTH
    probe, _ = _make_3d_sim(
        steps=steps,
        boundaries=[PML(thickness=pml_thickness, formulation="cpml")],
    )
    source_index = _center_index(tuple(np.asarray(probe.fields.Ez).shape))
    source_a = _PointElectricCurrentSource("Ez", source_index, frequency_scale=3.0e14)
    source_b = _PointElectricCurrentSource("Ez", source_index, frequency_scale=3.0e14)
    reference, _ = _make_3d_sim(
        steps=steps,
        sources=[source_a],
        boundaries=[PML(thickness=pml_thickness, formulation="cpml")],
    )
    compiled, _ = _make_3d_sim(
        steps=steps,
        sources=[source_b],
        boundaries=[PML(thickness=pml_thickness, formulation="cpml")],
    )

    _run_reference(reference, steps)
    compiled.run_compiled(num_steps=steps, progress=False)

    _assert_fields_close(reference, compiled, atol=6e-5, rtol=6e-4)


def test_step_and_compiled_match_tm_mode_source_snapshots_without_fallback():
    steps = 20
    snapshot_interval = 5
    reference, _ = _make_2d_tm_mode_source_sim(steps=steps)
    compiled, _ = _make_2d_tm_mode_source_sim(steps=steps)

    reference_snapshots = []
    reference_steps = []
    for _ in range(steps):
        assert reference.step() is True
        if reference.current_step % snapshot_interval == 0:
            reference_snapshots.append(np.asarray(reference.fields.Ez, dtype=np.float32))
            reference_steps.append(reference.current_step)

    def _unexpected_step():
        raise AssertionError("TM snapshot runs should stay on the compiled engine path")

    compiled.step = _unexpected_step
    result = compiled.run_compiled(
        num_steps=steps,
        progress=False,
        snapshot_field="Ez",
        snapshot_interval=snapshot_interval,
        store_snapshots=True,
    )

    compiled_snapshots = [
        np.asarray(frame["field"], dtype=np.float32) / 1e-6
        for frame in result["snapshots"]
    ]
    compiled_steps = [int(frame["step"]) for frame in result["snapshots"]]

    assert compiled_steps == reference_steps
    assert len(compiled_snapshots) == len(reference_snapshots)
    for ref_frame, cmp_frame in zip(
        reference_snapshots, compiled_snapshots, strict=True
    ):
        np.testing.assert_allclose(ref_frame, cmp_frame, atol=3e-5, rtol=3e-4)

    _assert_fields_close(reference, compiled, atol=3e-5, rtol=3e-4)


def test_private_jit_step_builders_are_disabled():
    sim, _ = _make_2d_sim(plane_2d="yz", steps=2)

    with pytest.raises(NotImplementedError, match="deprecated"):
        sim._create_jit_step()
    with pytest.raises(NotImplementedError, match="deprecated"):
        sim._create_jit_step_h()
    with pytest.raises(NotImplementedError, match="deprecated"):
        sim._create_jit_step_e()
