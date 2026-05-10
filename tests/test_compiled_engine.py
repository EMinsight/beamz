import jax
import jax.numpy as jnp
import numpy as np
import pytest
from types import SimpleNamespace

from beamz import (
    LIGHT_SPEED,
    PEC,
    PML,
    Design,
    GaussianSource,
    Material,
    ModeSource,
    Monitor,
    Rectangle,
    Simulation,
    calc_optimal_fdtd_params,
    ramped_cosine,
    um,
)
from beamz.const import EPS_0
from beamz.devices.monitors.compiler import CompiledMonitorSpec
from beamz.devices.sources.compiler import (
    _as_slab_spec,
    _compile_mode_source_3d,
    _sample_waveform,
)
from beamz.simulation.boundaries import initialize_full_pec_3d_state
from beamz.simulation.compiled import (
    CompiledRunConfig,
    CompiledSimulation,
    EngineState,
    MonitorState,
)

pytestmark = [pytest.mark.compiled, pytest.mark.component]


@pytest.fixture
def small_sim_params():
    wl = 1.55 * um
    dx, dt = calc_optimal_fdtd_params(
        wl, 1.0, dims=2, safety_factor=0.95, points_per_wavelength=10
    )
    domain = 5.0 * wl
    steps = 120
    t = np.arange(0, steps * dt, dt)
    freq = LIGHT_SPEED / wl
    signal = ramped_cosine(
        t,
        amplitude=1.0,
        frequency=freq,
        ramp_duration=2 / freq,
        t_max=t[-1] * 0.4,
    )
    return wl, dx, dt, domain, steps, t, signal


def _engine_state_for_sim(sim: Simulation) -> EngineState:
    if (not sim.is_3d) and sim.plane_2d == "xy":
        tm_ez = sim.fields.Ez
        tm_hx = sim.fields.Hx
        tm_hy = sim.fields.Hy
    else:
        tm_ez = jnp.zeros((0, 0), dtype=sim.fields.Ez.dtype)
        tm_hx = jnp.zeros((0, 0), dtype=sim.fields.Hx.dtype)
        tm_hy = jnp.zeros((0, 0), dtype=sim.fields.Hy.dtype)
    if sim.is_3d:
        try:
            fp_state = (
                sim.fields.full_pec_3d_state
                if sim.fields.full_pec_3d_state is not None
                else initialize_full_pec_3d_state(sim.fields)
            )
        except Exception:
            fp_ex = jnp.zeros((0, 0, 0), dtype=sim.fields.Ex.dtype)
            fp_ey = jnp.zeros((0, 0, 0), dtype=sim.fields.Ey.dtype)
            fp_ez = jnp.zeros((0, 0, 0), dtype=sim.fields.Ez.dtype)
            fp_hx = jnp.zeros((0, 0, 0), dtype=sim.fields.Hx.dtype)
            fp_hy = jnp.zeros((0, 0, 0), dtype=sim.fields.Hy.dtype)
            fp_hz = jnp.zeros((0, 0, 0), dtype=sim.fields.Hz.dtype)
        else:
            fp_ex = fp_state.Ex
            fp_ey = fp_state.Ey
            fp_ez = fp_state.Ez
            fp_hx = fp_state.Hx
            fp_hy = fp_state.Hy
            fp_hz = fp_state.Hz
    else:
        fp_ex = jnp.zeros((0, 0, 0), dtype=sim.fields.Ex.dtype)
        fp_ey = jnp.zeros((0, 0, 0), dtype=sim.fields.Ey.dtype)
        fp_ez = jnp.zeros((0, 0, 0), dtype=sim.fields.Ez.dtype)
        fp_hx = jnp.zeros((0, 0, 0), dtype=sim.fields.Hx.dtype)
        fp_hy = jnp.zeros((0, 0, 0), dtype=sim.fields.Hy.dtype)
        fp_hz = jnp.zeros((0, 0, 0), dtype=sim.fields.Hz.dtype)

    return EngineState(
        ex=sim.fields.Ex,
        ey=sim.fields.Ey,
        ez=sim.fields.Ez,
        hx=sim.fields.Hx,
        hy=sim.fields.Hy,
        hz=sim.fields.Hz,
        tm_ez=tm_ez,
        tm_hx=tm_hx,
        tm_hy=tm_hy,
        fp_ex=fp_ex,
        fp_ey=fp_ey,
        fp_ez=fp_ez,
        fp_hx=fp_hx,
        fp_hy=fp_hy,
        fp_hz=fp_hz,
        cpml_psi_h_terms=jnp.zeros((2, 0, 0), dtype=sim.fields.Hx.dtype),
        cpml_psi_e_terms=jnp.zeros((2, 0, 0), dtype=sim.fields.Ez.dtype),
        cpml3d_psi_h_terms=tuple(
            jnp.zeros((0, 0, 0), dtype=sim.fields.Hx.dtype) for _ in range(6)
        ),
        cpml3d_psi_e_terms=tuple(
            jnp.zeros((0, 0, 0), dtype=sim.fields.Ez.dtype) for _ in range(6)
        ),
        t=jnp.asarray(sim.t, dtype=jnp.float32),
        current_step=jnp.asarray(sim.current_step, dtype=jnp.int32),
    )


class _PlaneCurrentSheetSource3D:
    def __init__(self, signal, ix: int, iy0: int, iy1: int, iz0: int, iz1: int):
        self.signal = signal
        self.ix = int(ix)
        self.iy0 = int(iy0)
        self.iy1 = int(iy1)
        self.iz0 = int(iz0)
        self.iz1 = int(iz1)

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
        idx = (slice(self.iz0, self.iz1), slice(self.iy0, self.iy1), self.ix)
        eps_region = np.asarray(fields.eps_z[idx], dtype=np.float32)
        sig_region = np.asarray(fields.sig_z[idx], dtype=np.float32)
        denom = 1.0 + sig_region * (float(dt) / (2.0 * EPS_0 * eps_region))
        source_coeff = (float(dt) / (EPS_0 * eps_region)) / denom
        coeff = -source_coeff
        waveform = _sample_waveform(
            lambda t_sample, _dt: self.signal(float(t_sample)),
            t0=t0,
            dt=dt,
            num_steps=num_steps,
            offset_fn=lambda t, dt_: t + 0.5 * dt_,
            total_steps=total_steps,
        )
        return (
            _as_slab_spec(
                component="Ez",
                timing="e",
                index=idx,
                coeff=coeff,
                waveform=waveform,
                target_shape=tuple(fields.Ez.shape),
            ),
        )


def test_run_compiled_supports_3d_custom_current_source():
    class _CurrentSource:
        def __init__(self, signal, voxel_indices, voxel_weights):
            self.signal = signal
            self._voxel_weights = np.asarray(voxel_weights, dtype=np.float32)
            z_idx = np.asarray([idx[0] for idx in voxel_indices], dtype=np.int32)
            y_idx = np.asarray([idx[1] for idx in voxel_indices], dtype=np.int32)
            x_idx = np.asarray([idx[2] for idx in voxel_indices], dtype=np.int32)
            self._indices = (z_idx, y_idx, x_idx)

        def get_source_terms(self, fields, t, dt, current_step, resolution, design):
            del fields, current_step, resolution, design
            signal_value = float(self.signal(float(t) + 0.5 * float(dt)))
            values = -self._voxel_weights * signal_value
            return {"Ez": (values, self._indices)}, {}

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
            eps_region = np.asarray(fields.eps_z[self._indices], dtype=np.float32)
            sig_region = np.asarray(fields.sig_z[self._indices], dtype=np.float32)
            denom = 1.0 + sig_region * (float(dt) / (2.0 * EPS_0 * eps_region))
            source_coeff = (float(dt) / (EPS_0 * eps_region)) / denom
            coeff = -self._voxel_weights * source_coeff
            waveform = _sample_waveform(
                lambda t_sample, _dt: self.signal(float(t_sample)),
                t0=t0,
                dt=dt,
                num_steps=num_steps,
                offset_fn=lambda t, dt_: t + 0.5 * dt_,
                total_steps=total_steps,
            )
            return (
                _as_slab_spec(
                    component="Ez",
                    timing="e",
                    index=self._indices,
                    coeff=coeff,
                    waveform=waveform,
                    target_shape=tuple(fields.Ez.shape),
                ),
            )

    wl = 1.55 * um
    dx, dt = calc_optimal_fdtd_params(
        wl, 1.0, dims=3, safety_factor=0.95, points_per_wavelength=8
    )
    steps = 18
    t = np.arange(0, steps * dt, dt)
    freq = LIGHT_SPEED / wl
    center = 1.25 * wl
    sigma_t = 0.45 / freq

    def signal(t_s: float) -> float:
        envelope = np.exp(-0.5 * ((float(t_s) - center / LIGHT_SPEED) / sigma_t) ** 2)
        carrier = np.cos(2.0 * np.pi * freq * float(t_s))
        return float(envelope * carrier)

    domain = 2.4 * wl
    design = Design(
        width=domain,
        height=domain,
        depth=domain,
        material=Material(permittivity=1.0),
    )
    design += Rectangle(
        position=(0.8 * wl, 0.8 * wl, 0.8 * wl),
        width=0.6 * wl,
        height=0.7 * wl,
        depth=0.5 * wl,
        material=Material(permittivity=2.5),
    )

    grid_n = int(round(domain / dx))
    source_center = np.array([grid_n // 2, grid_n // 2, grid_n // 2], dtype=np.int32)
    voxel_indices = []
    voxel_weights = []
    for dz in (-1, 0):
        for dy in (-1, 0):
            for dx_idx in (-1, 0):
                idx = tuple(
                    (
                        source_center + np.array([dz, dy, dx_idx], dtype=np.int32)
                    ).tolist()
                )
                voxel_indices.append(idx)
                voxel_weights.append(1.0 / 8.0)
    source_b = _CurrentSource(signal, voxel_indices, voxel_weights)
    sim_compiled = Simulation(
        design=design.copy(),
        sources=[source_b],
        boundaries=[PEC(edges="all")],
        time=t,
        resolution=dx,
    )

    sim_compiled.run_compiled(progress=False)

    assert sim_compiled.current_step == len(t)
    for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        arr = np.asarray(getattr(sim_compiled.fields, component))
        assert arr.size > 0
        assert np.isfinite(arr).all()
    assert float(np.max(np.abs(np.asarray(sim_compiled.fields.Ez)))) > 0.0


def test_run_compiled_supports_2d_cpml_small_case(small_sim_params):
    wl, dx, _dt, domain, _steps, t, signal = small_sim_params
    design = Design(width=domain, height=domain, material=Material(permittivity=1.0))

    source = GaussianSource(
        position=(domain / 2, domain / 2), width=wl / 6, signal=signal
    )
    sim = Simulation(
        design=design,
        sources=[source],
        boundaries=[PML(thickness=1.0 * wl, formulation="cpml")],
        time=t,
        resolution=dx,
    )

    sim.run_compiled(progress=False)

    for component in ("Ez", "Hx", "Hy"):
        arr = np.asarray(getattr(sim.fields, component))
        assert arr.size > 0
        assert np.isfinite(arr).all()


def test_run_compiled_supports_3d_cpml_small_case():
    wl = 1.55 * um
    dx, dt = calc_optimal_fdtd_params(
        wl, 1.0, dims=3, safety_factor=0.95, points_per_wavelength=8
    )
    domain = 3.0 * wl
    steps = 36
    t = np.arange(0, steps * dt, dt)
    freq = LIGHT_SPEED / wl
    signal = ramped_cosine(
        t,
        amplitude=1.0,
        frequency=freq,
        ramp_duration=2 / freq,
        t_max=t[-1] * 0.35,
    )

    design = Design(
        width=domain,
        height=domain,
        depth=domain,
        material=Material(permittivity=1.0),
    )
    source = GaussianSource(
        position=(domain / 2, domain / 2, domain / 2),
        width=wl / 5,
        signal=signal,
    )
    sim = Simulation(
        design=design,
        sources=[source],
        boundaries=[PML(thickness=1.0 * wl, formulation="cpml")],
        time=t,
        resolution=dx,
    )

    sim.run_compiled(progress=False)

    for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        arr = np.asarray(getattr(sim.fields, component))
        assert arr.size > 0
        assert np.isfinite(arr).all()


def test_split_3d_cpml_boundaries_preserve_identity_kappa_in_compiled_terms():
    wl = 1.55 * um
    dx, dt = calc_optimal_fdtd_params(
        wl, 1.0, dims=3, safety_factor=0.95, points_per_wavelength=8
    )
    time = np.arange(0, 2 * dt, dt)
    design = Design(
        width=6.0 * wl,
        height=6.0 * wl,
        depth=6.0 * wl,
        material=Material(permittivity=1.0),
    )
    sim = Simulation(
        design=design,
        sources=[],
        boundaries=[
            PML(
                edges=["left", "right", "top", "bottom"],
                thickness=1.0 * wl,
                formulation="cpml",
            ),
            PML(
                edges=["front", "back"],
                thickness=1.0 * wl,
                formulation="cpml",
            ),
        ],
        time=time,
        resolution=dx,
    )
    program = sim.compile(num_steps=1)

    cz = program.cpml3d_inv_kappa_e_terms[4].shape[0] // 2
    cy = program.cpml3d_inv_kappa_e_terms[4].shape[1] // 2
    cx = program.cpml3d_inv_kappa_e_terms[4].shape[2] // 2

    assert np.asarray(sim.pml_data["kappa_x"], dtype=np.float64)[
        cz, cy, cx
    ] == pytest.approx(1.0)
    assert np.asarray(sim.pml_data["kappa_y"], dtype=np.float64)[
        cz, cy, cx
    ] == pytest.approx(1.0)
    assert np.asarray(program.cpml3d_inv_kappa_e_terms[4], dtype=np.float64)[
        cz, cy, cx
    ] == pytest.approx(1.0)
    assert np.asarray(program.cpml3d_inv_kappa_h_terms[3], dtype=np.float64)[
        cz, cy, cx
    ] == pytest.approx(1.0)


def test_compiled_3d_cpml_profiles_match_expected_x_boundary_embedding():
    wl = 1.55 * um
    dx, dt = calc_optimal_fdtd_params(
        wl, 1.0, dims=3, safety_factor=0.95, points_per_wavelength=8
    )
    time = np.arange(0, 2 * dt, dt)
    thickness = 1.0 * wl
    design = Design(
        width=4.0 * wl,
        height=2.0 * wl,
        depth=2.0 * wl,
        material=Material(permittivity=1.0),
    )
    sim = Simulation(
        design=design,
        sources=[],
        boundaries=[
            PML(
                edges="all",
                thickness=thickness,
                formulation="cpml",
                kappa_max=4.0,
                alpha_max=300.0,
            )
        ],
        time=time,
        resolution=dx,
    )
    program = sim.compile(num_steps=1)

    nx = int(sim.fields.permittivity.shape[2])
    ny = int(sim.fields.permittivity.shape[1])
    nz = int(sim.fields.permittivity.shape[0])
    pml_cells = int(round(thickness / dx))

    def expected_profile(count: int, *, sample_kind: str):
        sigma = np.zeros((count,), dtype=np.float32)
        kappa = np.ones((count,), dtype=np.float32)
        alpha = np.zeros((count,), dtype=np.float32)
        sigma_max = float(sim.boundaries[0].sigma_max)

        if sample_kind == "E":
            low_d = np.arange(pml_cells - 1, -1, -1, dtype=np.float32)[
                : min(count, pml_cells)
            ]
            high_d = np.insert(
                np.arange(0.5, pml_cells - 0.5, 1.0, dtype=np.float32),
                0,
                0.0,
            )[: min(count, pml_cells)]
        else:
            low_d = np.append(
                np.arange(pml_cells - 1.5, -0.5, -1.0, dtype=np.float32),
                0.0,
            )[: min(count, pml_cells)]
            high_d = np.arange(0.0, pml_cells, 1.0, dtype=np.float32)[
                : min(count, pml_cells)
            ]

        for side, d in (("low", low_d), ("high", high_d)):
            u = np.clip(d / max(float(pml_cells), 1e-30), 0.0, 1.0)
            side_sigma = sigma_max * np.power(u, 3.0)
            side_kappa = 1.0 + (4.0 - 1.0) * np.power(u, 3.0)
            side_alpha = 300.0 * np.power(1.0 - u, 1.0)
            if side == "low":
                sigma[: len(d)] = np.maximum(sigma[: len(d)], side_sigma)
                kappa[: len(d)] = np.maximum(kappa[: len(d)], side_kappa)
                alpha[: len(d)] = np.maximum(alpha[: len(d)], side_alpha)
            else:
                sigma[-len(d) :] = np.maximum(sigma[-len(d) :], side_sigma)
                kappa[-len(d) :] = np.maximum(kappa[-len(d) :], side_kappa)
                alpha[-len(d) :] = np.maximum(alpha[-len(d) :], side_alpha)
        return sigma, kappa, alpha

    sigma_e_x, kappa_e_x, alpha_e_x = expected_profile(nx, sample_kind="E")
    sigma_h_x, kappa_h_x, alpha_h_x = expected_profile(max(nx - 1, 0), sample_kind="H")

    sigma_e_native = sigma_e_x[None, None, :].repeat(nz - 1, axis=0).repeat(ny, axis=1)
    kappa_e_native = kappa_e_x[None, None, :].repeat(nz - 1, axis=0).repeat(ny, axis=1)
    alpha_e_native = alpha_e_x[None, None, :].repeat(nz - 1, axis=0).repeat(ny, axis=1)
    sigma_h_native = sigma_h_x[None, None, :].repeat(nz - 1, axis=0).repeat(ny, axis=1)
    kappa_h_native = kappa_h_x[None, None, :].repeat(nz - 1, axis=0).repeat(ny, axis=1)
    alpha_h_native = alpha_h_x[None, None, :].repeat(nz - 1, axis=0).repeat(ny, axis=1)

    decay_e = (sigma_e_native / kappa_e_native + alpha_e_native) * (dt / EPS_0)
    b_e = np.expm1(-decay_e) + 1.0
    a_e = np.nan_to_num(
        ((b_e - 1.0) * sigma_e_native)
        / np.maximum(
            (sigma_e_native + kappa_e_native * alpha_e_native) * kappa_e_native, 1e-30
        )
    )
    decay_h = (sigma_h_native / kappa_h_native + alpha_h_native) * (dt / EPS_0)
    b_h = np.expm1(-decay_h) + 1.0
    a_h = np.nan_to_num(
        ((b_h - 1.0) * sigma_h_native)
        / np.maximum(
            (sigma_h_native + kappa_h_native * alpha_h_native) * kappa_h_native, 1e-30
        )
    )

    np.testing.assert_allclose(
        np.asarray(program.cpml3d_inv_kappa_e_terms[4]),
        1.0 / kappa_e_native,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(program.cpml3d_b_e_terms[4]), b_e, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(program.cpml3d_a_e_terms[4]), a_e, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(program.cpml3d_inv_kappa_h_terms[3]),
        1.0 / kappa_h_native,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(program.cpml3d_b_h_terms[3]), b_h, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(program.cpml3d_a_h_terms[3]), a_h, rtol=1e-6, atol=1e-6
    )


def test_compiled_monitor_power_is_populated(small_sim_params):
    wl, dx, _dt, domain, _steps, t, signal = small_sim_params
    design = Design(width=domain, height=domain, material=Material(permittivity=1.0))

    source = GaussianSource(
        position=(domain / 2, domain / 2), width=wl / 6, signal=signal
    )
    monitor = Monitor(
        start=(domain * 0.35, domain * 0.35),
        end=(domain * 0.35, domain * 0.65),
        record_interval=3,
    )

    sim = Simulation(
        design=design,
        sources=[source],
        monitors=[monitor],
        boundaries=[PML(thickness=1.2 * wl)],
        time=t,
        resolution=dx,
    )

    sim.run_compiled(progress=False)

    assert len(monitor.power_history) > 0
    assert len(monitor.power_timestamps) == len(monitor.power_history)
    assert np.isfinite(np.asarray(monitor.power_history)).all()


def test_compiled_monitor_accumulates_across_chunks(small_sim_params):
    wl, dx, _dt, domain, _steps, t, signal = small_sim_params
    design = Design(width=domain, height=domain, material=Material(permittivity=1.0))

    source_a = GaussianSource(
        position=(domain / 2, domain / 2), width=wl / 6, signal=signal
    )
    monitor_a = Monitor(
        start=(domain * 0.35, domain * 0.35),
        end=(domain * 0.35, domain * 0.65),
        record_interval=3,
    )
    sim_full = Simulation(
        design=design.copy(),
        sources=[source_a],
        monitors=[monitor_a],
        boundaries=[PML(thickness=1.2 * wl)],
        time=t,
        resolution=dx,
    )

    source_b = GaussianSource(
        position=(domain / 2, domain / 2), width=wl / 6, signal=signal
    )
    monitor_b = Monitor(
        start=(domain * 0.35, domain * 0.35),
        end=(domain * 0.35, domain * 0.65),
        record_interval=3,
    )
    sim_chunked = Simulation(
        design=design.copy(),
        sources=[source_b],
        monitors=[monitor_b],
        boundaries=[PML(thickness=1.2 * wl)],
        time=t,
        resolution=dx,
    )

    sim_full.run_compiled(num_steps=40, progress=False)
    sim_chunked.run_compiled(
        num_steps=40,
        record_interval=10,  # force chunked execution path
        record_fields=["Ez"],
        progress=False,
    )

    p_full = np.asarray(monitor_a.power_history)
    p_chunked = np.asarray(monitor_b.power_history)
    t_full = np.asarray(monitor_a.power_timestamps)
    t_chunked = np.asarray(monitor_b.power_timestamps)

    assert p_full.size > 0
    assert p_chunked.size == p_full.size
    assert t_chunked.size == t_full.size
    assert np.allclose(p_chunked, p_full, rtol=5e-3, atol=5e-5)
    assert np.allclose(t_chunked, t_full, rtol=0.0, atol=0.0)


def test_run_snapshot_path_does_not_fall_back_to_python_step(small_sim_params):
    wl, dx, _dt, domain, _steps, t, signal = small_sim_params
    design = Design(width=domain, height=domain, material=Material(permittivity=1.0))
    source = GaussianSource(
        position=(domain / 2, domain / 2), width=wl / 6, signal=signal
    )
    sim = Simulation(
        design=design,
        sources=[source],
        boundaries=[PML(thickness=1.2 * wl)],
        time=t,
        resolution=dx,
    )

    def _unexpected_step():
        raise AssertionError("snapshot runs should stay on the compiled engine path")

    sim.step = _unexpected_step
    result = sim.run(snapshot_field="Ez", snapshot_interval=8, progress=False)

    assert result is not None
    assert len(result["snapshots"]) > 0
    assert result["snapshots"][0]["step"] == 8


def test_compiled_frequency_monitor_matches_direct_sum(small_sim_params):
    wl, dx, dt, domain, _steps, t, signal = small_sim_params
    design = Design(width=domain, height=domain, material=Material(permittivity=1.0))

    source = GaussianSource(
        position=(domain / 2, domain / 2), width=wl / 6, signal=signal
    )
    freq = LIGHT_SPEED / wl
    monitor = Monitor(
        start=(domain * 0.35, domain * 0.35),
        end=(domain * 0.35, domain * 0.65),
        record_interval=1,
        frequency_points=[freq],
        frequency_record_interval=1,
    )

    sim = Simulation(
        design=design,
        sources=[source],
        monitors=[monitor],
        boundaries=[PML(thickness=1.2 * wl)],
        time=t,
        resolution=dx,
    )
    sim.run_compiled(num_steps=60, progress=False)

    assert monitor.frequency_flux_spectrum.shape == (1,)
    assert np.isfinite(monitor.frequency_flux_spectrum).all()

    power = np.asarray(monitor.power_history, dtype=np.float64)
    ts = np.asarray(monitor.power_timestamps, dtype=np.float64)
    direct = np.sum(power * np.exp(-1j * 2.0 * np.pi * freq * ts)) * dt
    assert np.allclose(
        monitor.frequency_flux_spectrum[0],
        direct,
        rtol=5e-3,
        atol=5e-6,
    )


def test_compiled_frequency_monitor_accumulates_across_chunks(small_sim_params):
    wl, dx, _dt, domain, _steps, t, signal = small_sim_params
    design = Design(width=domain, height=domain, material=Material(permittivity=1.0))
    freqs = [LIGHT_SPEED / wl, 1.1 * LIGHT_SPEED / wl]

    source_a = GaussianSource(
        position=(domain / 2, domain / 2), width=wl / 6, signal=signal
    )
    monitor_a = Monitor(
        start=(domain * 0.35, domain * 0.35),
        end=(domain * 0.35, domain * 0.65),
        record_interval=2,
        frequency_points=freqs,
        frequency_record_interval=1,
    )
    sim_full = Simulation(
        design=design.copy(),
        sources=[source_a],
        monitors=[monitor_a],
        boundaries=[PML(thickness=1.2 * wl)],
        time=t,
        resolution=dx,
    )

    source_b = GaussianSource(
        position=(domain / 2, domain / 2), width=wl / 6, signal=signal
    )
    monitor_b = Monitor(
        start=(domain * 0.35, domain * 0.35),
        end=(domain * 0.35, domain * 0.65),
        record_interval=2,
        frequency_points=freqs,
        frequency_record_interval=1,
    )
    sim_chunked = Simulation(
        design=design.copy(),
        sources=[source_b],
        monitors=[monitor_b],
        boundaries=[PML(thickness=1.2 * wl)],
        time=t,
        resolution=dx,
    )

    sim_full.run_compiled(num_steps=50, progress=False)
    sim_chunked.run_compiled(
        num_steps=50,
        record_interval=10,
        record_fields=["Ez"],
        progress=False,
    )

    s_full = np.asarray(monitor_a.frequency_flux_spectrum)
    s_chunked = np.asarray(monitor_b.frequency_flux_spectrum)
    assert s_full.shape == (2,)
    assert s_chunked.shape == s_full.shape
    assert np.allclose(s_chunked, s_full, rtol=5e-3, atol=5e-6)


def test_compiled_frequency_monitor_3d_populated():
    wl = 1.55 * um
    dx, dt = calc_optimal_fdtd_params(
        wl, 1.0, dims=3, safety_factor=0.95, points_per_wavelength=6
    )
    domain = 2.0 * wl
    depth = 1.5 * wl
    t = np.arange(0, 24 * dt, dt)
    freq = LIGHT_SPEED / wl
    signal = ramped_cosine(
        t,
        amplitude=1.0,
        frequency=freq,
        ramp_duration=2 / freq,
        t_max=t[-1] * 0.6,
    )

    design = Design(
        width=domain,
        height=domain,
        depth=depth,
        material=Material(permittivity=1.0),
    )
    source = GaussianSource(
        position=(domain * 0.45, domain * 0.5, depth * 0.5),
        width=wl / 5,
        signal=signal,
    )
    monitor = Monitor(
        design=design,
        start=(domain * 0.65, domain * 0.2, depth * 0.2),
        plane_normal="x",
        plane_position=domain * 0.65,
        size=(domain * 0.6, depth * 0.6),
        record_interval=2,
        frequency_points=[freq],
        frequency_record_interval=1,
        record_fields=False,
    )
    sim = Simulation(
        design=design,
        sources=[source],
        monitors=[monitor],
        boundaries=[PML(thickness=0.6 * wl, edges="all")],
        time=t,
        resolution=dx,
    )
    sim.run_compiled(num_steps=12, progress=False)

    spec = np.asarray(monitor.frequency_flux_spectrum)
    assert spec.shape == (1,)
    assert np.isfinite(spec).all()
    assert len(monitor.power_history) > 0
    assert np.isfinite(np.asarray(monitor.power_history)).all()


def test_compiled_dft_component_monitor_populated(small_sim_params):
    wl, dx, _dt, domain, _steps, t, signal = small_sim_params
    design = Design(width=domain, height=domain, material=Material(permittivity=1.0))
    source = GaussianSource(
        position=(domain / 2, domain / 2), width=wl / 6, signal=signal
    )
    freq = LIGHT_SPEED / wl
    monitor = Monitor(
        start=(domain * 0.35, domain * 0.35),
        end=(domain * 0.35, domain * 0.65),
        record_fields=False,
        dft_enabled=True,
        dft_frequencies=[freq],
        dft_components=("Ez", "Hy"),
        dft_t_start=0.0,
        dft_t_end=float(t[-1]),
        dft_window="rect",
        dft_record_every_step=True,
        record_interval=2,
    )
    sim = Simulation(
        design=design,
        sources=[source],
        monitors=[monitor],
        boundaries=[PML(thickness=1.2 * wl)],
        time=t,
        resolution=dx,
    )
    sim.run_compiled(num_steps=60, progress=False)

    ez_dft = np.asarray(monitor.get_dft_component("Ez"))
    hy_dft = np.asarray(monitor.get_dft_component("Hy"))
    assert ez_dft.shape[0] == 1
    assert hy_dft.shape == ez_dft.shape
    assert ez_dft.shape[1] > 0
    assert np.isfinite(ez_dft).all()
    assert np.isfinite(hy_dft).all()
    assert np.max(np.abs(ez_dft)) > 0.0
    assert np.max(np.abs(hy_dft)) > 0.0
    np.testing.assert_allclose(
        np.asarray(monitor.frequency_flux_spectrum, dtype=np.complex128),
        monitor.get_dft_flux().astype(np.complex128),
        rtol=1e-6,
        atol=1e-12,
    )


def test_compiled_static_monitor_dft_uses_current_sample_phase():
    program = CompiledSimulation.__new__(CompiledSimulation)
    program.config = CompiledRunConfig(
        resolution=1.0,
        dt=1.0,
        num_steps=1,
        plane_2d="xy",
        is_3d=False,
    )
    program.monitor_specs = (
        CompiledMonitorSpec(
            name="m",
            monitor_index=0,
            is_3d=False,
            record_interval=1,
            accumulate_power=False,
            power_scale=1.0,
            accumulate_frequency=True,
            freq_record_interval=1,
            freq_count=1,
            freq_hz=jnp.asarray([1.0], dtype=jnp.float32),
            freq_rot_re=jnp.asarray([0.0], dtype=jnp.float32),
            freq_rot_im=jnp.asarray([-1.0], dtype=jnp.float32),
            dft_enabled=True,
            dft_record_interval=1,
            dft_t_start=0.0,
            dft_t_end=1.0,
            dft_window_code=0,
            dft_point_count=1,
            dft_component_mask=jnp.asarray([0, 0, 1, 0, 0, 0], dtype=jnp.float32),
            x_ex=jnp.asarray([0], dtype=jnp.int32),
            y_ex=jnp.asarray([0], dtype=jnp.int32),
            valid_ex=jnp.asarray([0.0], dtype=jnp.float32),
            x_ey=jnp.asarray([0], dtype=jnp.int32),
            y_ey=jnp.asarray([0], dtype=jnp.int32),
            valid_ey=jnp.asarray([0.0], dtype=jnp.float32),
            x_ez=jnp.asarray([0], dtype=jnp.int32),
            y_ez=jnp.asarray([0], dtype=jnp.int32),
            valid_ez=jnp.asarray([1.0], dtype=jnp.float32),
            x_hx=jnp.asarray([0], dtype=jnp.int32),
            y_hx=jnp.asarray([0], dtype=jnp.int32),
            valid_hx=jnp.asarray([0.0], dtype=jnp.float32),
            x_hy=jnp.asarray([0], dtype=jnp.int32),
            y_hy=jnp.asarray([0], dtype=jnp.int32),
            valid_hy=jnp.asarray([0.0], dtype=jnp.float32),
            x_hz=jnp.asarray([0], dtype=jnp.int32),
            y_hz=jnp.asarray([0], dtype=jnp.int32),
            valid_hz=jnp.asarray([0.0], dtype=jnp.float32),
        ),
    )

    monitor_state = MonitorState(
        powers=jnp.zeros((1, 1), dtype=jnp.float32),
        timestamps=jnp.zeros((1, 1), dtype=jnp.float32),
        counts=jnp.zeros((1,), dtype=jnp.int32),
        freq_flux_re=jnp.zeros((1, 1), dtype=jnp.float32),
        freq_flux_im=jnp.zeros((1, 1), dtype=jnp.float32),
        freq_phase_re=jnp.ones((1, 1), dtype=jnp.float32),
        freq_phase_im=jnp.zeros((1, 1), dtype=jnp.float32),
        dft_vec_re=jnp.zeros((1, 6, 1, 1), dtype=jnp.float32),
        dft_vec_im=jnp.zeros((1, 6, 1, 1), dtype=jnp.float32),
        dft_weight_sum=jnp.zeros((1, 1), dtype=jnp.float32),
    )

    updated = program._update_monitors(
        monitor_state,
        abs_step=jnp.asarray(0, dtype=jnp.int32),
        t_phys=jnp.asarray(0.0, dtype=jnp.float32),
        dt_scalar=jnp.asarray(1.0, dtype=jnp.float32),
        ex=jnp.zeros((1, 1), dtype=jnp.float32),
        ey=jnp.zeros((1, 1), dtype=jnp.float32),
        ez=jnp.asarray([[2.0]], dtype=jnp.float32),
        hx=jnp.zeros((1, 1), dtype=jnp.float32),
        hy=jnp.zeros((1, 1), dtype=jnp.float32),
        hz=jnp.zeros((1, 1), dtype=jnp.float32),
        monitors_2d=program.monitor_specs,
    )

    np.testing.assert_allclose(
        updated.dft_vec_re[0, 2, 0, 0], 2.0, rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(
        updated.dft_vec_im[0, 2, 0, 0], 0.0, rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(updated.freq_phase_re[0, 0], 0.0, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(updated.freq_phase_im[0, 0], -1.0, rtol=1e-7, atol=1e-7)


def test_compiled_static_monitor_physical_dft_uses_centered_tm_xy_sampling():
    program = CompiledSimulation.__new__(CompiledSimulation)
    program.config = CompiledRunConfig(
        resolution=1.0,
        dt=1.0,
        num_steps=1,
        plane_2d="xy",
        is_3d=False,
    )
    program.monitor_specs = (
        CompiledMonitorSpec(
            name="m_physical",
            monitor_index=0,
            is_3d=False,
            record_interval=1,
            accumulate_power=False,
            power_scale=1.0,
            accumulate_frequency=True,
            freq_record_interval=1,
            freq_count=1,
            freq_hz=jnp.asarray([0.0], dtype=jnp.float32),
            freq_rot_re=jnp.asarray([1.0], dtype=jnp.float32),
            freq_rot_im=jnp.asarray([0.0], dtype=jnp.float32),
            dft_enabled=True,
            dft_record_interval=1,
            dft_t_start=0.0,
            dft_t_end=1.0,
            dft_window_code=0,
            dft_normalization_code=1,
            dft_length_unit=float(LIGHT_SPEED),
            dft_centered_tm_xy_sampling=True,
            dft_point_count=1,
            dft_component_mask=jnp.asarray([0, 0, 1, 0, 0, 0], dtype=jnp.float32),
            dft_target_x=jnp.asarray([0.5], dtype=jnp.float32),
            dft_target_y=jnp.asarray([0.5], dtype=jnp.float32),
            x_ex=jnp.asarray([0], dtype=jnp.int32),
            y_ex=jnp.asarray([0], dtype=jnp.int32),
            valid_ex=jnp.asarray([0.0], dtype=jnp.float32),
            x_ey=jnp.asarray([0], dtype=jnp.int32),
            y_ey=jnp.asarray([0], dtype=jnp.int32),
            valid_ey=jnp.asarray([0.0], dtype=jnp.float32),
            x_ez=jnp.asarray([0], dtype=jnp.int32),
            y_ez=jnp.asarray([0], dtype=jnp.int32),
            valid_ez=jnp.asarray([0.0], dtype=jnp.float32),
            x_hx=jnp.asarray([0], dtype=jnp.int32),
            y_hx=jnp.asarray([0], dtype=jnp.int32),
            valid_hx=jnp.asarray([0.0], dtype=jnp.float32),
            x_hy=jnp.asarray([0], dtype=jnp.int32),
            y_hy=jnp.asarray([0], dtype=jnp.int32),
            valid_hy=jnp.asarray([0.0], dtype=jnp.float32),
            x_hz=jnp.asarray([0], dtype=jnp.int32),
            y_hz=jnp.asarray([0], dtype=jnp.int32),
            valid_hz=jnp.asarray([0.0], dtype=jnp.float32),
        ),
    )

    monitor_state = MonitorState(
        powers=jnp.zeros((1, 1), dtype=jnp.float32),
        timestamps=jnp.zeros((1, 1), dtype=jnp.float32),
        counts=jnp.zeros((1,), dtype=jnp.int32),
        freq_flux_re=jnp.zeros((1, 1), dtype=jnp.float32),
        freq_flux_im=jnp.zeros((1, 1), dtype=jnp.float32),
        freq_phase_re=jnp.ones((1, 1), dtype=jnp.float32),
        freq_phase_im=jnp.zeros((1, 1), dtype=jnp.float32),
        dft_vec_re=jnp.zeros((1, 6, 1, 1), dtype=jnp.float32),
        dft_vec_im=jnp.zeros((1, 6, 1, 1), dtype=jnp.float32),
        dft_weight_sum=jnp.zeros((1, 1), dtype=jnp.float32),
    )

    tm_ez = jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)

    updated = program._update_monitors(
        monitor_state,
        abs_step=jnp.asarray(0, dtype=jnp.int32),
        t_phys=jnp.asarray(0.0, dtype=jnp.float32),
        dt_scalar=jnp.asarray(1.0, dtype=jnp.float32),
        ex=jnp.zeros((1, 1), dtype=jnp.float32),
        ey=jnp.zeros((1, 1), dtype=jnp.float32),
        ez=jnp.zeros((1, 1), dtype=jnp.float32),
        hx=jnp.zeros((1, 1), dtype=jnp.float32),
        hy=jnp.zeros((1, 1), dtype=jnp.float32),
        hz=jnp.zeros((1, 1), dtype=jnp.float32),
        tm_ez=tm_ez,
        tm_hx=jnp.zeros((1, 2), dtype=jnp.float32),
        tm_hy=jnp.zeros((2, 1), dtype=jnp.float32),
        monitors_2d=program.monitor_specs,
    )

    expected = 2.5 / np.sqrt(2.0 * np.pi)
    np.testing.assert_allclose(
        updated.dft_vec_re[0, 2, 0, 0], expected, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        updated.dft_vec_im[0, 2, 0, 0], 0.0, rtol=1e-6, atol=1e-6
    )


def test_compiled_program_compiles_once(small_sim_params):
    _wl, _dx, _dt, _domain, _steps, _t, _signal = small_sim_params

    wl = 1.2 * um
    dx, dt = calc_optimal_fdtd_params(
        wl, 1.0, dims=2, safety_factor=0.95, points_per_wavelength=8
    )
    t = np.arange(0, 40 * dt, dt)
    design = Design(width=4 * wl, height=4 * wl, material=Material(permittivity=1.0))

    sim = Simulation(
        design=design,
        sources=[],
        monitors=[],
        boundaries=[PML(thickness=1.0 * wl)],
        time=t,
        resolution=dx,
    )

    program = sim.compile(num_steps=20)
    assert program.compile_count == 0

    eng0 = _engine_state_for_sim(sim)
    mon0 = MonitorState(
        powers=jnp.zeros((0, 0), dtype=jnp.float32),
        timestamps=jnp.zeros((0, 0), dtype=jnp.float32),
        counts=jnp.zeros((0,), dtype=jnp.int32),
        freq_flux_re=jnp.zeros((0, 0), dtype=jnp.float32),
        freq_flux_im=jnp.zeros((0, 0), dtype=jnp.float32),
        freq_phase_re=jnp.zeros((0, 0), dtype=jnp.float32),
        freq_phase_im=jnp.zeros((0, 0), dtype=jnp.float32),
        dft_vec_re=jnp.zeros((0, 0, 0, 0), dtype=jnp.float32),
        dft_vec_im=jnp.zeros((0, 0, 0, 0), dtype=jnp.float32),
        dft_weight_sum=jnp.zeros((0, 0), dtype=jnp.float32),
    )

    eng1, _, _, _ = program.run(eng0, mon0)
    assert program.compile_count == 1

    # Recreate states since donation invalidates buffers.
    eng1_input = EngineState(
        ex=eng1.ex,
        ey=eng1.ey,
        ez=eng1.ez,
        hx=eng1.hx,
        hy=eng1.hy,
        hz=eng1.hz,
        tm_ez=eng1.tm_ez,
        tm_hx=eng1.tm_hx,
        tm_hy=eng1.tm_hy,
        fp_ex=eng1.fp_ex,
        fp_ey=eng1.fp_ey,
        fp_ez=eng1.fp_ez,
        fp_hx=eng1.fp_hx,
        fp_hy=eng1.fp_hy,
        fp_hz=eng1.fp_hz,
        cpml_psi_h_terms=eng1.cpml_psi_h_terms,
        cpml_psi_e_terms=eng1.cpml_psi_e_terms,
        cpml3d_psi_h_terms=eng1.cpml3d_psi_h_terms,
        cpml3d_psi_e_terms=eng1.cpml3d_psi_e_terms,
        t=eng1.t,
        current_step=eng1.current_step,
    )
    mon1 = MonitorState(
        powers=jnp.zeros((0, 0), dtype=jnp.float32),
        timestamps=jnp.zeros((0, 0), dtype=jnp.float32),
        counts=jnp.zeros((0,), dtype=jnp.int32),
        freq_flux_re=jnp.zeros((0, 0), dtype=jnp.float32),
        freq_flux_im=jnp.zeros((0, 0), dtype=jnp.float32),
        freq_phase_re=jnp.zeros((0, 0), dtype=jnp.float32),
        freq_phase_im=jnp.zeros((0, 0), dtype=jnp.float32),
        dft_vec_re=jnp.zeros((0, 0, 0, 0), dtype=jnp.float32),
        dft_vec_im=jnp.zeros((0, 0, 0, 0), dtype=jnp.float32),
        dft_weight_sum=jnp.zeros((0, 0), dtype=jnp.float32),
    )
    program.run(eng1_input, mon1)
    assert program.compile_count == 1


def test_compiled_jaxpr_has_no_host_callbacks(small_sim_params):
    wl, dx, _dt, domain, _steps, t, signal = small_sim_params
    design = Design(width=domain, height=domain, material=Material(permittivity=1.0))
    source = GaussianSource(
        position=(domain / 2, domain / 2), width=wl / 6, signal=signal
    )

    sim = Simulation(
        design=design,
        sources=[source],
        boundaries=[PML(thickness=1.2 * wl)],
        time=t,
        resolution=dx,
    )

    program = sim.compile(num_steps=8)
    eng0 = _engine_state_for_sim(sim)
    mon0 = MonitorState(
        powers=jnp.zeros((0, 0), dtype=jnp.float32),
        timestamps=jnp.zeros((0, 0), dtype=jnp.float32),
        counts=jnp.zeros((0,), dtype=jnp.int32),
        freq_flux_re=jnp.zeros((0, 0), dtype=jnp.float32),
        freq_flux_im=jnp.zeros((0, 0), dtype=jnp.float32),
        freq_phase_re=jnp.zeros((0, 0), dtype=jnp.float32),
        freq_phase_im=jnp.zeros((0, 0), dtype=jnp.float32),
        dft_vec_re=jnp.zeros((0, 0, 0, 0), dtype=jnp.float32),
        dft_vec_im=jnp.zeros((0, 0, 0, 0), dtype=jnp.float32),
        dft_weight_sum=jnp.zeros((0, 0), dtype=jnp.float32),
    )

    program._build_scan()
    jaxpr = jax.make_jaxpr(program._compiled_scan)(
        eng0, mon0, program._update_coefficients()
    )
    assert "host_callback" not in str(jaxpr).lower()


def test_compile_mode_source_builds_e_and_h_specs():
    wl = 1.55 * um
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

    t = np.arange(0, 80 * dt, dt)
    freq = LIGHT_SPEED / wl
    signal = ramped_cosine(
        t,
        amplitude=0.1,
        frequency=freq,
        ramp_duration=2 / freq,
        t_max=t[-1] * 0.5,
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
        time=t,
        resolution=dx,
    )

    program = sim.compile(num_steps=20)
    assert any(spec.timing == "h" for spec in program.source_specs)
    assert any(spec.timing == "e" for spec in program.source_specs)

    sim.run_compiled(num_steps=20, progress=False)
    assert np.isfinite(np.asarray(sim.fields.Ez)).all()


def test_compile_3d_mode_source_scales_e_terms_by_component_eps():
    fields = SimpleNamespace(
        permittivity=jnp.full((2, 2, 2), 99.0),
        permeability=jnp.ones((2, 2, 2)),
        Ex=jnp.zeros((2, 2, 1)),
        Ey=jnp.zeros((2, 1, 2)),
        Ez=jnp.zeros((1, 2, 2)),
        Hx=jnp.zeros((1, 1, 2)),
        Hy=jnp.zeros((1, 2, 1)),
        Hz=jnp.zeros((2, 1, 1)),
        eps_x=jnp.full((2, 2, 1), 2.0),
        eps_y=jnp.full((2, 1, 2), 3.0),
        eps_z=jnp.full((1, 2, 2), 4.0),
    )
    one = np.ones((1, 1, 1), dtype=np.float32)
    source = SimpleNamespace(
        _axis="z",
        pol="te",
        _direction_sign=1.0,
        _Ex_profile=one,
        _Ey_profile=one,
        _Ez_profile=None,
        _Hx_profile=one,
        _Hy_profile=one,
        _Hz_profile=None,
        _Ex_indices=(slice(0, 1), slice(0, 1), slice(0, 1)),
        _Ey_indices=(slice(0, 1), slice(0, 1), slice(0, 1)),
        _Ez_indices=None,
        _Hx_indices=(slice(0, 1), slice(0, 1), slice(0, 1)),
        _Hy_indices=(slice(0, 1), slice(0, 1), slice(0, 1)),
        _Hz_indices=None,
    )

    specs = _compile_mode_source_3d(
        source,
        fields,
        dt=5.0,
        resolution=7.0,
        h_waveform=jnp.ones((1,), dtype=jnp.float32),
        e_waveform=jnp.ones((1,), dtype=jnp.float32),
    )
    e_specs = {spec.component: spec for spec in specs if spec.timing == "e"}

    assert np.asarray(e_specs["Ex"].coeff).item() == pytest.approx(
        -5.0 / (EPS_0 * 2.0 * 7.0)
    )
    assert np.asarray(e_specs["Ey"].coeff).item() == pytest.approx(
        5.0 / (EPS_0 * 3.0 * 7.0)
    )


def test_cache_reuse_across_equal_chunks(small_sim_params):
    """Equal-sized chunks should reuse the same compiled program (compile_count == 1)."""
    wl, dx, _dt, domain, _steps, t, signal = small_sim_params
    design = Design(width=domain, height=domain, material=Material(permittivity=1.0))
    source = GaussianSource(
        position=(domain / 2, domain / 2), width=wl / 6, signal=signal
    )

    sim = Simulation(
        design=design,
        sources=[source],
        boundaries=[PML(thickness=1.2 * wl)],
        time=t,
        resolution=dx,
    )

    # Run with record_interval to force multiple equal-sized chunks.
    chunk_size = 30
    sim.run_compiled(num_steps=90, record_interval=chunk_size, progress=False)

    # The program should have been compiled only once (all chunks are size 30).
    assert sim._compiled_program is not None
    assert sim._compiled_program.compile_count == 1


def test_waveform_absolute_indexing_correctness(small_sim_params):
    """Chunked execution with absolute waveform indexing should match single-shot."""
    wl, dx, _dt, domain, _steps, t, signal = small_sim_params
    design = Design(width=domain, height=domain, material=Material(permittivity=1.0))

    source_a = GaussianSource(
        position=(domain / 2, domain / 2), width=wl / 6, signal=signal
    )
    source_b = GaussianSource(
        position=(domain / 2, domain / 2), width=wl / 6, signal=signal
    )

    sim_single = Simulation(
        design=design.copy(),
        sources=[source_a],
        boundaries=[PML(thickness=1.2 * wl)],
        time=t,
        resolution=dx,
    )
    sim_chunked = Simulation(
        design=design.copy(),
        sources=[source_b],
        boundaries=[PML(thickness=1.2 * wl)],
        time=t,
        resolution=dx,
    )

    # Single-shot: run all 120 steps at once.
    sim_single.run_compiled(num_steps=120, progress=False)

    # Chunked: run 4 chunks of 30 steps each.
    sim_chunked.run_compiled(num_steps=120, record_interval=30, progress=False)

    ez_single = np.asarray(sim_single.fields.Ez)
    ez_chunked = np.asarray(sim_chunked.fields.Ez)

    assert sim_single.current_step == sim_chunked.current_step
    assert np.allclose(ez_single, ez_chunked, rtol=1e-5, atol=1e-6)
