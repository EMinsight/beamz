"""Modal objectives must use the same basis as detached forward analysis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import beamz as bz
from beamz.design import MaterialGrid
from beamz.optimization.modes import build_mode_projector


@pytest.mark.parametrize("axis", ("x", "y", "z"))
def test_fixed_projector_matches_mode_analysis_and_has_correct_vjp(axis):
    dx = 100e-9
    shape = (20, 20, 20)
    z, y, x = np.indices(shape)
    transverse = {"x": (z, y), "y": (z, x), "z": (y, x)}[axis]
    core = np.logical_and.reduce([(v >= 8) & (v < 12) for v in transverse])
    material = MaterialGrid(
        permittivity=np.where(core, 4.0, 1.0).astype(np.float32),
        conductivity=np.float32(0),
        permeability=np.float32(1),
        resolution=dx,
        shape=shape,
    )
    size = [1e-6] * 3
    normal = "xyz".index(axis)
    size[normal] = 0.0
    source_center = [1e-6] * 3
    source_center[normal] = 0.5e-6
    monitor_center = [1e-6] * 3
    monitor_center[normal] = 1.4e-6
    freq = bz.LIGHT_SPEED / 1e-6
    dt = 0.8 * dx / (bz.LIGHT_SPEED * np.sqrt(3))
    simulation = bz.Simulation(
        material_grid=material,
        time=np.arange(160) * dt,
        sources=[
            bz.ModeSource(
                center=tuple(source_center),
                size=tuple(size),
                direction="+",
                mode_spec=bz.ModeSpec(polarization="te"),
                source_time=bz.GaussianPulse(freq0=freq, fwidth=freq / 4),
            )
        ],
        monitors=[
            bz.ModeMonitor(
                center=tuple(monitor_center),
                size=tuple(size),
                mode_spec=bz.ModeSpec(polarization="te", num_modes=2),
                freqs=[freq, freq * 1.03],
                name="port",
            )
        ],
        boundaries=[bz.PML(thickness=0.3e-6, formulation="cpml")],
        normalize_source=None,
    )
    result = simulation.run(backend="jax", performance=False)
    solver = simulation.differentiable(backend="jax", checkpoint_interval=16)
    fields = solver.monitor_fields(solver(), "port")
    project = build_mode_projector(result, "port")
    np.testing.assert_allclose(
        project(fields),
        result.mode("port").amps.values,
        rtol=3e-4,
        atol=1e-7,
    )
    # An independent finite difference also checks complex conjugation and the
    # coupled forward/backward basis in the differentiable projection.
    rng = np.random.default_rng(12)
    direction = (
        jnp.asarray(
            (rng.normal(size=fields.shape) + 1j * rng.normal(size=fields.shape)).astype(
                np.complex64
            )
        )
        * jnp.linalg.norm(fields)
        / np.sqrt(fields.size)
    )

    def objective(scale):
        return jnp.sum(jnp.abs(project(fields + scale * direction)) ** 2)

    derivative = jax.grad(objective)(jnp.float32(0))
    h = 0.01
    finite_difference = (objective(h) - objective(-h)) / (2 * h)
    np.testing.assert_allclose(derivative, finite_difference, rtol=0.003, atol=1e-8)
    with pytest.raises(ValueError, match="Modal fields must have shape"):
        project(fields[:, :, :-1])
