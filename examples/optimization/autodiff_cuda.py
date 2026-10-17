"""Optimize a 3D dielectric volume with checkpointed, end-to-end autodiff.

Run from the repository root:
    python -m examples.optimization.autodiff_cuda --backend cuda --iterations 8
Use --backend jax for the portable reference solver.
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np
import optax

import beamz as bz
from beamz.design import MaterialGrid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("jax", "cuda", "auto"), default="jax")
    parser.add_argument("--iterations", type=int, default=8)
    args = parser.parse_args()
    shape = (12, 14, 20)
    dx = 80e-9
    dt = 0.8 * dx / (bz.LIGHT_SPEED * np.sqrt(3.0))
    timesteps = 48
    time = np.arange(timesteps, dtype=np.float64) * dt
    grid = MaterialGrid(
        np.ones(shape, np.float32), np.float32(0.0), np.float32(1.0), dx, shape
    )
    probe = bz.Simulation(material_grid=grid, time=time)
    target = probe.initial_state().ez.shape
    source = bz.CustomSource(
        component="Ez",
        timing="e",
        index=(slice(4, 9), slice(5, 10), slice(4, 5)),
        coeff=np.full((5, 5, 1), 0.02, dtype=np.float32),
        waveform=(
            np.sin(2 * np.pi * 2e14 * time)
            * np.exp(-(((np.arange(timesteps) - 12) / 6) ** 2))
        ).astype(np.float32),
        target_shape=target,
    )
    monitor = bz.FluxMonitor(
        center=(15 * dx, 7 * dx, 6 * dx),
        size=(0.0, 6 * dx, 6 * dx),
        freqs=(2e14,),
        name="output",
    )
    simulation = bz.Simulation(
        material_grid=grid,
        time=time,
        sources=[source],
        monitors=[monitor],
        boundaries=[bz.PML(thickness=3 * dx, formulation="cpml")],
    )
    solver = simulation.differentiable(backend=args.backend, checkpoint_interval=7)
    mask = jnp.zeros(shape, dtype=jnp.bool_).at[4:9, 5:10, 7:14].set(True)
    density = jnp.zeros(shape, dtype=jnp.float32)

    def permittivity(parameters):
        # A smooth density-to-material map composes directly with the solver VJP.
        return jnp.where(mask, 1.0 + 3.0 * jax.nn.sigmoid(parameters), 1.0)

    baseline = solver.monitor_flux(solver(permittivity(density)), "output")[0]
    normalization = jnp.maximum(jnp.abs(baseline), 1e-30)

    def objective(parameters):
        state = solver(permittivity(parameters))
        return -solver.monitor_flux(state, "output")[0] / normalization

    optimizer = optax.adam(0.05)
    optimizer_state = optimizer.init(density)

    @jax.jit
    def optimize(parameters, state):
        loss, gradient = jax.value_and_grad(objective)(parameters)
        updates, state = optimizer.update(gradient, state, parameters)
        return (
            optax.apply_updates(parameters, updates),
            state,
            loss,
            jnp.linalg.norm(gradient),
        )

    for step in range(args.iterations):
        density, optimizer_state, loss, norm = optimize(density, optimizer_state)
        print(
            f"step={step + 1} normalized_flux={float(-loss):.6f} gradient_norm={float(norm):.6g}"
        )


if __name__ == "__main__":
    main()
