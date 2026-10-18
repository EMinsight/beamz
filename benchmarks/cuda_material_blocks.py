"""Paired native timings of constant blocks versus the preceding material layout.

Identical operators, states, timesteps, CPML and DFTs; synchronized alternating
runs, with setup/JIT/capture excluded. Baseline restores dense tensor diagonals
or the previous diagonal byte-ID/dense layout, using the same native binary.
"""

import argparse
import json
import platform
import time

import jax
import jax.numpy as jnp
import numpy as np

import beamz as bz
from beamz.design.discretization import MaterialGrid
from beamz.design.raster import (
    ExtrudedPolygon,
    Grid,
    Material,
    Object,
    Polygon,
    RasterOptions,
    Scene,
    rasterize,
)
from beamz.simulation.compile import _pack_cuda_lossless_e_coefficients
from beamz.simulation.execute import build_program_scan, initial_program_state
from beamz.simulation.sharding import place_tree


def simulation(size, steps, *, full=False, graded=False, bulk=False, pec=False):
    length = size * 100e-9
    u = np.linspace(0, 1, size + 1)
    edges = length * (u + 0.08 * np.sin(2 * np.pi * u)) if graded else length * u
    angle = np.pi / 6
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    vertices = (
        np.array([[-0.5, -1 / 3], [0.5, -1 / 3], [0.5, 1 / 3], [-0.5, 1 / 3]])
        * 0.25
        @ rotation.T
        + 0.5
    ) * length
    scene = (
        Scene((Material(((6, 0.2, 0.1), (0.2, 5, 0.3), (0.1, 0.3, 4))),), ())
        if bulk
        else Scene(
            (Material(), Material(6)),
            (
                Object(
                    ExtrudedPolygon(Polygon(tuple(map(tuple, vertices))), 0, length),
                    1,
                ),
            ),
        )
    )
    material = MaterialGrid.from_raster_result(
        rasterize(
            scene,
            Grid(edges, edges, length * u),
            options=RasterOptions(
                smoothing="farjadpour_full" if full or bulk else "farjadpour_diagonal"
            ),
        )
    )
    return bz.Simulation(
        material_grid=material,
        time=np.arange(steps + 1) * material.grid.cfl_time_step(0.6),
        boundaries=(
            [bz.PEC(edges="all")]
            if pec
            else [bz.PML(thickness=300e-9 if size < 64 else 800e-9, formulation="cpml")]
        ),
        monitors=[
            bz.FluxMonitor(
                center=(0.5 * length, 0.5 * length, z * length),
                size=(0.7 * length, 0.7 * length, 0),
                freqs=[bz.LIGHT_SPEED / 1.55e-6],
                name=f"flux{i}",
            )
            for i, z in enumerate((0.35, 0.65))
        ],
        setup_device="cpu",
    )


def restore_previous_layout(coeffs, state):
    """Independent host reconstruction, also used for native equality checks."""
    changes = {}
    full = bool(coeffs.e_inverse_offdiagonal.size)
    dense = []
    for c in "xyz":
        shape = getattr(state, "e" + c).shape
        source = getattr(coeffs, "e_source_" + c)
        value = getattr(coeffs, ("e_inverse_diagonal_" if full else "e_decay_") + c)
        if source.ndim == 2 and source.dtype == jnp.int32:
            headers = np.asarray(source).ravel()
            linear = np.arange(np.prod(shape))
            header = headers[linear // 32]
            offsets = np.where(header < 0, ~header, header + linear % 32)
            value = np.asarray(value)[offsets].reshape(shape)
        elif full:
            value = np.broadcast_to(np.asarray(value), shape).copy()
        elif source.ndim == 1 and source.dtype == jnp.int32:
            words = np.asarray(source).view(np.uint32)
            linear = np.arange(np.prod(shape))
            ids = (words[linear // 4] >> (8 * (linear % 4))) & 255
            value = np.asarray(value)[ids].reshape(shape)
        else:
            value = source
        if full:
            changes["e_inverse_diagonal_" + c] = jnp.asarray(value)
            changes["e_source_" + c] = jnp.float32(1)
        else:
            dense.append(jnp.asarray(value))
    if not full:
        packed = _pack_cuda_lossless_e_coefficients(
            (jnp.float32(1),) * 3, dense, blocks=False
        )
        decays, sources = (
            packed if packed is not None else ((jnp.float32(1),) * 3, dense)
        )
        for c, decay, source in zip("xyz", decays, sources, strict=True):
            changes["e_decay_" + c] = decay
            changes["e_source_" + c] = source
    return coeffs._replace(**changes)


def coefficient_bytes(coeffs):
    full = bool(coeffs.e_inverse_offdiagonal.size)
    arrays = [
        getattr(coeffs, prefix + c)
        for prefix in (
            ("e_inverse_diagonal_", "e_source_") if full else ("e_decay_", "e_source_")
        )
        for c in "xyz"
    ]
    # Shared constant pools count once; include every distinct resident buffer.
    unique = {a.unsafe_buffer_pointer(): a.nbytes for a in arrays}
    return sum(unique.values())


def measure(size, steps, repeats, **scene):
    sim = simulation(size, steps, **scene)
    p = sim.compile(num_steps=steps, backend="cuda_streamed")
    state = initial_program_state(p, t=0.0, current_step=0)
    coords = np.meshgrid(
        *(np.linspace(-1, 1, n) for n in state.ex.shape), indexing="ij"
    )
    pulse = np.exp(-sum(c * c for c in coords) / 0.03) * np.cos(12 * coords[0])
    state = state._replace(ex=jnp.asarray(pulse, dtype=state.ex.dtype))
    coeffs = {
        "blocks": place_tree(p, p.coefficients),
        "previous": place_tree(p, restore_previous_layout(p.coefficients, state)),
    }
    functions, outputs, samples = {}, {}, {}
    compile_seconds = {}
    for name, values in coeffs.items():
        start = time.perf_counter()
        functions[name] = build_program_scan(p).lower(state, values).compile()
        compile_seconds[name] = time.perf_counter() - start
        for _ in range(3):
            outputs[name] = jax.block_until_ready(functions[name](state, values))
        samples[name] = []
    for a, b in zip(
        jax.tree.leaves(outputs["previous"]),
        jax.tree.leaves(outputs["blocks"]),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    for repeat in range(repeats):
        for name in (
            ("blocks", "previous") if repeat % 2 == 0 else ("previous", "blocks")
        ):
            start = time.perf_counter()
            jax.block_until_ready(functions[name](state, coeffs[name]))
            samples[name].append((time.perf_counter() - start) / steps)
    medians = {name: float(np.median(v)) for name, v in samples.items()}
    headers = [getattr(coeffs["blocks"], "e_source_" + c) for c in "xyz"]
    return dict(
        size=size,
        steps=steps,
        repeats=repeats,
        **scene,
        seconds_per_step=medians,
        speedup=medians["previous"] / medians["blocks"],
        bitwise_equal=True,
        coefficient_bytes={name: coefficient_bytes(c) for name, c in coeffs.items()},
        constant_block_fraction=[
            float(np.mean(np.asarray(h) < 0)) if h.ndim == 2 else None for h in headers
        ],
        samples=samples,
        compile_seconds=compile_seconds,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    for full, bulk in ((False, False), (True, False), (True, True)):
        for graded in (False, True):
            row = measure(
                args.size, args.steps, args.repeats, full=full, bulk=bulk, graded=graded
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
    with open(args.output, "w") as handle:
        json.dump(
            dict(
                jax=jax.__version__,
                devices=[d.device_kind for d in jax.devices()],
                platform=platform.platform(),
                rows=rows,
            ),
            handle,
            indent=2,
        )
