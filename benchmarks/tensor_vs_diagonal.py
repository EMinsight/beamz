"""Compare full and diagonal Farjadpour smoothing in the production JAX scan.

The scene is an isotropic, z-extruded rectangular guide rotated 30 degrees.
Both methods use identical grids, initial fields, CPML, timesteps and DFT monitors.
Their interface constitutive operators differ, so field equality is not expected.
Timing excludes material setup, compilation and warmup, and synchronizes devices.
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
from beamz.simulation.execute import build_program_scan, initial_program_state
from beamz.simulation.sharding import place_tree
from beamz.simulation.tensor_regions import TensorRegion


def measure(
    size=128, steps=100, repeats=7, graded=False, core_fraction=0.25, cuda=False
):
    length = size * 100e-9
    u = np.linspace(0, 1, size + 1)
    transverse = length * (u + 0.08 * np.sin(2 * np.pi * u)) if graded else length * u
    grid = Grid(transverse, transverse, length * u)
    angle = np.pi / 6
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    vertices = (
        np.array([[-0.5, -1 / 3], [0.5, -1 / 3], [0.5, 1 / 3], [-0.5, 1 / 3]])
        * core_fraction
        @ rotation.T
        + 0.5
    ) * length
    scene = Scene(
        (Material(), Material(6.0)),
        (Object(ExtrudedPolygon(Polygon(tuple(map(tuple, vertices))), 0, length), 1),),
    )
    plans, states, coeffs, funcs = {}, {}, {}, {}
    setup_seconds, compile_seconds, support = {}, {}, {}
    outputs, errors, memory = {}, {}, {}
    variants = [
        ("farjadpour_diagonal", "farjadpour_diagonal", "jax"),
        ("farjadpour_full", "farjadpour_full", "jax"),
    ]
    if cuda:
        variants += [
            ("cuda_diagonal", "farjadpour_diagonal", "cuda_streamed"),
            ("cuda_full", "farjadpour_full", "cuda_streamed"),
        ]
    for name, smoothing, backend in variants:
        start = time.perf_counter()
        material = MaterialGrid.from_raster_result(
            rasterize(scene, grid, options=RasterOptions(smoothing=smoothing))
        )
        dt = material.grid.cfl_time_step(0.6)
        sim = bz.Simulation(
            material_grid=material,
            time=np.arange(steps + 1) * dt,
            boundaries=[bz.PML(thickness=800e-9, formulation="cpml")],
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
        plans[name] = sim.compile(num_steps=steps, backend=backend)
        coeffs[name] = place_tree(plans[name], plans[name].coefficients)
        state = initial_program_state(plans[name], t=0.0, current_step=0)
        coords = np.meshgrid(
            *(np.linspace(-1, 1, n) for n in state.ex.shape), indexing="ij"
        )
        pulse = np.exp(-sum(c * c for c in coords) / 0.03) * np.cos(12 * coords[0])
        states[name] = state._replace(ex=jnp.asarray(pulse, dtype=state.ex.dtype))
        jax.block_until_ready((states[name], coeffs[name]))
        setup_seconds[name] = time.perf_counter() - start
        cross = coeffs[name].e_inverse_offdiagonal
        packed = isinstance(cross, TensorRegion)
        values = np.asarray(cross.values if packed else cross)
        support[name] = dict(
            packed=packed,
            cross_bytes=values.nbytes,
            nonzero_coefficients=int(np.count_nonzero(values)),
            bounds=cross.bounds if packed else None,
            dt=float(dt),
        )
        if smoothing == "farjadpour_diagonal":
            assert values.size == 0, "Diagonal baseline must have no tensor coupling"
        else:
            assert packed and np.any(values != 0), "Full run must exercise coupling"
        start = time.perf_counter()
        funcs[name] = (
            build_program_scan(plans[name]).lower(states[name], coeffs[name]).compile()
        )
        compile_seconds[name] = time.perf_counter() - start
        stats = funcs[name].memory_analysis()
        memory[name] = dict(
            arguments=stats.argument_size_in_bytes,
            outputs=stats.output_size_in_bytes,
            temporaries=stats.temp_size_in_bytes,
        )
        for _ in range(2):
            output = jax.block_until_ready(funcs[name](states[name], coeffs[name]))
        outputs[name] = output
        for field in ("ex", "ey", "ez", "hx", "hy", "hz", "dft_vec_re", "dft_vec_im"):
            assert np.isfinite(np.asarray(getattr(output, field))).all(), (name, field)
    if cuda:
        for name, reference in (
            ("cuda_full", "farjadpour_full"),
            ("cuda_diagonal", "farjadpour_diagonal"),
        ):
            errors[name] = {}
            for field in (
                "ex",
                "ey",
                "ez",
                "hx",
                "hy",
                "hz",
                "dft_vec_re",
                "dft_vec_im",
            ):
                a, b = (
                    np.asarray(getattr(outputs[k], field)) for k in (reference, name)
                )
                error = float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30))
                assert error < 2e-5, (name, field, error)
                errors[name][field] = error
    assert support["farjadpour_full"]["dt"] == support["farjadpour_diagonal"]["dt"]
    names = tuple(funcs)
    samples = {name: [] for name in names}
    for repeat in range(repeats):
        for name in names if repeat % 2 == 0 else names[::-1]:
            start = time.perf_counter()
            jax.block_until_ready(funcs[name](states[name], coeffs[name]))
            samples[name].append((time.perf_counter() - start) / steps)
    medians = {name: float(np.median(values)) for name, values in samples.items()}
    return dict(
        size=size,
        steps=steps,
        repeats=repeats,
        graded=graded,
        core_fraction=core_fraction,
        seconds_per_step=medians,
        full_over_diagonal=medians["farjadpour_full"] / medians["farjadpour_diagonal"],
        samples=samples,
        setup_seconds=setup_seconds,
        compile_seconds=compile_seconds,
        support=support,
        relative_errors=errors,
        memory=memory,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--core-fraction", type=float, default=0.25)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--cuda", action="store_true", help="Also compare native streamed CUDA"
    )
    args = parser.parse_args()
    rows = []
    for graded in (False, True):
        row = measure(
            args.size, args.steps, args.repeats, graded, args.core_fraction, args.cuda
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    with open(args.output, "w") as handle:
        json.dump(
            dict(
                jax=jax.__version__,
                devices=list(map(str, jax.devices())),
                device_kinds=[device.device_kind for device in jax.devices()],
                platform=platform.platform(),
                rows=rows,
            ),
            handle,
            indent=2,
        )
