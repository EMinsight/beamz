"""Compare the original and compact JAX tensor updates with synchronized timings.

Run with PYTHONPATH=. and JAX_PLATFORMS=cpu or cuda. Coefficients are dynamic
arguments, never embedded constants; setup/compilation are excluded from timings.
The isolated stencil feeds fields back into its synthetic curls to prevent
loop-invariant hoisting. --full-step measures the production Maxwell/CPML/DFT scan.
The dense reference is frozen from commit 66b40500; no production monkeypatch
survives construction of the two independently compiled benchmark executables.
"""

import argparse
import json
import platform
import time

import jax
import jax.numpy as jnp
import numpy as np

from beamz.const import EPS_0
from beamz.lattice import component_shape_3d
from beamz.simulation.kernels import advance_e_centered_tensor, collocate_yee_component
from beamz.simulation.tensor_regions import TensorRegion, compact_tensor_region


def full_step_measure(
    size=64, steps=100, repeats=7, graded=True, core_fraction=0.25, cuda=False
):
    """Production scan: six fields, curls, CPML, and two DFT flux monitors."""
    from dataclasses import replace

    import beamz as bz
    from beamz.design.discretization import MaterialGrid
    from beamz.design.raster import (
        Box,
        Grid,
        Material,
        Object,
        RasterOptions,
        Scene,
        rasterize,
    )
    from beamz.simulation import kernels
    from beamz.simulation.execute import build_program_scan, initial_program_state
    from beamz.simulation.sharding import place_tree

    length = size * 100e-9
    u = np.linspace(0, 1, size + 1)
    edges = [length * u, length * u, length * u]
    if graded:
        edges[1:] = [length * (u + 0.08 * np.sin(2 * np.pi * u))] * 2
    eps = ((6.0, 0.2, 0.1), (0.2, 5.0, 0.3), (0.1, 0.3, 4.0))
    material = MaterialGrid.from_raster_result(
        rasterize(
            Scene(
                (Material(), Material(epsilon_r=eps)),
                (
                    Object(
                        Box(
                            (
                                0,
                                (0.5 - core_fraction / 2) * length,
                                (0.5 - core_fraction / 2) * length,
                            ),
                            (
                                length,
                                (0.5 + core_fraction / 2) * length,
                                (0.5 + core_fraction / 2) * length,
                            ),
                        ),
                        1,
                    ),
                ),
            ),
            Grid(*edges),
            options=RasterOptions(smoothing="farjadpour_full"),
        )
    )
    dt = material.grid.cfl_time_step(0.6)
    sim = bz.Simulation(
        material_grid=material,
        time=np.arange(steps + 1) * dt,
        boundaries=[bz.PML(thickness=8 * 100e-9, formulation="cpml")],
        monitors=[
            bz.FluxMonitor(
                center=(x * length, 0.5 * length, 0.5 * length),
                size=(0, 0.7 * length, 0.7 * length),
                freqs=[bz.LIGHT_SPEED / 1.55e-6],
                name=f"flux{i}",
            )
            for i, x in enumerate((0.35, 0.65))
        ],
        setup_device="cpu",
    )
    program = sim.compile(num_steps=steps, backend="jax")
    coeffs = place_tree(program, program.coefficients)
    packed = coeffs.e_inverse_offdiagonal
    if isinstance(packed, TensorRegion):
        dense_cross = jnp.pad(
            packed.values,
            tuple(
                (lo, n - hi)
                for (lo, hi), n in zip(packed.bounds, packed.node_shape, strict=True)
            )
            + ((0, 0),),
        )
    else:
        dense_cross = packed
    dense_coeffs = coeffs._replace(e_inverse_offdiagonal=dense_cross)
    dense_program = replace(program, coefficients=dense_coeffs)
    state = initial_program_state(program, t=0.0, current_step=0)
    coords = np.meshgrid(
        *(np.linspace(-1, 1, n) for n in state.ey.shape), indexing="ij"
    )
    pulse = np.exp(-sum(c * c for c in coords) / 0.03) * np.cos(12 * coords[2])
    state = state._replace(ey=jnp.asarray(pulse, dtype=state.ey.dtype))
    funcs, outputs, compilation, memory = {}, {}, {}, {}
    optimized_update = kernels.advance_e_centered_tensor
    variants = (
        ("dense", dense_program, dense_coeffs, reference_dense_tensor),
        ("compact", program, coeffs, optimized_update),
    )
    if cuda:
        native = sim.compile(num_steps=steps, backend="cuda_streamed")
        variants += (
            ("cuda", native, place_tree(native, native.coefficients), optimized_update),
        )
    try:
        for name, plan, coefficients, update in variants:
            kernels.advance_e_centered_tensor = update
            start = time.perf_counter()
            funcs[name] = build_program_scan(plan).lower(state, coefficients).compile()
            compilation[name] = time.perf_counter() - start
            stats = funcs[name].memory_analysis()
            memory[name] = dict(
                arguments=stats.argument_size_in_bytes,
                outputs=stats.output_size_in_bytes,
                temporaries=stats.temp_size_in_bytes,
            )
            for _ in range(2):
                outputs[name] = jax.block_until_ready(funcs[name](state, coefficients))
    finally:
        kernels.advance_e_centered_tensor = optimized_update
    errors = {}
    for field in ("ex", "ey", "ez", "hx", "hy", "hz", "dft_vec_re", "dft_vec_im"):
        a, b = (
            np.asarray(getattr(outputs[name], field)) for name in ("dense", "compact")
        )
        errors[field] = float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30))
        assert errors[field] < 2e-5, (field, errors[field])
    cuda_errors = {}
    if cuda:
        for field in ("ex", "ey", "ez", "hx", "hy", "hz", "dft_vec_re", "dft_vec_im"):
            a, b = (
                np.asarray(getattr(outputs[name], field))
                for name in ("compact", "cuda")
            )
            error = float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30))
            assert error < 2e-5, (field, error)
            cuda_errors[field] = error
    samples = {name: [] for name, *_ in variants}
    for repeat in range(repeats):
        for name, _, coefficients, _ in variants if repeat % 2 == 0 else variants[::-1]:
            start = time.perf_counter()
            jax.block_until_ready(funcs[name](state, coefficients))
            samples[name].append((time.perf_counter() - start) / steps)
    medians = {k: float(np.median(v)) for k, v in samples.items()}
    return dict(
        size=size,
        steps=steps,
        graded=graded,
        case="full_fdtd",
        core_fraction=core_fraction,
        compact=isinstance(packed, TensorRegion),
        seconds_per_step=medians,
        speedup=medians["dense"] / medians["compact"],
        samples=samples,
        compile_seconds=compilation,
        memory=memory,
        relative_errors=errors,
        cuda_relative_errors=cuda_errors,
    )


def reference_dense_tensor(
    fields,
    curls,
    inverse_diagonals,
    inverse_offdiagonal,
    components,
    dt,
    *,
    cell_widths=None,
):
    """Advance E with diagonal terms on E sites and cross terms on dual centers."""

    scale = jnp.asarray(dt, dtype=fields[0].dtype) / jnp.asarray(
        EPS_0, dtype=fields[0].dtype
    )
    node_shape = inverse_offdiagonal.shape[:-1]
    centered_curls = tuple(
        collocate_yee_component(
            curl, source, "Node", node_shape, cell_widths=cell_widths
        )
        for curl, source in zip(curls, components, strict=True)
    )
    cross_component = {(0, 1): 0, (0, 2): 1, (1, 2): 2}
    updated = []
    for row, (field, diagonal, target) in enumerate(
        zip(fields, inverse_diagonals, components, strict=True)
    ):
        coupled = diagonal * curls[row]
        for column, centered_curl in enumerate(centered_curls):
            if row == column:
                continue
            pair = (min(row, column), max(row, column))
            centered = inverse_offdiagonal[..., cross_component[pair]] * centered_curl
            coupled = coupled + collocate_yee_component(
                centered, "Node", target, field.shape, cell_widths=cell_widths
            )
        updated.append(field + scale * coupled)
    return tuple(updated)


def measure(size=96, steps=50, repeats=7, case="guide", graded=False):
    shape = (size,) * 3
    names = ("Ex", "Ey", "Ez")
    rng = np.random.default_rng(361)
    fields = tuple(
        jnp.asarray(rng.normal(size=component_shape_3d(k, shape)), dtype=jnp.float32)
        for k in names
    )
    curls = tuple(
        jnp.asarray(rng.normal(size=v.shape), dtype=jnp.float32) for v in fields
    )
    diagonal = tuple(jnp.full_like(v, 0.5) for v in fields)
    cross = np.zeros((size + 1,) * 3 + (3,), np.float32)
    if case == "guide":
        lo, hi = size * 3 // 8, size * 5 // 8
        cross[lo:hi, lo:hi, :] = (0.1, 0.05, -0.08)
    elif case == "interface":
        cross[size // 2 : size // 2 + 2, :, :] = (0.1, 0.05, -0.08)
    else:
        cross[:] = (0.1, 0.05, -0.08)
    compact = compact_tensor_region(cross)
    compact = jax.tree.map(jnp.asarray, compact)
    dense = jnp.asarray(cross)
    widths = (
        tuple(
            jnp.asarray(rng.uniform(0.5, 1.5, size), dtype=jnp.float32)
            for _ in range(3)
        )
        if graded
        else None
    )

    def make_scan(update):
        def scan(fields, curls, diagonal, cross, widths):
            return jax.lax.fori_loop(
                0,
                steps,
                lambda _, e: update(
                    e,
                    tuple(c + 0.01 * v for c, v in zip(curls, e, strict=True)),
                    diagonal,
                    cross,
                    names,
                    EPS_0 * 0.01,
                    cell_widths=widths,
                ),
                fields,
            )

        return scan

    funcs, outputs, compile_seconds = {}, {}, {}
    for name, values, update in (
        ("dense", dense, reference_dense_tensor),
        ("compact", compact, advance_e_centered_tensor),
    ):
        start = time.perf_counter()
        funcs[name] = (
            jax.jit(make_scan(update), compiler_options={"xla_gpu_autotune_level": 0})
            .lower(fields, curls, diagonal, values, widths)
            .compile()
        )
        compile_seconds[name] = time.perf_counter() - start
        for _ in range(2):
            outputs[name] = jax.block_until_ready(
                funcs[name](fields, curls, diagonal, values, widths)
            )
    error = max(
        float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        for a, b in zip(outputs["dense"], outputs["compact"], strict=True)
    )
    np.testing.assert_allclose(
        np.concatenate([np.ravel(v) for v in outputs["dense"]]),
        np.concatenate([np.ravel(v) for v in outputs["compact"]]),
        rtol=2e-5,
        atol=2e-5,
    )
    samples = {"dense": [], "compact": []}
    for repeat in range(repeats):
        order = (("dense", dense), ("compact", compact))
        for name, values in order if repeat % 2 == 0 else order[::-1]:
            start = time.perf_counter()
            jax.block_until_ready(funcs[name](fields, curls, diagonal, values, widths))
            samples[name].append((time.perf_counter() - start) / steps)
    medians = {k: float(np.median(v)) for k, v in samples.items()}
    return dict(
        size=size,
        steps=steps,
        case=case,
        graded=graded,
        compact=isinstance(compact, TensorRegion),
        cross_bytes_dense=cross.nbytes,
        cross_bytes_compact=sum(v.nbytes for v in jax.tree.leaves(compact)),
        seconds_per_step=medians,
        speedup=medians["dense"] / medians["compact"],
        samples=samples,
        compile_seconds=compile_seconds,
        max_abs_error=error,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", required=True)
    parser.add_argument("--full-step", action="store_true")
    parser.add_argument("--core-fraction", type=float, default=0.25)
    parser.add_argument(
        "--cuda", action="store_true", help="Include native CUDA in --full-step"
    )
    args = parser.parse_args()
    if args.cuda and not args.full_step:
        parser.error("--cuda requires --full-step")
    rows = []
    for case in () if args.full_step else ("guide", "interface", "bulk"):
        for graded in (False, True):
            row = measure(args.size, args.steps, args.repeats, case, graded)
            rows.append(row)
            print(case, graded, row["speedup"], flush=True)
    if args.full_step:
        for graded in (False, True):
            row = full_step_measure(
                args.size,
                args.steps,
                args.repeats,
                graded,
                args.core_fraction,
                args.cuda,
            )
            rows.append(row)
            print("full_fdtd", graded, row["speedup"], flush=True)
            if args.cuda:
                timing = row["seconds_per_step"]
                print(
                    "native_cuda_speedup",
                    timing["compact"] / timing["cuda"],
                    flush=True,
                )
    report = dict(
        jax=jax.__version__,
        devices=list(map(str, jax.devices())),
        device_kinds=[device.device_kind for device in jax.devices()],
        baseline_revision="66b40500",
        platform=platform.platform(),
        rows=rows,
    )
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=2)
