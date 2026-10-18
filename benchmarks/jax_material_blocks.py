"""JAX material compression: paired propagation timing and executable memory.

Use --memory-worker with one --variant in a fresh process to measure allocator
peak usage independently, including setup, compilation and execution.
"""

import argparse
import json
import platform
import time
from contextlib import nullcontext
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from beamz.simulation.execute import build_program_scan, initial_program_state
from beamz.simulation.material_blocks import MaterialBlocks
from beamz.simulation.sharding import place_tree
from benchmarks.cuda_material_blocks import simulation


def dense_coefficients(coeffs, state):
    """Reconstruct the preceding layout on the host, including uniform grids."""
    changes = {}
    for prefix in ("e_permittivity_", "e_inverse_diagonal_"):
        for c in "xyz":
            value = getattr(coeffs, prefix + c)
            if not value.size:
                continue
            shape = getattr(state, "e" + c).shape
            if isinstance(value, MaterialBlocks):
                pool, headers = np.asarray(value.values), np.asarray(value.headers)
                if headers.dtype == np.uint8:
                    value = pool[headers].reshape(shape)
                else:
                    linear = np.arange(np.prod(shape), dtype=np.int32)
                    h = headers[linear >> 5]
                    value = pool[np.where(h < 0, ~h, h + (linear & 31))].reshape(shape)
            else:
                value = np.broadcast_to(np.asarray(value), shape).copy()
            changes[prefix + c] = value
    return coeffs._replace(**changes)


def unique_bytes(tree):
    arrays = {a.unsafe_buffer_pointer(): a.nbytes for a in jax.tree.leaves(tree)}
    return sum(arrays.values())


def executable_memory(executable):
    stats = executable.memory_analysis()
    result = {
        "arguments": stats.argument_size_in_bytes,
        "outputs": stats.output_size_in_bytes,
        "temporaries": stats.temp_size_in_bytes,
        "aliases": stats.alias_size_in_bytes,
    }
    result["total"] = (
        result["arguments"]
        + result["outputs"]
        + result["temporaries"]
        - result["aliases"]
    )
    return result


def prepare(size, steps, full, bulk, graded, host_material_grid=False):
    # setup_device="cpu" does not migrate a caller's pre-existing GPU arrays.
    with (
        jax.default_device(jax.devices("cpu")[0])
        if host_material_grid
        else nullcontext()
    ):
        sim = simulation(size, steps, full=full, bulk=bulk, graded=graded)
    plan = sim.compile(num_steps=steps, backend="jax")
    state = initial_program_state(plan, t=0.0, current_step=0)
    coords = np.meshgrid(
        *(np.linspace(-1, 1, n) for n in state.ex.shape), indexing="ij"
    )
    pulse = np.exp(-sum(c * c for c in coords) / 0.03) * np.cos(12 * coords[0])
    state = state._replace(ex=jnp.asarray(pulse, dtype=state.ex.dtype))
    return plan, place_tree(plan, state)


def measure(size=128, steps=100, repeats=9, full=False, bulk=False, graded=False):
    start = time.perf_counter()
    p, state = prepare(size, steps, full, bulk, graded)
    setup_seconds = time.perf_counter() - start
    coefficients = {
        "compressed": place_tree(p, p.coefficients),
        "dense": place_tree(p, dense_coefficients(p.coefficients, state)),
    }
    functions, outputs, samples, memory, compile_seconds = {}, {}, {}, {}, {}
    for name, coeffs in coefficients.items():
        start = time.perf_counter()
        functions[name] = build_program_scan(p).lower(state, coeffs).compile()
        compile_seconds[name] = time.perf_counter() - start
        memory[name] = executable_memory(functions[name])
        memory[name]["all_supplied_coefficients"] = unique_bytes(coeffs)
        # The compiler omits unused argument leaves. Keep their live device
        # allocations visible rather than implying they have been freed.
        memory[name]["unused_supplied_inputs"] = max(
            0, unique_bytes((state, coeffs)) - memory[name]["arguments"]
        )
        memory[name]["with_unused_inputs"] = (
            memory[name]["total"] + memory[name]["unused_supplied_inputs"]
        )
        samples[name] = []
        for _ in range(3):
            outputs[name] = jax.block_until_ready(functions[name](state, coeffs))
    errors, equal = {}, True
    for field in (
        "ex",
        "ey",
        "ez",
        "hx",
        "hy",
        "hz",
        "cpml_psi_e_terms",
        "cpml_psi_h_terms",
        "dft_vec_re",
        "dft_vec_im",
    ):
        values = [
            np.concatenate(
                [np.asarray(v).ravel() for v in jax.tree.leaves(getattr(output, field))]
                or [np.empty(0)]
            )
            for output in (outputs["compressed"], outputs["dense"])
        ]
        a, b = values
        equal &= np.array_equal(a, b)
        error = float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))
        assert np.isfinite(a).all() and error < 1e-5, (field, error)
        errors[field] = error
    for repeat in range(repeats):
        for name in (
            ("compressed", "dense") if repeat % 2 == 0 else ("dense", "compressed")
        ):
            start = time.perf_counter()
            jax.block_until_ready(functions[name](state, coefficients[name]))
            samples[name].append((time.perf_counter() - start) / steps)
    medians = {name: float(np.median(v)) for name, v in samples.items()}
    return dict(
        size=size,
        steps=steps,
        repeats=repeats,
        full=full,
        bulk=bulk,
        graded=graded,
        seconds_per_step=medians,
        speedup=medians["dense"] / medians["compressed"],
        executable_memory=memory,
        bitwise_equal=bool(equal),
        relative_errors=errors,
        samples=samples,
        setup_seconds=setup_seconds,
        compile_seconds=compile_seconds,
        encodings={
            prefix + c: (
                str(getattr(p.coefficients, prefix + c).headers.dtype)
                if isinstance(getattr(p.coefficients, prefix + c), MaterialBlocks)
                else "scalar"
                if getattr(p.coefficients, prefix + c).ndim == 0
                else "dense"
            )
            for prefix in ("e_permittivity_", "e_inverse_diagonal_")
            for c in "xyz"
        },
    )


def memory_worker(size, steps, full, bulk, graded, variant, host_material_grid=False):
    if variant == "dense":
        # Restore the original setup path as well as the runtime layout: do not
        # retain compressed device buffers alongside the baseline's dense ones.
        with patch("beamz.simulation.material_blocks.compact_material", lambda x: x):
            p, state = prepare(size, steps, full, bulk, graded, host_material_grid)
    else:
        p, state = prepare(size, steps, full, bulk, graded, host_material_grid)
    coeffs = place_tree(p, p.coefficients)
    device = jax.devices()[0]
    before = device.memory_stats()
    executable = build_program_scan(p).lower(state, coeffs).compile()
    out = jax.block_until_ready(executable(state, coeffs))
    return dict(
        size=size,
        steps=steps,
        full=full,
        bulk=bulk,
        graded=graded,
        variant=variant,
        host_material_grid=host_material_grid,
        before_compile=before,
        after_run=device.memory_stats(),
        executable_memory=executable_memory(executable),
        checksum=float(jnp.sum(out.ex**2)),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--output", required=True)
    parser.add_argument("--memory-worker", action="store_true")
    parser.add_argument(
        "--variant", choices=("compressed", "dense"), default="compressed"
    )
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--bulk", action="store_true")
    parser.add_argument("--graded", action="store_true")
    parser.add_argument("--host-material-grid", action="store_true")
    args = parser.parse_args()
    if args.memory_worker:
        result = memory_worker(
            args.size,
            args.steps,
            args.full,
            args.bulk,
            args.graded,
            args.variant,
            args.host_material_grid,
        )
        print(json.dumps(result), flush=True)
    else:
        rows = []
        for full, bulk in ((False, False), (True, False), (True, True)):
            for graded in (False, True):
                row = measure(args.size, args.steps, args.repeats, full, bulk, graded)
                rows.append(row)
                print(json.dumps(row), flush=True)
        result = dict(
            jax=jax.__version__,
            devices=[d.device_kind for d in jax.devices()],
            platform=platform.platform(),
            rows=rows,
        )
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)
