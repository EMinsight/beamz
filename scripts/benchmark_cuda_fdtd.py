"""Matched GPU benchmark for ordinary and differentiable FDTD.

Run from the repository root, for example::

    XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/benchmark_cuda_fdtd.py \
        --shape 64 128 192 --steps 512 --output results/cuda_performance/fdtd.json

Both backends use FP32 on the same GPU, the same sources, CPML, DFT monitors,
initial fields, material map and checkpoint interval. Compilation, device
placement and warmup are excluded from synchronized execution samples.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np
from cuda_benchmark_utils import provenance, speedup_interval

from beamz.design import MaterialGrid
from beamz.simulation.execute import build_scan, initial_program_state
from tests.performance.h100_workloads import H100Workload


def benchmark(args):
    if jax.default_backend() != "gpu":
        raise RuntimeError("Both JAX and native CUDA must run on the same GPU")
    jax.config.update("jax_enable_compilation_cache", False)
    workload = H100Workload(
        name="cuda_performance",
        shape_zyx=tuple(args.shape),
        timesteps=args.steps,
        pml_cells=args.pml_cells,
        heterogeneous=not args.bare,
        cpml=not args.bare,
        source=not args.bare,
        monitor=not args.bare,
    )
    simulation = workload.build()
    if args.conductivity:
        simulation = simulation.updated_copy(
            material_grid=MaterialGrid(
                permittivity=simulation.material_grid.permittivity,
                conductivity=np.float32(args.conductivity),
                permeability=np.float32(1.0),
                resolution=workload.resolution,
                shape=workload.shape_zyx,
            )
        )
    rng = np.random.default_rng(823)
    epsilon = jax.device_put(np.asarray(simulation.material_grid.permittivity))
    compiled, measurements, inputs = {}, {}, {}
    for backend in ("cuda", "jax"):
        program = simulation.compile(backend=backend)
        state = initial_program_state(program, t=0.0, current_step=0)
        state = (
            state._replace(
                **{
                    name: jnp.asarray(
                        rng.normal(size=getattr(state, name).shape).astype(np.float32)
                        * 1e-5
                    )
                    for name in ("ex", "ey", "ez", "hx", "hy", "hz")
                }
            )
            if backend == "cuda"
            else inputs["cuda"]["standard"][0]
        )
        # Non-donating executables preserve this shared initial state each run.
        inputs[backend] = {"standard": (state, program.coefficients)}
        functions = {"standard": build_scan(program, donate_state=False)}
        if not args.standard_only:
            solver = simulation.differentiable(
                backend=backend, checkpoint_interval=args.checkpoint
            )

            def objective(eps, solver=solver, state=state):
                out = solver(eps, state=state)
                # A nonzero objective exercises all fields and DFT accumulators.
                fields = sum(
                    jnp.mean(getattr(out, name) ** 2) for name in ("ex", "ey", "ez")
                )
                return fields + jnp.sum(out.dft_vec_re**2 + out.dft_vec_im**2)

            functions.update(
                forward=jax.jit(objective),
                value_and_grad=jax.jit(jax.value_and_grad(objective)),
            )
            inputs[backend].update(forward=(epsilon,), value_and_grad=(epsilon,))
        compiled[backend], measurements[backend] = {}, {}
        for kind, function in functions.items():
            start = time.perf_counter()
            executable = function.lower(*inputs[backend][kind]).compile()
            compile_s = time.perf_counter() - start
            compiled[backend][kind] = executable
            info = executable.memory_analysis()
            measurements[backend][kind] = dict(
                compile_s=compile_s,
                temporary_bytes=info.temp_size_in_bytes,
                samples_s=[],
            )
            for _ in range(args.warmup):
                jax.block_until_ready(executable(*inputs[backend][kind]))
            print(f"Compiled/warmed {backend} {kind}: {compile_s:.3f}s", flush=True)
    errors = {}
    for kind in compiled["cuda"]:
        native = compiled["cuda"][kind](*inputs["cuda"][kind])
        reference = compiled["jax"][kind](*inputs["jax"][kind])
        jax.block_until_ready((native, reference))
        relative_errors = []
        for actual, expected in zip(
            jax.tree.leaves(native), jax.tree.leaves(reference), strict=True
        ):
            a, e = np.asarray(actual), np.asarray(expected)
            if np.issubdtype(e.dtype, np.inexact):
                relative = np.linalg.norm((a - e).ravel()) / max(
                    np.linalg.norm(e.ravel()), 1e-30
                )
                relative_errors.append(float(relative))
                assert relative < 5e-4, (kind, relative)
                np.testing.assert_allclose(
                    a,
                    e,
                    rtol=5e-4,
                    atol=max(1e-12, float(np.max(np.abs(e), initial=0)) * 5e-4),
                )
            else:
                np.testing.assert_array_equal(a, e)
        errors[kind] = max(relative_errors, default=0.0)
        print(f"{kind} maximum leaf relative error: {errors[kind]:.3g}", flush=True)
        del native, reference
        for sample in range(args.samples):
            order = ("cuda", "jax") if sample % 2 == 0 else ("jax", "cuda")
            for backend in order:
                start = time.perf_counter()
                result = compiled[backend][kind](*inputs[backend][kind])
                jax.block_until_ready(result)
                measurements[backend][kind]["samples_s"].append(
                    time.perf_counter() - start
                )
                del result
        if args.profile:
            for backend in ("cuda", "jax"):
                with jax.profiler.trace(
                    str(args.profile / backend / kind), create_perfetto_trace=True
                ):
                    jax.block_until_ready(
                        compiled[backend][kind](*inputs[backend][kind])
                    )
    for modes in measurements.values():
        for result in modes.values():
            samples = result["samples_s"]
            result.update(
                median_s=float(np.median(samples)),
                min_s=min(samples),
                max_s=max(samples),
            )
    speedups = {
        kind: measurements["jax"][kind]["median_s"]
        / measurements["cuda"][kind]["median_s"]
        for kind in compiled["cuda"]
    }
    metadata = dict(
        shape_zyx=args.shape,
        steps=args.steps,
        checkpoint_interval=args.checkpoint,
        pml_cells=args.pml_cells if not args.bare else 0,
        bare=args.bare,
        conductivity_s_per_m=args.conductivity,
        dtype="float32",
        jax_version=jax.__version__,
        devices=[str(d) for d in jax.devices()],
        gpu=subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
        ).strip(),
        warmup_calls=args.warmup,
        samples=args.samples,
        compilation_cache_enabled=False,
        provenance=provenance(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                metadata=metadata,
                measurements=measurements,
                maximum_relative_errors=errors,
                cuda_speedup_over_jax=speedups,
                speedup_95_percent_intervals={
                    kind: speedup_interval(
                        measurements["cuda"][kind]["samples_s"],
                        measurements["jax"][kind]["samples_s"],
                    )
                    for kind in compiled["cuda"]
                },
            ),
            indent=2,
        )
    )
    print(json.dumps(speedups, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape", nargs=3, type=int, default=(44, 100, 100), metavar=("NZ", "NY", "NX")
    )
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--pml-cells", type=int, default=6)
    parser.add_argument("--checkpoint", type=int, default=32)
    parser.add_argument("--conductivity", type=float, default=0.0)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--bare", action="store_true")
    parser.add_argument("--standard-only", action="store_true")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    benchmark(parser.parse_args())
