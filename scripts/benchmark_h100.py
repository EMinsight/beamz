#!/usr/bin/env python3
"""Run the canonical BeamZ H100 throughput workloads and emit schema-v2 JSON."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import jax
import jaxlib

import beamz
from beamz.simulation import observe as monitor_runtime
from beamz.simulation import sharding as sharding_runtime
from beamz.simulation.execute import build_scan, initial_program_state
from tests.performance.benchmark_schema import BenchmarkRecord
from tests.performance.h100_workloads import H100_WORKLOADS


def _block(state) -> None:
    state.ez.block_until_ready()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _peak_memory_bytes(devices, fallback: int) -> int:
    total = 0
    found = False
    for device in devices:
        stats = device.memory_stats() or {}
        values = [
            int(stats[key])
            for key in ("peak_bytes_in_use", "bytes_in_use")
            if key in stats
        ]
        if values:
            found = True
            total += max(values)
    return total if found and total > 0 else int(fallback)


def _time_call(callable_):
    started = time.perf_counter()
    value = callable_()
    _block(value)
    return value, time.perf_counter() - started


def run_benchmark(args: argparse.Namespace) -> BenchmarkRecord:
    devices = jax.devices()
    if not devices:
        raise RuntimeError("JAX reported no execution devices")
    if not args.allow_cpu and not all(device.platform == "gpu" for device in devices):
        raise RuntimeError("H100 benchmark requires GPU devices; use --allow-cpu for smoke tests")

    workload = H100_WORKLOADS[args.workload].resized(
        shape_zyx=None if args.shape is None else tuple(args.shape),
        timesteps=args.timesteps,
    )
    sim = workload.build()
    sim.clear_compiled_cache()
    program = sim.compile(num_steps=workload.timesteps, sharding=args.sharding)
    state = initial_program_state(
        program,
        t=float(sim.time[0]),
        current_step=0,
        monitor_steps=workload.timesteps,
    )
    state = sharding_runtime.prepare_state(
        program,
        state,
        replicated_fields=(*monitor_runtime.MONITOR_FIELDS, "t", "current_step"),
    )
    coefficients = sharding_runtime.place_tree(program, program.coefficients)
    scan = build_scan(program, donate_state=False)

    started = time.perf_counter()
    lowered = scan.lower(state, coefficients)
    trace_lower_s = time.perf_counter() - started
    started = time.perf_counter()
    executable = lowered.compile()
    compile_s = time.perf_counter() - started

    # One unreported launch primes allocator and clocks before the measured samples.
    warm_state = executable(state, coefficients)
    _block(warm_state)
    kernel_samples = tuple(
        _time_call(lambda: executable(state, coefficients))[1]
        for _ in range(args.samples)
    )

    # Public-path latency includes input placement, state allocation and result decode.
    warm_run = sim.advance(num_steps=workload.timesteps, sharding=args.sharding)
    _block(warm_run.state)
    end_to_end_samples = tuple(
        _time_call(
            lambda: sim.advance(num_steps=workload.timesteps, sharding=args.sharding).state
        )[1]
        for _ in range(args.samples)
    )
    memory_fallback = sim.memory_estimate(
        num_steps=workload.timesteps,
        sharding=args.sharding,
    )["total_bytes"]
    features = workload.feature_labels
    return BenchmarkRecord(
        beamz_commit=_git_commit(),
        beamz_version=beamz.__version__,
        python_version=platform.python_version(),
        jax_version=jax.__version__,
        jaxlib_version=jaxlib.__version__,
        workload=workload.name,
        backend="jax",
        device="; ".join(sorted({device.device_kind for device in devices})),
        device_count=len(devices),
        precision="float32",
        grid_dimensions=workload.shape_zyx,
        timesteps=workload.timesteps,
        boundaries=features["boundaries"],
        sources=features["sources"],
        monitors=features["monitors"],
        trace_lower_s=trace_lower_s,
        compile_s=compile_s,
        warm_runtime_samples_s=kernel_samples,
        warm_end_to_end_samples_s=end_to_end_samples,
        peak_memory_bytes=_peak_memory_bytes(devices, memory_fallback),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=H100_WORKLOADS, default="realistic_3d")
    parser.add_argument("--shape", nargs=3, type=int, metavar=("NZ", "NY", "NX"))
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--sharding", default=None)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="allow a non-comparable smoke run when no accelerator is available",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.samples < 3:
        raise SystemExit("--samples must be at least three")
    if args.timesteps is None:
        args.timesteps = H100_WORKLOADS[args.workload].timesteps
    record = run_benchmark(args)
    payload = json.dumps(record.as_dict(), indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

