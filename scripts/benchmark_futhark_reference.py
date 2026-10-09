#!/usr/bin/env python3
"""Run the compiled Futhark reference with the CUDA benchmark's workload/timing.

Use the reference checkout's Python environment and put it on PYTHONPATH.
This script imports the current workload builder, so shape, source, boundary,
monitors and output lifetimes match benchmark_cuda_realistic.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import jax
import numpy as np
from benchmark_cuda_realistic import build_simulation

import beamz as bz
from beamz.simulation.execute import build_scan, initial_program_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", nargs=3, type=int, required=True)
    parser.add_argument("--boundary", choices=("all", "right", "pec"), default="all")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--pml", type=int, default=12)
    parser.add_argument("--monitors", type=int, default=2)
    parser.add_argument("--frequencies", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.material, args.source, args.monitor_type = "binary", "mode", "field"
    os.environ["BEAMZ_FUTHARK_TEMPORAL"] = "2"
    start = time.perf_counter()
    sim = build_simulation(args)
    program = sim.compile(num_steps=args.steps, backend="futhark")
    state = initial_program_state(
        program, t=0, current_step=0, monitor_steps=args.steps
    )
    jax.block_until_ready(state)
    setup_s = time.perf_counter() - start
    start = time.perf_counter()
    executable = (
        build_scan(program, donate_state=False)
        .lower(state, program.coefficients)
        .compile()
    )
    compile_s = time.perf_counter() - start
    start = time.perf_counter()
    result = executable(state, program.coefficients)
    jax.block_until_ready(result)
    first_run_s = time.perf_counter() - start
    for _ in range(args.warmups):
        jax.block_until_ready(executable(state, program.coefficients))
    timings = []
    for _ in range(args.samples):
        start = time.perf_counter()
        result = executable(state, program.coefficients)
        jax.block_until_ready(result)
        timings.append(time.perf_counter() - start)
    root = Path(bz.__file__).resolve().parents[1]
    from beamz.simulation.futhark.runtime import library_path

    library = library_path("cuda")
    data = {
        "backend": "futhark",
        "allocator": {
            "preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", ""),
            "memory_fraction": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", ""),
            "implementation": os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR", ""),
        },
        "temporal_steps": 2,
        "shape": args.shape,
        "steps": args.steps,
        "boundary": args.boundary,
        "pml": args.pml,
        "monitors": args.monitors,
        "frequencies": args.frequencies,
        "source": args.source,
        "material": args.material,
        "source_root": str(root),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "workload_sha256": hashlib.sha256(
            Path(__file__).with_name("benchmark_cuda_realistic.py").read_bytes()
        ).hexdigest(),
        "extension_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "temporal_source_sha256": hashlib.sha256(
            (root / "futhark/temporal.fut").read_bytes()
        ).hexdigest(),
        "device": str(jax.devices()[0].device_kind),
        "jax": jax.__version__,
        "setup_s": setup_s,
        "compile_s": compile_s,
        "first_run_s": first_run_s,
        "samples_s": timings,
        "median_gcups": float(
            np.prod(args.shape) * args.steps / np.median(timings) / 1e9
        ),
        "gpu": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,clocks.current.sm,clocks.current.memory,power.draw,power.limit,temperature.gpu,memory.used",
                "--format=csv",
            ],
            text=True,
        ).strip(),
    }
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data), flush=True)


if __name__ == "__main__":
    main()
