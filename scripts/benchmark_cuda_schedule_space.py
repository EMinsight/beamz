#!/usr/bin/env python3
"""Explore CUDA launch choices and measure complete time to result.

One fresh process per workload; candidates share material setup. Dependency-plan
construction is timed separately so both cold preparation and cached costs can
be compared. GPU measurements run sequentially in balanced orders.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


def configurations():
    candidates = [{"name": "streamed", "backend": "cuda_streamed", "env": {}}]
    candidates += [
        {
            "name": f"streamed-pair-{tile}",
            "backend": "cuda_streamed",
            "env": {
                "BEAMZ_CUDA_TEMPORAL_STEPS": "2",
                "BEAMZ_CUDA_PAIR_TILE": tile,
            },
        }
        for tile in ("16x8x16", "32x4x16", "32x8x16")
    ]
    candidates += [
        {
            "name": "streamed-capacity",
            "backend": "cuda_streamed",
            "env": {"BEAMZ_CUDA_MEMORY_POLICY": "capacity"},
        }
    ]
    candidates += [
        {
            "name": f"streamed-cpml-pair-{tile}",
            "backend": "cuda_streamed",
            "env": {"BEAMZ_CUDA_CPML_TEMPORAL": "1", "BEAMZ_CUDA_CPML_PAIR_TILE": tile},
        }
        for tile in ("16x8x16", "16x8x4", "oriented")
    ]
    candidates += [
        {
            "name": f"streamed-axes-{axes}",
            "backend": "cuda_streamed",
            "env": {"BEAMZ_CUDA_STORAGE_AXES": axes},
        }
        for axes in ("120", "201")
    ]
    candidates += [
        {
            "name": "streamed-spatial",
            "backend": "cuda_streamed",
            "env": {"BEAMZ_CUDA_CPML_SPATIAL": "1", "BEAMZ_CUDA_CPML_CORE_FUSION": "1"},
        },
        {
            "name": "streamed-fused",
            "backend": "cuda_streamed",
            "env": {"BEAMZ_CUDA_CPML_CORE_FUSION": "1"},
        },
    ]
    candidates += [
        {
            "name": f"mstep-{tile}-{segment}",
            "backend": "cuda_mstep",
            "env": {"BEAMZ_CUDA_MSTEP_TILE": tile, "BEAMZ_CUDA_MSTEP_SEGMENT": segment},
        }
        for tile in ("16x32", "8x32", "8x16", "16x16")
        for segment in ("whole", "32", "64", "128", "256")
    ]
    return candidates


def workloads():
    shapes = {
        "tiny": (32, 48, 64),
        "small": (64, 96, 128),
        "crossover": (96, 160, 256),
        "medium": (128, 256, 512),
        "large": (256, 512, 512),
        "narrow_x": (512, 256, 64),
        "narrow_z": (64, 512, 256),
        "thin_z": (48, 512, 512),
        "thin_y": (256, 48, 512),
        "thin_x": (256, 512, 48),
        "odd": (127, 193, 257),
        "cube": (128, 128, 128),
    }
    base = dict(
        steps=256,
        pml=12,
        monitors=2,
        frequencies=3,
        material="binary",
        source="mode",
        boundary="all",
        monitor_type="field",
    )
    cases = [
        dict(base, name=name, shape=shape, pml=4 if name == "tiny" else 12)
        for name, shape in shapes.items()
    ]
    for name, changes in {
        "pec": dict(boundary="pec"),
        "right": dict(boundary="right"),
        "cpml4": dict(pml=4),
        "cpml20": dict(pml=20),
        "smooth": dict(material="smooth"),
        "lossy": dict(material="lossy", source="gaussian"),
        "gaussian": dict(source="gaussian"),
        "no_monitors": dict(monitors=0),
        "dense_monitors": dict(monitors=3, frequencies=8),
    }.items():
        cases.append(dict(base, name=name, shape=shapes["medium"], **changes))
    return cases


def worker(case, candidates, output, samples, rounds):
    from types import SimpleNamespace

    import beamz._cuda as cuda
    import jax

    from beamz.simulation.cuda import mstep
    from beamz.simulation.execute import build_scan, initial_program_state
    from scripts.benchmark_cuda_realistic import build_simulation

    t = time.perf_counter()
    sim = build_simulation(SimpleNamespace(**case))
    base_program = sim.compile(num_steps=case["steps"], backend="cuda_streamed")
    state = initial_program_state(
        base_program, t=0, current_step=0, monitor_steps=case["steps"]
    )
    jax.block_until_ready(state)
    setup = time.perf_counter() - t
    report = {
        "case": case,
        "device": str(jax.devices()[0].device_kind),
        "jax": jax.__version__,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "extension_sha256": hashlib.sha256(
            Path(cuda.__file__).read_bytes()
        ).hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "shared_setup_s": setup,
        "records": [],
        "method": "Shared material setup; measured plan construction, lowering, XLA compilation, first execution and warm samples. Reversed order in alternating rounds.",
    }
    plan_cost = [0.0]
    original = mstep.build_mstep_plan

    def timed_plan(*args, **kwargs):
        t = time.perf_counter()
        p = original(*args, **kwargs)
        elapsed = time.perf_counter() - t
        plan_cost[0] = max(plan_cost[0], elapsed)
        last_plan.update(elapsed=elapsed, bytes=p.summary()["plan_bytes"])
        return p

    mstep.build_mstep_plan = timed_plan
    clean_env = {k: v for k, v in os.environ.items() if k.startswith("BEAMZ_CUDA_")}
    for round_index in range(rounds):
        ordered = candidates if round_index % 2 == 0 else candidates[::-1]
        for choice in ordered:
            for key in list(os.environ):
                if key.startswith("BEAMZ_CUDA_"):
                    del os.environ[key]
            os.environ.update(choice["env"])
            os.environ["BEAMZ_CUDA_AUTOTUNE"] = "off"
            last_plan = {}
            entry = {"configuration": choice, "round": round_index, "status": "ok"}
            try:
                t = time.perf_counter()
                program = sim.compile(
                    num_steps=case["steps"], backend=choice["backend"]
                )
                entry["program_setup_s"] = time.perf_counter() - t
                t = time.perf_counter()
                scan = build_scan(program, donate_state=False)
                entry["build_scan_s"] = time.perf_counter() - t
                entry["plan_build_s"] = last_plan.get("elapsed", 0.0)
                entry["plan_bytes"] = last_plan.get("bytes", 0)
                entry["decision"] = scan.cuda_schedule_decision
                entry["cuda_flags"] = program.config.cuda_flags
                entry["storage_axes"] = program.config.cuda_storage_axes
                t = time.perf_counter()
                lowered = scan.lower(state, program.coefficients)
                entry["lower_s"] = time.perf_counter() - t
                t = time.perf_counter()
                executable = lowered.compile()
                entry["compile_s"] = time.perf_counter() - t
                entry["memory"] = str(executable.memory_analysis())
                t = time.perf_counter()
                result = jax.block_until_ready(executable(state, program.coefficients))
                entry["first_run_s"] = time.perf_counter() - t
                jax.block_until_ready(executable(state, program.coefficients))
                timings = []
                for _ in range(samples):
                    t = time.perf_counter()
                    result = jax.block_until_ready(
                        executable(state, program.coefficients)
                    )
                    timings.append(time.perf_counter() - t)
                entry["samples_s"] = timings
                entry["median_run_s"] = statistics.median(timings)
                entry["warm_gcups"] = (
                    math.prod(case["shape"])
                    * case["steps"]
                    / entry["median_run_s"]
                    / 1e9
                )
                entry["actual_preparation_s"] = sum(
                    entry[k]
                    for k in ("program_setup_s", "build_scan_s", "lower_s", "compile_s")
                )
                entry["cold_preparation_s"] = (
                    entry["actual_preparation_s"]
                    - entry["plan_build_s"]
                    + (plan_cost[0] if choice["backend"] == "cuda_mstep" else 0)
                )
                entry["cold_time_to_result_s"] = (
                    entry["cold_preparation_s"] + entry["first_run_s"]
                )
                del executable, lowered, scan, result, program
            except Exception as exc:
                entry.update(status="unavailable", error=f"{type(exc).__name__}: {exc}")
            report["records"].append(entry)
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                f"{case['name']} round {round_index + 1} {choice['name']}: {entry.get('warm_gcups', entry.get('error'))}",
                flush=True,
            )
            gc.collect()
    for key in list(os.environ):
        if key.startswith("BEAMZ_CUDA_"):
            del os.environ[key]
    os.environ.update(clean_env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--case-json")
    parser.add_argument("--config-json")
    args = parser.parse_args()
    if min(args.steps, args.samples, args.rounds) < 1:
        parser.error("steps, samples and rounds must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        case = json.loads(args.case_json)
        worker(
            case,
            json.loads(args.config_json),
            args.output_dir / f"{case['name']}-{case['steps']}.json",
            args.samples,
            args.rounds,
        )
        return
    cases = [
        dict(c, steps=args.steps)
        for c in workloads()
        if args.cases is None or c["name"] in args.cases
    ]
    choices = [
        c for c in configurations() if args.configs is None or c["name"] in args.configs
    ]
    if not cases or not choices:
        parser.error("no matching cases/configurations")
    env = {k: v for k, v in os.environ.items() if not k.startswith("BEAMZ_CUDA_")}
    env.update(
        PYTHONPATH=str(Path(__file__).resolve().parents[1]),
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        XLA_PYTHON_CLIENT_MEM_FRACTION="0.6",
    )
    for case in cases:
        subprocess.run(
            [
                sys.executable,
                __file__,
                "--worker",
                "--case-json",
                json.dumps(case),
                "--config-json",
                json.dumps(choices),
                "--output-dir",
                str(args.output_dir.resolve()),
                "--samples",
                str(args.samples),
                "--rounds",
                str(args.rounds),
            ],
            env=env,
            check=True,
        )


if __name__ == "__main__":
    main()
