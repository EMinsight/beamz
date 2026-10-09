#!/usr/bin/env python3
"""Compare explicit CUDA schedules in fresh processes on identical workloads.

Warm execution includes sources, CPML and DFT monitors. First-use costs and
memory are recorded separately. No automatic switching threshold is installed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=["32x64x96", "64x128x256", "128x256x512", "256x512x512", "512x256x64"],
        help="Physical z,y,x cell counts; each shape uses both schedules",
    )
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--pml", type=int, default=12)
    parser.add_argument("--boundary", choices=("all", "right", "pec"), default="all")
    parser.add_argument("--source", choices=("mode", "gaussian"), default="mode")
    parser.add_argument("--monitors", type=int, default=2)
    parser.add_argument(
        "--material", choices=("binary", "smooth", "scalar", "lossy"), default="binary"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("rounds must be positive")
    shapes = []
    for value in args.shapes:
        try:
            shape = tuple(int(v) for v in value.split("x"))
        except ValueError:
            parser.error(f"invalid shape {value!r}")
        if len(shape) != 3 or min(shape) <= 2 * args.pml:
            parser.error(f"shape {value!r} needs a nonempty CPML interior")
        shapes.append(shape)
    root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Compare the default established policy with the explicit new policy.
    # Avoid inheriting diagnostic scheduling overrides from the caller.
    env = {k: v for k, v in os.environ.items() if not k.startswith("BEAMZ_CUDA_")}
    env["PYTHONPATH"] = str(root)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    records = []
    for round_index in range(args.rounds):
        backends = ("cuda_streamed", "cuda_mstep")
        if round_index % 2:
            backends = backends[::-1]
        for shape in shapes:
            label = "x".join(map(str, shape))
            for backend in backends:
                output = args.output_dir / f"{label}-{backend}-{round_index}.json"
                print(f"round {round_index + 1}: {label} {backend}", flush=True)
                command = [
                    sys.executable,
                    str(root / "scripts/benchmark_cuda_realistic.py"),
                    "--backend",
                    backend,
                    "--shape",
                    *map(str, shape),
                    "--steps",
                    str(args.steps),
                    "--samples",
                    str(args.samples),
                    "--warmups",
                    "3",
                    "--pml",
                    str(args.pml),
                    "--boundary",
                    args.boundary,
                    "--source",
                    args.source,
                    "--monitors",
                    str(args.monitors),
                    "--material",
                    args.material,
                    "--output",
                    str(output.resolve()),
                ]
                with output.with_suffix(".log").open("w") as log:
                    subprocess.run(
                        command, cwd=root, env=env, stdout=log, stderr=log, check=True
                    )
                data = json.loads(output.read_text())
                records.append(data)
                print(f"  {data['median_gcups']:.3f} GCUPS", flush=True)
    (args.output_dir / "results.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
