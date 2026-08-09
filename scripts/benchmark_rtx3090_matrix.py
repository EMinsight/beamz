#!/usr/bin/env python3
"""Benchmark the cumulative BeamZ feature envelope on an RTX 3090."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from benchmark_rtx3090_cuda import run_comparison
from rtx3090_benchmark import RTX3090Matrix, write_matrix_artifacts

PROFILES = (
    "uniform_pec",
    "heterogeneous_pec",
    "heterogeneous_cpml",
    "axis_uniform_pec",
    "rectilinear_pec",
    "rectilinear_cpml",
    "pec_source",
    "pec_source_monitor",
    "cpml_source",
    "cpml_source_monitor",
    "conductive_pec",
    "sponge",
    "mixed_boundaries",
    "multiple_sources",
    "h_source",
    "multiple_monitors",
    "ragged_monitors",
    "scheduled_windowed_monitor",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape", nargs=3, type=int, default=(64, 96, 128), metavar=("NZ", "NY", "NX")
    )
    parser.add_argument("--timesteps", type=int, default=160)
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmarks/results/rtx3090/matrix")
    )
    parser.add_argument("--allow-other-gpu", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.samples < 3 or args.warmups < 1 or args.timesteps < 2:
        raise SystemExit("matrix requires >=3 samples, >=1 warmup, and >=2 timesteps")
    comparisons = []
    for profile in PROFILES:
        comparison = run_comparison(
            SimpleNamespace(
                shape=tuple(args.shape),
                timesteps=args.timesteps,
                samples=args.samples,
                warmups=args.warmups,
                backend="cuda_streamed",
                allow_other_gpu=args.allow_other_gpu,
                profile=profile,
            )
        )
        comparisons.append((profile, comparison))
        print(f"{profile}: {comparison.runtime_speedup:.3f}x")
    paths = write_matrix_artifacts(RTX3090Matrix(tuple(comparisons)), args.output_dir)
    for kind, path in paths.items():
        print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
