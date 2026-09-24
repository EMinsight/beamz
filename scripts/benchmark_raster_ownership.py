"""Run the pinned PR #230 crossing adapter with a selectable native raster build.

Extract the benchmark helper/case files from 372143da into --benchmark-root.
Use a fresh process per run; --native selects a saved baseline extension.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _digest(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def _save(directory, simulation, results, scattering, *, execution_backend):
    from beamz.design.raster.importers._beamz import from_beamz

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    grid = simulation.grid
    np.savez_compressed(
        directory / "grid.npz", **dict(zip("xyz", grid.edges, strict=True))
    )
    scene = from_beamz(simulation.design, padded_size=grid.extent)
    (directory / "scene.json").write_text(scene.to_json())
    source = simulation.sources[0]
    signal, quadrature = source.source_time.sample(simulation.time)
    data = {
        "backend": execution_backend,
        "grid_shape": list(grid.shape),
        "grid_edges_sha256": [_digest(a) for a in grid.edges],
        "time_sha256": _digest(simulation.time),
        "source_signal_sha256": _digest(signal),
        "source_quadrature_sha256": _digest(quadrature),
        "source_center": list(source.center),
        "monitors": [
            {"name": m.name, "center": list(m.center), "size": list(m.size)}
            for m in simulation.monitors
        ],
        "raster_options": dataclasses.asdict(simulation.raster_options),
        "boundaries": [repr(b) for b in simulation.boundaries],
        "termination": dataclasses.asdict(results.termination),
        "frequencies": np.asarray(scattering.frequencies).tolist(),
        "s_parameters": {
            f"{k[0]}_{k[1]}": [[float(v.real), float(v.imag)] for v in np.asarray(a)]
            for k, a in scattering.s_matrix.items()
        },
        "valid_mask": np.asarray(scattering.diagnostics["valid_mask"]).tolist(),
    }
    (directory / "details.json").write_text(
        json.dumps(data, indent=2, default=str) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--ppw", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    if args.native is not None:
        name = "beamz.design.raster._native"
        spec = importlib.util.spec_from_file_location(name, args.native.resolve())
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        sys.modules[name] = native
    differential = importlib.import_module("tests.differential")
    differential.__path__.append(
        str(args.benchmark_root.resolve() / "tests/differential")
    )
    crossing = importlib.import_module("tests.differential.passive_soi.crossing")
    # Preserve the pinned adapter's diagnostics monitor and all numerical settings;
    # replace only its plot/array writer with compact reproducibility metadata.
    crossing._save_crossing_artifacts = _save
    result = crossing.run_crossing_benchmark(
        resolution_ppw=args.ppw,
        backend="cuda_streamed",
        progress=True,
        artifact_dir=args.output,
    )
    encoded = json.dumps(dataclasses.asdict(result), indent=2) + "\n"
    (args.output / "result.json").write_text(encoded)
    print(encoded)


if __name__ == "__main__":
    main()
