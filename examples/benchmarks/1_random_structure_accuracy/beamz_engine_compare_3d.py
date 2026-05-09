from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import random_structure_compare as rsc


DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results_beamz_engine_compare"


def _run_beamz_single_compiled(
    cfg: rsc.RandomBenchmarkConfig,
    *,
    structure_specs: list[dict[str, Any]],
    structure_index: int,
    seed: int,
    output_dir: Path,
    field_grid: str,
) -> dict[str, Any]:
    from beamz import PEC, Simulation

    signal = rsc._gaussian_modulated_signal_fn(cfg)
    design = rsc._build_beamz_design(cfg, structure_specs)
    source_voxels, source_weights = rsc._beamz_ez_source_voxels(
        cfg,
        width_um=cfg.source_width_um * cfg.meep_source_width_scale,
    )
    source = rsc._BeamzCurrentSource(
        signal=signal,
        voxel_indices=source_voxels,
        voxel_weights=source_weights,
    )
    time_axis = np.arange(cfg.num_steps, dtype=np.float64) * cfg.dt_s
    sim = Simulation(
        design=design,
        sources=[source],
        boundaries=[PEC(edges="all")],
        time=time_axis,
        resolution=cfg.dx_m,
    )

    snapshots: list[dict[str, Any]] = []
    start = time.perf_counter()
    for target_step in cfg.snapshot_steps:
        step_delta = int(target_step - sim.current_step)
        if step_delta > 0:
            sim.run_compiled(num_steps=step_delta, progress=False)
        if field_grid == rsc.FIELD_GRID_RAW_YEE:
            field_data = rsc._extract_raw_beamz_fields(sim)
            axis_order = "zyx_raw_yee"
        elif field_grid == rsc.FIELD_GRID_MEEP_SAMPLED:
            field_data = rsc._extract_meep_sampled_beamz_fields(sim)
            axis_order = "zyx_meep_sampled"
        else:
            field_data = rsc._extract_centered_beamz_fields(sim)
            axis_order = "zyx_centered"
        snapshots.append(
            {
                "step": int(sim.current_step),
                "time_s": float(sim.t),
                "fields": field_data,
            }
        )
    runtime_s = time.perf_counter() - start

    stem = f"beamz_compiled_fields{rsc._solver_output_stem('', field_grid).replace('_fields', '')}"
    out_path = output_dir / f"structure_{structure_index:03d}" / f"{stem}.npz"
    return rsc._write_solver_output(
        output_path=out_path,
        solver_name="beamz_compiled",
        cfg=cfg,
        structure_index=structure_index,
        seed=seed,
        structure_specs=structure_specs,
        permittivity=np.asarray(sim.fields.permittivity),
        snapshots=snapshots,
        runtime_s=runtime_s,
        extra_metadata={
            "axis_order": axis_order,
            "field_grid": field_grid,
            "component_shapes": rsc._component_shape_metadata(snapshots),
            "component_coordinates_um": rsc._beamz_component_coordinates_um(cfg, field_grid),
        },
    )


def _run_beamz_single_step(
    cfg: rsc.RandomBenchmarkConfig,
    *,
    structure_specs: list[dict[str, Any]],
    structure_index: int,
    seed: int,
    output_dir: Path,
    field_grid: str,
) -> dict[str, Any]:
    result = rsc.run_beamz_single(
        cfg,
        structure_specs=structure_specs,
        structure_index=structure_index,
        seed=seed,
        output_dir=output_dir,
        field_grid=field_grid,
    )
    structure_dir = output_dir / f"structure_{structure_index:03d}"
    src_npz = structure_dir / f"beamz_fields{_stem_suffix(field_grid)}.npz"
    src_json = structure_dir / f"beamz_fields{_stem_suffix(field_grid)}.json"
    dst_npz = structure_dir / f"beamz_step_fields{_stem_suffix(field_grid)}.npz"
    dst_json = structure_dir / f"beamz_step_fields{_stem_suffix(field_grid)}.json"
    if src_npz.exists():
        src_npz.replace(dst_npz)
    if src_json.exists():
        payload = json.loads(src_json.read_text())
        payload["solver"] = "beamz_step"
        dst_json.write_text(json.dumps(payload, indent=2))
        src_json.unlink()
    result = dict(result)
    result["solver"] = "beamz_step"
    result["data_file"] = dst_npz.name
    return result


def _stem_suffix(field_grid: str) -> str:
    return {
        rsc.FIELD_GRID_CENTERED: "",
        rsc.FIELD_GRID_RAW_YEE: "_raw_yee",
        rsc.FIELD_GRID_MEEP_SAMPLED: "_meep_sampled",
    }[field_grid]


def _compare_solver_outputs(
    *,
    structure_dir: Path,
    field_grid: str,
) -> dict[str, Any]:
    suffix = _stem_suffix(field_grid)
    step = np.load(structure_dir / f"beamz_step_fields{suffix}.npz")
    compiled = np.load(structure_dir / f"beamz_compiled_fields{suffix}.npz")

    component_metrics: dict[str, Any] = {}
    max_rel = 0.0
    for component in ("Ex", "Ey", "Ez"):
        step_arr = np.asarray(step[component], dtype=np.float64)
        compiled_arr = np.asarray(compiled[component], dtype=np.float64)
        diff = compiled_arr - step_arr
        rel_by_snapshot = []
        for snap_idx in range(step_arr.shape[0]):
            ref = float(np.linalg.norm(step_arr[snap_idx].ravel()))
            err = float(np.linalg.norm(diff[snap_idx].ravel()))
            rel = err / max(ref, 1e-30)
            rel_by_snapshot.append(rel)
            max_rel = max(max_rel, rel)
        component_metrics[component] = {
            "relative_l2_by_snapshot": rel_by_snapshot,
            "max_relative_l2": float(max(rel_by_snapshot) if rel_by_snapshot else 0.0),
            "mean_relative_l2": float(np.mean(rel_by_snapshot) if rel_by_snapshot else 0.0),
        }

    perm_step = np.asarray(step["permittivity"], dtype=np.float64)
    perm_compiled = np.asarray(compiled["permittivity"], dtype=np.float64)
    perm_diff = perm_compiled - perm_step
    perm_ref = float(np.linalg.norm(perm_step.ravel()))
    perm_err = float(np.linalg.norm(perm_diff.ravel()))

    return {
        "field_grid": field_grid,
        "components": component_metrics,
        "max_relative_l2_overall": float(max_rel),
        "permittivity_relative_l2": float(perm_err / max(perm_ref, 1e-30)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare BeamZ 3D step() and compiled engines on the validated random-structure benchmark setup."
    )
    parser.add_argument("--num-structures", type=int, default=1)
    parser.add_argument("--structure-index", type=int, default=None)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--resolution-ppw", type=int, default=8)
    parser.add_argument("--num-primitives", type=int, default=4)
    parser.add_argument("--polygon-permittivity", type=float, default=12.0)
    parser.add_argument("--num-snapshots", type=int, default=3)
    parser.add_argument("--total-cycles", type=float, default=4.0)
    parser.add_argument("--wavelength-um", type=float, default=1.0)
    parser.add_argument("--courant-safety", type=float, default=0.95)
    parser.add_argument("--meep-source-width-scale", type=float, default=1.0)
    parser.add_argument("--meep-source-amplitude-scale", type=float, default=7.37e-07)
    parser.add_argument(
        "--geometry-mode",
        choices=("random-primitives", "random-polygon"),
        default="random-primitives",
    )
    parser.add_argument(
        "--geometry-source",
        choices=("native", "beamz-raster"),
        default="native",
    )
    parser.add_argument(
        "--field-grid",
        choices=(rsc.FIELD_GRID_CENTERED, rsc.FIELD_GRID_RAW_YEE, rsc.FIELD_GRID_MEEP_SAMPLED),
        default=rsc.FIELD_GRID_MEEP_SAMPLED,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--emit-json-only", action="store_true")
    args = parser.parse_args()

    cfg = rsc.RandomBenchmarkConfig(
        resolution_ppw=args.resolution_ppw,
        geometry_mode=args.geometry_mode,
        geometry_source=args.geometry_source,
        num_primitives=args.num_primitives,
        num_snapshots=args.num_snapshots,
        total_cycles=args.total_cycles,
        wavelength_um=args.wavelength_um,
        courant_safety=args.courant_safety,
        meep_source_width_scale=args.meep_source_width_scale,
        meep_source_amplitude_scale=args.meep_source_amplitude_scale,
        polygon_permittivity=args.polygon_permittivity,
    )

    if args.structure_index is not None:
        indices = [int(args.structure_index)]
    else:
        indices = list(range(max(1, int(args.num_structures))))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_structures = []
    for structure_index in indices:
        seed = int(args.start_seed) + int(structure_index)
        structure_specs = rsc._random_structure_specs(cfg, seed=seed)
        structure_dir = output_dir / f"structure_{structure_index:03d}"
        structure_dir.mkdir(parents=True, exist_ok=True)
        (structure_dir / "structure.json").write_text(
            json.dumps(
                {
                    "structure_index": int(structure_index),
                    "seed": int(seed),
                    "config": asdict(cfg),
                    "structure_specs": structure_specs,
                },
                indent=2,
            )
        )

        step_result = _run_beamz_single_step(
            cfg,
            structure_specs=structure_specs,
            structure_index=structure_index,
            seed=seed,
            output_dir=output_dir,
            field_grid=args.field_grid,
        )
        compiled_result = _run_beamz_single_compiled(
            cfg,
            structure_specs=structure_specs,
            structure_index=structure_index,
            seed=seed,
            output_dir=output_dir,
            field_grid=args.field_grid,
        )
        comparison = _compare_solver_outputs(
            structure_dir=structure_dir,
            field_grid=args.field_grid,
        )
        manifest_structures.append(
            {
                "structure_index": int(structure_index),
                "seed": int(seed),
                "results": {
                    "beamz_step": step_result,
                    "beamz_compiled": compiled_result,
                },
                "comparison": comparison,
            }
        )

    manifest = {
        "config": asdict(cfg),
        "field_grid": args.field_grid,
        "structures": manifest_structures,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    if args.emit_json_only:
        print(json.dumps(manifest))
    else:
        print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
