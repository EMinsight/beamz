from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results_beamz_engine_compare"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _stem_suffix(field_grid: str) -> str:
    return {
        "centered": "",
        "raw-yee": "_raw_yee",
        "meep-sampled": "_meep_sampled",
    }[field_grid]


def _slice_for_axis(data: np.ndarray, axis: int, index: int) -> np.ndarray:
    return np.take(np.asarray(data), indices=index, axis=axis)


def _best_fixed_slice_index(
    step_fields: np.ndarray,
    compiled_fields: np.ndarray,
    axis: int,
) -> int:
    reduce_axes = tuple(i for i in range(1, step_fields.ndim) if i != (axis + 1))
    step_score = np.sum(np.abs(step_fields), axis=reduce_axes)
    compiled_score = np.sum(np.abs(compiled_fields), axis=reduce_axes)
    total_score = np.sum(step_score + compiled_score, axis=0)
    return int(np.argmax(total_score))


def _extent_from_center_coords(x_um: np.ndarray, y_um: np.ndarray) -> tuple[float, float, float, float]:
    def _edges(vals: np.ndarray) -> tuple[float, float]:
        if vals.size == 1:
            return float(vals[0] - 0.5), float(vals[0] + 0.5)
        step = float(np.median(np.diff(vals)))
        return float(vals[0] - 0.5 * step), float(vals[-1] + 0.5 * step)

    x0, x1 = _edges(x_um)
    y0, y1 = _edges(y_um)
    return (x0, x1, y0, y1)


def _plot_material(
    *,
    step_slice: np.ndarray,
    compiled_slice: np.ndarray,
    x_um: np.ndarray,
    y_um: np.ndarray,
    output_path: Path,
) -> None:
    diff = compiled_slice - step_slice
    vmax = float(max(np.max(step_slice), np.max(compiled_slice), 1e-12))
    dmax = float(max(np.max(np.abs(diff)), 1e-12))
    extent = _extent_from_center_coords(x_um, y_um)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    panels = [
        (step_slice, "Step permittivity", "viridis", 1.0, vmax),
        (compiled_slice, "Compiled permittivity", "viridis", 1.0, vmax),
        (diff, "Compiled - Step", "coolwarm", -dmax, dmax),
    ]
    for ax, (arr, title, cmap, vmin, vmax_i) in zip(axes, panels, strict=False):
        im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax_i, extent=extent)
        ax.set_title(title)
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_component(
    *,
    step_slice: np.ndarray,
    compiled_slice: np.ndarray,
    x_um: np.ndarray,
    y_um: np.ndarray,
    component: str,
    step_index: int,
    rel_l2: float,
    output_path: Path,
) -> None:
    diff = compiled_slice - step_slice
    vmax_step = float(max(np.max(np.abs(step_slice)), 1e-12))
    vmax_comp = float(max(np.max(np.abs(compiled_slice)), 1e-12))
    scale = float(max(vmax_step, vmax_comp, 1e-12))
    diff_norm = diff / scale
    extent = _extent_from_center_coords(x_um, y_um)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    panels = [
        (step_slice, f"Step {component}", "RdBu_r", -vmax_step, vmax_step),
        (compiled_slice, f"Compiled {component}", "RdBu_r", -vmax_comp, vmax_comp),
        (diff_norm, f"Normalized error\nrel L2={rel_l2:.3e}", "coolwarm", -1.0, 1.0),
    ]
    for ax, (arr, title, cmap, vmin, vmax_i) in zip(axes, panels, strict=False):
        im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax_i, extent=extent)
        ax.set_title(title)
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{component} z-slice at snapshot step {step_index}")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot BeamZ step vs compiled 3D benchmark field comparisons."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--structure-index", type=int, default=0)
    parser.add_argument("--component", choices=("Ex", "Ey", "Ez"), default="Ez")
    parser.add_argument("--field-grid", choices=("centered", "raw-yee", "meep-sampled"), default="meep-sampled")
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    structure_dir = results_dir / f"structure_{args.structure_index:03d}"
    plot_dir = structure_dir / f"plots_{args.field_grid.replace('-', '_')}"
    plot_dir.mkdir(parents=True, exist_ok=True)
    suffix = _stem_suffix(args.field_grid)

    step_meta = _load_json(structure_dir / f"beamz_step_fields{suffix}.json")
    compiled_meta = _load_json(structure_dir / f"beamz_compiled_fields{suffix}.json")
    step = np.load(structure_dir / f"beamz_step_fields{suffix}.npz")
    compiled = np.load(structure_dir / f"beamz_compiled_fields{suffix}.npz")

    step_fields = np.asarray(step[args.component], dtype=np.float32)
    compiled_fields = np.asarray(compiled[args.component], dtype=np.float32)
    if step_fields.shape != compiled_fields.shape:
        raise ValueError(f"Field shape mismatch: {step_fields.shape} vs {compiled_fields.shape}")

    coords = step_meta["component_coordinates_um"][args.component]
    x_um = np.asarray(coords["x"], dtype=np.float64)
    y_um = np.asarray(coords["y"], dtype=np.float64)

    perm = np.asarray(step["permittivity"], dtype=np.float32)
    perm_coords = step_meta["component_coordinates_um"]["Ez"]
    perm_x_um = np.asarray(perm_coords["x"], dtype=np.float64)
    perm_y_um = np.asarray(perm_coords["y"], dtype=np.float64)
    perm_z = perm.shape[0] // 2
    _plot_material(
        step_slice=_slice_for_axis(perm, 0, perm_z),
        compiled_slice=_slice_for_axis(np.asarray(compiled["permittivity"], dtype=np.float32), 0, perm_z),
        x_um=perm_x_um,
        y_um=perm_y_um,
        output_path=plot_dir / "permittivity_slice_comparison.png",
    )

    z_index = _best_fixed_slice_index(step_fields, compiled_fields, axis=0)
    times = np.asarray(step["snapshot_times_s"], dtype=np.float64)
    steps = np.asarray(step["snapshot_steps"], dtype=np.int32)
    for snap_idx in range(step_fields.shape[0]):
        step_slice = _slice_for_axis(step_fields[snap_idx], 0, z_index)
        compiled_slice = _slice_for_axis(compiled_fields[snap_idx], 0, z_index)
        ref = float(np.linalg.norm(step_slice.ravel()))
        err = float(np.linalg.norm((compiled_slice - step_slice).ravel()))
        rel_l2 = err / max(ref, 1e-30)
        output_path = plot_dir / f"{args.component.lower()}_snapshot_{snap_idx:02d}.png"
        _plot_component(
            step_slice=step_slice,
            compiled_slice=compiled_slice,
            x_um=x_um,
            y_um=y_um,
            component=args.component,
            step_index=int(steps[snap_idx]),
            rel_l2=rel_l2,
            output_path=output_path,
        )
        print(
            json.dumps(
                {
                    "snapshot_index": int(snap_idx),
                    "step": int(steps[snap_idx]),
                    "time_s": float(times[snap_idx]),
                    "relative_l2": float(rel_l2),
                }
            )
        )

    del compiled_meta
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
