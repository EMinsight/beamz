from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"
FIELD_GRID_CENTERED = "centered"
FIELD_GRID_RAW_YEE = "raw-yee"
FIELD_GRID_MEEP_SAMPLED = "meep-sampled"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _pick_center_slice(volume: np.ndarray, axis: int) -> np.ndarray:
    index = volume.shape[axis] // 2
    return np.take(volume, indices=index, axis=axis)


def _canonical_source_mask(cfg: dict, grid_shape: tuple[int, int, int]) -> np.ndarray:
    nz, ny, nx = grid_shape
    res_um = float(cfg["wavelength_um"]) / float(cfg["resolution_ppw"])
    x = (np.arange(nx, dtype=np.float32) + 0.5) * res_um
    y = (np.arange(ny, dtype=np.float32) + 0.5) * res_um
    z = (np.arange(nz, dtype=np.float32) + 0.5) * res_um
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    dist_sq = (
        (xx - float(cfg["source_x_um"])) ** 2
        + (yy - float(cfg["source_y_um"])) ** 2
        + (zz - float(cfg["source_z_um"])) ** 2
    )
    return np.asarray(
        np.exp(-dist_sq / (2.0 * float(cfg["source_width_um"]) ** 2)),
        dtype=np.float32,
    )


def _source_index_coords(cfg: dict) -> tuple[float, float, float]:
    res_um = float(cfg["wavelength_um"]) / float(cfg["resolution_ppw"])
    return (
        float(cfg["source_x_um"]) / res_um - 0.5,
        float(cfg["source_y_um"]) / res_um - 0.5,
        float(cfg["source_z_um"]) / res_um - 0.5,
    )


def _beamz_raw_component_source_xy(component: str, cfg: dict) -> tuple[float, float]:
    """Projected source marker on Beamz's raw Yee storage lattice."""
    res_um = float(cfg["wavelength_um"]) / float(cfg["resolution_ppw"])
    base_x = float(cfg["source_x_um"]) / res_um - 1.0
    base_y = float(cfg["source_y_um"]) / res_um - 1.0
    del component
    return (base_x, base_y)


def _beamz_raw_component_source_z(component: str, cfg: dict) -> float:
    res_um = float(cfg["wavelength_um"]) / float(cfg["resolution_ppw"])
    base_z = float(cfg["source_z_um"]) / res_um - 1.0
    del component
    return base_z


def _best_slice_index(beamz_volume: np.ndarray, meep_volume: np.ndarray, axis: int) -> int:
    beamz_score = np.sum(np.abs(beamz_volume), axis=tuple(i for i in range(3) if i != axis))
    meep_score = np.sum(np.abs(meep_volume), axis=tuple(i for i in range(3) if i != axis))
    return int(np.argmax(beamz_score + meep_score))


def _best_fixed_slice_index(
    beamz_fields: np.ndarray,
    meep_fields: np.ndarray,
    axis: int,
) -> int:
    """Pick one slice index to use across all time steps for a given component."""
    reduce_axes = tuple(i for i in range(1, beamz_fields.ndim) if i != (axis + 1))
    beamz_score = np.sum(np.abs(beamz_fields), axis=reduce_axes)
    meep_score = np.sum(np.abs(meep_fields), axis=reduce_axes)
    total_score = np.sum(beamz_score + meep_score, axis=0)
    return int(np.argmax(total_score))


def _structure_fixed_slice_index(
    structure_specs: list[dict],
    z_coords_um: np.ndarray,
) -> int | None:
    if not structure_specs:
        return None
    z_centers = []
    for spec in structure_specs:
        if spec["type"] == "polygon_prism":
            z_centers.append(float(spec["z_um"]) + 0.5 * float(spec["depth_um"]))
        elif spec["type"] == "block":
            z_centers.append(float(spec["position_um"][2]) + 0.5 * float(spec["size_um"][2]))
        elif spec["type"] == "sphere":
            z_centers.append(float(spec["center_um"][2]))
    if not z_centers:
        return None
    target = float(np.median(np.asarray(z_centers, dtype=np.float64)))
    return int(np.argmin(np.abs(z_coords_um - target)))


def _slice_for_axis(data: np.ndarray, axis: int, index: int) -> np.ndarray:
    return np.take(np.asarray(data), indices=index, axis=axis)


def _slice_coords_for_axis(
    coords_zyx_um: dict[str, list[float]],
    axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    if axis != 0:
        raise NotImplementedError("Only z-slice plotting is currently supported.")
    y_um = np.asarray(coords_zyx_um["y"], dtype=np.float64)
    x_um = np.asarray(coords_zyx_um["x"], dtype=np.float64)
    return x_um, y_um


def _extent_from_center_coords(x_um: np.ndarray, y_um: np.ndarray) -> tuple[float, float, float, float]:
    def _edges(vals: np.ndarray) -> tuple[float, float]:
        if vals.size == 1:
            delta = 0.5
            return float(vals[0] - delta), float(vals[0] + delta)
        step = float(np.median(np.diff(vals)))
        return float(vals[0] - 0.5 * step), float(vals[-1] + 0.5 * step)

    x0, x1 = _edges(x_um)
    y0, y1 = _edges(y_um)
    return (x0, x1, y0, y1)


def _physical_centroid_um(arr: np.ndarray, x_um: np.ndarray, y_um: np.ndarray) -> tuple[float, float]:
    abs_arr = np.abs(np.asarray(arr, dtype=np.float64))
    total = float(abs_arr.sum())
    if total <= 0.0:
        return (float("nan"), float("nan"))
    yy, xx = np.meshgrid(y_um, x_um, indexing="ij")
    cx = float((abs_arr * xx).sum() / total)
    cy = float((abs_arr * yy).sum() / total)
    return (cx, cy)


def _draw_structure_overlays(ax, structure_specs: list[dict], *, slice_z_um: float | None) -> None:
    for spec in structure_specs:
        linestyle = "--"
        alpha = 0.9
        if slice_z_um is not None:
            z0 = None
            z1 = None
            if spec["type"] == "polygon_prism":
                z0 = float(spec["z_um"])
                z1 = z0 + float(spec["depth_um"])
            elif spec["type"] == "block":
                z0 = float(spec["position_um"][2])
                z1 = z0 + float(spec["size_um"][2])
            elif spec["type"] == "sphere":
                cz = float(spec["center_um"][2])
                r = float(spec["radius_um"])
                z0 = cz - r
                z1 = cz + r
            if z0 is not None and z1 is not None and z0 <= slice_z_um <= z1:
                linestyle = "-"

        if spec["type"] == "polygon_prism":
            pts = np.asarray(spec["vertices_um"], dtype=np.float64)
            pts = np.vstack([pts, pts[0]])
            ax.plot(pts[:, 0], pts[:, 1], color="black", linewidth=1.3, linestyle=linestyle, alpha=alpha)
        elif spec["type"] == "block":
            x0, y0, _ = spec["position_um"]
            sx, sy, _ = spec["size_um"]
            xs = [x0, x0 + sx, x0 + sx, x0, x0]
            ys = [y0, y0, y0 + sy, y0 + sy, y0]
            ax.plot(xs, ys, color="black", linewidth=1.3, linestyle=linestyle, alpha=alpha)
        elif spec["type"] == "sphere":
            cx, cy, _ = spec["center_um"]
            r = float(spec["radius_um"])
            theta = np.linspace(0.0, 2.0 * np.pi, 128)
            ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta), color="black", linewidth=1.3, linestyle=linestyle, alpha=alpha)


def _permittivity_z_coords_um(shape: tuple[int, int, int], cfg: dict) -> np.ndarray:
    dx_um = float(cfg["wavelength_um"]) / float(cfg["resolution_ppw"])
    return (np.arange(shape[0], dtype=np.float64) + 0.5) * dx_um


def _material_slice_index(
    structure_specs: list[dict],
    permittivity_shape: tuple[int, int, int],
    cfg: dict,
) -> int:
    z_coords_um = _permittivity_z_coords_um(permittivity_shape, cfg)
    if not structure_specs:
        return int(len(z_coords_um) // 2)
    z_centers = []
    for spec in structure_specs:
        if spec["type"] == "polygon_prism":
            z_centers.append(float(spec["z_um"]) + 0.5 * float(spec["depth_um"]))
        elif spec["type"] == "block":
            z_centers.append(float(spec["position_um"][2]) + 0.5 * float(spec["size_um"][2]))
        elif spec["type"] == "sphere":
            z_centers.append(float(spec["center_um"][2]))
    if not z_centers:
        return int(len(z_coords_um) // 2)
    target = float(np.median(np.asarray(z_centers, dtype=np.float64)))
    return int(np.argmin(np.abs(z_coords_um - target)))


def _plot_one(
    *,
    beamz_slice: np.ndarray,
    meep_slice: np.ndarray,
    diff_slice: np.ndarray,
    beamz_x_um: np.ndarray,
    beamz_y_um: np.ndarray,
    meep_x_um: np.ndarray,
    meep_y_um: np.ndarray,
    permittivity_slice: np.ndarray,
    background_eps: float,
    source_xy_um: tuple[float, float],
    structure_specs: list[dict],
    slice_z_um: float | None,
    title: str,
    output_path: Path,
) -> None:
    vmax_beamz = float(max(np.max(np.abs(beamz_slice)), 1e-12))
    vmax_meep = float(max(np.max(np.abs(meep_slice)), 1e-12))
    err_max = float(np.max(np.abs(diff_slice)))
    if err_max <= 0.0:
        err_max = 1e-12
    ref_scale = float(max(vmax_beamz, vmax_meep, 1e-12))
    diff_norm = diff_slice / ref_scale
    rel_max = float(np.max(np.abs(diff_norm)))
    if rel_max <= 0.0:
        rel_max = 1e-12

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    panels = [
        (
            beamz_slice,
            f"Beamz | max={vmax_beamz:.2e}",
            "RdBu_r",
            -vmax_beamz,
            vmax_beamz,
            _extent_from_center_coords(beamz_x_um, beamz_y_um),
            _physical_centroid_um(beamz_slice, beamz_x_um, beamz_y_um),
        ),
        (
            meep_slice,
            f"Meep | max={vmax_meep:.2e}",
            "RdBu_r",
            -vmax_meep,
            vmax_meep,
            _extent_from_center_coords(meep_x_um, meep_y_um),
            _physical_centroid_um(meep_slice, meep_x_um, meep_y_um),
        ),
        (
            diff_norm,
            f"Normalized Error\nmax|err|={err_max:.2e}, max rel={rel_max:.2e}",
            "coolwarm",
            -rel_max,
            rel_max,
            _extent_from_center_coords(beamz_x_um, beamz_y_um),
            (float("nan"), float("nan")),
        ),
    ]
    for ax, (arr, label, cmap, vmin, vmax, extent, centroid) in zip(axes, panels, strict=False):
        im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
        ax.set_title(label)
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        _draw_structure_overlays(ax, structure_specs, slice_z_um=slice_z_um)
        ax.scatter(
            [source_xy_um[0]],
            [source_xy_um[1]],
            marker="x",
            s=80,
            c="black",
            linewidths=1.5,
        )
        if np.isfinite(centroid[0]) and np.isfinite(centroid[1]):
            ax.scatter([centroid[0]], [centroid[1]], marker="+", s=90, c="lime", linewidths=1.5)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if label.startswith("Normalized Error"):
            cbar.set_label("Normalized error")

    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_material_comparison(
    *,
    beamz_slice: np.ndarray,
    meep_slice: np.ndarray,
    beamz_x_um: np.ndarray,
    beamz_y_um: np.ndarray,
    meep_x_um: np.ndarray,
    meep_y_um: np.ndarray,
    structure_specs: list[dict],
    slice_z_um: float,
    title: str,
    output_path: Path,
) -> None:
    diff_slice = beamz_slice - meep_slice
    vmax = float(max(np.max(beamz_slice), np.max(meep_slice), 1e-12))
    err_max = float(np.max(np.abs(diff_slice)))
    if err_max <= 0.0:
        err_max = 1e-12

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    panels = [
        (
            beamz_slice,
            f"Beamz eps | max={float(np.max(beamz_slice)):.2f}",
            "viridis",
            1.0,
            vmax,
            _extent_from_center_coords(beamz_x_um, beamz_y_um),
        ),
        (
            meep_slice,
            f"Meep eps | max={float(np.max(meep_slice)):.2f}",
            "viridis",
            1.0,
            vmax,
            _extent_from_center_coords(meep_x_um, meep_y_um),
        ),
        (
            diff_slice,
            f"Eps error | max|err|={err_max:.2e}",
            "coolwarm",
            -err_max,
            err_max,
            _extent_from_center_coords(beamz_x_um, beamz_y_um),
        ),
    ]
    for ax, (arr, label, cmap, vmin, vmax_i, extent) in zip(axes, panels, strict=False):
        im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax_i, extent=extent)
        ax.set_title(label)
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        _draw_structure_overlays(ax, structure_specs, slice_z_um=slice_z_um)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if label.startswith("Eps error"):
            cbar.set_label("Permittivity error")

    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_source_diagnostic(
    *,
    source_mask: np.ndarray,
    source_xy: tuple[float, float],
    source_z: float,
    output_path: Path,
) -> None:
    z_index = int(round(source_z))
    z_index = max(0, min(source_mask.shape[0] - 1, z_index))
    xy = source_mask[z_index]
    xz = source_mask[:, int(round(source_xy[1]))]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), constrained_layout=True)
    panels = [
        (xy, "Canonical source mask (xy @ source z)"),
        (xz, "Canonical source mask (xz @ source y)"),
    ]
    for ax, (arr, label) in zip(axes, panels, strict=False):
        im = ax.imshow(arr, origin="lower", cmap="magma")
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    axes[0].scatter([source_xy[0]], [source_xy[1]], marker="x", s=80, c="cyan", linewidths=1.5)
    axes[1].scatter([source_xy[0]], [source_z], marker="x", s=80, c="cyan", linewidths=1.5)
    fig.suptitle("Configured source location in canonical Beamz-style zyx grid")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot Beamz/Meep field comparisons for saved random-structure benchmarks."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )
    parser.add_argument(
        "--structure-index",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--component",
        choices=("Ex", "Ey", "Ez"),
        default="Ez",
    )
    parser.add_argument(
        "--field-grid",
        choices=(FIELD_GRID_CENTERED, FIELD_GRID_RAW_YEE, FIELD_GRID_MEEP_SAMPLED),
        default=FIELD_GRID_CENTERED,
    )
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    structure_dir = results_dir / f"structure_{args.structure_index:03d}"
    output_dir = {
        FIELD_GRID_CENTERED: structure_dir / "plots",
        FIELD_GRID_RAW_YEE: structure_dir / "plots_raw_yee",
        FIELD_GRID_MEEP_SAMPLED: structure_dir / "plots_meep_sampled",
    }[args.field_grid]

    stem_suffix = {
        FIELD_GRID_CENTERED: "",
        FIELD_GRID_RAW_YEE: "_raw_yee",
        FIELD_GRID_MEEP_SAMPLED: "_meep_sampled",
    }[args.field_grid]

    beamz_meta = _load_json(structure_dir / f"beamz_fields{stem_suffix}.json")
    meep_meta = _load_json(structure_dir / f"meep_fields{stem_suffix}.json")
    structure_payload = _load_json(structure_dir / "structure.json")
    structure_specs = structure_payload.get("structure_specs", [])
    beamz = np.load(structure_dir / f"beamz_fields{stem_suffix}.npz")
    meep = np.load(structure_dir / f"meep_fields{stem_suffix}.npz")

    beamz_fields = np.asarray(beamz[args.component], dtype=np.float32)
    meep_fields = np.asarray(meep[args.component], dtype=np.float32)
    beamz_permittivity = np.asarray(beamz["permittivity"], dtype=np.float32)
    meep_permittivity = np.asarray(meep["permittivity"], dtype=np.float32)
    times = np.asarray(beamz["snapshot_times_s"], dtype=np.float64)
    steps = np.asarray(beamz["snapshot_steps"], dtype=np.int32)
    source_mask = _canonical_source_mask(beamz_meta["config"], tuple(beamz["permittivity"].shape))
    source_x_idx, source_y_idx, source_z_idx = _source_index_coords(beamz_meta["config"])
    source_xy_um = (
        float(beamz_meta["config"]["source_x_um"]),
        float(beamz_meta["config"]["source_y_um"]),
    )
    beamz_coords = beamz_meta["component_coordinates_um"][args.component]
    meep_coords = meep_meta["component_coordinates_um"][args.component]
    background_eps = float(beamz_meta["config"]["background_index"]) ** 2

    if beamz_fields.shape != meep_fields.shape:
        raise ValueError(
            f"Field shape mismatch: beamz {beamz_fields.shape} vs meep {meep_fields.shape}"
        )

    generated = []
    source_plot = output_dir / "source_mask_diagnostic.png"
    _plot_source_diagnostic(
        source_mask=source_mask,
        source_xy=(source_x_idx, source_y_idx),
        source_z=source_z_idx,
        output_path=source_plot,
    )
    generated.append(str(source_plot))
    material_slice_index = _material_slice_index(
        structure_specs,
        tuple(beamz_permittivity.shape),
        beamz_meta["config"],
    )
    material_z_coords_um = _permittivity_z_coords_um(tuple(beamz_permittivity.shape), beamz_meta["config"])
    material_slice_z_um = float(material_z_coords_um[material_slice_index])
    material_x_um = (np.arange(beamz_permittivity.shape[2], dtype=np.float64) + 0.5) * (
        float(beamz_meta["config"]["wavelength_um"]) / float(beamz_meta["config"]["resolution_ppw"])
    )
    material_y_um = (np.arange(beamz_permittivity.shape[1], dtype=np.float64) + 0.5) * (
        float(beamz_meta["config"]["wavelength_um"]) / float(beamz_meta["config"]["resolution_ppw"])
    )
    material_output = output_dir / "permittivity_slice_comparison.png"
    _plot_material_comparison(
        beamz_slice=_slice_for_axis(beamz_permittivity, axis=0, index=material_slice_index),
        meep_slice=_slice_for_axis(meep_permittivity, axis=0, index=material_slice_index),
        beamz_x_um=material_x_um,
        beamz_y_um=material_y_um,
        meep_x_um=material_x_um,
        meep_y_um=material_y_um,
        structure_specs=structure_specs,
        slice_z_um=material_slice_z_um,
        title=(
            f"Structure {args.structure_index:03d} | Permittivity | "
            f"z-slice {material_slice_index} ({material_slice_z_um:.3f} um)"
        ),
        output_path=material_output,
    )
    generated.append(str(material_output))
    axis = 0
    z_coords_um = np.asarray(beamz_coords["z"], dtype=np.float64)
    fixed_slice_index = _structure_fixed_slice_index(structure_specs, z_coords_um)
    if fixed_slice_index is None:
        fixed_slice_index = _best_fixed_slice_index(beamz_fields, meep_fields, axis=axis)
    for i in range(beamz_fields.shape[0]):
        slice_index = fixed_slice_index
        beamz_slice = _slice_for_axis(beamz_fields[i], axis, slice_index)
        meep_slice = _slice_for_axis(meep_fields[i], axis, slice_index)
        diff_slice = beamz_slice - meep_slice
        permittivity_slice = _slice_for_axis(beamz_permittivity, axis, slice_index)
        beamz_x_um, beamz_y_um = _slice_coords_for_axis(beamz_coords, axis)
        meep_x_um, meep_y_um = _slice_coords_for_axis(meep_coords, axis)
        slice_z_um = float(np.asarray(beamz_coords["z"], dtype=np.float64)[slice_index])
        output_path = output_dir / f"{args.component.lower()}_snapshot_{i:02d}.png"
        _plot_one(
            beamz_slice=beamz_slice,
            meep_slice=meep_slice,
            diff_slice=diff_slice,
            beamz_x_um=beamz_x_um,
            beamz_y_um=beamz_y_um,
            meep_x_um=meep_x_um,
            meep_y_um=meep_y_um,
            permittivity_slice=permittivity_slice,
            background_eps=background_eps,
            source_xy_um=source_xy_um,
            structure_specs=structure_specs,
            slice_z_um=slice_z_um,
            title=(
                f"Structure {args.structure_index:03d} | {args.component} | "
                f"step {int(steps[i])} | t = {times[i] * 1e15:.2f} fs | z-slice {slice_index}"
            ),
            output_path=output_path,
        )
        generated.append(str(output_path))

    print(json.dumps({"generated_plots": generated}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
