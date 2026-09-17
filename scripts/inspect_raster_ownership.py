"""Inspect selected native supports around the pinned crossing's slab and step.

Read scene.json/grid.npz saved by benchmark_raster_ownership.py. Write spatial
JSONL, GEOS cross-checks, explicit sample counts, and cell/Ex/Ey/Ez figures.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]


def _selection(axes):
    centers = [(a[:-1] + a[1:]) / 2 for a in axes]
    supports, slices = [], []
    for component, edges in (
        ("cell", (False, False, False)),
        ("ex", (False, True, True)),
        ("ey", (True, False, True)),
        ("ez", (True, True, False)),
    ):
        locations = [
            a if e else c for a, c, e in zip(axes, centers, edges, strict=True)
        ]
        xs = np.flatnonzero((locations[0] >= 3e-6) & (locations[0] <= 7e-6))
        ys = np.flatnonzero((locations[1] >= 3.5e-6) & (locations[1] <= 7.5e-6))
        zs = np.flatnonzero((locations[2] >= 1.95e-6) & (locations[2] <= 2.3e-6))
        iz = int(np.argmin(abs(locations[2] - 2.15e-6)))
        iy = int(np.argmin(abs(locations[1] - 6e-6)))
        indices = {(int(x), int(y), iz) for y in ys for x in xs}
        indices.update((int(x), iy, int(z)) for z in zs for x in xs)
        supports.extend(
            {"component": component, "index": list(i)} for i in sorted(indices)
        )
        slices.append(
            {
                "component": component,
                "xy_z_index": iz,
                "xy_z_m": float(locations[2][iz]),
                "xz_y_index": iy,
                "xz_y_m": float(locations[1][iy]),
                "supports": len(indices),
            }
        )
    return supports, slices


def _geos_layers(scene):
    objects = []
    heights = set()
    for obj in scene["objects"]:
        geometry = obj["geometry"]
        if geometry["type"] == "box":
            lo, hi = geometry["bounds"]["min"], geometry["bounds"]["max"]
            polygon, zmin, zmax = box(lo[0], lo[1], hi[0], hi[1]), lo[2], hi[2]
        else:
            assert geometry["type"] == "extruded_polygon"
            polygon = Polygon(
                geometry["polygon"]["exterior"], geometry["polygon"]["holes"]
            )
            zmin, zmax = geometry["z_min"], geometry["z_max"]
        objects.append(
            (obj["priority"], obj["id"], obj["material_id"], polygon, zmin, zmax)
        )
        heights.update((zmin, zmax))
    heights = sorted(heights)
    layers = []
    for low, high in zip(heights[:-1], heights[1:], strict=True):
        covered = Polygon()
        regions = [[] for _ in scene["materials"]]
        for _, _, material, polygon, zmin, zmax in sorted(
            objects, key=lambda o: o[:2], reverse=True
        ):
            if zmin < high and zmax > low:
                regions[material].append(polygon.difference(covered))
                covered = covered.union(polygon)
        layers.append((low, high, [unary_union(r) for r in regions]))
    return layers


def _summarize(records, scene):
    layers = _geos_layers(scene)
    errors = []
    counts = {c: Counter() for c in ("cell", "ex", "ey", "ez")}
    for record in records:
        lo, hi = record["bounds"]["min"], record["bounds"]["max"]
        area = box(lo[0], lo[1], hi[0], hi[1])
        volumes = np.zeros(len(scene["materials"]))
        for bottom, top, regions in layers:
            depth = max(0, min(hi[2], top) - max(lo[2], bottom))
            if depth:
                volumes += depth * np.array(
                    [r.intersection(area).area for r in regions]
                )
        fractions = volumes / np.prod(np.array(hi) - lo)
        fractions[scene["background_material"]] += 1 - fractions.sum()
        errors.append(float(np.max(abs(fractions - record["fractions"]))))
        count = counts[record["component"]]
        count["selected_supports"] += 1
        mixed = sum(f > 1e-12 for f in record["fractions"]) > 1
        count["mixed_supports"] += mixed
        count["smoothed_supports"] += record["smoothed"]
        if record["fallback"]:
            count[record["fallback"]] += 1
    return {
        "counts": counts,
        "max_fraction_error_vs_geos": max(errors),
        "p95_fraction_error_vs_geos": float(np.percentile(errors, 95)),
        "denominator": "unique selected supports, separately for cell/Ex/Ey/Ez; not all grid cells",
    }


def _plot(records, slices, output, plane):
    fig, axes = plt.subplots(4, 3, figsize=(12, 13), constrained_layout=True)
    reasons = [
        None,
        "MultipleOrientations",
        "MultipleObjects",
        "UnresolvedGeometry",
        "MissingSurfaceEvidence",
    ]
    cmap = ListedColormap(
        ["#dddddd", "#0077bb", "#ee7733", "#cc3311", "#aa3377", "#009988"]
    )
    for row, selection in enumerate(slices):
        transverse, fixed = (1, 2) if plane == "xy" else (2, 1)
        target = selection["xy_z_index" if plane == "xy" else "xz_y_index"]
        subset = [
            r
            for r in records
            if r["component"] == selection["component"] and r["index"][fixed] == target
        ]
        xs = sorted({r["index"][0] for r in subset})
        ys = sorted({r["index"][transverse] for r in subset})
        values = [np.full((len(ys), len(xs)), np.nan) for _ in range(3)]
        for r in subset:
            y, x = ys.index(r["index"][transverse]), xs.index(r["index"][0])
            values[0][y, x] = r["fractions"][1]
            if r["smoothed"]:
                values[1][y, x] = abs(r["normal"][2])
            values[2][y, x] = (
                1
                if r["smoothed"]
                else (0 if r["fallback"] is None else reasons.index(r["fallback"]) + 1)
            )

        def support_edges(indices, axis, subset=subset):
            edges = [
                min(r["bounds"]["min"][axis] for r in subset if r["index"][axis] == i)
                for i in indices
            ]
            edges.append(max(r["bounds"]["max"][axis] for r in subset))
            return np.array(edges) * 1e6

        x_edges = support_edges(xs, 0)
        y_edges = support_edges(ys, transverse)
        for col, (value, title) in enumerate(
            zip(
                values,
                ["Silicon fraction", "Smoothed normal |nz|", "Smoothing / fallback"],
                strict=True,
            )
        ):
            ax = axes[row, col]
            kwargs = (
                {"cmap": cmap, "norm": BoundaryNorm(np.arange(-0.5, 6.5), 6)}
                if col == 2
                else {"vmin": 0, "vmax": 1}
            )
            img = ax.pcolormesh(x_edges, y_edges, value, shading="flat", **kwargs)
            if col == 1:
                arrows = [
                    r
                    for r in subset
                    if r["smoothed"]
                    and r["index"][0] % 3 == 0
                    and r["index"][transverse] % 3 == 0
                ]
                if arrows:
                    centers = (
                        np.array(
                            [
                                (np.array(r["bounds"]["min"]) + r["bounds"]["max"]) / 2
                                for r in arrows
                            ]
                        )
                        * 1e6
                    )
                    normals = np.array([r["normal"] for r in arrows])
                    ax.quiver(
                        centers[:, 0],
                        centers[:, transverse],
                        normals[:, 0],
                        normals[:, transverse],
                        color="black",
                        pivot="mid",
                        scale=30,
                    )
            ax.set_title(f"{selection['component'].upper()} · {title}")
            ax.set_xlabel("x (µm)")
            ax.set_ylabel(f"{'y' if plane == 'xy' else 'z'} (µm)")
            cb = fig.colorbar(img, ax=ax, fraction=0.046)
            if col == 2:
                cb.set_ticks(
                    [0, 1, 2, 3, 4, 5],
                    labels=[
                        "uniform",
                        "smoothed",
                        "corner",
                        "objects",
                        "unresolved",
                        "missing",
                    ],
                )
    fig.suptitle(
        f"10 PPW · resolved ownership · {plane.upper()} supports near slab step"
    )
    fig.savefig(output / f"spatial_{plane}.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scene = json.loads((args.run / "scene.json").read_text())
    with np.load(args.run / "grid.npz") as grid:
        axes = [grid[a] for a in "xyz"]
    supports, slices = _selection(axes)
    request = {
        "scene": scene,
        "grid": dict(
            zip(
                ["x_edges", "y_edges", "z_edges"],
                [a.tolist() for a in axes],
                strict=True,
            )
        ),
        "supports": supports,
    }
    input_path, output_path = (
        args.output / "inspection.json",
        args.output / "spatial.jsonl",
    )
    input_path.write_text(json.dumps(request))
    env = {
        **os.environ,
        "BEAMZ_RASTER_INSPECTION_INPUT": str(input_path.resolve()),
        "BEAMZ_RASTER_INSPECTION_OUTPUT": str(output_path.resolve()),
    }
    subprocess.run(
        [
            "cargo",
            "test",
            "--release",
            "-p",
            "fdtd-raster-core",
            "write_spatial_ownership_diagnostics",
            "--",
            "--ignored",
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    (args.output / "spatial_selection.json").write_text(
        json.dumps(slices, indent=2) + "\n"
    )
    (args.output / "spatial_summary.json").write_text(
        json.dumps(_summarize(records, scene), indent=2) + "\n"
    )
    for plane in ("xy", "xz"):
        _plot(records, slices, args.output, plane)


if __name__ == "__main__":
    main()
