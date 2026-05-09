from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _centroid_xyz_um(arr_zyx: np.ndarray, coords_zyx_um: dict[str, list[float]]) -> tuple[float, float, float]:
    arr = np.abs(np.asarray(arr_zyx, dtype=np.float64))
    total = float(arr.sum())
    if total <= 0.0:
        return (float("nan"), float("nan"), float("nan"))
    z_um = np.asarray(coords_zyx_um["z"], dtype=np.float64)
    y_um = np.asarray(coords_zyx_um["y"], dtype=np.float64)
    x_um = np.asarray(coords_zyx_um["x"], dtype=np.float64)
    zz, yy, xx = np.meshgrid(z_um, y_um, x_um, indexing="ij")
    cz = float((arr * zz).sum() / total)
    cy = float((arr * yy).sum() / total)
    cx = float((arr * xx).sum() / total)
    return (cx, cy, cz)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Beamz/Meep field centroids in physical coordinates.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--structure-index", type=int, default=0)
    parser.add_argument(
        "--field-grid",
        choices=("centered", "raw-yee", "meep-sampled"),
        default="raw-yee",
    )
    args = parser.parse_args()

    structure_dir = args.results_dir.resolve() / f"structure_{args.structure_index:03d}"
    stem_suffix = {
        "centered": "",
        "raw-yee": "_raw_yee",
        "meep-sampled": "_meep_sampled",
    }[args.field_grid]

    beamz_meta = _load_json(structure_dir / f"beamz_fields{stem_suffix}.json")
    meep_meta = _load_json(structure_dir / f"meep_fields{stem_suffix}.json")
    beamz = np.load(structure_dir / f"beamz_fields{stem_suffix}.npz")
    meep = np.load(structure_dir / f"meep_fields{stem_suffix}.npz")

    source_xyz_um = {
        "x_um": float(beamz_meta["config"]["source_x_um"]),
        "y_um": float(beamz_meta["config"]["source_y_um"]),
        "z_um": float(beamz_meta["config"]["source_z_um"]),
    }

    report: dict[str, object] = {
        "structure_index": args.structure_index,
        "field_grid": args.field_grid,
        "source_center_um": source_xyz_um,
        "components": {},
    }

    for component in ("Ex", "Ey", "Ez"):
        beamz_coords = beamz_meta["component_coordinates_um"][component]
        meep_coords = meep_meta["component_coordinates_um"][component]
        component_report = []
        for i in range(int(beamz[component].shape[0])):
            beamz_centroid = _centroid_xyz_um(beamz[component][i], beamz_coords)
            meep_centroid = _centroid_xyz_um(meep[component][i], meep_coords)
            component_report.append(
                {
                    "snapshot_index": i,
                    "beamz_centroid_um": {
                        "x_um": beamz_centroid[0],
                        "y_um": beamz_centroid[1],
                        "z_um": beamz_centroid[2],
                    },
                    "meep_centroid_um": {
                        "x_um": meep_centroid[0],
                        "y_um": meep_centroid[1],
                        "z_um": meep_centroid[2],
                    },
                    "beamz_minus_meep_um": {
                        "x_um": beamz_centroid[0] - meep_centroid[0],
                        "y_um": beamz_centroid[1] - meep_centroid[1],
                        "z_um": beamz_centroid[2] - meep_centroid[2],
                    },
                    "beamz_minus_source_um": {
                        "x_um": beamz_centroid[0] - source_xyz_um["x_um"],
                        "y_um": beamz_centroid[1] - source_xyz_um["y_um"],
                        "z_um": beamz_centroid[2] - source_xyz_um["z_um"],
                    },
                    "meep_minus_source_um": {
                        "x_um": meep_centroid[0] - source_xyz_um["x_um"],
                        "y_um": meep_centroid[1] - source_xyz_um["y_um"],
                        "z_um": meep_centroid[2] - source_xyz_um["z_um"],
                    },
                }
            )
        report["components"][component] = component_report

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
