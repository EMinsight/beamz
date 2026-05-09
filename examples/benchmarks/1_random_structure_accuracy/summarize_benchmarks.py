from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_npz(path: Path) -> np.lib.npyio.NpzFile:
    return np.load(path)


def _component_paths(structure_dir: Path, field_grid: str) -> tuple[Path, Path, dict]:
    suffix = {
        "centered": "_fields_centered",
        "raw-yee": "_fields_raw_yee",
        "meep-sampled": "_fields_meep_sampled",
    }[field_grid]
    beamz_json = structure_dir / f"beamz{suffix}.json"
    meep_json = structure_dir / f"meep{suffix}.json"
    beamz_npz = structure_dir / f"beamz{suffix}.npz"
    meep_npz = structure_dir / f"meep{suffix}.npz"
    meta = _load_json(beamz_json)
    return beamz_npz, meep_npz, meta


def _metrics(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    diff = a - b
    denom = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1e-30)
    max_rel = float(np.max(np.abs(diff)) / denom)
    l2_rel = float(np.linalg.norm(diff.ravel()) / max(np.linalg.norm(b.ravel()), 1e-30))
    corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    return max_rel, l2_rel, corr


def summarize_case(results_dir: Path, field_grid: str) -> tuple[list[dict], dict]:
    structure_dir = results_dir / "structure_000"
    beamz_npz, meep_npz, meta = _component_paths(structure_dir, field_grid)
    beamz = _load_npz(beamz_npz)
    meep = _load_npz(meep_npz)
    rows: list[dict] = []
    for component in ("Ex", "Ey", "Ez"):
        for snap_idx, step in enumerate(meta["snapshots"]):
            max_rel, l2_rel, corr = _metrics(
                np.asarray(beamz[component][snap_idx], dtype=np.float64),
                np.asarray(meep[component][snap_idx], dtype=np.float64),
            )
            rows.append(
                {
                    "case": results_dir.name,
                    "component": component,
                    "snapshot_index": snap_idx,
                    "step": int(step["step"]),
                    "time_fs": float(step["time_s"]) * 1e15,
                    "max_rel_error": max_rel,
                    "l2_rel_error": l2_rel,
                    "correlation": corr,
                }
            )
    return rows, meta


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Beamz vs Meep Benchmark Summary",
        "",
        "| Case | Component | Step | Time (fs) | Max rel err | L2 rel err | Corr |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {row['component']} | {row['step']} | "
            f"{row['time_fs']:.2f} | {row['max_rel_error']:.4e} | "
            f"{row['l2_rel_error']:.4e} | {row['correlation']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def plot_summary(path: Path, rows: list[dict]) -> None:
    cases = []
    for row in rows:
        if row["case"] not in cases:
            cases.append(row["case"])
    components = ["Ex", "Ey", "Ez"]
    fig, axes = plt.subplots(1, len(cases), figsize=(6 * len(cases), 4), sharey=True)
    if len(cases) == 1:
        axes = [axes]

    for ax, case in zip(axes, cases, strict=False):
        case_rows = [r for r in rows if r["case"] == case]
        steps = sorted({r["step"] for r in case_rows})
        x = np.arange(len(steps))
        width = 0.24
        for idx, component in enumerate(components):
            vals = [
                next(
                    r["max_rel_error"]
                    for r in case_rows
                    if r["component"] == component and r["step"] == step
                )
                for step in steps
            ]
            ax.bar(x + (idx - 1) * width, vals, width=width, label=component)
        ax.set_title(case)
        ax.set_xticks(x, [str(step) for step in steps])
        ax.set_xlabel("Step")
        ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Max relative error")
    axes[-1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Beamz/Meep benchmark cases.")
    parser.add_argument("--field-grid", default="meep-sampled", choices=("centered", "raw-yee", "meep-sampled"))
    parser.add_argument(
        "--cases",
        nargs="+",
        type=Path,
        default=[
            Path("examples/benchmarks/1_random_structure_accuracy/results_uniform_meep_sampled_v12"),
            Path("examples/benchmarks/1_random_structure_accuracy/results_polygon_meep_sampled_shared_raster_v3"),
        ],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/benchmarks/1_random_structure_accuracy/summary_outputs"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for case in args.cases:
        rows, _meta = summarize_case(case, args.field_grid)
        all_rows.extend(rows)

    csv_path = args.output_dir / "benchmark_summary.csv"
    md_path = args.output_dir / "benchmark_summary.md"
    png_path = args.output_dir / "benchmark_summary_max_rel.png"
    write_csv(csv_path, all_rows)
    write_markdown(md_path, all_rows)
    plot_summary(png_path, all_rows)

    print(
        json.dumps(
            {
                "csv": str(csv_path.resolve()),
                "markdown": str(md_path.resolve()),
                "plot": str(png_path.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
