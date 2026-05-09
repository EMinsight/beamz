from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _outline_xy(specs: list[dict]) -> np.ndarray | None:
    for spec in specs:
        if spec.get("type") == "polygon":
            pts = np.asarray(spec["vertices_um"], dtype=np.float64)
            if pts.shape[0] >= 3:
                return np.vstack([pts, pts[:1]])
    return None


def _plot_material(
    output_dir: Path,
    beamz: np.lib.npyio.NpzFile,
    meep: np.lib.npyio.NpzFile,
    structure: dict,
    beamz_meta: dict,
) -> None:
    outline = _outline_xy(structure.get("structure_specs", []))
    beamz_eps = np.asarray(beamz["permittivity"], dtype=np.float32)
    meep_eps = np.asarray(meep["permittivity"], dtype=np.float32)
    diff = beamz_eps - meep_eps
    vmax = max(float(np.max(np.abs(diff))), 1e-9)
    dx_um = float(beamz_meta["config"]["wavelength_um"]) / float(
        beamz_meta["config"]["resolution_ppw"]
    )
    extent = [0.0, beamz_eps.shape[1] * dx_um, 0.0, beamz_eps.shape[0] * dx_um]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, data, title, cmap, kwargs in (
        (axes[0], beamz_eps, "Beamz permittivity", "viridis", {}),
        (axes[1], meep_eps, "Meep permittivity", "viridis", {}),
        (axes[2], diff, "Beamz - Meep permittivity", "coolwarm", {"vmin": -vmax, "vmax": vmax}),
    ):
        im = ax.imshow(data, origin="lower", extent=extent, cmap=cmap, **kwargs)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if outline is not None:
            ax.plot(outline[:, 0], outline[:, 1], color="black", linewidth=1.4)
        ax.set_title(title)
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
    fig.tight_layout()
    fig.savefig(output_dir / "permittivity_slice_comparison.png", dpi=180)
    plt.close(fig)


def _plot_component(output_dir: Path, component: str, beamz: np.lib.npyio.NpzFile, meep: np.lib.npyio.NpzFile, beamz_meta: dict, structure: dict) -> None:
    outline = _outline_xy(structure.get("structure_specs", []))
    source_x = float(beamz_meta["config"]["source_x_um"])
    source_y = float(beamz_meta["config"]["source_y_um"])
    dx_um = float(beamz_meta["config"]["wavelength_um"]) / float(
        beamz_meta["config"]["resolution_ppw"]
    )
    arr0 = np.asarray(beamz[component][0])
    extent = [0.0, arr0.shape[1] * dx_um, 0.0, arr0.shape[0] * dx_um]

    for snap_idx, snap in enumerate(beamz_meta["snapshots"]):
        b = np.asarray(beamz[component][snap_idx], dtype=np.float32)
        m = np.asarray(meep[component][snap_idx], dtype=np.float32)
        diff = b - m
        denom = max(float(np.max(np.abs(b))), float(np.max(np.abs(m))), 1e-30)
        diff_norm = diff / denom
        max_abs_err = float(np.max(np.abs(diff)))
        max_rel_err = float(np.max(np.abs(diff_norm)))
        vmax = max(float(np.max(np.abs(diff_norm))), 1e-9)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, data, title, cmap, kwargs in (
            (axes[0], b, f"Beamz {component}", "RdBu_r", {}),
            (axes[1], m, f"Meep {component}", "RdBu_r", {}),
            (
                axes[2],
                diff_norm,
                f"Normalized error\nmax|err|={max_abs_err:.3e}, max rel={max_rel_err:.3f}",
                "coolwarm",
                {"vmin": -vmax, "vmax": vmax},
            ),
        ):
            im = ax.imshow(data, origin="lower", extent=extent, cmap=cmap, **kwargs)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if outline is not None:
                ax.plot(outline[:, 0], outline[:, 1], color="black", linewidth=1.4)
            ax.plot(source_x, source_y, marker="x", color="black", markersize=8, mew=1.3)
            ax.set_xlabel("x (um)")
            ax.set_ylabel("y (um)")
            ax.set_title(title)
        fig.suptitle(f"{component} at step {snap['step']} ({snap['time_s'] * 1e15:.2f} fs)")
        fig.tight_layout()
        fig.savefig(output_dir / f"{component.lower()}_snapshot_{snap_idx:02d}.png", dpi=180)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot 2D Beamz/Meep field comparisons.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--component", default=None)
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    structure_dir = results_dir / "structure_000"
    out_dir = structure_dir / "plots_meep_sampled"
    out_dir.mkdir(parents=True, exist_ok=True)

    beamz = np.load(structure_dir / "beamz_fields_meep_sampled.npz")
    meep = np.load(structure_dir / "meep_fields_meep_sampled.npz")
    beamz_meta = _load_json(structure_dir / "beamz_fields_meep_sampled.json")
    structure = _load_json(structure_dir / "structure.json")

    _plot_material(out_dir, beamz, meep, structure, beamz_meta)
    available_components = tuple(
        name
        for name in ("Ez", "Hx", "Hy", "Ex", "Ey", "Hz")
        if name in beamz.files and name in meep.files
    )
    if args.component is not None and args.component not in available_components:
        raise ValueError(
            f"Component {args.component!r} not available. Choices: {available_components}"
        )
    components = (args.component,) if args.component else available_components
    for component in components:
        _plot_component(out_dir, component, beamz, meep, beamz_meta, structure)

    print(
        json.dumps(
            {
                "generated_plots": [
                    str(path.resolve()) for path in sorted(out_dir.glob("*.png"))
                ]
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
