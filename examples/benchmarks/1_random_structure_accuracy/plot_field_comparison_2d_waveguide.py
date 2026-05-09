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
        if spec.get("type") == "block":
            x0, y0 = spec["position_um"]
            sx, sy = spec["size_um"]
            return np.asarray(
                [
                    [x0, y0],
                    [x0 + sx, y0],
                    [x0 + sx, y0 + sy],
                    [x0, y0 + sy],
                    [x0, y0],
                ],
                dtype=np.float64,
            )
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


def _crop_profile(
    y_um: np.ndarray,
    beamz_vals: np.ndarray,
    meep_vals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y_um, dtype=np.float64).reshape(-1)
    b = np.asarray(beamz_vals).reshape(-1)
    m = np.asarray(meep_vals).reshape(-1)
    n = min(y.size, b.size, m.size)
    return y[:n], b[:n], m[:n]


def _plot_dft_final(output_dir: Path, beamz: np.lib.npyio.NpzFile, meep: np.lib.npyio.NpzFile) -> None:
    y_um, b, m = _crop_profile(
        np.asarray(beamz["dft_monitor_y_um"], dtype=np.float64),
        np.asarray(beamz["dft_monitor_Ez"], dtype=np.complex128),
        np.asarray(meep["dft_monitor_Ez"], dtype=np.complex128),
    )
    mc = np.conj(m)
    bmag = np.abs(b)
    mmag = np.abs(m)
    ref = max(float(np.max(bmag)), float(np.max(mmag)), 1e-30)
    err_raw = np.abs(b - m) / ref
    err_conj = np.abs(b - mc) / ref
    phase_floor = 5e-2 * ref
    phase_mask = (bmag > phase_floor) & (mmag > phase_floor)
    phase_err_raw = np.full_like(y_um, np.nan, dtype=np.float64)
    phase_err_conj = np.full_like(y_um, np.nan, dtype=np.float64)
    raw_phase_rms = float("nan")
    conj_phase_rms = float("nan")
    if np.any(phase_mask):
        raw_delta = np.angle(b[phase_mask] * np.conj(m[phase_mask]))
        conj_delta = np.angle(b[phase_mask] * m[phase_mask])
        raw_offset = np.angle(np.mean(np.exp(1j * raw_delta)))
        conj_offset = np.angle(np.mean(np.exp(1j * conj_delta)))
        raw_resid = np.angle(np.exp(1j * (raw_delta - raw_offset)))
        conj_resid = np.angle(np.exp(1j * (conj_delta - conj_offset)))
        phase_err_raw[phase_mask] = raw_resid
        phase_err_conj[phase_mask] = conj_resid
        raw_phase_rms = float(np.sqrt(np.mean(raw_resid**2)))
        conj_phase_rms = float(np.sqrt(np.mean(conj_resid**2)))

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4), constrained_layout=True)
    axes[0].plot(y_um, bmag, label="Beamz", linewidth=1.8)
    axes[0].plot(y_um, mmag, label="Meep", linewidth=1.8, linestyle="--")
    axes[0].set_title("|DFT Ez|")
    axes[0].set_xlabel("monitor y (um)")
    axes[0].set_ylabel("magnitude")
    axes[0].legend()

    axes[1].axhline(0.0, color="0.7", linewidth=1.0)
    axes[1].plot(y_um, phase_err_raw, label="Beamz vs Meep", linewidth=1.8)
    axes[1].plot(
        y_um,
        phase_err_conj,
        label="Beamz vs conj(Meep)",
        linewidth=1.6,
        linestyle=":",
    )
    axes[1].set_title(
        "Phase residual after best global offset\n"
        f"raw rms={raw_phase_rms:.3f} rad, conj rms={conj_phase_rms:.3f} rad"
    )
    axes[1].set_xlabel("monitor y (um)")
    axes[1].set_ylabel("phase residual (rad)")
    axes[1].legend()

    axes[2].plot(y_um, err_raw, color="black", linewidth=1.8, label="Beamz - Meep")
    axes[2].plot(
        y_um,
        err_conj,
        color="tab:green",
        linewidth=1.8,
        linestyle="--",
        label="Beamz - conj(Meep)",
    )
    axes[2].set_title(
        "Normalized complex error\n"
        f"raw max={float(np.max(err_raw)):.3e}, conj max={float(np.max(err_conj)):.3e}"
    )
    axes[2].set_xlabel("monitor y (um)")
    axes[2].set_ylabel("|Beamz-Meep| / max(|.|)")
    axes[2].legend()

    fig.savefig(output_dir / "dft_monitor_ez_final.png", dpi=180)
    plt.close(fig)


def _plot_dft_running(output_dir: Path, beamz: np.lib.npyio.NpzFile, meep: np.lib.npyio.NpzFile) -> None:
    y_um, _, _ = _crop_profile(
        np.asarray(beamz["dft_monitor_y_um"], dtype=np.float64),
        np.asarray(beamz["dft_monitor_Ez"][...], dtype=np.complex128),
        np.asarray(meep["dft_monitor_Ez"][...], dtype=np.complex128),
    )
    t_beamz = np.asarray(beamz["dft_monitor_times_s"], dtype=np.float64)
    t_meep = np.asarray(meep["dft_monitor_times_s"], dtype=np.float64)
    b = np.asarray(beamz["dft_monitor_Ez_running"], dtype=np.complex128)
    m = np.asarray(meep["dft_monitor_Ez_running"], dtype=np.complex128)
    n_steps = min(b.shape[0], m.shape[0], t_beamz.size, t_meep.size)
    n_pts = min(b.shape[1], m.shape[1], y_um.size)
    b = b[:n_steps, :n_pts]
    m = m[:n_steps, :n_pts]
    y_um = y_um[:n_pts]
    t_fs = t_beamz[:n_steps] * 1e15
    sample_idx = np.arange(n_steps, dtype=np.float64)

    bmag = np.abs(b)
    mmag = np.abs(m)
    ref = max(float(np.max(bmag)), float(np.max(mmag)), 1e-30)
    err = np.abs(b - m) / ref
    extent = [float(y_um[0]), float(y_um[-1]), -0.5, float(n_steps) - 0.5]
    bmax = max(float(np.max(bmag)), 1e-30)
    mmax = max(float(np.max(mmag)), 1e-30)
    shared_vmax = max(bmax, mmax)
    err_vmax = max(float(np.max(err)), 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    for ax, data, title, cmap, vmin, vmax_i in (
        (axes[0], bmag, f"Beamz running |DFT Ez|\nmax={bmax:.3e}", "magma", 0.0, bmax),
        (axes[1], mmag, f"Meep running |DFT Ez|\nmax={mmax:.3e}", "magma", 0.0, mmax),
        (
            axes[2],
            err,
            f"Normalized running error\nmax={float(np.max(err)):.3e}",
            "viridis",
            0.0,
            err_vmax,
        ),
    ):
        im = ax.imshow(
            data,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax_i,
        )
        ax.set_title(title)
        ax.set_xlabel("monitor y (um)")
        ax.set_ylabel("time (fs)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(output_dir / "dft_monitor_ez_running.png", dpi=180)
    plt.close(fig)


def _plot_dft_running_actual(
    output_dir: Path, beamz: np.lib.npyio.NpzFile, meep: np.lib.npyio.NpzFile
) -> None:
    def _edges_from_centers(vals: np.ndarray) -> np.ndarray:
        arr = np.asarray(vals, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return np.asarray([0.0, 1.0], dtype=np.float64)
        if arr.size == 1:
            return np.asarray([arr[0] - 0.5, arr[0] + 0.5], dtype=np.float64)
        mids = 0.5 * (arr[:-1] + arr[1:])
        first = arr[0] - (mids[0] - arr[0])
        last = arr[-1] + (arr[-1] - mids[-1])
        return np.concatenate(([first], mids, [last]))

    y_um, _, _ = _crop_profile(
        np.asarray(beamz["dft_monitor_y_um"], dtype=np.float64),
        np.asarray(beamz["dft_monitor_Ez"][...], dtype=np.complex128),
        np.asarray(meep["dft_monitor_Ez"][...], dtype=np.complex128),
    )
    t_beamz = np.asarray(beamz["dft_monitor_times_s"], dtype=np.float64)
    t_meep = np.asarray(meep["dft_monitor_times_s"], dtype=np.float64)
    b = np.asarray(beamz["dft_monitor_Ez_running_actual"], dtype=np.complex128)
    m = np.asarray(meep["dft_monitor_Ez_running_actual"], dtype=np.complex128)
    n_pts = min(b.shape[1], m.shape[1], y_um.size)
    b_steps = min(b.shape[0], t_beamz.size)
    m_steps = min(m.shape[0], t_meep.size)
    n_steps = min(b_steps, m_steps)
    b = b[:b_steps, :n_pts]
    m = m[:m_steps, :n_pts]
    y_um = y_um[:n_pts]
    y_edges = _edges_from_centers(y_um)
    b_t_fs = t_beamz[:b_steps] * 1e15
    m_t_fs = t_meep[:m_steps] * 1e15

    bmag = np.abs(b)
    mmag = np.abs(m)
    b_common = b[:n_steps]
    m_common = m[:n_steps]
    ref = max(float(np.max(np.abs(b_common))), float(np.max(np.abs(m_common))), 1e-30)
    err = np.abs(b_common - m_common) / ref
    b_extent = [float(y_edges[0]), float(y_edges[-1]), -0.5, float(b_steps) - 0.5]
    m_extent = [float(y_edges[0]), float(y_edges[-1]), -0.5, float(m_steps) - 0.5]
    e_extent = [float(y_edges[0]), float(y_edges[-1]), -0.5, float(n_steps) - 0.5]
    bmax = max(float(np.max(bmag)), 1e-30)
    mmax = max(float(np.max(mmag)), 1e-30)
    shared_vmax = max(bmax, mmax)
    err_vmax = max(float(np.max(err)), 1e-12)
    rel = np.zeros((n_steps,), dtype=np.float64)
    for idx in range(n_steps):
        denom = max(
            float(np.linalg.norm(b_common[idx])),
            float(np.linalg.norm(m_common[idx])),
            1e-30,
        )
        rel[idx] = float(np.linalg.norm(b_common[idx] - m_common[idx]) / denom)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    for ax, data, title, cmap, vmin, vmax_i, extent, steps in (
        (
            axes[0],
            bmag,
            (
                "Beamz actual running |DFT Ez|\n"
                f"samples={b_steps}, t=[{float(b_t_fs[0]):.2f}, {float(b_t_fs[-1]):.2f}] fs, max={bmax:.3e}"
            ),
            "magma",
            0.0,
            shared_vmax,
            b_extent,
            b_steps,
        ),
        (
            axes[1],
            mmag,
            (
                "Meep actual running |DFT Ez|\n"
                f"samples={m_steps}, t=[{float(m_t_fs[0]):.2f}, {float(m_t_fs[-1]):.2f}] fs, max={mmax:.3e}"
            ),
            "magma",
            0.0,
            shared_vmax,
            m_extent,
            m_steps,
        ),
        (
            axes[2],
            err,
            (
                "Actual running error (common subset)\n"
                f"samples={n_steps}, final rel={float(rel[-1]):.3e}, max rel={float(np.max(rel)):.3e}"
            ),
            "viridis",
            0.0,
            err_vmax,
            e_extent,
            n_steps,
        ),
    ):
        im = ax.imshow(
            data,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax_i,
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xlabel("monitor y (um)")
        ax.set_ylabel("sample index")
        if steps > 1:
            tick_count = min(6, steps)
            tick_idx = np.linspace(0, steps - 1, tick_count, dtype=int)
            ax.set_yticks(tick_idx)
            ax.set_yticklabels([str(int(i)) for i in tick_idx])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(output_dir / "dft_monitor_ez_running_actual.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot 2D Beamz/Meep waveguide field comparisons.")
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

    if "dft_monitor_Ez" in beamz.files and "dft_monitor_Ez" in meep.files:
        _plot_dft_final(out_dir, beamz, meep)
    if "dft_monitor_Ez_running" in beamz.files and "dft_monitor_Ez_running" in meep.files:
        _plot_dft_running(out_dir, beamz, meep)
    if (
        "dft_monitor_Ez_running_actual" in beamz.files
        and "dft_monitor_Ez_running_actual" in meep.files
    ):
        _plot_dft_running_actual(out_dir, beamz, meep)

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
