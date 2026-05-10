"""Representative 3D dipole-radiation CPML validation case.

This mirrors the spirit of FDTDX's dipole-radiation validation: run a compact
3D dipole-like source in a homogeneous box with absorbing boundaries and check
that the radiated field is physically plausible.  Here we focus on rotational
symmetry in the transverse (x/y) plane for a z-polarized electric source.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from beamz import LIGHT_SPEED, Design, GaussianSource, Material, Monitor, PML, Simulation

SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    wavelength_um: float = 1.55
    num_freqs: int = 3
    inner_size_wl: float = 4.0
    pml_thickness_wl: float = 1.5
    resolution_ppw: int = 10
    courant_safety: float = 0.95
    source_width_wl: float = 0.18
    probe_radius_wl: float = 1.0
    pulse_sigma_periods: float = 1.5
    pulse_center_sigmas: float = 4.0
    cpml_kappa_max: float = 2.0
    cpml_alpha_max: float | None = None
    formulations: tuple[str, ...] = ("sigma", "cpml")

    @property
    def wavelength_m(self) -> float:
        return float(self.wavelength_um) * 1e-6

    @property
    def frequency_hz(self) -> float:
        return LIGHT_SPEED / self.wavelength_m

    @property
    def frequencies_hz(self) -> np.ndarray:
        if self.num_freqs <= 1:
            return np.asarray([self.frequency_hz], dtype=float)
        wavelengths = np.linspace(
            0.99 * self.wavelength_m,
            1.01 * self.wavelength_m,
            int(self.num_freqs),
            dtype=float,
        )
        return LIGHT_SPEED / wavelengths


@dataclass
class Result:
    formulation: str
    pml_thickness_wl: float
    cpml_kappa_max: float | None
    cpml_alpha_max: float | None
    probe_radius_um: float
    ex_db: float
    wx_db: float
    ny_db: float
    py_db: float
    xy_symmetry_rel: float
    x_pair_rel: float
    y_pair_rel: float
    late_energy_ratio: float
    late_energy_db: float
    steps: int


def _point_monitor(position, name: str, freqs: np.ndarray) -> Monitor:
    return Monitor(
        start=position,
        end=position,
        name=name,
        record_fields=False,
        dft_enabled=True,
        dft_frequencies=freqs,
        dft_components=("Ez",),
        dft_window="none",
        dft_record_every_step=True,
    )


def _center_component(mon: Monitor, component: str, freqs: np.ndarray, center_hz: float) -> complex:
    idx = int(np.argmin(np.abs(np.asarray(freqs, dtype=float) - float(center_hz))))
    arr = np.asarray(mon.get_dft_component(component), dtype=np.complex128).reshape(-1)
    return complex(arr[idx])


def _build_sim(cfg: Config, formulation: str, pml_thickness_wl: float, cpml_kappa_max: float | None, cpml_alpha_max: float | None):
    wl = cfg.wavelength_m
    pml_m = float(pml_thickness_wl) * wl
    inner_size_m = float(cfg.inner_size_wl) * wl
    size_m = inner_size_m + 2.0 * pml_m
    dx = wl / float(cfg.resolution_ppw)
    dt = float(cfg.courant_safety) * dx / (LIGHT_SPEED * np.sqrt(3.0))

    design = Design(
        width=size_m,
        height=size_m,
        depth=size_m,
        material=Material(1.0),
    )

    freq = cfg.frequency_hz
    period = 1.0 / freq
    sigma_t = float(cfg.pulse_sigma_periods) * period
    t0 = float(cfg.pulse_center_sigmas) * sigma_t
    max_inner_radius = 0.5 * math.sqrt(3.0) * inner_size_m
    total_time_s = t0 + 4.0 * sigma_t + max_inner_radius / LIGHT_SPEED + 6.0 * sigma_t
    time_axis = np.arange(0.0, total_time_s, dt, dtype=float)

    def signal(t_s: float) -> float:
        return float(
            np.exp(-0.5 * ((t_s - t0) / max(sigma_t, 1e-30)) ** 2)
            * np.cos(2.0 * np.pi * freq * (t_s - t0))
        )

    center = 0.5 * size_m
    radius = float(cfg.probe_radius_wl) * wl
    freqs = np.asarray(cfg.frequencies_hz, dtype=float)
    monitors = [
        _point_monitor((center + radius, center, center), "px", freqs),
        _point_monitor((center - radius, center, center), "nx", freqs),
        _point_monitor((center, center + radius, center), "py", freqs),
        _point_monitor((center, center - radius, center), "ny", freqs),
    ]

    kappa_arg = float(cfg.cpml_kappa_max if cpml_kappa_max is None else cpml_kappa_max)
    alpha_arg = cfg.cpml_alpha_max if cpml_alpha_max is None else cpml_alpha_max
    sim = Simulation(
        design=design,
        sources=[
            GaussianSource(
                position=(center, center, center),
                width=float(cfg.source_width_wl) * wl,
                signal=signal,
            )
        ],
        monitors=monitors,
        boundaries=[
            PML(
                edges="all",
                thickness=pml_m,
                formulation=formulation,
                kappa_max=kappa_arg,
                alpha_max=alpha_arg,
            )
        ],
        time=time_axis,
        resolution=dx,
    )
    return sim, monitors, freqs


def _save_snapshots(ez_hist: np.ndarray, formulation: str, pml_thickness_wl: float, cpml_kappa_max: float | None, cpml_alpha_max: float | None, out_dir: Path):
    peak_idx = int(np.argmax(np.mean(np.square(ez_hist), axis=(1, 2, 3))))
    late_idx = int(round(0.92 * (ez_hist.shape[0] - 1)))
    center_z = ez_hist.shape[1] // 2
    center_y = ez_hist.shape[2] // 2

    def _slug(v: float | None) -> str:
        if v is None:
            return "na"
        return str(v).replace(".", "p")

    stem = (
        f"{formulation}_pml{pml_thickness_wl:.2f}wl_k{_slug(cpml_kappa_max)}_a{_slug(cpml_alpha_max)}"
        .replace("-", "m")
    )
    snapshots = [
        ("peak_xy", ez_hist[peak_idx, center_z]),
        ("peak_xz", ez_hist[peak_idx, :, center_y, :]),
        ("late_xy", ez_hist[late_idx, center_z]),
        ("late_xz", ez_hist[late_idx, :, center_y, :]),
    ]
    saved = []
    for name, arr in snapshots:
        fig, ax = plt.subplots(figsize=(4.4, 3.8), dpi=180)
        im = ax.imshow(np.asarray(arr), cmap="RdBu", origin="lower")
        ax.set_title(f"{formulation} {name}")
        ax.set_xlabel("x index")
        ax.set_ylabel("y index" if name.endswith("xy") else "z index")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        path = out_dir / f"{stem}_{name}.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        saved.append(str(path))
    return saved


def run_case(
    cfg: Config,
    formulation: str,
    pml_thickness_wl: float,
    cpml_kappa_max: float | None,
    cpml_alpha_max: float | None,
    snapshot_dir: Path | None = None,
) -> tuple[Result, list[str]]:
    sim, monitors, freqs = _build_sim(cfg, formulation, pml_thickness_wl, cpml_kappa_max, cpml_alpha_max)
    out = sim.run_compiled(record_interval=1, record_fields=["Ez"], progress=False)
    ez_hist = np.asarray(out["fields"]["Ez"], dtype=np.float64)

    probe_vals = {
        mon.name: abs(_center_component(mon, "Ez", freqs, cfg.frequency_hz))
        for mon in monitors
    }
    vals = np.asarray([probe_vals["px"], probe_vals["nx"], probe_vals["py"], probe_vals["ny"]], dtype=float)
    mean_val = float(np.mean(vals))
    xy_symmetry_rel = float(np.max(np.abs(vals - mean_val)) / max(mean_val, 1e-30))
    x_pair_rel = float(abs(probe_vals["px"] - probe_vals["nx"]) / max(0.5 * (probe_vals["px"] + probe_vals["nx"]), 1e-30))
    y_pair_rel = float(abs(probe_vals["py"] - probe_vals["ny"]) / max(0.5 * (probe_vals["py"] + probe_vals["ny"]), 1e-30))

    peak_energy = float(np.max(np.mean(np.square(ez_hist), axis=(1, 2, 3))))
    late_start = int(0.85 * ez_hist.shape[0])
    late_energy = float(np.mean(np.mean(np.square(ez_hist[late_start:]), axis=(1, 2, 3))))
    late_ratio = late_energy / max(peak_energy, 1e-30)

    def to_db(v: float) -> float:
        return 20.0 * math.log10(max(float(v), 1e-30))

    saved_snapshots: list[str] = []
    if snapshot_dir is not None:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        saved_snapshots = _save_snapshots(
            ez_hist,
            formulation=formulation,
            pml_thickness_wl=pml_thickness_wl,
            cpml_kappa_max=cpml_kappa_max,
            cpml_alpha_max=cpml_alpha_max,
            out_dir=snapshot_dir,
        )

    return Result(
        formulation=str(formulation),
        pml_thickness_wl=float(pml_thickness_wl),
        cpml_kappa_max=None if cpml_kappa_max is None else float(cpml_kappa_max),
        cpml_alpha_max=None if cpml_alpha_max is None else float(cpml_alpha_max),
        probe_radius_um=float(cfg.probe_radius_wl) * cfg.wavelength_um,
        ex_db=to_db(probe_vals["px"]),
        wx_db=to_db(probe_vals["nx"]),
        ny_db=to_db(probe_vals["ny"]),
        py_db=to_db(probe_vals["py"]),
        xy_symmetry_rel=xy_symmetry_rel,
        x_pair_rel=x_pair_rel,
        y_pair_rel=y_pair_rel,
        late_energy_ratio=float(late_ratio),
        late_energy_db=10.0 * math.log10(max(late_ratio, 1e-30)),
        steps=int(ez_hist.shape[0]),
    ), saved_snapshots


def _plot(results: list[Result], out_dir: Path) -> Path:
    labels = []
    for r in results:
        if r.formulation == "cpml":
            labels.append(f"cpml\n{r.pml_thickness_wl:.2f}λ\nκ={r.cpml_kappa_max:g}\nα={r.cpml_alpha_max:g}")
        else:
            labels.append(f"{r.formulation}\n{r.pml_thickness_wl:.2f}λ")
    x = np.arange(len(results), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), dpi=220)

    axes[0].bar(x - 0.15, [r.ex_db for r in results], width=0.15, label="+x")
    axes[0].bar(x, [r.wx_db for r in results], width=0.15, label="-x")
    axes[0].bar(x + 0.15, [r.py_db for r in results], width=0.15, label="+y")
    axes[0].bar(x + 0.30, [r.ny_db for r in results], width=0.15, label="-y")
    axes[0].set_xticks(x + 0.075)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("|Ez| at probes (dB)")
    axes[0].set_title("3D Dipole Radiation Probe Levels")
    axes[0].grid(alpha=0.3)
    axes[0].legend(frameon=False)

    axes[1].plot(x, [100.0 * r.xy_symmetry_rel for r in results], marker="o", label="XY spread")
    axes[1].plot(x, [100.0 * r.x_pair_rel for r in results], marker="o", label="X pair")
    axes[1].plot(x, [100.0 * r.y_pair_rel for r in results], marker="o", label="Y pair")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Relative mismatch (%)")
    axes[1].set_title("Transverse Symmetry Error")
    axes[1].grid(alpha=0.3)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    out = out_dir / "dipole_radiation_symmetry_3d.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results_dipole_radiation_symmetry_3d")
    parser.add_argument("--formulations", type=str, default="sponge,cpml")
    parser.add_argument("--pml-thicknesses-wl", type=str, default="1.0")
    parser.add_argument("--cpml-kappa-values", type=str, default="4.0,6.0")
    parser.add_argument("--cpml-alpha-values", type=str, default="300.0,150.0")
    parser.add_argument("--save-snapshots", action="store_true")
    args = parser.parse_args()

    cfg = Config(formulations=tuple(tok.strip() for tok in args.formulations.split(",") if tok.strip()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    thicknesses = tuple(float(tok.strip()) for tok in args.pml_thicknesses_wl.split(",") if tok.strip())
    cpml_kappas = tuple(float(tok.strip()) for tok in args.cpml_kappa_values.split(",") if tok.strip())
    cpml_alphas = tuple(float(tok.strip()) for tok in args.cpml_alpha_values.split(",") if tok.strip())
    results: list[Result] = []
    all_snapshots: list[str] = []
    snapshot_dir = args.output_dir / "snapshots" if args.save_snapshots else None
    for formulation in cfg.formulations:
        for thickness in thicknesses:
            if formulation == "cpml":
                for kappa in cpml_kappas:
                    for alpha in cpml_alphas:
                        result, saved = run_case(cfg, formulation, thickness, kappa, alpha, snapshot_dir=snapshot_dir)
                        results.append(result)
                        all_snapshots.extend(saved)
            else:
                result, saved = run_case(cfg, formulation, thickness, None, None, snapshot_dir=snapshot_dir)
                results.append(result)
                all_snapshots.extend(saved)
    plot_path = _plot(results, args.output_dir)
    manifest = {
        "config": asdict(cfg),
        "sweep": {
            "pml_thicknesses_wl": list(thicknesses),
            "cpml_kappa_values": list(cpml_kappas),
            "cpml_alpha_values": list(cpml_alphas),
        },
        "results": [asdict(r) for r in results],
        "plot": str(plot_path),
        "snapshots": all_snapshots,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {manifest_path}")
    print(f"Wrote {plot_path}")
    for r in results:
        print(
            f"{r.formulation} pml={r.pml_thickness_wl:.2f}λ "
            f"k={r.cpml_kappa_max if r.cpml_kappa_max is not None else '-'} "
            f"a={r.cpml_alpha_max if r.cpml_alpha_max is not None else '-'}: "
            f"|Ez|(+x,-x,+y,-y)=({r.ex_db:.2f}, {r.wx_db:.2f}, {r.py_db:.2f}, {r.ny_db:.2f}) dB, "
            f"xy_sym={100.0 * r.xy_symmetry_rel:.2f}%, "
            f"x_pair={100.0 * r.x_pair_rel:.2f}%, "
            f"y_pair={100.0 * r.y_pair_rel:.2f}%, "
            f"late={r.late_energy_db:.2f} dB"
        )


if __name__ == "__main__":
    main()
