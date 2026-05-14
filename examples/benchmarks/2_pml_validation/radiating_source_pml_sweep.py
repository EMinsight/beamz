"""Radiating-source absorber benchmark for CPML versus graded absorber.

This benchmark launches a short Gaussian pulse from a compact source in vacuum
and measures the residual field energy left in the non-PML interior after the
wavefront should have left the domain. It is harsher on absorbers than a simple
normal-incidence plane-wave termination because the outgoing radiation spans a
broad angular spectrum.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from beamz import LIGHT_SPEED, Design, GaussianSource, Material, PML, Simulation

SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SweepConfig:
    wavelength_um: float = 1.55
    inner_width_wl: float = 6.0
    inner_height_wl: float = 6.0
    source_width_wl: float = 0.20
    resolution_ppw: int = 14
    courant_safety: float = 0.95
    pulse_sigma_periods: float = 1.5
    pulse_center_sigmas: float = 4.0
    reference_window_fraction: float = 0.15
    cpml_kappa_max: float = 3.0
    cpml_alpha_max: float | None = 0.0
    pml_formulations: tuple[str, ...] = ("sigma", "cpml")
    pml_thicknesses_wl: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)

    @property
    def wavelength_m(self) -> float:
        return float(self.wavelength_um) * 1e-6

    @property
    def frequency_hz(self) -> float:
        return LIGHT_SPEED / self.wavelength_m


@dataclass
class CaseResult:
    formulation: str
    pml_thickness_wl: float
    pml_thickness_um: float
    residual_energy_ratio: float
    residual_energy_db: float
    peak_energy: float
    late_energy: float
    dx_nm: float
    dt_fs: float
    num_steps: int


def _run_case(cfg: SweepConfig, formulation: str, pml_thickness_wl: float):
    wl = cfg.wavelength_m
    pml_m = float(pml_thickness_wl) * wl
    inner_width_m = float(cfg.inner_width_wl) * wl
    inner_height_m = float(cfg.inner_height_wl) * wl
    width_m = inner_width_m + 2.0 * pml_m
    height_m = inner_height_m + 2.0 * pml_m
    dx = wl / float(cfg.resolution_ppw)
    dt = float(cfg.courant_safety) * dx / (LIGHT_SPEED * np.sqrt(2.0))

    design = Design(width=width_m, height=height_m, material=Material(1.0))
    freq = cfg.frequency_hz
    period = 1.0 / freq
    sigma_t = float(cfg.pulse_sigma_periods) * period
    t0 = float(cfg.pulse_center_sigmas) * sigma_t
    max_inner_radius = 0.5 * math.hypot(inner_width_m, inner_height_m)
    total_time_s = t0 + 4.0 * sigma_t + max_inner_radius / LIGHT_SPEED + 6.0 * sigma_t
    time_axis = np.arange(0.0, total_time_s, dt, dtype=float)

    def signal(t_s: float) -> float:
        return float(
            np.exp(-0.5 * ((t_s - t0) / max(sigma_t, 1e-30)) ** 2)
            * np.cos(2.0 * np.pi * freq * (t_s - t0))
        )

    sim = Simulation(
        design=design,
        sources=[
            GaussianSource(
                position=(0.5 * width_m, 0.5 * height_m),
                width=float(cfg.source_width_wl) * wl,
                signal=signal,
            )
        ],
        boundaries=[
            PML(
                edges="all",
                thickness=pml_m,
                formulation=formulation,
                kappa_max=float(cfg.cpml_kappa_max),
                alpha_max=cfg.cpml_alpha_max,
            )
        ],
        time=time_axis,
        resolution=dx,
    )

    out = sim.run_compiled(record_interval=1, record_fields=["Ez"], progress=False)
    ez_hist = np.asarray(out["fields"]["Ez"], dtype=np.float64)
    energies = np.mean(np.square(ez_hist), axis=(1, 2))

    start_ix = int(round(pml_m / dx))
    end_ix = int(round((width_m - pml_m) / dx))
    start_iy = int(round(pml_m / dx))
    end_iy = int(round((height_m - pml_m) / dx))
    interior_hist = ez_hist[:, start_iy:end_iy, start_ix:end_ix]
    interior_energies = np.mean(np.square(interior_hist), axis=(1, 2))

    peak_energy = float(np.max(interior_energies))
    late_start = int((1.0 - float(cfg.reference_window_fraction)) * interior_energies.shape[0])
    late_energy = float(np.mean(interior_energies[late_start:]))
    residual_ratio = late_energy / max(peak_energy, 1e-30)

    result = CaseResult(
        formulation=str(formulation),
        pml_thickness_wl=float(pml_thickness_wl),
        pml_thickness_um=float(pml_thickness_wl) * cfg.wavelength_um,
        residual_energy_ratio=float(residual_ratio),
        residual_energy_db=10.0 * math.log10(max(residual_ratio, 1e-30)),
        peak_energy=peak_energy,
        late_energy=late_energy,
        dx_nm=float(dx / 1e-9),
        dt_fs=float(dt / 1e-15),
        num_steps=int(ez_hist.shape[0]),
    )
    debug = {
        "times_s": np.arange(ez_hist.shape[0], dtype=float) * dt,
        "total_energy": energies,
        "interior_energy": interior_energies,
    }
    return result, debug


def _plot_results(results: list[CaseResult], out_dir: Path):
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=220)
    formulations = sorted({r.formulation for r in results})
    for formulation in formulations:
        subset = [r for r in results if r.formulation == formulation]
        x = np.asarray([r.pml_thickness_wl for r in subset], dtype=float)
        y = np.asarray([r.residual_energy_db for r in subset], dtype=float)
        ax.plot(x, y, marker="o", lw=2.0, label=formulation)
    ax.set_title("Radiating-Source Residual Energy vs PML Thickness")
    ax.set_xlabel("PML thickness (wavelengths)")
    ax.set_ylabel("Late / peak interior energy (dB)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path = out_dir / "radiating_source_pml_sweep.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def _plot_energy_trace(debug: dict, out_dir: Path, *, formulation: str, thickness_wl: float):
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=220)
    times_fs = np.asarray(debug["times_s"], dtype=float) / 1e-15
    ax.plot(times_fs, np.asarray(debug["total_energy"]), lw=1.0, label="total Ez energy")
    ax.plot(times_fs, np.asarray(debug["interior_energy"]), lw=1.2, label="interior Ez energy")
    ax.set_title(f"Radiating Energy Trace ({formulation}, {thickness_wl:.2f} λ)")
    ax.set_xlabel("Time (fs)")
    ax.set_ylabel("Mean squared Ez")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path = out_dir / f"radiating_energy_trace_{formulation}_{thickness_wl:.2f}wl.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep radiating-source residual energy versus absorber thickness.")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results_radiating_source_pml_sweep")
    parser.add_argument("--formulations", type=str, default="sponge,cpml")
    parser.add_argument("--pml-thicknesses-wl", type=str, default="0.5,1.0,1.5,2.0")
    parser.add_argument("--cpml-kappa-max", type=float, default=SweepConfig().cpml_kappa_max)
    parser.add_argument("--cpml-alpha-max", type=float, default=SweepConfig().cpml_alpha_max)
    args = parser.parse_args()

    cfg = SweepConfig(
        pml_formulations=tuple(token.strip() for token in str(args.formulations).split(",") if token.strip()),
        pml_thicknesses_wl=tuple(float(token.strip()) for token in str(args.pml_thicknesses_wl).split(",") if token.strip()),
        cpml_kappa_max=float(args.cpml_kappa_max),
        cpml_alpha_max=None if args.cpml_alpha_max is None else float(args.cpml_alpha_max),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[CaseResult] = []
    trace_plots: list[str] = []
    mid_idx = len(cfg.pml_thicknesses_wl) // 2
    for formulation in cfg.pml_formulations:
        for thickness_wl in cfg.pml_thicknesses_wl:
            print(f"{formulation} thickness={thickness_wl:.2f} λ")
            case, debug = _run_case(cfg, formulation, thickness_wl)
            results.append(case)
            if math.isclose(thickness_wl, cfg.pml_thicknesses_wl[mid_idx]):
                trace_plots.append(
                    str(_plot_energy_trace(debug, args.output_dir, formulation=formulation, thickness_wl=thickness_wl))
                )

    plot_path = _plot_results(results, args.output_dir)
    manifest = {
        "config": asdict(cfg),
        "results": [asdict(r) for r in results],
        "plot": str(plot_path),
        "trace_plots": trace_plots,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {manifest_path}")
    print(f"Wrote {plot_path}")
    for res in results:
        print(
            f"  {res.formulation} {res.pml_thickness_wl:.2f} λ:"
            f" residual={res.residual_energy_db:.2f} dB"
            f" (ratio={res.residual_energy_ratio:.3e})"
        )


if __name__ == "__main__":
    main()
