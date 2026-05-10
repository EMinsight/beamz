"""Oblique-incidence reflection benchmark for absorber boundaries.

This benchmark extends the normal-incidence plane-wave test by launching a
phased Ez current sheet that approximates an oblique plane wave in homogeneous
vacuum. We compare each case against a wider reference domain with the same
discretization and source, then measure the first reflected return on a probe
line near the source side.

This is the next important absorber gate after normal incidence:
- CPML should remain stable and converge with thickness at nonzero angle
- CPML should ideally outperform the legacy graded-conductivity absorber here
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from beamz import EPS_0, LIGHT_SPEED, Design, Material, PML, Simulation
from beamz.devices.sources.compiler import _as_slab_spec, _sample_waveform

SCRIPT_DIR = Path(__file__).resolve().parent


class ObliqueLineCurrentSource:
    """Phased Ez point-source sheet that approximates an oblique plane wave."""

    def __init__(self, base_signal, ix: int, y_positions_m: np.ndarray, angle_deg: float):
        self.base_signal = base_signal
        self.ix = int(ix)
        self.y_positions_m = np.asarray(y_positions_m, dtype=float)
        self.angle_rad = math.radians(float(angle_deg))

    def compile_source_specs(
        self,
        *,
        fields,
        dt: float,
        num_steps: int,
        t0: float,
        resolution: float,
        total_steps: int | None = None,
    ):
        del resolution
        specs = []
        iy_positions = np.arange(self.y_positions_m.shape[0], dtype=int)
        eps_col = np.asarray(fields.permittivity[:, self.ix], dtype=np.float32)
        time_delays = (self.y_positions_m - float(np.mean(self.y_positions_m))) * math.sin(self.angle_rad) / LIGHT_SPEED
        apod = np.hanning(max(len(iy_positions), 2))
        if len(iy_positions) == 1:
            apod = np.ones((1,), dtype=float)

        for local_i, iy in enumerate(iy_positions):
            coeff = -(float(dt) / (EPS_0 * float(eps_col[iy]))) * float(apod[local_i])
            delay = float(time_delays[local_i])
            waveform = _sample_waveform(
                lambda t_sample, _dt, delay=delay: self.base_signal(float(t_sample) - delay),
                t0=t0,
                dt=dt,
                num_steps=num_steps,
                offset_fn=lambda t, dt_: t + 0.5 * dt_,
                total_steps=total_steps,
            )
            specs.append(
                _as_slab_spec(
                    component="Ez",
                    timing="pre_e",
                    index=(int(iy), self.ix),
                    coeff=np.asarray([[coeff]], dtype=np.float32),
                    waveform=waveform,
                    target_shape=tuple(fields.Ez.shape),
                )
            )
        return tuple(specs)


@dataclass(frozen=True)
class SweepConfig:
    wavelength_um: float = 1.55
    angle_deg: float = 30.0
    inner_width_wl: float = 7.0
    inner_height_wl: float = 4.0
    source_offset_wl: float = 0.50
    probe_offset_wl: float = 1.40
    source_margin_wl: float = 0.25
    resolution_ppw: int = 14
    courant_safety: float = 0.95
    pulse_sigma_periods: float = 1.5
    pulse_center_sigmas: float = 4.0
    reference_extra_width_wl: float = 12.0
    cpml_kappa_max: float = 2.0
    cpml_alpha_max: float | None = None
    pml_formulations: tuple[str, ...] = ("sigma", "cpml")
    pml_thicknesses_wl: tuple[float, ...] = (0.5, 1.0, 1.5)

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
    angle_deg: float
    reflection_ratio: float
    reflection_db: float
    incident_peak: float
    reflected_peak: float
    dx_nm: float
    dt_fs: float
    num_steps: int


def _simulate_probe(
    cfg: SweepConfig,
    formulation: str,
    pml_thickness_wl: float,
    *,
    inner_width_m: float,
    dx: float,
    dt: float,
    total_time_s: float | None = None,
):
    wl = cfg.wavelength_m
    pml_m = float(pml_thickness_wl) * wl
    inner_height_m = float(cfg.inner_height_wl) * wl
    width_m = inner_width_m + 2.0 * pml_m
    height_m = inner_height_m + 2.0 * pml_m
    design = Design(width=width_m, height=height_m, material=Material(1.0))

    freq = cfg.frequency_hz
    period = 1.0 / freq
    sigma_t = float(cfg.pulse_sigma_periods) * period
    t0 = float(cfg.pulse_center_sigmas) * sigma_t

    cos_theta = max(math.cos(math.radians(float(cfg.angle_deg))), 1e-6)
    x_src = pml_m + float(cfg.source_offset_wl) * wl
    x_probe = pml_m + float(cfg.probe_offset_wl) * wl
    x_right = width_m - pml_m
    right_travel = (x_right - x_src + x_right - x_probe) / (LIGHT_SPEED * cos_theta)
    if total_time_s is None:
        total_time_s = t0 + 4.0 * sigma_t + right_travel + 4.0 * sigma_t
    time_axis = np.arange(0.0, total_time_s, dt, dtype=float)

    def signal(t_s: float) -> float:
        return float(
            np.exp(-0.5 * ((t_s - t0) / max(sigma_t, 1e-30)) ** 2)
            * np.cos(2.0 * np.pi * freq * (t_s - t0))
        )

    y0 = pml_m + float(cfg.source_margin_wl) * wl
    y1 = height_m - pml_m - float(cfg.source_margin_wl) * wl
    iy0 = int(round(y0 / dx))
    iy1 = int(round(y1 / dx))
    ix_src = int(round(x_src / dx))
    ix_probe = int(round(x_probe / dx))
    probe_slice = slice(iy0, iy1)
    y_positions_m = np.arange(iy0, iy1, dtype=float) * dx

    sim = Simulation(
        design=design,
        sources=[ObliqueLineCurrentSource(signal, ix_src, y_positions_m, cfg.angle_deg)],
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
    probe = ez_hist[:, probe_slice, ix_probe]
    times = np.arange(probe.shape[0], dtype=float) * dt
    incident_center = t0 + (x_probe - x_src) / (LIGHT_SPEED * cos_theta)
    reflected_center = t0 + right_travel
    window_half = 2.5 * sigma_t / cos_theta
    return {
        "probe": probe,
        "times_s": times,
        "incident_center_s": incident_center,
        "reflected_center_s": reflected_center,
        "window_half_s": window_half,
        "dx_nm": float(dx / 1e-9),
        "dt_fs": float(dt / 1e-15),
        "num_steps": int(probe.shape[0]),
    }


def _run_case(cfg: SweepConfig, formulation: str, pml_thickness_wl: float):
    wl = cfg.wavelength_m
    inner_width_m = float(cfg.inner_width_wl) * wl
    dx = wl / float(cfg.resolution_ppw)
    dt = float(cfg.courant_safety) * dx / (LIGHT_SPEED * np.sqrt(2.0))
    case = _simulate_probe(
        cfg,
        formulation,
        pml_thickness_wl,
        inner_width_m=inner_width_m,
        dx=dx,
        dt=dt,
    )
    reference = _simulate_probe(
        cfg,
        formulation,
        pml_thickness_wl,
        inner_width_m=(float(cfg.inner_width_wl) + float(cfg.reference_extra_width_wl)) * wl,
        dx=dx,
        dt=dt,
        total_time_s=float(case["times_s"][-1]) + float(dt),
    )

    times = np.asarray(case["times_s"], dtype=float)
    probe = np.asarray(case["probe"], dtype=np.float64)
    ref_probe = np.asarray(reference["probe"], dtype=np.float64)
    reflected = probe - ref_probe[: probe.shape[0]]
    incident_center = float(case["incident_center_s"])
    reflected_center = float(case["reflected_center_s"])
    window_half = float(case["window_half_s"])
    incident_mask = (times >= incident_center - window_half) & (times <= incident_center + window_half)
    reflected_mask = (times >= reflected_center - window_half) & (times <= reflected_center + window_half)

    incident_norms = np.linalg.norm(ref_probe[: probe.shape[0]], axis=1)
    reflected_norms = np.linalg.norm(reflected, axis=1)
    incident_peak = float(np.max(incident_norms[incident_mask])) if np.any(incident_mask) else 0.0
    reflected_peak = float(np.max(reflected_norms[reflected_mask])) if np.any(reflected_mask) else 0.0
    reflection_ratio = reflected_peak / max(incident_peak, 1e-30)

    result = CaseResult(
        formulation=str(formulation),
        pml_thickness_wl=float(pml_thickness_wl),
        pml_thickness_um=float(pml_thickness_wl) * cfg.wavelength_um,
        angle_deg=float(cfg.angle_deg),
        reflection_ratio=float(reflection_ratio),
        reflection_db=20.0 * math.log10(max(reflection_ratio, 1e-30)),
        incident_peak=incident_peak,
        reflected_peak=reflected_peak,
        dx_nm=float(case["dx_nm"]),
        dt_fs=float(case["dt_fs"]),
        num_steps=int(case["num_steps"]),
    )
    debug = {
        "times_s": times,
        "incident_norms": incident_norms,
        "reflected_norms": reflected_norms,
        "incident_center_s": incident_center,
        "reflected_center_s": reflected_center,
        "window_half_s": window_half,
    }
    return result, debug


def _plot_results(results: list[CaseResult], out_dir: Path):
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=220)
    formulations = sorted({r.formulation for r in results})
    for formulation in formulations:
        subset = [r for r in results if r.formulation == formulation]
        x = np.asarray([r.pml_thickness_wl for r in subset], dtype=float)
        y = np.asarray([r.reflection_db for r in subset], dtype=float)
        ax.plot(x, y, marker="o", lw=2.0, label=formulation)
    ax.set_title("Oblique Reflection vs PML Thickness")
    ax.set_xlabel("PML thickness (wavelengths)")
    ax.set_ylabel("Reflected / incident line norm (dB)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path = out_dir / "oblique_plane_wave_pml_sweep.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def _plot_probe_trace(debug: dict, out_dir: Path, *, formulation: str, thickness_wl: float):
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=220)
    times_fs = np.asarray(debug["times_s"], dtype=float) / 1e-15
    ax.plot(times_fs, np.asarray(debug["incident_norms"]), lw=1.2, label="reference line norm")
    ax.plot(times_fs, np.asarray(debug["reflected_norms"]), lw=1.2, label="reflected line norm")
    ax.axvline(float(debug["incident_center_s"]) / 1e-15, color="tab:green", ls="--", lw=1.0, label="incident")
    ax.axvline(float(debug["reflected_center_s"]) / 1e-15, color="tab:red", ls="--", lw=1.0, label="reflected")
    ax.set_title(f"Oblique Probe Trace ({formulation}, {thickness_wl:.2f} λ)")
    ax.set_xlabel("Time (fs)")
    ax.set_ylabel("Line-field norm")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path = out_dir / f"oblique_probe_trace_{formulation}_{thickness_wl:.2f}wl.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep oblique plane-wave reflection versus absorber thickness.")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results_oblique_plane_wave_pml_sweep")
    parser.add_argument("--formulations", type=str, default="sponge,cpml")
    parser.add_argument("--pml-thicknesses-wl", type=str, default="0.5,1.0,1.5")
    parser.add_argument("--angle-deg", type=float, default=SweepConfig().angle_deg)
    parser.add_argument("--cpml-kappa-max", type=float, default=SweepConfig().cpml_kappa_max)
    parser.add_argument("--cpml-alpha-max", type=float, default=SweepConfig().cpml_alpha_max)
    args = parser.parse_args()

    cfg = SweepConfig(
        pml_formulations=tuple(token.strip() for token in str(args.formulations).split(",") if token.strip()),
        pml_thicknesses_wl=tuple(float(token.strip()) for token in str(args.pml_thicknesses_wl).split(",") if token.strip()),
        angle_deg=float(args.angle_deg),
        cpml_kappa_max=float(args.cpml_kappa_max),
        cpml_alpha_max=None if args.cpml_alpha_max is None else float(args.cpml_alpha_max),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[CaseResult] = []
    probe_plots: list[str] = []
    for formulation in cfg.pml_formulations:
        for thickness_wl in cfg.pml_thicknesses_wl:
            print(f"{formulation} thickness={thickness_wl:.2f} λ angle={cfg.angle_deg:.1f}°")
            case, debug = _run_case(cfg, formulation, thickness_wl)
            results.append(case)
            if math.isclose(thickness_wl, cfg.pml_thicknesses_wl[len(cfg.pml_thicknesses_wl) // 2]):
                probe_plots.append(
                    str(_plot_probe_trace(debug, args.output_dir, formulation=formulation, thickness_wl=thickness_wl))
                )

    plot_path = _plot_results(results, args.output_dir)
    manifest = {
        "config": asdict(cfg),
        "results": [asdict(r) for r in results],
        "plot": str(plot_path),
        "probe_plots": probe_plots,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {manifest_path}")
    print(f"Wrote {plot_path}")
    for result in results:
        print(
            f"  {result.formulation} {result.pml_thickness_wl:.2f} λ @ {result.angle_deg:.1f}°: "
            f"refl={result.reflection_db:.2f} dB (ratio={result.reflection_ratio:.3e})"
        )


if __name__ == "__main__":
    main()
