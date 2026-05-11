"""3D normal-incidence plane-wave reflection benchmark for absorber boundaries.

This is the 3D analogue of ``plane_wave_pml_sweep.py``. A short uniform ``Ez``
current sheet on a ``yz`` plane launches an approximately planar pulse in
homogeneous vacuum. We record the ``Ez`` field on a downstream probe plane,
compare against a wider reference domain with identical discretization, and
measure the reflected return amplitude relative to the incident amplitude.

This should be the primary 3D absorber gate because it isolates the x-directed
boundary reflection more cleanly than the radiating-source residual benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from beamz import EPS_0, LIGHT_SPEED, MU_0, Design, Material, PML, Simulation
from beamz.devices.sources.compiler import _as_slab_spec, _sample_waveform

SCRIPT_DIR = Path(__file__).resolve().parent
ETA_0 = math.sqrt(MU_0 / EPS_0)


class PlaneCurrentSheetSource:
    """Uniform Ez current sheet used to launch an approximately planar pulse."""

    def __init__(self, signal, ix: int, iy0: int, iy1: int, iz0: int, iz1: int):
        self.signal = signal
        self.ix = int(ix)
        self.iy0 = int(iy0)
        self.iy1 = int(iy1)
        self.iz0 = int(iz0)
        self.iz1 = int(iz1)

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
        idx = (slice(self.iz0, self.iz1), slice(self.iy0, self.iy1), self.ix)
        eps_region = np.asarray(fields.eps_z[idx], dtype=np.float32)
        sig_region = np.asarray(fields.sig_z[idx], dtype=np.float32)
        denom = 1.0 + sig_region * (float(dt) / (2.0 * EPS_0 * eps_region))
        source_coeff = (float(dt) / (EPS_0 * eps_region)) / denom
        coeff = -source_coeff
        waveform = _sample_waveform(
            lambda t_sample, _dt: self.signal(float(t_sample)),
            t0=t0,
            dt=dt,
            num_steps=num_steps,
            offset_fn=lambda t, dt_: t + 0.5 * dt_,
            total_steps=total_steps,
        )
        return (
            _as_slab_spec(
                component="Ez",
                timing="e",
                index=idx,
                coeff=coeff,
                waveform=waveform,
                target_shape=tuple(fields.Ez.shape),
            ),
        )


@dataclass(frozen=True)
class SweepConfig:
    wavelength_um: float = 1.55
    inner_width_wl: float = 6.0
    inner_height_wl: float = 1.5
    inner_depth_wl: float = 1.5
    source_offset_wl: float = 0.40
    probe_offset_wl: float = 1.20
    reference_extra_width_wl: float = 12.0
    transverse_margin_wl: float = 0.0
    resolution_ppw: int = 8
    courant_safety: float = 0.95
    pulse_sigma_periods: float = 1.5
    pulse_center_sigmas: float = 4.0
    cpml_kappa_max: float = 3.0
    cpml_alpha_max: float | None = 0.0
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
    inner_depth_m = float(cfg.inner_depth_wl) * wl
    width_m = inner_width_m + 2.0 * pml_m
    height_m = inner_height_m + 2.0 * pml_m
    depth_m = inner_depth_m + 2.0 * pml_m

    design = Design(
        width=width_m,
        height=height_m,
        depth=depth_m,
        material=Material(1.0),
    )

    freq = cfg.frequency_hz
    period = 1.0 / freq
    sigma_t = float(cfg.pulse_sigma_periods) * period
    t0 = float(cfg.pulse_center_sigmas) * sigma_t

    x_src = pml_m + float(cfg.source_offset_wl) * wl
    x_probe = pml_m + float(cfg.probe_offset_wl) * wl
    x_right = width_m - pml_m
    left_travel = (x_src - pml_m) + (x_probe - pml_m)
    right_travel = (x_probe - x_src) + 2.0 * (x_right - x_probe)
    if total_time_s is None:
        total_time_s = (
            t0
            + 4.0 * sigma_t
            + max(left_travel, right_travel) / LIGHT_SPEED
            + 4.0 * sigma_t
        )
    time_axis = np.arange(0.0, total_time_s, dt, dtype=float)

    def signal(t_s: float) -> float:
        return float(
            np.exp(-0.5 * ((t_s - t0) / max(sigma_t, 1e-30)) ** 2)
            * np.cos(2.0 * np.pi * freq * (t_s - t0))
        )

    margin_m = float(cfg.transverse_margin_wl) * wl
    iy0 = int(round((pml_m + margin_m) / dx))
    iy1 = int(round((height_m - pml_m - margin_m) / dx))
    iz0 = int(round((pml_m + margin_m) / dx))
    iz1 = int(round((depth_m - pml_m - margin_m) / dx))
    ix_src = int(round(x_src / dx))
    ix_probe = int(round(x_probe / dx))

    sim = Simulation(
        design=design,
        sources=[PlaneCurrentSheetSource(signal, ix_src, iy0, iy1, iz0, iz1)],
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

    out = sim.run_compiled(record_interval=1, record_fields=["Ez", "Hy"], progress=False)
    ez_hist = np.asarray(out["fields"]["Ez"], dtype=np.float64)
    hy_hist = np.asarray(out["fields"]["Hy"], dtype=np.float64)
    probe_ez = ez_hist[:, iz0:iz1, iy0:iy1, ix_probe]
    probe_hy = hy_hist[:, iz0:iz1, iy0:iy1, ix_probe]
    times = np.arange(probe_ez.shape[0], dtype=float) * dt

    incident_center = t0 + (x_probe - x_src) / LIGHT_SPEED
    reflected_center = t0 + right_travel / LIGHT_SPEED
    window_half = 2.5 * sigma_t

    return {
        "probe_ez": probe_ez,
        "probe_hy": probe_hy,
        "times_s": times,
        "incident_center_s": incident_center,
        "reflected_center_s": reflected_center,
        "window_half_s": window_half,
        "dx_nm": float(dx / 1e-9),
        "dt_fs": float(dt / 1e-15),
        "num_steps": int(probe_ez.shape[0]),
    }


def _run_case(cfg: SweepConfig, formulation: str, pml_thickness_wl: float):
    wl = cfg.wavelength_m
    inner_width_m = float(cfg.inner_width_wl) * wl
    dx = wl / float(cfg.resolution_ppw)
    dt = float(cfg.courant_safety) * dx / (LIGHT_SPEED * np.sqrt(3.0))

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
    probe_ez = np.asarray(case["probe_ez"], dtype=np.float64)
    ref_probe_ez = np.asarray(reference["probe_ez"], dtype=np.float64)
    probe_hy = np.asarray(case["probe_hy"], dtype=np.float64)
    ref_probe_hy = np.asarray(reference["probe_hy"], dtype=np.float64)
    reflected_ez = probe_ez - ref_probe_ez[: probe_ez.shape[0]]
    reflected_hy = probe_hy - ref_probe_hy[: probe_hy.shape[0]]
    incident_center = float(case["incident_center_s"])
    reflected_center = float(case["reflected_center_s"])
    window_half = float(case["window_half_s"])

    incident_mask = (times >= incident_center - window_half) & (times <= incident_center + window_half)
    reflected_mask = (times >= reflected_center - window_half) & (
        times <= reflected_center + window_half
    )

    incident_norms = np.sqrt(
        np.sum(np.square(ref_probe_ez[: probe_ez.shape[0]]), axis=(1, 2))
        + np.sum(np.square(ETA_0 * ref_probe_hy[: probe_hy.shape[0]]), axis=(1, 2))
    )
    reflected_norms = np.sqrt(
        np.sum(np.square(reflected_ez), axis=(1, 2))
        + np.sum(np.square(ETA_0 * reflected_hy), axis=(1, 2))
    )
    incident_peak = float(np.max(incident_norms[incident_mask])) if np.any(incident_mask) else 0.0
    reflected_peak = float(np.max(reflected_norms[reflected_mask])) if np.any(reflected_mask) else 0.0
    reflection_ratio = reflected_peak / max(incident_peak, 1e-30)

    result = CaseResult(
        formulation=str(formulation),
        pml_thickness_wl=float(pml_thickness_wl),
        pml_thickness_um=float(pml_thickness_wl) * cfg.wavelength_um,
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
    ax.set_title("3D Normal-Incidence Reflection vs PML Thickness")
    ax.set_xlabel("PML thickness (wavelengths)")
    ax.set_ylabel("Reflected / incident plane norm (dB)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path = out_dir / "plane_wave_pml_sweep_3d.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def _plot_probe_trace(debug: dict, out_dir: Path, *, formulation: str, thickness_wl: float):
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=220)
    times_fs = np.asarray(debug["times_s"], dtype=float) / 1e-15
    ax.plot(times_fs, np.asarray(debug["incident_norms"]), lw=1.2, label="reference plane norm")
    ax.plot(times_fs, np.asarray(debug["reflected_norms"]), lw=1.2, label="reflected plane norm")
    ax.axvline(float(debug["incident_center_s"]) / 1e-15, color="tab:green", ls="--", lw=1.0, label="incident")
    ax.axvline(float(debug["reflected_center_s"]) / 1e-15, color="tab:red", ls="--", lw=1.0, label="reflected")
    ax.set_title(f"3D Plane Probe Trace ({formulation}, {thickness_wl:.2f} λ)")
    ax.set_xlabel("Time (fs)")
    ax.set_ylabel("Plane-field norm")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path = out_dir / f"plane_wave_probe_trace_3d_{formulation}_{thickness_wl:.2f}wl.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep 3D normal-incidence plane-wave reflection versus absorber thickness.")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results_plane_wave_pml_sweep_3d")
    parser.add_argument("--formulations", type=str, default="sponge,cpml")
    parser.add_argument("--pml-thicknesses-wl", type=str, default="0.5,1.0,1.5")
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
                    str(_plot_probe_trace(debug, args.output_dir, formulation=formulation, thickness_wl=thickness_wl))
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
            f" reflection={res.reflection_db:.2f} dB"
            f" (ratio={res.reflection_ratio:.3e})"
        )


if __name__ == "__main__":
    main()
