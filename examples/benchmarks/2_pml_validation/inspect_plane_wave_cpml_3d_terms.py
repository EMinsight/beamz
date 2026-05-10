"""Inspect 3D CPML term evolution on the plane-wave benchmark.

This diagnostic reuses the 3D normal-incidence plane-wave benchmark geometry,
runs the compiled solver with full field recording, and replays the CPML term
updates offline. It dumps the raw derivative terms, psi auxiliaries, corrected
terms, and sigma/kappa/alpha profiles on the right-boundary centerline so we can
see where the CPML branch first behaves unexpectedly.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from beamz import EPS_0, LIGHT_SPEED, Design, Material, PML, Simulation
from beamz.devices.sources.compiler import _as_slab_spec, _sample_waveform

SCRIPT_DIR = Path(__file__).resolve().parent


class PlaneCurrentSheetSource:
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
class InspectConfig:
    wavelength_um: float = 1.55
    pml_thickness_wl: float = 1.5
    inner_width_wl: float = 6.0
    inner_height_wl: float = 1.5
    inner_depth_wl: float = 1.5
    source_offset_wl: float = 0.40
    transverse_margin_wl: float = 0.0
    resolution_ppw: int = 8
    courant_safety: float = 0.95
    pulse_sigma_periods: float = 1.5
    pulse_center_sigmas: float = 4.0
    cpml_kappa_max: float = 2.0
    cpml_alpha_max: float | None = None

    @property
    def wavelength_m(self) -> float:
        return float(self.wavelength_um) * 1e-6

    @property
    def frequency_hz(self) -> float:
        return LIGHT_SPEED / self.wavelength_m


def _cpml_ab_from_profiles(sigma, kappa, alpha, dt):
    kappa = np.maximum(kappa, 1.0)
    decay = (sigma / kappa + alpha) * (dt / EPS_0)
    b = np.expm1(-decay) + 1.0
    denom = sigma + kappa * alpha
    a = np.nan_to_num(
        ((b - 1.0) * sigma) / np.maximum(denom * kappa, 1e-30),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return a, b


def _build_simulation(cfg: InspectConfig):
    wl = cfg.wavelength_m
    pml_m = float(cfg.pml_thickness_wl) * wl
    inner_width_m = float(cfg.inner_width_wl) * wl
    inner_height_m = float(cfg.inner_height_wl) * wl
    inner_depth_m = float(cfg.inner_depth_wl) * wl
    width_m = inner_width_m + 2.0 * pml_m
    height_m = inner_height_m + 2.0 * pml_m
    depth_m = inner_depth_m + 2.0 * pml_m
    dx = wl / float(cfg.resolution_ppw)
    dt = float(cfg.courant_safety) * dx / (LIGHT_SPEED * np.sqrt(3.0))

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
    x_right = width_m - pml_m
    right_travel = x_right - x_src
    total_time_s = t0 + 4.0 * sigma_t + 2.0 * right_travel / LIGHT_SPEED + 4.0 * sigma_t
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

    sim = Simulation(
        design=design,
        sources=[PlaneCurrentSheetSource(signal, ix_src, iy0, iy1, iz0, iz1)],
        boundaries=[
            PML(
                edges="all",
                thickness=pml_m,
                formulation="cpml",
                kappa_max=float(cfg.cpml_kappa_max),
                alpha_max=cfg.cpml_alpha_max,
            )
        ],
        time=time_axis,
        resolution=dx,
    )
    meta = {
        "wl_m": wl,
        "dx": dx,
        "dt": dt,
        "t0": t0,
        "x_src": x_src,
        "pml_m": pml_m,
        "width_m": width_m,
        "iy0": iy0,
        "iy1": iy1,
        "iz0": iz0,
        "iz1": iz1,
    }
    return sim, meta


def _full_from_native_e(ex, ey, ez):
    nz = ex.shape[0]
    ny = ex.shape[1]
    nx = ey.shape[2]
    ex_full = np.zeros((nz, ny, nx), dtype=np.float64)
    ey_full = np.zeros((nz, ny, nx), dtype=np.float64)
    ez_full = np.zeros((nz, ny, nx), dtype=np.float64)
    ex_full[:, :, :-1] = ex
    ey_full[:, :-1, :] = ey
    ez_full[:-1, :, :] = ez
    return ex_full, ey_full, ez_full


def _full_from_native_h(hx, hy, hz):
    nz = hz.shape[0]
    ny = hy.shape[1]
    nx = hx.shape[2]
    hx_full = np.zeros((nz, ny, nx), dtype=np.float64)
    hy_full = np.zeros((nz, ny, nx), dtype=np.float64)
    hz_full = np.zeros((nz, ny, nx), dtype=np.float64)
    hx_full[:-1, :-1, :] = hx
    hy_full[:-1, :, :-1] = hy
    hz_full[:, :-1, :-1] = hz
    return hx_full, hy_full, hz_full


def _d_terms_h(ex, ey, ez, resolution):
    ex_full, ey_full, ez_full = _full_from_native_e(ex, ey, ez)
    e_pad = np.pad(np.stack((ex_full, ey_full, ez_full), axis=0), ((0, 0), (1, 1), (1, 1), (1, 1)))
    return np.stack(
        (
            (np.roll(e_pad[2], -1, axis=1) - e_pad[2])[1:-1, 1:-1, 1:-1] / resolution,
            (np.roll(e_pad[1], -1, axis=0) - e_pad[1])[1:-1, 1:-1, 1:-1] / resolution,
            (np.roll(e_pad[0], -1, axis=0) - e_pad[0])[1:-1, 1:-1, 1:-1] / resolution,
            (np.roll(e_pad[2], -1, axis=2) - e_pad[2])[1:-1, 1:-1, 1:-1] / resolution,
            (np.roll(e_pad[1], -1, axis=2) - e_pad[1])[1:-1, 1:-1, 1:-1] / resolution,
            (np.roll(e_pad[0], -1, axis=1) - e_pad[0])[1:-1, 1:-1, 1:-1] / resolution,
        ),
        axis=0,
    )


def _d_terms_e(hx, hy, hz, resolution):
    hx_full, hy_full, hz_full = _full_from_native_h(hx, hy, hz)
    h_pad = np.pad(np.stack((hx_full, hy_full, hz_full), axis=0), ((0, 0), (1, 1), (1, 1), (1, 1)))
    return np.stack(
        (
            (h_pad[2] - np.roll(h_pad[2], 1, axis=1))[1:-1, 1:-1, 1:-1] / resolution,
            (h_pad[1] - np.roll(h_pad[1], 1, axis=0))[1:-1, 1:-1, 1:-1] / resolution,
            (h_pad[0] - np.roll(h_pad[0], 1, axis=0))[1:-1, 1:-1, 1:-1] / resolution,
            (h_pad[2] - np.roll(h_pad[2], 1, axis=2))[1:-1, 1:-1, 1:-1] / resolution,
            (h_pad[1] - np.roll(h_pad[1], 1, axis=2))[1:-1, 1:-1, 1:-1] / resolution,
            (h_pad[0] - np.roll(h_pad[0], 1, axis=1))[1:-1, 1:-1, 1:-1] / resolution,
        ),
        axis=0,
    )


def _plot_centerline(result: dict, out_dir: Path):
    x_um = np.asarray(result["x_um"], dtype=float)
    interface_x_um = float(result["interface_x_um"])

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=220, sharex=True)
    ax = axes[0, 0]
    ax.plot(x_um, np.asarray(result["sigma_h_term3"]), label="sigma")
    ax.plot(x_um, np.asarray(result["kappa_h_term3"]), label="kappa")
    ax.plot(x_um, np.asarray(result["alpha_h_term3"]), label="alpha")
    ax.axvline(interface_x_um, color="k", ls="--", lw=1.0)
    ax.set_title("H-side term 3 coefficients (dEz/dx -> Hy)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(x_um, np.asarray(result["sigma_e_term4"]), label="sigma")
    ax.plot(x_um, np.asarray(result["kappa_e_term4"]), label="kappa")
    ax.plot(x_um, np.asarray(result["alpha_e_term4"]), label="alpha")
    ax.axvline(interface_x_um, color="k", ls="--", lw=1.0)
    ax.set_title("E-side term 4 coefficients (dHy/dx -> Ez)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(x_um, np.asarray(result["d_h_term3"]), label="raw d")
    ax.plot(x_um, np.asarray(result["psi_h_term3"]), label="psi")
    ax.plot(x_um, np.asarray(result["corr_h_term3"]), label="corrected")
    ax.axvline(interface_x_um, color="k", ls="--", lw=1.0)
    ax.set_title("H-side term 3 centerline")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(x_um, np.asarray(result["d_e_term4"]), label="raw d")
    ax.plot(x_um, np.asarray(result["psi_e_term4"]), label="psi")
    ax.plot(x_um, np.asarray(result["corr_e_term4"]), label="corrected")
    ax.axvline(interface_x_um, color="k", ls="--", lw=1.0)
    ax.set_title("E-side term 4 centerline")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    for ax in axes[1]:
        ax.set_xlabel("x (um)")
    fig.tight_layout()
    out_path = out_dir / "cpml_3d_term_centerline.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect 3D CPML term evolution on the plane-wave benchmark.")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results_inspect_plane_wave_cpml_3d_terms")
    parser.add_argument("--pml-thickness-wl", type=float, default=InspectConfig().pml_thickness_wl)
    parser.add_argument("--resolution-ppw", type=int, default=InspectConfig().resolution_ppw)
    args = parser.parse_args()

    cfg = InspectConfig(
        pml_thickness_wl=float(args.pml_thickness_wl),
        resolution_ppw=int(args.resolution_ppw),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sim, meta = _build_simulation(cfg)
    program = sim.compile(num_steps=sim.num_steps)
    out = sim.run_compiled(
        record_interval=1,
        record_fields=["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"],
        progress=False,
    )

    ex_hist = np.asarray(out["fields"]["Ex"], dtype=np.float64)
    ey_hist = np.asarray(out["fields"]["Ey"], dtype=np.float64)
    ez_hist = np.asarray(out["fields"]["Ez"], dtype=np.float64)
    hx_hist = np.asarray(out["fields"]["Hx"], dtype=np.float64)
    hy_hist = np.asarray(out["fields"]["Hy"], dtype=np.float64)
    hz_hist = np.asarray(out["fields"]["Hz"], dtype=np.float64)
    times = np.arange(ez_hist.shape[0], dtype=float) * meta["dt"]

    pml_m = float(meta["pml_m"])
    width_m = float(meta["width_m"])
    x_src = float(meta["x_src"])
    dx = float(meta["dx"])
    interface_x_m = width_m - pml_m
    interface_idx = int(round(interface_x_m / dx))
    arrival_t = float(meta["t0"]) + (interface_x_m - x_src) / LIGHT_SPEED
    target_step = int(np.argmin(np.abs(times - arrival_t)))

    sigma_h = np.asarray(program.cpml3d_sigma_h_terms, dtype=np.float64)
    kappa_h = np.asarray(program.cpml3d_kappa_h_terms, dtype=np.float64)
    alpha_h = np.asarray(program.cpml3d_alpha_h_terms, dtype=np.float64)
    sigma_e = np.asarray(program.cpml3d_sigma_e_terms, dtype=np.float64)
    kappa_e = np.asarray(program.cpml3d_kappa_e_terms, dtype=np.float64)
    alpha_e = np.asarray(program.cpml3d_alpha_e_terms, dtype=np.float64)

    psi_h = np.zeros_like(sigma_h)
    psi_e = np.zeros_like(sigma_e)
    selected = None
    for n in range(ez_hist.shape[0]):
        d_h = _d_terms_h(ex_hist[n], ey_hist[n], ez_hist[n], dx)
        a_h, b_h = _cpml_ab_from_profiles(sigma_h, kappa_h, alpha_h, meta["dt"])
        psi_h = b_h * psi_h + a_h * d_h
        corr_h = d_h / np.maximum(kappa_h, 1.0) + psi_h

        d_e = _d_terms_e(hx_hist[n], hy_hist[n], hz_hist[n], dx)
        a_e, b_e = _cpml_ab_from_profiles(sigma_e, kappa_e, alpha_e, meta["dt"])
        psi_e = b_e * psi_e + a_e * d_e
        corr_e = d_e / np.maximum(kappa_e, 1.0) + psi_e

        if n == target_step:
            selected = {
                "d_h": d_h,
                "psi_h": psi_h.copy(),
                "corr_h": corr_h,
                "d_e": d_e,
                "psi_e": psi_e.copy(),
                "corr_e": corr_e,
            }
            break

    if selected is None:
        raise RuntimeError("Failed to capture target step diagnostics.")

    cy = sigma_h.shape[2] // 2
    cz = sigma_h.shape[1] // 2
    x_um = np.arange(sigma_h.shape[3], dtype=float) * dx * 1e6
    summary = {
        "config": asdict(cfg),
        "target_step": int(target_step),
        "target_time_fs": float(times[target_step] / 1e-15),
        "arrival_time_fs": float(arrival_t / 1e-15),
        "interface_x_um": float(interface_x_m * 1e6),
        "interface_idx": int(interface_idx),
        "center_indices": {"z": int(cz), "y": int(cy)},
        "x_um": x_um.tolist(),
        "sigma_h_term3": sigma_h[3, cz, cy, :].tolist(),
        "kappa_h_term3": kappa_h[3, cz, cy, :].tolist(),
        "alpha_h_term3": alpha_h[3, cz, cy, :].tolist(),
        "sigma_e_term4": sigma_e[4, cz, cy, :].tolist(),
        "kappa_e_term4": kappa_e[4, cz, cy, :].tolist(),
        "alpha_e_term4": alpha_e[4, cz, cy, :].tolist(),
        "d_h_term3": selected["d_h"][3, cz, cy, :].tolist(),
        "psi_h_term3": selected["psi_h"][3, cz, cy, :].tolist(),
        "corr_h_term3": selected["corr_h"][3, cz, cy, :].tolist(),
        "d_e_term4": selected["d_e"][4, cz, cy, :].tolist(),
        "psi_e_term4": selected["psi_e"][4, cz, cy, :].tolist(),
        "corr_e_term4": selected["corr_e"][4, cz, cy, :].tolist(),
    }
    plot_path = _plot_centerline(summary, args.output_dir)
    summary["plot"] = str(plot_path)
    manifest_path = args.output_dir / "summary.json"
    manifest_path.write_text(json.dumps(summary, indent=2))

    print(f"Wrote {manifest_path}")
    print(f"Wrote {plot_path}")
    print(
        f"target_step={summary['target_step']} target_time={summary['target_time_fs']:.2f} fs "
        f"arrival={summary['arrival_time_fs']:.2f} fs interface_idx={summary['interface_idx']}"
    )


if __name__ == "__main__":
    main()
