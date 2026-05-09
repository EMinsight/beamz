"""BeamZ analogue of FDTDX's simple waveguide transmission test.

This benchmark builds a straight 3D dielectric waveguide, launches the
fundamental mode with a broadband ModeSource, and measures modal transmission
at several planes downstream using the in-simulation DFT monitors.

It is not an exact replica of FDTDX's slab-waveguide + periodic-y setup because
BeamZ does not currently expose the same periodic boundary API here. Instead it
uses a straight rectangular guide with generous transverse cladding clearance on
the compiled runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import time as pytime
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from beamz import LIGHT_SPEED, Design, Material, ModeSource, Monitor, PML, PortSpec, Rectangle, Simulation
from beamz.const import µm
from beamz.devices.sources.signals import gaussian_band_pulse

SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    wavelength_um: float = 1.55
    num_freqs: int = 3
    dx_nm: float = 50.0
    courant_safety: float = 0.99
    pml_cells: int = 10
    domain_x_um: float = 3.0
    domain_y_um: float = 2.0
    domain_z_um: float = 2.0
    core_width_um: float = 0.60
    core_thickness_um: float = 0.25
    n_core: float = 3.5
    n_clad: float = 1.5
    source_x_um: float = 0.60
    det_source_x_um: float = 0.65
    det_near_x_um: float = 1.00
    det_mid_x_um: float = 1.25
    det_far_x_um: float = 1.50
    mode_span_y_um: float = 1.20
    mode_span_z_um: float = 1.20
    run_after_sources_uoc: float = 60.0
    decay_ratio: float = 1e-4
    lookback_records: int = 20
    formulations: tuple[str, ...] = ("sigma", "cpml")
    cpml_kappa_max: float = 3.0
    cpml_alpha_max: float | None = 0.0

    @property
    def wavelength_m(self) -> float:
        return float(self.wavelength_um) * 1e-6

    @property
    def dx_m(self) -> float:
        return float(self.dx_nm) * 1e-9

    @property
    def dt_s(self) -> float:
        return float(self.courant_safety) * self.dx_m / (LIGHT_SPEED * math.sqrt(3.0))

    @property
    def pml_um(self) -> float:
        return self.pml_cells * self.dx_nm * 1e-3

    @property
    def frequencies_hz(self) -> np.ndarray:
        if self.num_freqs <= 1:
            return np.asarray([LIGHT_SPEED / self.wavelength_m], dtype=float)
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
    s11_db: float
    s_near_db: float
    s_mid_db: float
    s_far_db: float
    power_sum: float
    loss_est: float
    steps: int
    runtime_s: float


def _x_plane(x_um: float, cfg: Config):
    x = float(x_um) * µm
    y0 = 0.5 * (float(cfg.domain_y_um) - float(cfg.mode_span_y_um)) * µm
    y1 = 0.5 * (float(cfg.domain_y_um) + float(cfg.mode_span_y_um)) * µm
    z0 = 0.5 * (float(cfg.domain_z_um) - float(cfg.mode_span_z_um)) * µm
    z1 = 0.5 * (float(cfg.domain_z_um) + float(cfg.mode_span_z_um)) * µm
    return (x, y0, z0), (x, y1, z1)


def _build_sim(cfg: Config, formulation: str):
    design = Design(
        width=float(cfg.domain_x_um) * µm,
        height=float(cfg.domain_y_um) * µm,
        depth=float(cfg.domain_z_um) * µm,
        material=Material(cfg.n_clad**2),
    )
    design += Rectangle(
        position=(
            0.0,
            0.5 * (float(cfg.domain_y_um) - float(cfg.core_width_um)) * µm,
            0.5 * (float(cfg.domain_z_um) - float(cfg.core_thickness_um)) * µm,
        ),
        width=float(cfg.domain_x_um) * µm,
        height=float(cfg.core_width_um) * µm,
        depth=float(cfg.core_thickness_um) * µm,
        material=Material(cfg.n_core**2),
    )

    grid = design.rasterize(resolution=cfg.dx_m)
    freqs = np.asarray(cfg.frequencies_hz, dtype=float)
    pulse = gaussian_band_pulse(
        freqs,
        carrier_frequency=LIGHT_SPEED / cfg.wavelength_m,
        dt=cfg.dt_s,
        run_after_sources_uoc=cfg.run_after_sources_uoc,
        max_output_distance_um=max(
            float(cfg.det_near_x_um - cfg.source_x_um),
            float(cfg.det_mid_x_um - cfg.source_x_um),
            float(cfg.det_far_x_um - cfg.source_x_um),
        ),
    )

    source = ModeSource(
        grid=grid,
        center=(
            float(cfg.source_x_um) * µm,
            0.5 * float(cfg.domain_y_um) * µm,
            0.5 * float(cfg.domain_z_um) * µm,
        ),
        width=float(cfg.mode_span_y_um) * µm,
        height=float(cfg.mode_span_z_um) * µm,
        wavelength=cfg.wavelength_m,
        pol="te",
        signal=pulse.signal,
        direction="+x",
    )
    source.initialize(grid.permittivity, cfg.dx_m)

    monitor_cfg = dict(
        record_fields=False,
        dft_enabled=True,
        dft_frequencies=freqs,
        dft_components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        dft_window="none",
        dft_record_every_step=True,
    )
    monitors = [
        Monitor(start=_x_plane(cfg.det_source_x_um, cfg)[0], end=_x_plane(cfg.det_source_x_um, cfg)[1], name="src", **monitor_cfg),
        Monitor(start=_x_plane(cfg.det_near_x_um, cfg)[0], end=_x_plane(cfg.det_near_x_um, cfg)[1], name="near", **monitor_cfg),
        Monitor(start=_x_plane(cfg.det_mid_x_um, cfg)[0], end=_x_plane(cfg.det_mid_x_um, cfg)[1], name="mid", **monitor_cfg),
        Monitor(start=_x_plane(cfg.det_far_x_um, cfg)[0], end=_x_plane(cfg.det_far_x_um, cfg)[1], name="far", **monitor_cfg),
    ]

    sim = Simulation(
        design=design,
        sources=[source],
        monitors=monitors,
        boundaries=[PML(edges="all", thickness=float(cfg.pml_um) * µm, formulation=formulation, kappa_max=cfg.cpml_kappa_max, alpha_max=cfg.cpml_alpha_max)],
        time=pulse.time,
        resolution=cfg.dx_m,
    )
    return sim, monitors, freqs, pulse


def run_case(cfg: Config, formulation: str) -> Result:
    sim, monitors, freqs, pulse = _build_sim(cfg, formulation)
    t0 = pytime.perf_counter()
    steps = sim.run_compiled_until_decay(
        monitors,
        min_time_s=float(pulse.source_end_time + pulse.tail_time),
        lookback_records=cfg.lookback_records,
        decay_ratio=cfg.decay_ratio,
        progress=False,
    )
    runtime_s = max(pytime.perf_counter() - t0, 1e-12)

    ports = [
        PortSpec(
            name="o1",
            monitor_name="src",
            direction="+x",
            polarization="te",
            mode_index=0,
            incident_wave="plus",
            scattered_wave="minus",
        ),
        PortSpec(
            name="near",
            monitor_name="near",
            direction="+x",
            polarization="te",
            mode_index=0,
            incident_wave="minus",
            scattered_wave="plus",
        ),
        PortSpec(
            name="mid",
            monitor_name="mid",
            direction="+x",
            polarization="te",
            mode_index=0,
            incident_wave="minus",
            scattered_wave="plus",
        ),
        PortSpec(
            name="far",
            monitor_name="far",
            direction="+x",
            polarization="te",
            mode_index=0,
            incident_wave="minus",
            scattered_wave="plus",
        ),
    ]
    result = sim.get_S_matrix_modal_dft(
        source_port="o1",
        ports=ports,
        output_ports=["o1", "near", "mid", "far"],
        frequencies=freqs,
        as_sax=False,
        return_diagnostics=True,
        min_incident_db=-50.0,
    )
    center_idx = int(np.argmin(np.abs(freqs - LIGHT_SPEED / cfg.wavelength_m)))
    s_matrix = result["s_matrix"]
    diag = result["diagnostics"]
    return Result(
        formulation=str(formulation),
        s11_db=20.0 * math.log10(max(abs(complex(np.asarray(s_matrix[("o1", "o1")])[center_idx])), 1e-12)),
        s_near_db=20.0 * math.log10(max(abs(complex(np.asarray(s_matrix[("near", "o1")])[center_idx])), 1e-12)),
        s_mid_db=20.0 * math.log10(max(abs(complex(np.asarray(s_matrix[("mid", "o1")])[center_idx])), 1e-12)),
        s_far_db=20.0 * math.log10(max(abs(complex(np.asarray(s_matrix[("far", "o1")])[center_idx])), 1e-12)),
        power_sum=float(np.asarray(diag["power_sum"], dtype=float)[center_idx]),
        loss_est=float(np.asarray(diag["loss_est"], dtype=float)[center_idx]),
        steps=int(steps),
        runtime_s=float(runtime_s),
    )


def _plot(results: list[Result], out_dir: Path) -> Path:
    x = np.arange(len(results), dtype=float)
    labels = [r.formulation for r in results]
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=220)
    ax.plot(x, [r.s_near_db for r in results], marker="o", label="Snear")
    ax.plot(x, [r.s_mid_db for r in results], marker="o", label="Smid")
    ax.plot(x, [r.s_far_db for r in results], marker="o", label="Sfar")
    ax.plot(x, [r.s11_db for r in results], marker="o", label="S11")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("3D Straight Waveguide Transmission")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = out_dir / "simple_waveguide_transmission_3d.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results_simple_waveguide_transmission_3d")
    parser.add_argument("--formulations", type=str, default="sigma,cpml")
    args = parser.parse_args()

    cfg = Config(formulations=tuple(tok.strip() for tok in args.formulations.split(",") if tok.strip()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [run_case(cfg, formulation) for formulation in cfg.formulations]
    plot_path = _plot(results, args.output_dir)
    manifest = {
        "config": asdict(cfg),
        "results": [asdict(r) for r in results],
        "plot": str(plot_path),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {manifest_path}")
    print(f"Wrote {plot_path}")
    for r in results:
        print(
            f"{r.formulation}: "
            f"S11={r.s11_db:.2f} dB, "
            f"Snear={r.s_near_db:.2f} dB, "
            f"Smid={r.s_mid_db:.2f} dB, "
            f"Sfar={r.s_far_db:.2f} dB, "
            f"power_sum={r.power_sum:.4f}, "
            f"loss={r.loss_est:.4f}, "
            f"steps={r.steps}, wall={r.runtime_s:.2f}s"
        )


if __name__ == "__main__":
    main()
