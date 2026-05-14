"""Straight-waveguide absorber benchmark versus PML thickness.

This is a deliberately small 2D modal benchmark intended to answer one
question: does BeamZ's current boundary absorber converge like a real PML for a
uniform waveguide?

For each PML thickness, the script runs the same straight-waveguide simulation
and records:
- S11 at the source-side monitor
- S21 at the output monitor
- guided-power sum and estimated loss from modal decomposition
- monitor tail/peak ratio used as a late-time reflection proxy
- steps/runtime to adaptive decay

This is the primary regression gate for replacing the current graded-sigma
absorber with a proper PML implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from beamz import (
    LIGHT_SPEED,
    Design,
    Material,
    ModeSource,
    Monitor,
    PML,
    PortSpec,
    Rectangle,
    Simulation,
    calc_optimal_fdtd_params,
    µm,
)
SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SweepConfig:
    wavelength_um: float = 1.55
    num_freqs: int = 3
    waveguide_width_um: float = 0.60
    n_core: float = 2.04
    n_clad: float = 1.444
    guide_length_um: float = 8.0
    vertical_clearance_um: float = 1.6
    source_offset_um: float = 0.45
    input_monitor_offset_um: float = 1.10
    output_monitor_offset_um: float = 1.10
    port_margin_um: float = 0.50
    resolution_ppw: int = 14
    courant_safety: float = 0.95
    pulse_sigma_periods: float = 6.0
    pulse_center_sigmas: float = 4.0
    settle_transit_multiples: float = 6.0
    decay_ratio: float = 1e-4
    lookback_records: int = 20
    pml_formulation: str = "sigma"
    cpml_kappa_max: float = 8.0
    cpml_alpha_max: float | None = None
    pml_thicknesses_wl: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)

    @property
    def wavelength_m(self) -> float:
        return float(self.wavelength_um) * 1e-6

    @property
    def frequency_hz(self) -> float:
        return LIGHT_SPEED / self.wavelength_m

    @property
    def waveguide_width_m(self) -> float:
        return float(self.waveguide_width_um) * 1e-6

    @property
    def source_offset_m(self) -> float:
        return float(self.source_offset_um) * 1e-6

    @property
    def input_monitor_offset_m(self) -> float:
        return float(self.input_monitor_offset_um) * 1e-6

    @property
    def output_monitor_offset_m(self) -> float:
        return float(self.output_monitor_offset_um) * 1e-6

    @property
    def guide_length_m(self) -> float:
        return float(self.guide_length_um) * 1e-6

    @property
    def vertical_clearance_m(self) -> float:
        return float(self.vertical_clearance_um) * 1e-6

    @property
    def port_margin_m(self) -> float:
        return float(self.port_margin_um) * 1e-6

    @property
    def dft_frequencies_hz(self) -> np.ndarray:
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
class CaseResult:
    pml_thickness_wl: float
    pml_thickness_um: float
    width_um: float
    height_um: float
    dx_nm: float
    dt_fs: float
    steps_to_decay: int
    runtime_s: float
    s11_db_center: float
    s21_db_center: float
    power_sum_center: float
    loss_est_center: float
    source_dominance_db: float
    output_dominance_db: float
    max_tail_ratio: float
    source_tail_ratio: float
    output_tail_ratio: float
    pml_formulation: str


def _monitor_line(x_m: float, y_center_m: float, span_m: float):
    return (
        (float(x_m), float(y_center_m - 0.5 * span_m)),
        (float(x_m), float(y_center_m + 0.5 * span_m)),
    )


def _wave_dominance_db(a_plus: np.ndarray, a_minus: np.ndarray, selector: str) -> float:
    sel = np.asarray(a_plus if selector == "plus" else a_minus, dtype=np.complex128)
    opp = np.asarray(a_minus if selector == "plus" else a_plus, dtype=np.complex128)
    p_sel = float(np.mean(np.abs(sel) ** 2))
    p_opp = float(np.mean(np.abs(opp) ** 2))
    return 10.0 * np.log10(max(p_sel, 1e-18) / max(p_opp, 1e-18))


def _tail_ratio(monitor: Monitor, lookback_records: int) -> float:
    hist = np.abs(np.asarray(getattr(monitor, "power_history", ()), dtype=float))
    if hist.size == 0:
        return float("nan")
    peak = max(float(np.max(hist)), 1e-30)
    tail = float(np.max(hist[-max(2, int(lookback_records)) :]))
    return tail / peak


def _build_case(cfg: SweepConfig, pml_thickness_wl: float):
    pml_m = float(pml_thickness_wl) * cfg.wavelength_m
    width_m = cfg.guide_length_m + 2.0 * pml_m
    height_m = cfg.waveguide_width_m + 2.0 * (cfg.vertical_clearance_m + pml_m)

    dx, dt = calc_optimal_fdtd_params(
        cfg.wavelength_m,
        max(cfg.n_core, cfg.n_clad),
        dims=2,
        safety_factor=cfg.courant_safety,
        points_per_wavelength=cfg.resolution_ppw,
        width=width_m,
        height=height_m,
    )

    design = Design(
        width=width_m,
        height=height_m,
        material=Material(cfg.n_clad**2),
    )
    y0 = 0.5 * (height_m - cfg.waveguide_width_m)
    design += Rectangle(
        position=(0.0, y0),
        width=width_m,
        height=cfg.waveguide_width_m,
        material=Material(cfg.n_core**2),
    )
    grid = design.rasterize(resolution=dx)

    freqs = np.asarray(cfg.dft_frequencies_hz, dtype=float)
    period = 1.0 / cfg.frequency_hz
    sigma_t = float(cfg.pulse_sigma_periods) * period
    t0 = float(cfg.pulse_center_sigmas) * sigma_t
    n_eff_guess = 0.5 * (cfg.n_core + cfg.n_clad)
    transit_time = n_eff_guess * width_m / LIGHT_SPEED
    min_time_s = t0 + 4.0 * sigma_t + float(cfg.settle_transit_multiples) * transit_time
    total_time_s = min_time_s + 2.0 * transit_time
    num_steps = max(64, int(np.ceil(total_time_s / dt)) + 1)
    time_axis = np.arange(num_steps, dtype=float) * dt
    signal = np.exp(-0.5 * ((time_axis - t0) / max(sigma_t, 1e-30)) ** 2) * np.cos(
        2.0 * np.pi * cfg.frequency_hz * (time_axis - t0)
    )
    pulse = SimpleNamespace(
        signal=np.asarray(signal, dtype=float),
        time=np.asarray(time_axis, dtype=float),
        source_end_time=float(t0 + 4.0 * sigma_t),
        tail_time=float(cfg.settle_transit_multiples) * transit_time,
    )

    y_center = 0.5 * height_m
    source_span = max(cfg.waveguide_width_m + 2.0 * cfg.port_margin_m, 3.0 * cfg.waveguide_width_m)
    source_center = (pml_m + cfg.source_offset_m, y_center)
    source = ModeSource(
        grid=grid,
        center=source_center,
        width=source_span,
        wavelength=cfg.wavelength_m,
        pol="tm",
        signal=pulse.signal,
        direction="+x",
    )
    source.initialize(grid.permittivity, dx)

    o1_line = _monitor_line(
        pml_m + cfg.input_monitor_offset_m,
        y_center,
        source_span,
    )
    o2_line = _monitor_line(
        width_m - pml_m - cfg.output_monitor_offset_m,
        y_center,
        source_span,
    )
    monitor_cfg = dict(
        record_fields=False,
        dft_enabled=True,
        dft_frequencies=freqs,
        dft_components=("Ez", "Hx", "Hy"),
        dft_window="none",
        dft_record_every_step=True,
    )
    m_o1 = Monitor(start=o1_line[0], end=o1_line[1], name="o1", **monitor_cfg)
    m_o2 = Monitor(start=o2_line[0], end=o2_line[1], name="o2", **monitor_cfg)

    sim = Simulation(
        design=design,
        sources=[source],
        monitors=[m_o1, m_o2],
        boundaries=[
            PML(
                edges="all",
                thickness=pml_m,
                formulation=cfg.pml_formulation,
                kappa_max=float(cfg.cpml_kappa_max),
                alpha_max=cfg.cpml_alpha_max,
            )
        ],
        time=pulse.time,
        resolution=dx,
    )
    return sim, m_o1, m_o2, freqs, pulse, width_m, height_m, dx, dt


def _run_until_decay(sim: Simulation, monitors, *, min_time_s: float, lookback_records: int, decay_ratio: float) -> int:
    total_steps = int(sim.num_steps - sim.current_step)
    if total_steps <= 0:
        return 0
    lookback_records = max(2, int(lookback_records))
    min_steps = int(np.ceil(max(0.0, float(min_time_s)) / max(float(sim.dt), 1e-30)))
    peak = 0.0
    steps_done = 0

    while steps_done < total_steps:
        if not sim.step():
            break
        steps_done += 1
        histories = [
            np.abs(np.asarray(mon.power_history, dtype=np.float64))
            for mon in monitors
            if len(mon.power_history)
        ]
        tail = np.inf
        if histories:
            peak = max(peak, max(float(np.max(hist)) for hist in histories))
            tail = max(float(np.max(hist[-lookback_records:])) for hist in histories)
        if (
            steps_done >= min_steps
            and peak > 0.0
            and np.isfinite(tail)
            and tail <= float(decay_ratio) * peak
        ):
            break
    return steps_done


def run_case(cfg: SweepConfig, pml_thickness_wl: float) -> CaseResult:
    sim, m_o1, m_o2, freqs, pulse, width_m, height_m, dx, dt = _build_case(
        cfg, pml_thickness_wl
    )
    wall_t0 = time.perf_counter()
    steps = sim.run_compiled_until_decay(
        [m_o1, m_o2],
        min_time_s=pulse.source_end_time + pulse.tail_time,
        lookback_records=cfg.lookback_records,
        decay_ratio=cfg.decay_ratio,
        progress=False,
    )
    runtime_s = max(time.perf_counter() - wall_t0, 1e-12)

    result = sim.get_S_matrix_modal_dft(
        source_port="o1",
        ports=[
            PortSpec(
                name="o1",
                monitor_name="o1",
                direction="+x",
                polarization="tm",
                incident_wave="plus",
                scattered_wave="minus",
            ),
            PortSpec(
                name="o2",
                monitor_name="o2",
                direction="+x",
                polarization="tm",
                incident_wave="minus",
                scattered_wave="plus",
            ),
        ],
        output_ports=["o1", "o2"],
        frequencies=freqs,
        as_sax=False,
        return_diagnostics=True,
        min_incident_db=-60.0,
    )
    center_idx = int(np.argmin(np.abs(freqs - cfg.frequency_hz)))
    s11 = complex(np.asarray(result["s_matrix"][("o1", "o1")], dtype=np.complex128)[center_idx])
    s21 = complex(np.asarray(result["s_matrix"][("o2", "o1")], dtype=np.complex128)[center_idx])
    diag = result["diagnostics"]
    waves_o1 = diag["waves"]["o1"]
    waves_o2 = diag["waves"]["o2"]
    return CaseResult(
        pml_thickness_wl=float(pml_thickness_wl),
        pml_thickness_um=float(pml_thickness_wl) * cfg.wavelength_um,
        width_um=width_m / 1e-6,
        height_um=height_m / 1e-6,
        dx_nm=dx / 1e-9,
        dt_fs=dt / 1e-15,
        steps_to_decay=int(steps),
        runtime_s=float(runtime_s),
        s11_db_center=20.0 * math.log10(max(abs(s11), 1e-12)),
        s21_db_center=20.0 * math.log10(max(abs(s21), 1e-12)),
        power_sum_center=float(np.asarray(diag["power_sum"], dtype=float)[center_idx]),
        loss_est_center=float(np.asarray(diag["loss_est"], dtype=float)[center_idx]),
        source_dominance_db=_wave_dominance_db(
            waves_o1["a_plus"], waves_o1["a_minus"], "plus"
        ),
        output_dominance_db=_wave_dominance_db(
            waves_o2["a_plus"], waves_o2["a_minus"], "plus"
        ),
        max_tail_ratio=max(_tail_ratio(m_o1, cfg.lookback_records), _tail_ratio(m_o2, cfg.lookback_records)),
        source_tail_ratio=_tail_ratio(m_o1, cfg.lookback_records),
        output_tail_ratio=_tail_ratio(m_o2, cfg.lookback_records),
        pml_formulation=str(cfg.pml_formulation),
    )


def _plot_results(results: list[CaseResult], out_dir: Path) -> Path:
    x = np.asarray([r.pml_thickness_wl for r in results], dtype=float)
    s11 = np.asarray([r.s11_db_center for r in results], dtype=float)
    s21 = np.asarray([r.s21_db_center for r in results], dtype=float)
    power_sum = np.asarray([r.power_sum_center for r in results], dtype=float)
    loss_est = np.asarray([r.loss_est_center for r in results], dtype=float)
    tail = np.asarray([r.max_tail_ratio for r in results], dtype=float)
    steps = np.asarray([r.steps_to_decay for r in results], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.4), dpi=220)
    ax = axes.ravel()

    ax[0].plot(x, s21, marker="o", lw=2.0, label="S21")
    ax[0].plot(x, s11, marker="o", lw=2.0, label="S11")
    ax[0].axhline(0.0, color="black", lw=1.0, ls="--", alpha=0.5)
    ax[0].set_title("Center-Frequency S-Parameters")
    ax[0].set_xlabel("PML thickness (wavelengths)")
    ax[0].set_ylabel("Magnitude (dB)")
    ax[0].grid(alpha=0.3)
    ax[0].legend(frameon=False)

    ax[1].plot(x, power_sum, marker="o", lw=2.0, label="power sum")
    ax[1].plot(x, loss_est, marker="o", lw=2.0, label="loss est")
    ax[1].axhline(1.0, color="black", lw=1.0, ls="--", alpha=0.5)
    ax[1].set_title("Modal Power Accounting")
    ax[1].set_xlabel("PML thickness (wavelengths)")
    ax[1].set_ylabel("Unitless")
    ax[1].grid(alpha=0.3)
    ax[1].legend(frameon=False)

    ax[2].plot(x, tail, marker="o", lw=2.0, color="tab:red")
    ax[2].set_title("Late-Time Tail / Peak")
    ax[2].set_xlabel("PML thickness (wavelengths)")
    ax[2].set_ylabel("Tail ratio")
    ax[2].set_yscale("log")
    ax[2].grid(alpha=0.3, which="both")

    ax[3].plot(x, steps, marker="o", lw=2.0, color="tab:purple")
    ax[3].set_title("Adaptive Steps to Decay")
    ax[3].set_xlabel("PML thickness (wavelengths)")
    ax[3].set_ylabel("Steps")
    ax[3].grid(alpha=0.3)

    fig.suptitle("Straight-Waveguide Absorber Benchmark", fontsize=14)
    fig.tight_layout()
    out_path = out_dir / "straight_waveguide_pml_sweep.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def _parse_thicknesses(raw: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if raw is None or not str(raw).strip():
        return tuple(float(v) for v in default)
    values = []
    for part in str(raw).split(","):
        token = part.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("No PML thicknesses parsed.")
    return tuple(values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep BeamZ's current absorber over PML thickness on a tiny straight-waveguide benchmark."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "results_straight_waveguide_pml_sweep",
    )
    parser.add_argument(
        "--pml-thicknesses-wl",
        type=str,
        default=None,
        help="Comma-separated PML thicknesses in wavelengths, e.g. 0.25,0.5,1.0,1.5",
    )
    parser.add_argument(
        "--formulation",
        type=str,
        default="sponge",
        choices=("sigma", "cpml"),
        help="Boundary absorber formulation to benchmark.",
    )
    parser.add_argument(
        "--cpml-kappa-max",
        type=float,
        default=SweepConfig().cpml_kappa_max,
        help="CPML kappa_max when --formulation=cpml.",
    )
    parser.add_argument(
        "--cpml-alpha-max",
        type=float,
        default=None,
        help="Optional CPML alpha_max override when --formulation=cpml.",
    )
    args = parser.parse_args()

    cfg = SweepConfig(
        pml_formulation=str(args.formulation).lower(),
        cpml_kappa_max=float(args.cpml_kappa_max),
        cpml_alpha_max=(
            None if args.cpml_alpha_max is None else float(args.cpml_alpha_max)
        ),
        pml_thicknesses_wl=_parse_thicknesses(
            args.pml_thicknesses_wl, SweepConfig().pml_thicknesses_wl
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    print("Running straight-waveguide absorber sweep...")
    for thickness_wl in cfg.pml_thicknesses_wl:
        print(f"  thickness={thickness_wl:.2f} λ")
        results.append(run_case(cfg, float(thickness_wl)))

    plot_path = _plot_results(results, args.output_dir)
    manifest = {
        "config": asdict(cfg),
        "results": [asdict(r) for r in results],
        "plot": str(plot_path),
        "notes": {
            "boundary_model": str(cfg.pml_formulation),
            "intended_use": "Regression gate for replacing BeamZ's legacy absorber with a proper UPML/CPML-style implementation.",
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Wrote {manifest_path}")
    print(f"Wrote {plot_path}")
    for row in results:
        print(
            "  "
            f"{row.pml_thickness_wl:.2f} λ: "
            f"S21={row.s21_db_center:.2f} dB, "
            f"S11={row.s11_db_center:.2f} dB, "
            f"power_sum={row.power_sum_center:.4f}, "
            f"tail={row.max_tail_ratio:.3e}"
        )


if __name__ == "__main__":
    main()
