"""Short-time downstream field-symmetry history for a 3D straight guide.

This diagnostic complements the local constitutive benchmark by sampling one
physical monitor plane downstream of a `ModeSource` after selected step counts.
It uses a tiny straight guide, sigma PML, and low PPW so we can see whether the
guide-level field asymmetry is present immediately or only after propagation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from beamz import LIGHT_SPEED, ModeSource, Monitor, PML, Simulation, calc_optimal_fdtd_params, ramped_cosine
from beamz.const import µm

from straight_waveguide_field_symmetry_3d import (
    SweepConfig,
    _axis_pos,
    _build_rectangular_guide_quantized,
    _mirror_metrics,
    _monitor_plane,
    _move_along,
    _plane_axes_for_axis,
    _poynting_from_plane,
)

SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class HistoryConfig:
    wavelength_um: float = 1.55
    ppw: int = 6
    direction: str = "+x"
    polarization: str = "te"
    step_counts: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    monitor_offset_cells: int = 4


def _run_history_case(cfg: HistoryConfig) -> dict[str, object]:
    base_cfg = SweepConfig(wavelength_um=float(cfg.wavelength_um), ppw=int(cfg.ppw))
    axis = str(cfg.direction)[1]
    dx, dt = calc_optimal_fdtd_params(
        base_cfg.wavelength_m,
        base_cfg.n_core,
        dims=3,
        safety_factor=0.9,
        points_per_wavelength=int(base_cfg.ppw),
        width=base_cfg.long_span_m if axis == "x" else base_cfg.transverse_span0_m,
        height=base_cfg.long_span_m if axis == "y" else base_cfg.transverse_span0_m if axis == "z" else base_cfg.transverse_span0_m,
        depth=base_cfg.long_span_m if axis == "z" else base_cfg.transverse_span1_m,
    )
    design, center, source_spans = _build_rectangular_guide_quantized(base_cfg, axis, dx=dx)
    center = list(center)
    axis_idx = _axis_pos(axis)
    axis_len = (design.width, design.height, design.depth)[axis_idx]
    if cfg.direction.startswith("+"):
        center[axis_idx] = base_cfg.pml_m + base_cfg.source_clearance_m
    else:
        center[axis_idx] = axis_len - base_cfg.pml_m - base_cfg.source_clearance_m
    center = tuple(center)

    freq = LIGHT_SPEED / base_cfg.wavelength_m
    t_total = float(base_cfg.run_cycles) / freq
    time = np.arange(0.0, t_total, dt)
    signal = ramped_cosine(
        time,
        amplitude=1.0,
        frequency=freq,
        ramp_duration=float(base_cfg.ramp_cycles) / freq,
        t_max=t_total,
    )
    grid = design.rasterize(resolution=dx)
    source = ModeSource(
        grid=grid,
        center=center,
        width=source_spans[0],
        height=source_spans[1],
        wavelength=base_cfg.wavelength_m,
        pol=str(cfg.polarization),
        signal=signal,
        direction=str(cfg.direction),
    )
    monitor_center = _move_along(center, cfg.direction, float(cfg.monitor_offset_cells) * dx)
    mon_start, mon_end = _monitor_plane(monitor_center, axis, source_spans[0], source_spans[1])
    monitor = Monitor(
        start=mon_start,
        end=mon_end,
        name="m",
        record_fields=True,
        dft_enabled=False,
    )
    sim = Simulation(
        design=design,
        sources=[source],
        monitors=[],
        boundaries=[
            PML(edges=["left", "right", "top", "bottom"], thickness=base_cfg.pml_m, formulation="sigma"),
            PML(edges=["front", "back"], thickness=base_cfg.pml_m, formulation="sigma"),
        ],
        time=time,
        resolution=dx,
    )

    coords0, coords1 = monitor.get_analysis_plane_coords_3d(
        dx=dx,
        dy=dx,
        dz=dx,
        field_shape=tuple(np.asarray(grid.permittivity).shape),
    )
    n0 = int(coords0.size)
    n1 = int(coords1.size)

    samples: list[dict[str, object]] = []
    target_steps = sorted({int(v) for v in cfg.step_counts if int(v) > 0})
    for target in target_steps:
        while int(sim.current_step) < target:
            sim.step()
        monitor.record_fields_3d(
            np.asarray(sim.fields.Ex),
            np.asarray(sim.fields.Ey),
            np.asarray(sim.fields.Ez),
            np.asarray(sim.fields.Hx),
            np.asarray(sim.fields.Hy),
            np.asarray(sim.fields.Hz),
            t=float(sim.t),
            dx=dx,
            dy=dx,
            dz=dx,
            step=int(sim.current_step),
        )
        component_planes = {
            comp: np.asarray(monitor.fields[comp][-1], dtype=np.complex128).reshape(n0, n1)
            for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
        }
        energy = sum(np.abs(arr) ** 2 for arr in component_planes.values())
        poynting = _poynting_from_plane(component_planes, axis)

        comp_axis0 = 0.0
        comp_axis1 = 0.0
        comp_rot = 0.0
        for arr in component_planes.values():
            metrics = _mirror_metrics(np.abs(arr))
            comp_axis0 = max(comp_axis0, metrics["axis0_pct"])
            comp_axis1 = max(comp_axis1, metrics["axis1_pct"])
            comp_rot = max(comp_rot, metrics["rot_pct"])

        samples.append(
            {
                "step": int(sim.current_step),
                "time_fs": float(sim.t) * 1e15,
                "energy": _mirror_metrics(energy),
                "poynting": _mirror_metrics(poynting),
                "max_component_mag": {
                    "axis0_pct": float(comp_axis0),
                    "axis1_pct": float(comp_axis1),
                    "rot_pct": float(comp_rot),
                },
            }
        )

    axis0, axis1 = _plane_axes_for_axis(axis)
    return {
        "config": asdict(cfg),
        "resolution_um": float(dx / µm),
        "grid_shape": tuple(int(v) for v in np.asarray(grid.permittivity).shape),
        "monitor_axis0": axis0,
        "monitor_axis1": axis1,
        "monitor_shape": (n0, n1),
        "samples": samples,
    }


def _plot_report(report: dict[str, object], out_path: Path) -> None:
    samples = report["samples"]
    steps = [entry["step"] for entry in samples]
    energy0 = [entry["energy"]["axis0_pct"] for entry in samples]
    energy1 = [entry["energy"]["axis1_pct"] for entry in samples]
    p0 = [entry["poynting"]["axis0_pct"] for entry in samples]
    p1 = [entry["poynting"]["axis1_pct"] for entry in samples]
    c0 = [entry["max_component_mag"]["axis0_pct"] for entry in samples]
    c1 = [entry["max_component_mag"]["axis1_pct"] for entry in samples]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
    axes[0].plot(steps, energy0, marker="o", label=report["monitor_axis0"])
    axes[0].plot(steps, energy1, marker="o", label=report["monitor_axis1"])
    axes[0].set_title("Energy Mirror Asymmetry")
    axes[0].set_ylabel("%")
    axes[0].legend()

    axes[1].plot(steps, p0, marker="o", label=report["monitor_axis0"])
    axes[1].plot(steps, p1, marker="o", label=report["monitor_axis1"])
    axes[1].set_title("Poynting Mirror Asymmetry")
    axes[1].legend()

    axes[2].plot(steps, c0, marker="o", label=report["monitor_axis0"])
    axes[2].plot(steps, c1, marker="o", label=report["monitor_axis1"])
    axes[2].set_title("Worst Component Magnitude Asymmetry")
    axes[2].legend()

    for ax in axes:
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample downstream straight-guide field symmetry after selected step counts.")
    parser.add_argument("--ppw", type=int, default=6)
    parser.add_argument("--wavelength-um", type=float, default=1.55)
    parser.add_argument("--direction", default="+x")
    parser.add_argument("--polarization", default="te")
    parser.add_argument("--step-counts", default="1,2,4,8,16,32")
    parser.add_argument("--monitor-offset-cells", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "results_straight_waveguide_step_symmetry_3d",
    )
    args = parser.parse_args()

    step_counts = tuple(int(part) for part in str(args.step_counts).split(",") if part.strip())
    cfg = HistoryConfig(
        wavelength_um=float(args.wavelength_um),
        ppw=int(args.ppw),
        direction=str(args.direction),
        polarization=str(args.polarization),
        step_counts=step_counts,
        monitor_offset_cells=int(args.monitor_offset_cells),
    )
    report = _run_history_case(cfg)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "straight_waveguide_step_symmetry_3d.json"
    png_path = args.output_dir / "straight_waveguide_step_symmetry_3d.png"
    json_path.write_text(json.dumps(report, indent=2))
    _plot_report(report, png_path)
    print(json.dumps({"report": str(json_path.resolve()), "plot": str(png_path.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
