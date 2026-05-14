"""3D straight-waveguide DFT-field symmetry probe for ModeSource.

This benchmark isolates source/field asymmetry before any modal S-parameter
extraction. It launches a low-resolution 3D ModeSource into a small straight
guide, samples one downstream DFT monitor plane, and measures mirror symmetry
of the monitor-plane energy density and axial Poynting for all directions and
both polarizations.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np

from beamz import (
    LIGHT_SPEED,
    Design,
    Material,
    ModeSource,
    Monitor,
    PML,
    Rectangle,
    Simulation,
    calc_optimal_fdtd_params,
    ramped_cosine,
    µm,
)

SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SweepConfig:
    wavelength_um: float = 1.55
    ppw: int = 6
    n_core: float = 2.0
    n_clad: float = 1.0
    guide_span0_um: float = 0.55 * 1.55
    guide_span1_um: float = 0.35 * 1.55
    long_span_um: float = 5.5 * 1.55
    transverse_span0_um: float = 2.2 * 1.55
    transverse_span1_um: float = 2.0 * 1.55
    pml_um: float = 0.8 * 1.55
    source_clearance_um: float = 1.3 * 1.55
    monitor_offset_um: float = 0.9 * 1.55
    source_span_scale: float = 3.0
    run_cycles: float = 8.0
    ramp_cycles: float = 1.0
    directions: tuple[str, ...] = (
        "+x",
        "-x",
        "+y",
        "-y",
        "+z",
        "-z",
    )
    polarizations: tuple[str, ...] = ("te", "tm")

    @property
    def wavelength_m(self) -> float:
        return float(self.wavelength_um) * 1e-6

    @property
    def guide_span0_m(self) -> float:
        return float(self.guide_span0_um) * 1e-6

    @property
    def guide_span1_m(self) -> float:
        return float(self.guide_span1_um) * 1e-6

    @property
    def long_span_m(self) -> float:
        return float(self.long_span_um) * 1e-6

    @property
    def transverse_span0_m(self) -> float:
        return float(self.transverse_span0_um) * 1e-6

    @property
    def transverse_span1_m(self) -> float:
        return float(self.transverse_span1_um) * 1e-6

    @property
    def pml_m(self) -> float:
        return float(self.pml_um) * 1e-6

    @property
    def monitor_offset_m(self) -> float:
        return float(self.monitor_offset_um) * 1e-6

    @property
    def source_clearance_m(self) -> float:
        return float(self.source_clearance_um) * 1e-6


@dataclass(frozen=True)
class CaseResult:
    direction: str
    polarization: str
    ppw: int
    resolution_um: float
    grid_shape: tuple[int, int, int]
    monitor_axis0: str
    monitor_axis1: str
    monitor_shape: tuple[int, int]
    source_h_sym_axis0_pct: float
    source_h_sym_axis1_pct: float
    source_h_sym_rot_pct: float
    energy_sym_axis0_pct: float
    energy_sym_axis1_pct: float
    energy_sym_rot_pct: float
    poynting_sym_axis0_pct: float
    poynting_sym_axis1_pct: float
    poynting_sym_rot_pct: float
    max_component_mag_sym_axis0_pct: float
    max_component_mag_sym_axis1_pct: float
    max_component_mag_sym_rot_pct: float


def _case_slug(direction: str, pol: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_+-]+", "_", f"{direction}_{pol}").replace("+", "plus").replace("-", "minus")


def _plane_axes_for_axis(axis: str) -> tuple[str, str]:
    mapping = {
        "x": ("z", "y"),
        "y": ("z", "x"),
        "z": ("y", "x"),
    }
    return mapping[str(axis).lower()]


def _poynting_from_plane(components: dict[str, np.ndarray], axis: str) -> np.ndarray:
    axis = str(axis).lower()
    if axis == "x":
        return np.real(
            components["Ey"] * np.conjugate(components["Hz"])
            - components["Ez"] * np.conjugate(components["Hy"])
        )
    if axis == "y":
        return np.real(
            components["Ez"] * np.conjugate(components["Hx"])
            - components["Ex"] * np.conjugate(components["Hz"])
        )
    return np.real(
        components["Ex"] * np.conjugate(components["Hy"])
        - components["Ey"] * np.conjugate(components["Hx"])
    )


def _relative_symmetry_pct(values: np.ndarray, transformed: np.ndarray) -> float:
    ref = max(float(np.linalg.norm(values)), 1e-30)
    return 100.0 * float(np.linalg.norm(values - transformed)) / ref


def _mirror_metrics(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values)
    return {
        "axis0_pct": _relative_symmetry_pct(arr, np.flip(arr, axis=0)),
        "axis1_pct": _relative_symmetry_pct(arr, np.flip(arr, axis=1)),
        "rot_pct": _relative_symmetry_pct(arr, np.flip(arr, axis=(0, 1))),
    }


def _component_symmetry_report(component_planes: dict[str, np.ndarray]) -> dict[str, dict[str, dict[str, float]]]:
    report: dict[str, dict[str, dict[str, float]]] = {}
    for name, arr in component_planes.items():
        report[name] = {
            "real": _mirror_metrics(np.real(arr)),
            "imag": _mirror_metrics(np.imag(arr)),
            "magnitude": _mirror_metrics(np.abs(arr)),
        }
    return report


def _plot_component_planes(
    component_planes: dict[str, np.ndarray],
    *,
    out_path: Path,
    title: str,
) -> None:
    components = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    fig, axes = plt.subplots(len(components), 3, figsize=(11.5, 14.0), dpi=220)
    fig.suptitle(title, fontsize=14)
    for row, comp in enumerate(components):
        arr = np.asarray(component_planes[comp], dtype=np.complex128)
        real = np.real(arr)
        imag = np.imag(arr)
        mag = np.abs(arr)
        real_lim = max(float(np.max(np.abs(real))), 1e-30)
        imag_lim = max(float(np.max(np.abs(imag))), 1e-30)
        images = (
            (real, (-real_lim, real_lim), "RdBu_r", f"{comp} real"),
            (imag, (-imag_lim, imag_lim), "RdBu_r", f"{comp} imag"),
            (mag, None, "viridis", f"{comp} |.|"),
        )
        for col, (data, clim, cmap, label) in enumerate(images):
            ax = axes[row, col]
            im = ax.imshow(data, origin="lower", cmap=cmap, aspect="auto")
            if clim is not None:
                im.set_clim(*clim)
            ax.set_title(label, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    fig.subplots_adjust(top=0.965)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _save_case_artifacts(
    out_dir: Path,
    result: CaseResult,
    coords0: np.ndarray,
    coords1: np.ndarray,
    component_planes: dict[str, np.ndarray],
) -> dict[str, str]:
    case_dir = out_dir / _case_slug(result.direction, result.polarization)
    case_dir.mkdir(parents=True, exist_ok=True)
    npz_path = case_dir / "component_planes.npz"
    np.savez_compressed(
        npz_path,
        coords0=np.asarray(coords0, dtype=float),
        coords1=np.asarray(coords1, dtype=float),
        **{name: np.asarray(arr, dtype=np.complex128) for name, arr in component_planes.items()},
    )
    report = {
        "case": asdict(result),
        "components": _component_symmetry_report(component_planes),
    }
    report_path = case_dir / "component_symmetry.json"
    report_path.write_text(json.dumps(report, indent=2))
    panel_path = case_dir / "component_planes.png"
    _plot_component_planes(
        component_planes,
        out_path=panel_path,
        title=f"{result.direction}/{result.polarization} DFT monitor planes",
    )
    return {
        "case_dir": str(case_dir),
        "npz": str(npz_path),
        "report": str(report_path),
        "panel": str(panel_path),
    }


def _support_parity_metrics(fields, source: ModeSource, component_name: str, index_attr: str) -> dict[str, float]:
    arr = np.asarray(getattr(fields, component_name), dtype=np.complex128)[getattr(source, index_attr)]
    mag = np.abs(arr)
    return _mirror_metrics(mag)


def _source_h_injection_metrics(sim: Simulation, source: ModeSource) -> dict[str, float]:
    source.inject_h(
        sim.fields,
        t=0.0,
        dt=float(sim.dt),
        current_step=0,
        resolution=float(sim.resolution),
        design=sim.design,
    )
    axis0 = 0.0
    axis1 = 0.0
    rot = 0.0
    for comp, idx_attr in (("Hx", "_Hx_indices"), ("Hy", "_Hy_indices"), ("Hz", "_Hz_indices")):
        metrics = _support_parity_metrics(sim.fields, source, comp, idx_attr)
        axis0 = max(axis0, float(metrics["axis0_pct"]))
        axis1 = max(axis1, float(metrics["axis1_pct"]))
        rot = max(rot, float(metrics["rot_pct"]))
    sim.fields.Hx = 0.0 * sim.fields.Hx
    sim.fields.Hy = 0.0 * sim.fields.Hy
    sim.fields.Hz = 0.0 * sim.fields.Hz
    return {
        "axis0_pct": axis0,
        "axis1_pct": axis1,
        "rot_pct": rot,
    }


def _build_rectangular_guide(cfg: SweepConfig, axis: str) -> tuple[Design, tuple[float, float, float], tuple[float, float]]:
    raise RuntimeError("Use _build_rectangular_guide_quantized with an explicit resolution.")


def _quantize_cells(target_m: float, dx: float, *, min_cells: int = 2, parity: int | None = None) -> int:
    cells = max(int(min_cells), int(round(float(target_m) / float(dx))))
    if parity is None:
        return cells
    parity = int(parity) & 1
    if cells % 2 == parity:
        return cells
    if cells <= int(min_cells):
        return cells + 1
    lower = cells - 1
    upper = cells + 1
    if lower >= int(min_cells) and (lower % 2) == parity:
        return lower
    return upper


def _build_rectangular_guide_quantized(
    cfg: SweepConfig,
    axis: str,
    *,
    dx: float,
) -> tuple[Design, tuple[float, float, float], tuple[float, float]]:
    g0_cells_target = _quantize_cells(cfg.guide_span0_m, dx, min_cells=2)
    g1_cells_target = _quantize_cells(cfg.guide_span1_m, dx, min_cells=2)
    n_core = cfg.n_core
    n_clad = cfg.n_clad

    if axis == "x":
        width_cells = _quantize_cells(cfg.long_span_m, dx, min_cells=12)
        height_cells = _quantize_cells(cfg.transverse_span0_m, dx, min_cells=g0_cells_target + 8)
        depth_cells = _quantize_cells(cfg.transverse_span1_m, dx, min_cells=g1_cells_target + 8)
        g0_cells = _quantize_cells(cfg.guide_span0_m, dx, min_cells=2, parity=height_cells % 2)
        g1_cells = _quantize_cells(cfg.guide_span1_m, dx, min_cells=2, parity=depth_cells % 2)
        width = width_cells * dx
        height = height_cells * dx
        depth = depth_cells * dx
        g0 = g0_cells * dx
        g1 = g1_cells * dx
        center = (0.5 * width, 0.5 * height, 0.5 * depth)
        core = Rectangle(
            position=(0.0, center[1] - 0.5 * g0, center[2] - 0.5 * g1),
            width=width,
            height=g0,
            depth=g1,
            material=Material(n_core**2),
        )
        span0_cells = _quantize_cells(
            cfg.source_span_scale * g0,
            dx,
            min_cells=g0_cells + 4,
            parity=height_cells % 2,
        )
        span1_cells = _quantize_cells(
            cfg.source_span_scale * g1,
            dx,
            min_cells=g1_cells + 4,
            parity=depth_cells % 2,
        )
        source_spans = (span0_cells * dx, span1_cells * dx)
    elif axis == "y":
        width_cells = _quantize_cells(cfg.transverse_span0_m, dx, min_cells=g0_cells_target + 8)
        height_cells = _quantize_cells(cfg.long_span_m, dx, min_cells=12)
        depth_cells = _quantize_cells(cfg.transverse_span1_m, dx, min_cells=g1_cells_target + 8)
        g0_cells = _quantize_cells(cfg.guide_span0_m, dx, min_cells=2, parity=width_cells % 2)
        g1_cells = _quantize_cells(cfg.guide_span1_m, dx, min_cells=2, parity=depth_cells % 2)
        width = width_cells * dx
        height = height_cells * dx
        depth = depth_cells * dx
        g0 = g0_cells * dx
        g1 = g1_cells * dx
        center = (0.5 * width, 0.5 * height, 0.5 * depth)
        core = Rectangle(
            position=(center[0] - 0.5 * g0, 0.0, center[2] - 0.5 * g1),
            width=g0,
            height=height,
            depth=g1,
            material=Material(n_core**2),
        )
        span0_cells = _quantize_cells(
            cfg.source_span_scale * g0,
            dx,
            min_cells=g0_cells + 4,
            parity=width_cells % 2,
        )
        span1_cells = _quantize_cells(
            cfg.source_span_scale * g1,
            dx,
            min_cells=g1_cells + 4,
            parity=depth_cells % 2,
        )
        source_spans = (span0_cells * dx, span1_cells * dx)
    else:
        width_cells = _quantize_cells(cfg.transverse_span0_m, dx, min_cells=g0_cells_target + 8)
        height_cells = _quantize_cells(cfg.transverse_span1_m, dx, min_cells=g1_cells_target + 8)
        depth_cells = _quantize_cells(cfg.long_span_m, dx, min_cells=12)
        g0_cells = _quantize_cells(cfg.guide_span0_m, dx, min_cells=2, parity=width_cells % 2)
        g1_cells = _quantize_cells(cfg.guide_span1_m, dx, min_cells=2, parity=height_cells % 2)
        width = width_cells * dx
        height = height_cells * dx
        depth = depth_cells * dx
        g0 = g0_cells * dx
        g1 = g1_cells * dx
        center = (0.5 * width, 0.5 * height, 0.5 * depth)
        core = Rectangle(
            position=(center[0] - 0.5 * g0, center[1] - 0.5 * g1, 0.0),
            width=g0,
            height=g1,
            depth=depth,
            material=Material(n_core**2),
        )
        span0_cells = _quantize_cells(
            cfg.source_span_scale * g1,
            dx,
            min_cells=g1_cells + 4,
            parity=height_cells % 2,
        )
        span1_cells = _quantize_cells(
            cfg.source_span_scale * g0,
            dx,
            min_cells=g0_cells + 4,
            parity=width_cells % 2,
        )
        source_spans = (span0_cells * dx, span1_cells * dx)

    design = Design(
        width=width,
        height=height,
        depth=depth,
        material=Material(n_clad**2),
    )
    design += core
    return design, center, source_spans


def _move_along(center: tuple[float, float, float], direction: str, distance: float) -> tuple[float, float, float]:
    x, y, z = center
    if direction == "+x":
        return (x + distance, y, z)
    if direction == "-x":
        return (x - distance, y, z)
    if direction == "+y":
        return (x, y + distance, z)
    if direction == "-y":
        return (x, y - distance, z)
    if direction == "+z":
        return (x, y, z + distance)
    return (x, y, z - distance)


def _axis_pos(axis: str) -> int:
    return {"x": 0, "y": 1, "z": 2}[str(axis).lower()]


def _monitor_plane(center: tuple[float, float, float], axis: str, span0: float, span1: float):
    x, y, z = center
    if axis == "x":
        return (x, y - 0.5 * span0, z - 0.5 * span1), (x, y + 0.5 * span0, z + 0.5 * span1)
    if axis == "y":
        return (x - 0.5 * span0, y, z - 0.5 * span1), (x + 0.5 * span0, y, z + 0.5 * span1)
    return (x - 0.5 * span1, y - 0.5 * span0, z), (x + 0.5 * span1, y + 0.5 * span0, z)


def run_case(
    cfg: SweepConfig,
    direction: str,
    pol: str,
    *,
    output_dir: Path | None = None,
    save_component_planes: bool = False,
) -> tuple[CaseResult, dict[str, str] | None]:
    axis = direction[1]
    dx, dt = calc_optimal_fdtd_params(
        cfg.wavelength_m,
        cfg.n_core,
        dims=3,
        safety_factor=0.9,
        points_per_wavelength=int(cfg.ppw),
        width=cfg.long_span_m if axis == "x" else cfg.transverse_span0_m,
        height=cfg.long_span_m if axis == "y" else cfg.transverse_span0_m if axis == "z" else cfg.transverse_span0_m,
        depth=cfg.long_span_m if axis == "z" else cfg.transverse_span1_m,
    )
    design, center, source_spans = _build_rectangular_guide_quantized(cfg, axis, dx=dx)
    center = list(center)
    axis_idx = _axis_pos(axis)
    axis_len = (design.width, design.height, design.depth)[axis_idx]
    if direction.startswith("+"):
        center[axis_idx] = cfg.pml_m + cfg.source_clearance_m
    else:
        center[axis_idx] = axis_len - cfg.pml_m - cfg.source_clearance_m
    center = tuple(center)
    freq = LIGHT_SPEED / cfg.wavelength_m
    t_total = float(cfg.run_cycles) / freq
    time = np.arange(0.0, t_total, dt)
    signal = ramped_cosine(
        time,
        amplitude=1.0,
        frequency=freq,
        ramp_duration=float(cfg.ramp_cycles) / freq,
        t_max=t_total,
    )
    grid = design.rasterize(resolution=dx)
    source = ModeSource(
        grid=grid,
        center=center,
        width=source_spans[0],
        height=source_spans[1],
        wavelength=cfg.wavelength_m,
        pol=pol,
        signal=signal,
        direction=direction,
    )
    monitor_center = _move_along(center, direction, cfg.monitor_offset_m)
    mon_start, mon_end = _monitor_plane(monitor_center, axis, source_spans[0], source_spans[1])
    monitor = Monitor(
        start=mon_start,
        end=mon_end,
        name="m",
        record_fields=False,
        dft_enabled=True,
        dft_frequencies=np.asarray([freq], dtype=float),
        dft_components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        dft_window="none",
        dft_record_every_step=True,
    )
    sim = Simulation(
        design=design,
        sources=[source],
        monitors=[monitor],
        boundaries=[
            PML(edges=["left", "right", "top", "bottom"], thickness=cfg.pml_m, formulation="sponge"),
            PML(edges=["front", "back"], thickness=cfg.pml_m, formulation="sponge"),
        ],
        time=time,
        resolution=dx,
    )
    source_h_metrics = _source_h_injection_metrics(sim, source)
    sim.run_compiled(progress=False)

    coords0, coords1 = monitor.get_analysis_plane_coords_3d(
        dx=dx,
        dy=dx,
        dz=dx,
        field_shape=tuple(np.asarray(grid.permittivity).shape),
    )
    n0 = int(coords0.size)
    n1 = int(coords1.size)
    component_planes = {
        comp: np.asarray(monitor.get_dft_component(comp), dtype=np.complex128)[0].reshape(n0, n1)
        for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    }
    energy = sum(np.abs(arr) ** 2 for arr in component_planes.values())
    poynting = _poynting_from_plane(component_planes, axis)
    energy_metrics = _mirror_metrics(energy)
    poynting_metrics = _mirror_metrics(poynting)

    comp_axis0 = 0.0
    comp_axis1 = 0.0
    comp_rot = 0.0
    for arr in component_planes.values():
        mag = np.abs(arr)
        metrics = _mirror_metrics(mag)
        comp_axis0 = max(comp_axis0, metrics["axis0_pct"])
        comp_axis1 = max(comp_axis1, metrics["axis1_pct"])
        comp_rot = max(comp_rot, metrics["rot_pct"])

    axis0, axis1 = _plane_axes_for_axis(axis)
    result = CaseResult(
        direction=str(direction),
        polarization=str(pol),
        ppw=int(cfg.ppw),
        resolution_um=float(dx / µm),
        grid_shape=tuple(int(v) for v in np.asarray(grid.permittivity).shape),
        monitor_axis0=axis0,
        monitor_axis1=axis1,
        monitor_shape=(n0, n1),
        source_h_sym_axis0_pct=float(source_h_metrics["axis0_pct"]),
        source_h_sym_axis1_pct=float(source_h_metrics["axis1_pct"]),
        source_h_sym_rot_pct=float(source_h_metrics["rot_pct"]),
        energy_sym_axis0_pct=float(energy_metrics["axis0_pct"]),
        energy_sym_axis1_pct=float(energy_metrics["axis1_pct"]),
        energy_sym_rot_pct=float(energy_metrics["rot_pct"]),
        poynting_sym_axis0_pct=float(poynting_metrics["axis0_pct"]),
        poynting_sym_axis1_pct=float(poynting_metrics["axis1_pct"]),
        poynting_sym_rot_pct=float(poynting_metrics["rot_pct"]),
        max_component_mag_sym_axis0_pct=float(comp_axis0),
        max_component_mag_sym_axis1_pct=float(comp_axis1),
        max_component_mag_sym_rot_pct=float(comp_rot),
    )
    artifacts = None
    if save_component_planes and output_dir is not None:
        artifacts = _save_case_artifacts(
            output_dir,
            result,
            np.asarray(coords0, dtype=float),
            np.asarray(coords1, dtype=float),
            component_planes,
        )
    return result, artifacts


def plot_results(results: list[CaseResult], out_dir: Path) -> Path:
    labels = [f"{r.direction}/{r.polarization}" for r in results]
    x = np.arange(len(results), dtype=float)
    energy = [max(r.energy_sym_axis0_pct, r.energy_sym_axis1_pct) for r in results]
    poynt = [max(r.poynting_sym_axis0_pct, r.poynting_sym_axis1_pct) for r in results]
    comp = [max(r.max_component_mag_sym_axis0_pct, r.max_component_mag_sym_axis1_pct) for r in results]

    fig, axes = plt.subplots(3, 1, figsize=(11.0, 7.5), dpi=220, sharex=True)
    axes[0].bar(x, energy, color="#1f77b4")
    axes[0].set_ylabel("Percent")
    axes[0].set_title("Energy-Density Mirror Asymmetry")
    axes[0].grid(alpha=0.3, axis="y")
    axes[1].bar(x, poynt, color="#ff7f0e")
    axes[1].set_ylabel("Percent")
    axes[1].set_title("Axial Poynting Mirror Asymmetry")
    axes[1].grid(alpha=0.3, axis="y")
    axes[2].bar(x, comp, color="#2ca02c")
    axes[2].set_ylabel("Percent")
    axes[2].set_title("Worst Component-Magnitude Mirror Asymmetry")
    axes[2].grid(alpha=0.3, axis="y")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    out_path = out_dir / "straight_waveguide_field_symmetry_3d.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "results_straight_waveguide_field_symmetry_3d",
    )
    parser.add_argument("--ppw", type=int, default=6)
    parser.add_argument("--directions", type=str, default="+x,-x,+y,-y,+z,-z")
    parser.add_argument("--polarizations", type=str, default="te,tm")
    parser.add_argument("--save-component-planes", action="store_true")
    args = parser.parse_args()

    cfg = SweepConfig(
        ppw=int(args.ppw),
        directions=tuple(tok.strip() for tok in args.directions.split(",") if tok.strip()),
        polarizations=tuple(tok.strip() for tok in args.polarizations.split(",") if tok.strip()),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[CaseResult] = []
    artifacts: dict[str, dict[str, str]] = {}
    for direction in cfg.directions:
        for pol in cfg.polarizations:
            result, case_artifacts = run_case(
                cfg,
                direction,
                pol,
                output_dir=args.output_dir,
                save_component_planes=bool(args.save_component_planes),
            )
            results.append(result)
            if case_artifacts is not None:
                artifacts[f"{direction}/{pol}"] = case_artifacts
            print(
                "{d}/{p}: source={src:.2f}% energy={e:.2f}% poynt={s:.2f}% comp={c:.2f}%".format(
                    d=direction,
                    p=pol,
                    src=max(result.source_h_sym_axis0_pct, result.source_h_sym_axis1_pct),
                    e=max(result.energy_sym_axis0_pct, result.energy_sym_axis1_pct),
                    s=max(result.poynting_sym_axis0_pct, result.poynting_sym_axis1_pct),
                    c=max(
                        result.max_component_mag_sym_axis0_pct,
                        result.max_component_mag_sym_axis1_pct,
                    ),
                )
            )

    plot_path = plot_results(results, args.output_dir)
    manifest = {
        "config": asdict(cfg),
        "results": [asdict(r) for r in results],
        "plot": str(plot_path),
    }
    if artifacts:
        manifest["artifacts"] = artifacts
    manifest_path = args.output_dir / "straight_waveguide_field_symmetry_3d.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {manifest_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
