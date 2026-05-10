from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
WRONG_REPO_ROOT = REPO_ROOT.parent / "beamz"
sys.path = [
    p
    for p in sys.path
    if Path(p).resolve() not in {WRONG_REPO_ROOT, WRONG_REPO_ROOT.parent}
]
sys.path.insert(0, str(REPO_ROOT))
for module_name in list(sys.modules):
    if module_name == "beamz" or module_name.startswith("beamz."):
        del sys.modules[module_name]

SCRIPT_DIR = Path(__file__).resolve().parent
C0 = 299_792_458.0


@dataclass(frozen=True)
class WaveguideBenchmarkConfig:
    domain_x_um: float = 4.0
    domain_y_um: float = 2.2
    wavelength_um: float = 1.55
    resolution_ppw: int = 10
    courant_safety: float = 0.95
    total_cycles: float = 3.0
    pulse_center_cycles: float = 0.90
    pulse_sigma_cycles: float = 0.25
    clad_index: float = 1.44
    core_index: float = 3.47
    waveguide_width_um: float = 0.50
    pml_um: float = 0.35
    source_x_um: float = 1.05
    source_y_um: float = 1.10
    monitor_x_um: float = 2.45
    source_width_um: float = 0.10
    monitor_span_um: float = 1.10
    meep_source_amplitude_scale: float = 1.0
    error_tolerance: float = 1e-3
    geometry_source: str = "beamz-raster"
    num_snapshots: int = 4

    @property
    def dx_m(self) -> float:
        return (self.wavelength_um * 1e-6) / float(self.resolution_ppw)

    @property
    def frequency_hz(self) -> float:
        return C0 / (self.wavelength_um * 1e-6)

    @property
    def frequency_um_inv(self) -> float:
        return 1.0 / float(self.wavelength_um)

    @property
    def dt_s(self) -> float:
        return self.courant_safety * self.dx_m / (C0 * math.sqrt(2.0))

    @property
    def total_time_s(self) -> float:
        return self.total_cycles / self.frequency_hz

    @property
    def pulse_center_s(self) -> float:
        return self.pulse_center_cycles / self.frequency_hz

    @property
    def pulse_sigma_s(self) -> float:
        return self.pulse_sigma_cycles / self.frequency_hz

    @property
    def num_steps(self) -> int:
        return max(4, int(math.floor(self.total_time_s / self.dt_s)))

    @property
    def snapshot_steps(self) -> list[int]:
        raw = np.linspace(0.2, 0.8, self.num_snapshots)
        out: list[int] = []
        for frac in raw:
            step = int(round(frac * self.num_steps))
            step = min(max(step, 1), self.num_steps)
            if not out or step != out[-1]:
                out.append(step)
        return out

    @property
    def monitor_y0_um(self) -> float:
        return 0.5 * self.domain_y_um - 0.5 * self.monitor_span_um

    @property
    def monitor_y1_um(self) -> float:
        return 0.5 * self.domain_y_um + 0.5 * self.monitor_span_um


def _gaussian_modulated_signal_value(cfg: WaveguideBenchmarkConfig, t_s: float) -> float:
    envelope = math.exp(-0.5 * ((t_s - cfg.pulse_center_s) / cfg.pulse_sigma_s) ** 2)
    carrier = math.cos(2.0 * math.pi * cfg.frequency_hz * t_s)
    return float(envelope * carrier)


def _gaussian_modulated_signal_samples(cfg: WaveguideBenchmarkConfig) -> np.ndarray:
    t = np.arange(cfg.num_steps + 2, dtype=np.float64) * cfg.dt_s
    return np.asarray(
        [_gaussian_modulated_signal_value(cfg, float(tt)) for tt in t],
        dtype=np.float32,
    )


def _meep_signal_function(cfg: WaveguideBenchmarkConfig):
    def _signal(t_um: float) -> float:
        t_s = float(t_um) * 1e-6 / C0
        return _gaussian_modulated_signal_value(cfg, t_s)

    return _signal


def _meep_native_source_coords_2d(cfg: WaveguideBenchmarkConfig) -> dict[str, np.ndarray]:
    dx_um = cfg.dx_m * 1e6
    nx = int(round(cfg.domain_x_um / dx_um))
    ny = int(round(cfg.domain_y_um / dx_um))
    return {
        "y": np.arange(ny + 1, dtype=np.float64) * dx_um,
        "x": np.arange(nx + 1, dtype=np.float64) * dx_um,
    }


def _beamz_native_tm_coords_2d(cfg: WaveguideBenchmarkConfig) -> dict[str, np.ndarray]:
    dx_um = cfg.dx_m * 1e6
    nx = int(round(cfg.domain_x_um / dx_um))
    ny = int(round(cfg.domain_y_um / dx_um))
    return {
        "y": np.arange(ny + 1, dtype=np.float64) * dx_um,
        "x": np.arange(nx + 1, dtype=np.float64) * dx_um,
    }


def _support_pixels_2d(
    coords: dict[str, np.ndarray],
    *,
    center_um: tuple[float, float],
    width_um: float,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    x0, y0 = center_um
    sigma = float(width_um)
    x = np.asarray(coords["x"], dtype=np.float64)
    y = np.asarray(coords["y"], dtype=np.float64)
    dx_um = float(np.median(np.diff(x)))
    radius_cells = int(np.ceil(4.0 * sigma / dx_um))

    cx = int(np.argmin(np.abs(x - x0)))
    cy = int(np.argmin(np.abs(y - y0)))
    x_start, x_end = max(0, cx - radius_cells), min(x.size, cx + radius_cells + 1)
    y_start, y_end = max(0, cy - radius_cells), min(y.size, cy + radius_cells + 1)
    yy, xx = np.meshgrid(y[y_start:y_end], x[x_start:x_end], indexing="ij")
    dist_sq = (xx - x0) ** 2 + (yy - y0) ** 2
    weights = np.exp(-dist_sq / (2.0 * sigma**2))
    pixels: list[tuple[int, int]] = []
    for iy in range(y_start, y_end):
        for ix in range(x_start, x_end):
            pixels.append((iy, ix))
    return pixels, np.asarray(weights, dtype=np.float64).reshape(-1)


class _BeamzCurrentSource2D:
    def __init__(
        self,
        *,
        signal,
        pixel_indices: list[tuple[int, int]],
        pixel_weights: np.ndarray,
    ) -> None:
        self.signal = signal
        self._pixel_weights = np.asarray(pixel_weights, dtype=np.float32)
        y_idx = np.asarray([idx[0] for idx in pixel_indices], dtype=np.int32)
        x_idx = np.asarray([idx[1] for idx in pixel_indices], dtype=np.int32)
        self._indices = (y_idx, x_idx)

    def get_source_terms(self, fields, t, dt, current_step, resolution, design):
        del fields, current_step, resolution, design
        signal_value = float(self.signal(float(t) + 0.5 * float(dt)))
        values = -self._pixel_weights * signal_value
        return {"Ez": (values, self._indices)}, {}


def _waveguide_structure_spec(cfg: WaveguideBenchmarkConfig) -> list[dict[str, Any]]:
    return [
        {
            "type": "block",
            "position_um": [0.0, 0.5 * (cfg.domain_y_um - cfg.waveguide_width_um)],
            "size_um": [cfg.domain_x_um, cfg.waveguide_width_um],
            "index": cfg.core_index,
            "permittivity": cfg.core_index**2,
        }
    ]


def _build_beamz_design(cfg: WaveguideBenchmarkConfig):
    from beamz import Design, Material, Rectangle

    design = Design(
        width=cfg.domain_x_um * 1e-6,
        height=cfg.domain_y_um * 1e-6,
        material=Material(cfg.clad_index**2),
    )
    design += Rectangle(
        position=(
            0.0,
            0.5 * (cfg.domain_y_um - cfg.waveguide_width_um) * 1e-6,
        ),
        width=cfg.domain_x_um * 1e-6,
        height=cfg.waveguide_width_um * 1e-6,
        material=Material(cfg.core_index**2),
    )
    return design


def _rasterize_beamz_permittivity(cfg: WaveguideBenchmarkConfig) -> np.ndarray:
    from beamz import PML, Simulation

    design = _build_beamz_design(cfg)
    sim = Simulation(
        design=design,
        sources=[],
        monitors=[],
        boundaries=[PML(edges="all", thickness=cfg.pml_um * 1e-6)],
        time=np.asarray([0.0, cfg.dt_s], dtype=np.float64),
        resolution=cfg.dx_m,
    )
    return np.asarray(sim.fields.permittivity, dtype=np.float32)


def _monitor_line_endpoints_m(
    cfg: WaveguideBenchmarkConfig,
) -> tuple[tuple[float, float], tuple[float, float]]:
    return (
        (cfg.monitor_x_um * 1e-6, cfg.monitor_y0_um * 1e-6),
        (cfg.monitor_x_um * 1e-6, cfg.monitor_y1_um * 1e-6),
    )


def _monitor_sample_geometry(
    cfg: WaveguideBenchmarkConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    dx_um = cfg.dx_m * 1e6
    nx = int(round(cfg.domain_x_um / dx_um))
    ny = int(round(cfg.domain_y_um / dx_um))
    x_idx = int(round((cfg.monitor_x_um / dx_um) - 0.5))
    x_idx = max(0, min(x_idx, nx - 1))

    eps = 1e-12 * max(abs(cfg.monitor_y0_um), abs(cfg.monitor_y1_um), abs(dx_um), 1.0)
    y_start = int(math.floor((cfg.monitor_y0_um + eps) / dx_um))
    y_stop = int(math.ceil((cfg.monitor_y1_um - eps) / dx_um))
    y_start = max(0, min(y_start, ny))
    y_stop = max(y_start + 1, min(y_stop, ny))
    y_idx = np.arange(y_start, y_stop, dtype=np.int32)
    x_um = (np.full(y_idx.shape, x_idx, dtype=np.float64) + 0.5) * dx_um
    y_um = (y_idx.astype(np.float64) + 0.5) * dx_um
    return x_um, y_um, y_idx, x_idx


def _fit_real_scale(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64).ravel()
    cand = np.asarray(candidate, dtype=np.float64).ravel()
    denom = float(np.dot(cand, cand))
    if denom <= 1e-30:
        return 1.0
    return float(np.dot(ref, cand) / denom)


def _extent_from_center_coords(x_um: np.ndarray, y_um: np.ndarray) -> tuple[float, float, float, float]:
    def _edges(vals: np.ndarray) -> tuple[float, float]:
        if vals.size <= 1:
            delta = 0.5
            return float(vals[0] - delta), float(vals[0] + delta)
        step = float(np.median(np.diff(vals)))
        return float(vals[0] - 0.5 * step), float(vals[-1] + 0.5 * step)

    x0, x1 = _edges(np.asarray(x_um, dtype=np.float64))
    y0, y1 = _edges(np.asarray(y_um, dtype=np.float64))
    return (x0, x1, y0, y1)


def _field_coords_um(cfg: WaveguideBenchmarkConfig, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    dx_um = cfg.dx_m * 1e6
    ny, nx = shape
    x_um = (np.arange(nx, dtype=np.float64) + 0.5) * dx_um
    y_um = (np.arange(ny, dtype=np.float64) + 0.5) * dx_um
    return x_um, y_um


def _crop_to_common_2d(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    ny = min(a_arr.shape[0], b_arr.shape[0])
    nx = min(a_arr.shape[1], b_arr.shape[1])

    def _crop(arr: np.ndarray) -> np.ndarray:
        y0 = max((arr.shape[0] - ny) // 2, 0)
        x0 = max((arr.shape[1] - nx) // 2, 0)
        return np.asarray(arr[y0 : y0 + ny, x0 : x0 + nx])

    return _crop(a_arr), _crop(b_arr)


def _waveguide_outline_xy(cfg: WaveguideBenchmarkConfig) -> np.ndarray:
    y0 = 0.5 * (cfg.domain_y_um - cfg.waveguide_width_um)
    y1 = y0 + cfg.waveguide_width_um
    return np.asarray(
        [
            [0.0, y0],
            [cfg.domain_x_um, y0],
            [cfg.domain_x_um, y1],
            [0.0, y1],
            [0.0, y0],
        ],
        dtype=np.float64,
    )


def _plot_field_snapshot(
    *,
    beamz_ez: np.ndarray,
    meep_ez: np.ndarray,
    scaled_meep_ez: np.ndarray,
    cfg: WaveguideBenchmarkConfig,
    step: int,
    time_s: float,
    gain: float,
    output_path: Path,
) -> None:
    outline = _waveguide_outline_xy(cfg)
    beamz_arr, meep_scaled = _crop_to_common_2d(
        np.asarray(beamz_ez, dtype=np.float64),
        np.asarray(scaled_meep_ez, dtype=np.float64),
    )
    x_um, y_um = _field_coords_um(cfg, tuple(beamz_arr.shape))
    extent = _extent_from_center_coords(x_um, y_um)
    diff = beamz_arr - meep_scaled
    ref_scale = max(float(np.max(np.abs(beamz_arr))), float(np.max(np.abs(meep_scaled))), 1e-30)
    diff_norm = diff / ref_scale
    vmax = max(float(np.max(np.abs(beamz_arr))), float(np.max(np.abs(meep_scaled))), 1e-12)
    err_vmax = max(float(np.max(np.abs(diff_norm))), 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    panels = [
        (axes[0], beamz_arr, "BeamZ Ez", "RdBu_r", -vmax, vmax),
        (axes[1], meep_scaled, f"Meep Ez x gain ({gain:.3e})", "RdBu_r", -vmax, vmax),
        (
            axes[2],
            diff_norm,
            f"Normalized error\nmax rel={float(np.max(np.abs(diff_norm))):.3e}",
            "coolwarm",
            -err_vmax,
            err_vmax,
        ),
    ]
    for ax, data, title, cmap, vmin, vmax_i in panels:
        im = ax.imshow(data, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax_i)
        ax.plot(outline[:, 0], outline[:, 1], color="black", linewidth=1.2)
        ax.axvline(cfg.monitor_x_um, color="lime", linestyle="--", linewidth=1.0)
        ax.scatter([cfg.source_x_um], [cfg.source_y_um], marker="x", c="black", s=70, linewidths=1.4)
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Ez snapshot at step {step} ({time_s * 1e15:.2f} fs)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_line_history(
    *,
    beamz_history: np.ndarray,
    meep_history: np.ndarray,
    y_um: np.ndarray,
    times_s: np.ndarray,
    gain: float,
    output_path: Path,
) -> None:
    beamz_arr = np.asarray(beamz_history, dtype=np.float64)
    meep_scaled = float(gain) * np.asarray(meep_history, dtype=np.float64)
    diff = beamz_arr - meep_scaled

    t_fs = np.asarray(times_s, dtype=np.float64) * 1e15
    extent = [
        float(y_um[0]),
        float(y_um[-1]),
        float(t_fs[0]),
        float(t_fs[-1]),
    ]
    vmax = max(float(np.max(np.abs(beamz_arr))), float(np.max(np.abs(meep_scaled))), 1e-12)
    diff_norm = diff / max(vmax, 1e-30)
    err_vmax = max(float(np.max(np.abs(diff_norm))), 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    panels = [
        (axes[0], beamz_arr, "BeamZ Ez monitor history", "RdBu_r", -vmax, vmax),
        (axes[1], meep_scaled, f"Meep Ez monitor history x gain ({gain:.3e})", "RdBu_r", -vmax, vmax),
        (
            axes[2],
            diff_norm,
            f"Normalized error\nmax rel={float(np.max(np.abs(diff_norm))):.3e}",
            "coolwarm",
            -err_vmax,
            err_vmax,
        ),
    ]
    for ax, data, title, cmap, vmin, vmax_i in panels:
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_permittivity_comparison(
    *,
    beamz_eps: np.ndarray,
    meep_eps: np.ndarray,
    cfg: WaveguideBenchmarkConfig,
    output_path: Path,
) -> None:
    outline = _waveguide_outline_xy(cfg)
    x_um, y_um = _field_coords_um(cfg, tuple(np.asarray(beamz_eps).shape))
    extent = _extent_from_center_coords(x_um, y_um)
    diff = np.asarray(beamz_eps, dtype=np.float64) - np.asarray(meep_eps, dtype=np.float64)
    eps_max = max(float(np.max(beamz_eps)), float(np.max(meep_eps)), 1.0)
    err_vmax = max(float(np.max(np.abs(diff))), 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    panels = [
        (axes[0], beamz_eps, "BeamZ permittivity", "viridis", 1.0, eps_max),
        (axes[1], meep_eps, "Meep permittivity", "viridis", 1.0, eps_max),
        (axes[2], diff, "Permittivity difference", "coolwarm", -err_vmax, err_vmax),
    ]
    for ax, data, title, cmap, vmin, vmax_i in panels:
        im = ax.imshow(data, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax_i)
        ax.plot(outline[:, 0], outline[:, 1], color="black", linewidth=1.2)
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _generate_debug_plots(
    *,
    output_dir: Path,
    cfg: WaveguideBenchmarkConfig,
    gain: float,
) -> list[str]:
    structure_dir = output_dir / "structure_000"
    plot_dir = structure_dir / "plots"
    beamz = np.load(structure_dir / "beamz_monitor_history.npz")
    meep = np.load(structure_dir / "meep_monitor_history.npz")

    generated: list[str] = []
    permittivity_path = plot_dir / "permittivity_comparison.png"
    _plot_permittivity_comparison(
        beamz_eps=np.asarray(beamz["permittivity"], dtype=np.float32),
        meep_eps=np.asarray(meep["permittivity"], dtype=np.float32),
        cfg=cfg,
        output_path=permittivity_path,
    )
    generated.append(str(permittivity_path.resolve()))

    _plot_line_history(
        beamz_history=np.asarray(beamz["Ez_history"], dtype=np.float32),
        meep_history=np.asarray(meep["Ez_history"], dtype=np.float32),
        y_um=np.asarray(beamz["monitor_y_um"], dtype=np.float64),
        times_s=np.asarray(beamz["times_s"], dtype=np.float64),
        gain=gain,
        output_path=plot_dir / "ez_monitor_history.png",
    )
    generated.append(str((plot_dir / "ez_monitor_history.png").resolve()))

    beamz_steps = np.asarray(beamz["snapshot_steps"], dtype=np.int32)
    beamz_times = np.asarray(beamz["snapshot_times_s"], dtype=np.float64)
    beamz_snapshots = np.asarray(beamz["Ez_snapshots"], dtype=np.float32)
    meep_snapshots = np.asarray(meep["Ez_snapshots"], dtype=np.float32)
    for idx, step in enumerate(beamz_steps):
        output_path = plot_dir / f"ez_snapshot_{idx:02d}.png"
        _plot_field_snapshot(
            beamz_ez=beamz_snapshots[idx],
            meep_ez=meep_snapshots[idx],
            scaled_meep_ez=float(gain) * meep_snapshots[idx],
            cfg=cfg,
            step=int(step),
            time_s=float(beamz_times[idx]),
            gain=gain,
            output_path=output_path,
        )
        generated.append(str(output_path.resolve()))
    return generated


def _compare_histories(
    beamz_ez: np.ndarray,
    meep_ez: np.ndarray,
    *,
    tolerance: float,
) -> dict[str, Any]:
    beamz_ez = np.asarray(beamz_ez, dtype=np.float64)
    meep_ez = np.asarray(meep_ez, dtype=np.float64)
    n_steps = min(beamz_ez.shape[0], meep_ez.shape[0])
    beamz_ez = beamz_ez[:n_steps]
    meep_ez = meep_ez[:n_steps]

    gain = _fit_real_scale(beamz_ez, meep_ez)
    meep_scaled = gain * meep_ez

    step_norms = np.linalg.norm(beamz_ez, axis=1)
    active_floor = max(1e-18, 1e-6 * float(np.max(step_norms))) if step_norms.size else 1e-18
    active_mask = step_norms >= active_floor

    rel_l2 = np.zeros((n_steps,), dtype=np.float64)
    max_abs = np.zeros((n_steps,), dtype=np.float64)
    for idx in range(n_steps):
        diff = meep_scaled[idx] - beamz_ez[idx]
        max_abs[idx] = float(np.max(np.abs(diff))) if diff.size else 0.0
        if active_mask[idx]:
            rel_l2[idx] = float(np.linalg.norm(diff) / max(step_norms[idx], 1e-30))

    active_rel = rel_l2[active_mask]
    return {
        "global_gain": float(gain),
        "n_steps_compared": int(n_steps),
        "n_active_steps": int(np.count_nonzero(active_mask)),
        "active_threshold_norm": float(active_floor),
        "per_step_rel_l2": rel_l2.tolist(),
        "per_step_max_abs": max_abs.tolist(),
        "max_rel_l2_active": float(np.max(active_rel)) if active_rel.size else 0.0,
        "mean_rel_l2_active": float(np.mean(active_rel)) if active_rel.size else 0.0,
        "passes_tolerance": bool(
            (np.max(active_rel) <= float(tolerance)) if active_rel.size else False
        ),
        "tolerance": float(tolerance),
    }


def _compare_dft(
    beamz_dft: np.ndarray,
    meep_dft: np.ndarray,
    *,
    global_gain: float,
) -> dict[str, Any]:
    b = np.asarray(beamz_dft, dtype=np.complex128).reshape(-1)
    m = np.asarray(meep_dft, dtype=np.complex128).reshape(-1)
    n = min(b.size, m.size)
    b = b[:n]
    m = m[:n]
    denom = float(np.linalg.norm(b))
    rel_l2 = float(np.linalg.norm((global_gain * m) - b) / max(denom, 1e-30))
    return {
        "n_points_compared": int(n),
        "relative_l2_after_history_gain": rel_l2,
        "beamz_norm": denom,
        "meep_norm": float(np.linalg.norm(m)),
    }


def _write_backend_output(
    *,
    output_path: Path,
    solver_name: str,
    cfg: WaveguideBenchmarkConfig,
    structure_specs: list[dict[str, Any]],
    permittivity: np.ndarray,
    monitor_x_um: np.ndarray,
    monitor_y_um: np.ndarray,
    times_s: np.ndarray,
    ez_history: np.ndarray,
    hy_history: np.ndarray,
    dft_ez: np.ndarray,
    dft_hy: np.ndarray,
    snapshot_steps: np.ndarray,
    snapshot_times_s: np.ndarray,
    ez_snapshots: np.ndarray,
    runtime_s: float,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        permittivity=np.asarray(permittivity, dtype=np.float32),
        monitor_x_um=np.asarray(monitor_x_um, dtype=np.float64),
        monitor_y_um=np.asarray(monitor_y_um, dtype=np.float64),
        times_s=np.asarray(times_s, dtype=np.float64),
        Ez_history=np.asarray(ez_history, dtype=np.float32),
        Hy_history=np.asarray(hy_history, dtype=np.float32),
        dft_Ez=np.asarray(dft_ez, dtype=np.complex128),
        dft_Hy=np.asarray(dft_hy, dtype=np.complex128),
        snapshot_steps=np.asarray(snapshot_steps, dtype=np.int32),
        snapshot_times_s=np.asarray(snapshot_times_s, dtype=np.float64),
        Ez_snapshots=np.asarray(ez_snapshots, dtype=np.float32),
    )
    metadata = {
        "solver": solver_name,
        "config": asdict(cfg),
        "runtime_s": float(runtime_s),
        "data_file": output_path.name,
        "monitor_point_count": int(np.asarray(monitor_y_um).size),
        "history_steps": int(np.asarray(times_s).size),
        "snapshots": [
            {"step": int(step), "time_s": float(tt)}
            for step, tt in zip(
                np.asarray(snapshot_steps, dtype=np.int32),
                np.asarray(snapshot_times_s, dtype=np.float64),
                strict=True,
            )
        ],
        "structure_specs": structure_specs,
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    return metadata


def run_beamz_single(
    cfg: WaveguideBenchmarkConfig,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    from beamz import Monitor, PML, Simulation

    design = _build_beamz_design(cfg)
    signal = _meep_signal_function(cfg)

    ez_coords = _beamz_native_tm_coords_2d(cfg)
    pixels, weights = _support_pixels_2d(
        ez_coords,
        center_um=(cfg.source_x_um, cfg.source_y_um),
        width_um=cfg.source_width_um,
    )
    source = _BeamzCurrentSource2D(
        signal=signal,
        pixel_indices=pixels,
        pixel_weights=weights,
    )

    monitor_start, monitor_end = _monitor_line_endpoints_m(cfg)
    monitor = Monitor(
        design=design,
        start=monitor_start,
        end=monitor_end,
        name="wg_dft",
        record_fields=False,
        accumulate_power=False,
        dft_enabled=True,
        dft_frequencies=np.array([cfg.frequency_hz], dtype=np.float64),
        dft_components=("Ez", "Hy"),
        dft_window="rect",
        dft_record_interval=1,
    )

    time_axis = np.arange(cfg.num_steps + 1, dtype=np.float64) * cfg.dt_s
    sim = Simulation(
        design=design,
        sources=[source],
        monitors=[monitor],
        boundaries=[PML(edges="all", thickness=cfg.pml_um * 1e-6)],
        time=time_axis,
        resolution=cfg.dx_m,
    )

    x_um, y_um, y_idx, x_idx = _monitor_sample_geometry(cfg)

    ez_history: list[np.ndarray] = []
    hy_history: list[np.ndarray] = []
    times_s: list[float] = []
    snapshot_steps = set(cfg.snapshot_steps)
    snapshot_records: list[tuple[int, float, np.ndarray]] = []

    start = time.perf_counter()
    while sim.step():
        ez_full = np.asarray(sim.fields.Ez, dtype=np.float32)
        hy_full = np.asarray(sim.fields.Hy, dtype=np.float32)
        if int(sim.current_step) in snapshot_steps:
            snapshot_records.append(
                (int(sim.current_step), float(sim.t), np.asarray(ez_full, dtype=np.float32))
            )
        ez_history.append(np.asarray(ez_full[y_idx, x_idx], dtype=np.float32))
        hy_history.append(np.asarray(hy_full[y_idx, x_idx], dtype=np.float32))
        times_s.append(float(sim.t))
    runtime_s = time.perf_counter() - start

    dft_ez = np.asarray(monitor.get_dft_component("Ez"), dtype=np.complex128)
    dft_hy = np.asarray(monitor.get_dft_component("Hy"), dtype=np.complex128)

    out_path = output_dir / "structure_000" / "beamz_monitor_history.npz"
    metadata = _write_backend_output(
        output_path=out_path,
        solver_name="beamz",
        cfg=cfg,
        structure_specs=_waveguide_structure_spec(cfg),
        permittivity=np.asarray(sim.fields.permittivity, dtype=np.float32),
        monitor_x_um=x_um,
        monitor_y_um=y_um,
        times_s=np.asarray(times_s, dtype=np.float64),
        ez_history=np.asarray(ez_history, dtype=np.float32),
        hy_history=np.asarray(hy_history, dtype=np.float32),
        dft_ez=dft_ez,
        dft_hy=dft_hy,
        snapshot_steps=np.asarray([step for step, _, _ in snapshot_records], dtype=np.int32),
        snapshot_times_s=np.asarray([tt for _, tt, _ in snapshot_records], dtype=np.float64),
        ez_snapshots=np.stack([arr for _, _, arr in snapshot_records]).astype(np.float32),
        runtime_s=runtime_s,
    )
    return {
        **metadata,
        "monitor_x_um": x_um.tolist(),
        "monitor_y_um": y_um.tolist(),
        "times_s": np.asarray(times_s, dtype=np.float64).tolist(),
        "Ez_history": np.asarray(ez_history, dtype=np.float32).tolist(),
        "Hy_history": np.asarray(hy_history, dtype=np.float32).tolist(),
        "dft_Ez_real": np.real(dft_ez).tolist(),
        "dft_Ez_imag": np.imag(dft_ez).tolist(),
        "dft_Hy_real": np.real(dft_hy).tolist(),
        "dft_Hy_imag": np.imag(dft_hy).tolist(),
    }


def run_meep_single(
    cfg: WaveguideBenchmarkConfig,
    *,
    output_dir: Path,
    beamz_raster_file: Path | None,
) -> dict[str, Any]:
    import meep as mp

    resolution_px_per_um = 1.0 / (cfg.dx_m * 1e6)
    geometry = []
    default_material: Any = mp.Medium(index=cfg.clad_index)
    epsilon_func = None
    eps_averaging = True
    shared_beamz_eps_yx: np.ndarray | None = None
    if cfg.geometry_source == "beamz-raster":
        if beamz_raster_file is None:
            raise ValueError("beamz_raster_file is required for beamz-raster mode")
        shared_beamz_eps_yx = np.asarray(np.load(beamz_raster_file), dtype=np.float32)
        eps_grid = np.asarray(shared_beamz_eps_yx, dtype=np.float64)
        dx_um = cfg.dx_m * 1e6

        def epsilon_func(p):
            x_um = float(p.x) + 0.5 * cfg.domain_x_um
            y_um = float(p.y) + 0.5 * cfg.domain_y_um
            ix = int(np.clip(np.floor(x_um / dx_um), 0, eps_grid.shape[1] - 1))
            iy = int(np.clip(np.floor(y_um / dx_um), 0, eps_grid.shape[0] - 1))
            return float(eps_grid[iy, ix])

        eps_averaging = False
    else:
        geometry.append(
            mp.Block(
                size=mp.Vector3(cfg.domain_x_um, cfg.waveguide_width_um, mp.inf),
                center=mp.Vector3(0.0, 0.0),
                material=mp.Medium(index=cfg.core_index),
            )
        )

    signal_fn = _meep_signal_function(cfg)
    custom_src = mp.CustomSource(
        src_func=signal_fn,
        start_time=0.0,
        end_time=(cfg.num_steps + 1) * cfg.dt_s * C0 / 1e-6,
        center_frequency=cfg.frequency_um_inv,
        fwidth=(1.0 / max(2.0 * np.pi * cfg.pulse_sigma_s, 1e-30)) * (1e-6 / C0),
    )

    monitor_center = mp.Vector3(
        cfg.monitor_x_um - 0.5 * cfg.domain_x_um,
        0.0,
        0.0,
    )
    monitor_size = mp.Vector3(0.0, cfg.monitor_span_um, 0.0)

    ez_coords = _meep_native_source_coords_2d(cfg)
    pixels, weights = _support_pixels_2d(
        ez_coords,
        center_um=(cfg.source_x_um, cfg.source_y_um),
        width_um=cfg.source_width_um,
    )
    sources = []
    for (iy, ix), weight in zip(pixels, weights, strict=False):
        x_um = float(ez_coords["x"][ix])
        y_um = float(ez_coords["y"][iy])
        sources.append(
            mp.Source(
                src=custom_src,
                component=mp.Ez,
                center=mp.Vector3(
                    x_um - 0.5 * cfg.domain_x_um,
                    y_um - 0.5 * cfg.domain_y_um,
                    0.0,
                ),
                size=mp.Vector3(),
                amplitude=float(cfg.meep_source_amplitude_scale * weight),
            )
        )

    sim = mp.Simulation(
        cell_size=mp.Vector3(cfg.domain_x_um, cfg.domain_y_um, 0.0),
        geometry=geometry,
        sources=sources,
        boundary_layers=[mp.PML(thickness=cfg.pml_um)],
        default_material=default_material,
        epsilon_func=epsilon_func,
        eps_averaging=eps_averaging,
        resolution=resolution_px_per_um,
        Courant=cfg.courant_safety / math.sqrt(2.0),
        dimensions=2,
    )

    dft_monitor = sim.add_dft_fields(
        [mp.Ez, mp.Hy],
        cfg.frequency_um_inv,
        0.0,
        1,
        center=monitor_center,
        size=monitor_size,
    )

    x_um, y_um, y_idx, x_idx = _monitor_sample_geometry(cfg)
    point_positions = [
        mp.Vector3(
            float(x) - 0.5 * cfg.domain_x_um,
            float(y) - 0.5 * cfg.domain_y_um,
            0.0,
        )
        for x, y in zip(x_um, y_um, strict=True)
    ]

    ez_history: list[np.ndarray] = []
    hy_history: list[np.ndarray] = []
    times_s: list[float] = []
    snapshot_index_by_step = {step: idx for idx, step in enumerate(cfg.snapshot_steps)}
    snapshot_steps = np.asarray(cfg.snapshot_steps, dtype=np.int32)
    snapshot_times_s = np.asarray(cfg.snapshot_steps, dtype=np.float64) * cfg.dt_s
    ez_snapshots: list[np.ndarray | None] = [None] * len(snapshot_steps)
    dt_um = cfg.dt_s * C0 / 1e-6

    def _capture(sim_obj):
        t_um = float(sim_obj.meep_time())
        step = int(round(t_um / dt_um))
        sim_obj.fields.synchronize_magnetic_fields()
        try:
            ez_line = np.asarray(
                [sim_obj.get_field_point(mp.Ez, p) for p in point_positions],
                dtype=np.complex128,
            )
            hy_line = np.asarray(
                [sim_obj.get_field_point(mp.Hy, p) for p in point_positions],
                dtype=np.complex128,
            )
            snap_idx = snapshot_index_by_step.get(step)
            if snap_idx is not None:
                ez_snapshots[snap_idx] = np.transpose(
                    np.asarray(
                        sim_obj.get_array(
                            center=mp.Vector3(),
                            size=mp.Vector3(cfg.domain_x_um, cfg.domain_y_um, 0.0),
                            component=mp.Ez,
                            snap=True,
                        ),
                        dtype=np.float32,
                    ),
                    (1, 0),
                )
        finally:
            sim_obj.fields.restore_magnetic_fields()
        ez_history.append(np.real(ez_line).astype(np.float32))
        hy_history.append(np.real(hy_line).astype(np.float32))
        times_s.append(float(t_um * 1e-6 / C0))

    mp.verbosity(0)
    sim.init_sim()
    start = time.perf_counter()
    sim.run(mp.at_every(dt_um, _capture), until=(cfg.num_steps + 0.5) * dt_um)
    runtime_s = time.perf_counter() - start

    if shared_beamz_eps_yx is not None:
        permittivity = shared_beamz_eps_yx
    else:
        permittivity = np.transpose(
            np.asarray(
                sim.get_array(
                    center=mp.Vector3(),
                    size=mp.Vector3(cfg.domain_x_um, cfg.domain_y_um, 0.0),
                    component=mp.Dielectric,
                ),
                dtype=np.float32,
            ),
            (1, 0),
        )

    if any(snap is None for snap in ez_snapshots):
        raise RuntimeError("Meep run did not emit all requested Ez snapshots")

    dft_ez = np.asarray(sim.get_dft_array(dft_monitor, mp.Ez, 0), dtype=np.complex128)
    dft_hy = np.asarray(sim.get_dft_array(dft_monitor, mp.Hy, 0), dtype=np.complex128)

    out_path = output_dir / "structure_000" / "meep_monitor_history.npz"
    metadata = _write_backend_output(
        output_path=out_path,
        solver_name="meep",
        cfg=cfg,
        structure_specs=_waveguide_structure_spec(cfg),
        permittivity=permittivity,
        monitor_x_um=x_um,
        monitor_y_um=y_um,
        times_s=np.asarray(times_s, dtype=np.float64),
        ez_history=np.asarray(ez_history, dtype=np.float32),
        hy_history=np.asarray(hy_history, dtype=np.float32),
        dft_ez=dft_ez.reshape(1, -1),
        dft_hy=dft_hy.reshape(1, -1),
        snapshot_steps=snapshot_steps,
        snapshot_times_s=snapshot_times_s,
        ez_snapshots=np.stack([snap for snap in ez_snapshots if snap is not None]).astype(np.float32),
        runtime_s=runtime_s,
    )
    return {
        **metadata,
        "monitor_x_um": x_um.tolist(),
        "monitor_y_um": y_um.tolist(),
        "times_s": np.asarray(times_s, dtype=np.float64).tolist(),
        "Ez_history": np.asarray(ez_history, dtype=np.float32).tolist(),
        "Hy_history": np.asarray(hy_history, dtype=np.float32).tolist(),
        "dft_Ez_real": np.real(dft_ez).reshape(1, -1).tolist(),
        "dft_Ez_imag": np.imag(dft_ez).reshape(1, -1).tolist(),
        "dft_Hy_real": np.real(dft_hy).reshape(1, -1).tolist(),
        "dft_Hy_imag": np.imag(dft_hy).reshape(1, -1).tolist(),
    }


def _extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("No JSON object found in subprocess stdout.")


def _run_meep_subprocess(*, meep_env: str, args: list[str]) -> dict[str, Any]:
    cmd = [
        "conda",
        "run",
        "-n",
        meep_env,
        "python",
        str(Path(__file__).resolve()),
        "--backend",
        "meep",
        "--emit-json-only",
    ]
    cmd.extend(args)
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip() or f"exit status {proc.returncode}"
        raise RuntimeError(f"Meep subprocess failed for conda env '{meep_env}': {detail}")
    return _extract_json_object(proc.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Tiny BeamZ/Meep waveguide benchmark with a matched custom source, a DFT monitor, "
            "and per-step monitor-line field-history comparison."
        )
    )
    parser.add_argument("--backend", choices=("beamz", "meep", "both"), default="both")
    parser.add_argument("--resolution-ppw", type=int, default=10)
    parser.add_argument("--total-cycles", type=float, default=3.0)
    parser.add_argument("--wavelength-um", type=float, default=1.55)
    parser.add_argument("--courant-safety", type=float, default=0.95)
    parser.add_argument("--meep-source-amplitude-scale", type=float, default=1.0)
    parser.add_argument("--geometry-source", choices=("native", "beamz-raster"), default="beamz-raster")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "waveguide_results")
    parser.add_argument("--meep-env", default=os.getenv("BEAMZ_MEEP_ENV", "beamz-meep"))
    parser.add_argument("--beamz-raster-file", type=Path, default=None)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--emit-json-only", action="store_true")
    args = parser.parse_args()

    cfg = WaveguideBenchmarkConfig(
        resolution_ppw=args.resolution_ppw,
        total_cycles=args.total_cycles,
        wavelength_um=args.wavelength_um,
        courant_safety=args.courant_safety,
        meep_source_amplitude_scale=args.meep_source_amplitude_scale,
        geometry_source=args.geometry_source,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    structure_dir = output_dir / "structure_000"
    structure_dir.mkdir(parents=True, exist_ok=True)

    structure_specs = _waveguide_structure_spec(cfg)
    (structure_dir / "structure.json").write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "structure_specs": structure_specs,
            },
            indent=2,
        )
    )

    beamz_raster_file = args.beamz_raster_file
    if cfg.geometry_source == "beamz-raster" and beamz_raster_file is None:
        beamz_raster_file = structure_dir / "beamz_shared_permittivity.npy"
        np.save(beamz_raster_file, _rasterize_beamz_permittivity(cfg))

    try:
        import meep as _meep  # noqa: F401

        meep_available = True
    except Exception:
        meep_available = False

    results: dict[str, Any] = {}
    if args.backend in ("beamz", "both"):
        results["beamz"] = run_beamz_single(cfg, output_dir=output_dir)
    if args.backend in ("meep", "both"):
        if meep_available:
            results["meep"] = run_meep_single(
                cfg,
                output_dir=output_dir,
                beamz_raster_file=beamz_raster_file,
            )
        else:
            extra_args = []
            if beamz_raster_file is not None:
                extra_args.extend(["--beamz-raster-file", str(beamz_raster_file)])
            results["meep"] = _run_meep_subprocess(
                meep_env=args.meep_env,
                args=[
                    "--resolution-ppw",
                    str(cfg.resolution_ppw),
                    "--total-cycles",
                    str(cfg.total_cycles),
                    "--wavelength-um",
                    str(cfg.wavelength_um),
                    "--courant-safety",
                    str(cfg.courant_safety),
                    "--meep-source-amplitude-scale",
                    str(cfg.meep_source_amplitude_scale),
                    "--geometry-source",
                    str(cfg.geometry_source),
                    "--output-dir",
                    str(output_dir),
                    *extra_args,
                ],
            )["results"]["meep"]

    summary: dict[str, Any] = {
        "generated_at_epoch_s": time.time(),
        "config": asdict(cfg),
        "structure_specs": structure_specs,
        "results": results,
    }
    if "beamz" in results and "meep" in results:
        beamz_ez = np.asarray(results["beamz"]["Ez_history"], dtype=np.float64)
        meep_ez = np.asarray(results["meep"]["Ez_history"], dtype=np.float64)
        history_cmp = _compare_histories(
            beamz_ez,
            meep_ez,
            tolerance=cfg.error_tolerance,
        )
        summary["comparison"] = {
            "field_history_ez": history_cmp,
            "dft_monitor_ez": _compare_dft(
                np.asarray(results["beamz"]["dft_Ez_real"], dtype=np.float64)
                + 1j * np.asarray(results["beamz"]["dft_Ez_imag"], dtype=np.float64),
                np.asarray(results["meep"]["dft_Ez_real"], dtype=np.float64)
                + 1j * np.asarray(results["meep"]["dft_Ez_imag"], dtype=np.float64),
                global_gain=float(history_cmp["global_gain"]),
            ),
            "dft_monitor_hy": _compare_dft(
                np.asarray(results["beamz"]["dft_Hy_real"], dtype=np.float64)
                + 1j * np.asarray(results["beamz"]["dft_Hy_imag"], dtype=np.float64),
                np.asarray(results["meep"]["dft_Hy_real"], dtype=np.float64)
                + 1j * np.asarray(results["meep"]["dft_Hy_imag"], dtype=np.float64),
                global_gain=float(history_cmp["global_gain"]),
            ),
        }
        if not args.skip_plots:
            summary["plots"] = _generate_debug_plots(
                output_dir=output_dir,
                cfg=cfg,
                gain=float(history_cmp["global_gain"]),
            )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2))

    if args.emit_json_only:
        print(json.dumps(summary))
    else:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
