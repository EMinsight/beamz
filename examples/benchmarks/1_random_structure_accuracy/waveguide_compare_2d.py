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

import numpy as np

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
EPS_0 = 8.8541878128e-12
MU_0 = 1.25663706212e-6

SCRIPT_DIR = Path(__file__).resolve().parent
C0 = 299_792_458.0
Z0_OHM = math.sqrt(MU_0 / EPS_0)


@dataclass(frozen=True)
class Benchmark2DConfig:
    domain_um: float = 3.0
    wavelength_um: float = 1.0
    resolution_ppw: int = 8
    courant_safety: float = 0.95
    total_cycles: float = 4.0
    pulse_center_cycles: float = 1.25
    pulse_sigma_cycles: float = 0.35
    background_index: float = 1.0
    source_x_um: float = 0.75
    source_y_um: float = 1.5
    source_width_um: float = 0.18
    dft_monitor_x_um: float = 2.25
    dft_decimation_factor: int = 1
    waveguide_width_um: float = 0.60
    waveguide_index: float = math.sqrt(12.0)
    geometry_source: str = "beamz-raster"
    num_snapshots: int = 3
    meep_source_width_scale: float = 1.0
    meep_source_amplitude_scale: float = 5.887e-06

    @property
    def dx_m(self) -> float:
        return (self.wavelength_um * 1e-6) / float(self.resolution_ppw)

    @property
    def frequency_hz(self) -> float:
        return C0 / (self.wavelength_um * 1e-6)

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
    def domain_m(self) -> float:
        return self.domain_um * 1e-6

    @property
    def grid_shape(self) -> tuple[int, int]:
        n = int(round(self.domain_m / self.dx_m))
        return (n, n)

    @property
    def num_steps(self) -> int:
        return max(2, int(math.floor(self.total_time_s / self.dt_s)))

    @property
    def snapshot_steps(self) -> list[int]:
        raw = np.linspace(0.25, 0.75, self.num_snapshots)
        out = []
        for frac in raw:
            step = int(round(frac * (self.num_steps - 1)))
            step = min(max(step, 1), self.num_steps - 1)
            if not out or step != out[-1]:
                out.append(step)
        return out

    @property
    def snapshot_times_s(self) -> list[float]:
        return [step * self.dt_s for step in self.snapshot_steps]


def _gaussian_modulated_signal_value(cfg: Benchmark2DConfig, t_s: float) -> float:
    envelope = math.exp(-0.5 * ((t_s - cfg.pulse_center_s) / cfg.pulse_sigma_s) ** 2)
    carrier = math.cos(2.0 * math.pi * cfg.frequency_hz * t_s)
    return float(envelope * carrier)


def _gaussian_modulated_signal_fn(cfg: Benchmark2DConfig):
    def _signal(t_s: float) -> float:
        return _gaussian_modulated_signal_value(cfg, float(t_s))

    return _signal


def _waveguide_specs(cfg: Benchmark2DConfig) -> list[dict[str, Any]]:
    y0 = 0.5 * (cfg.domain_um - cfg.waveguide_width_um)
    return [
        {
            "type": "block",
            "position_um": [0.0, y0],
            "size_um": [cfg.domain_um, cfg.waveguide_width_um],
            "index": float(cfg.waveguide_index),
            "permittivity": float(cfg.waveguide_index**2),
        }
    ]


def _dft_monitor_line_endpoints_m(cfg: Benchmark2DConfig) -> tuple[tuple[float, float], tuple[float, float]]:
    return (
        (cfg.dft_monitor_x_um * 1e-6, 0.0),
        (cfg.dft_monitor_x_um * 1e-6, cfg.domain_um * 1e-6),
    )


def _dft_monitor_sample_geometry(
    cfg: Benchmark2DConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    ez_coords = _meep_sample_coords_2d("Ez", cfg)
    y = np.asarray(ez_coords["y"], dtype=np.float64)
    y_idx = np.arange(y.size, dtype=np.int32)
    x_um = np.full(y.shape, float(cfg.dft_monitor_x_um), dtype=np.float64)
    return x_um, y, y_idx, -1


def _sample_vertical_line_on_centered_grid(
    field_yx: np.ndarray,
    *,
    x_coords_um: np.ndarray,
    x_line_um: float,
) -> np.ndarray:
    field = np.asarray(field_yx, dtype=np.complex128)
    x_src = np.asarray(x_coords_um, dtype=np.float64).reshape(-1)
    out = np.empty((field.shape[0],), dtype=np.complex128)
    for iy in range(field.shape[0]):
        out[iy] = np.interp(x_line_um, x_src, field[iy].real) + 1j * np.interp(
            x_line_um,
            x_src,
            field[iy].imag,
        )
    return out


def _running_dft_from_history(
    history: np.ndarray,
    *,
    times_s: np.ndarray,
    frequency_hz: float,
    length_unit_m: float = 1e-6,
    decimation_factor: int = 1,
) -> np.ndarray:
    history_arr = np.asarray(history, dtype=np.complex128)
    if history_arr.ndim != 2:
        raise ValueError(f"Expected 2D history array, got shape {history_arr.shape}")
    times = np.asarray(times_s, dtype=np.float64).reshape(-1)
    if history_arr.shape[0] != times.size:
        raise ValueError(
            f"History/time length mismatch: {history_arr.shape[0]} vs {times.size}"
        )
    if times.size == 0:
        return np.zeros_like(history_arr, dtype=np.complex128)
    phase = np.exp(1j * 2.0 * np.pi * float(frequency_hz) * times)[:, None]
    accum = np.cumsum(history_arr * phase, axis=0)
    if times.size > 1:
        base_dt_s = float(np.median(np.diff(times)))
    else:
        base_dt_s = 0.0
    dt_norm = (
        float(base_dt_s)
        * float(C0)
        * max(1, int(decimation_factor))
        / max(float(length_unit_m), 1e-30)
    )
    return accum * (dt_norm / math.sqrt(2.0 * math.pi))


def _compare_complex_profiles(
    beamz: np.ndarray,
    meep: np.ndarray,
) -> dict[str, Any]:
    b = np.asarray(beamz, dtype=np.complex128).reshape(-1)
    m = np.asarray(meep, dtype=np.complex128).reshape(-1)
    n = min(b.size, m.size)
    b = b[:n]
    m = m[:n]
    denom = max(float(np.linalg.norm(b)), float(np.linalg.norm(m)), 1e-30)
    diff = b - m
    return {
        "n_points_compared": int(n),
        "relative_l2": float(np.linalg.norm(diff) / denom),
        "max_abs_error": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "beamz_norm": float(np.linalg.norm(b)),
        "meep_norm": float(np.linalg.norm(m)),
    }


def _compare_running_dft(
    beamz_running: np.ndarray,
    meep_running: np.ndarray,
) -> dict[str, Any]:
    b = np.asarray(beamz_running, dtype=np.complex128)
    m = np.asarray(meep_running, dtype=np.complex128)
    n_steps = min(b.shape[0], m.shape[0])
    n_pts = min(b.shape[1], m.shape[1])
    b = b[:n_steps, :n_pts]
    m = m[:n_steps, :n_pts]
    rel = np.zeros((n_steps,), dtype=np.float64)
    for i in range(n_steps):
        denom = max(float(np.linalg.norm(b[i])), float(np.linalg.norm(m[i])), 1e-30)
        rel[i] = float(np.linalg.norm(b[i] - m[i]) / denom)
    return {
        "n_steps_compared": int(n_steps),
        "n_points_compared": int(n_pts),
        "per_step_rel_l2": rel.tolist(),
        "max_rel_l2": float(np.max(rel)) if rel.size else 0.0,
        "mean_rel_l2": float(np.mean(rel)) if rel.size else 0.0,
        "final_rel_l2": float(rel[-1]) if rel.size else 0.0,
    }


def _meep_dft_decimation_factor(dft_monitor) -> int | None:
    chunks = getattr(dft_monitor, "chunks", None)
    factors: list[int] = []
    seen = set()
    while chunks is not None:
        ident = id(chunks)
        if ident in seen:
            break
        seen.add(ident)
        try:
            factors.append(int(chunks.get_decimation_factor()))
        except Exception:
            break
        chunks = getattr(chunks, "next_in_dft", None)
    return max(factors) if factors else None


def _canonicalize_meep_array_2d(arr: np.ndarray) -> np.ndarray:
    arr_np = np.asarray(arr, dtype=np.float32)
    if arr_np.ndim != 2:
        raise ValueError(f"Expected 2D Meep array, got shape {arr_np.shape}")
    return np.transpose(arr_np, (1, 0))


def _interp_axis_between_coords_2d(
    arr: np.ndarray,
    axis: int,
    src_coords: np.ndarray,
    target_coords: np.ndarray,
) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    moved = np.moveaxis(arr, axis, 0)
    flat = moved.reshape(moved.shape[0], -1)
    out_flat = np.empty((target_coords.size, flat.shape[1]), dtype=np.float32)
    for idx in range(flat.shape[1]):
        out_flat[:, idx] = np.interp(
            target_coords,
            src_coords,
            flat[:, idx],
            left=float(flat[0, idx]),
            right=float(flat[-1, idx]),
        )
    out = out_flat.reshape((target_coords.size, *moved.shape[1:]))
    return np.moveaxis(out, 0, axis)


def _beamz_raw_coords_2d(component: str, cfg: Benchmark2DConfig) -> dict[str, np.ndarray]:
    ny, nx = cfg.grid_shape
    dx_um = cfg.dx_m * 1e6
    if component == "Ex":
        return {
            "y": (np.arange(ny, dtype=np.float64) + 0.5) * dx_um,
            "x": np.arange(nx - 1, dtype=np.float64) * dx_um,
        }
    if component == "Ey":
        return {
            "y": np.arange(ny - 1, dtype=np.float64) * dx_um,
            "x": (np.arange(nx, dtype=np.float64) + 0.5) * dx_um,
        }
    if component == "Ez":
        return {
            "y": (np.arange(ny, dtype=np.float64) + 0.5) * dx_um,
            "x": (np.arange(nx, dtype=np.float64) + 0.5) * dx_um,
        }
    if component == "Hx":
        return {
            "y": (np.arange(ny, dtype=np.float64) + 0.5) * dx_um,
            "x": np.arange(nx - 1, dtype=np.float64) * dx_um,
        }
    if component == "Hy":
        return {
            "y": np.arange(ny - 1, dtype=np.float64) * dx_um,
            "x": (np.arange(nx, dtype=np.float64) + 0.5) * dx_um,
        }
    if component == "Hz":
        return {
            "y": np.arange(ny - 1, dtype=np.float64) * dx_um,
            "x": np.arange(nx - 1, dtype=np.float64) * dx_um,
        }
    raise ValueError(f"Unsupported component {component!r}")


def _meep_sample_coords_2d(component: str, cfg: Benchmark2DConfig) -> dict[str, np.ndarray]:
    ny, nx = cfg.grid_shape
    dx_um = cfg.dx_m * 1e6
    if component == "Ex":
        return {
            "y": (np.arange(ny, dtype=np.float64) + 0.5) * dx_um,
            "x": (np.arange(nx - 1, dtype=np.float64) + 0.5) * dx_um,
        }
    if component == "Ey":
        return {
            "y": (np.arange(ny - 1, dtype=np.float64) + 0.5) * dx_um,
            "x": (np.arange(nx, dtype=np.float64) + 0.5) * dx_um,
        }
    if component == "Ez":
        return {
            "y": (np.arange(ny, dtype=np.float64) + 0.5) * dx_um,
            "x": (np.arange(nx, dtype=np.float64) + 0.5) * dx_um,
        }
    if component == "Hx":
        return {
            # For this get_array(..., snap=True) window, Meep returns Hx on the
            # cell-centered output grid with the high-side y row omitted.
            "y": (np.arange(ny - 1, dtype=np.float64) + 0.5) * dx_um,
            "x": (np.arange(nx, dtype=np.float64) + 0.5) * dx_um,
        }
    if component == "Hy":
        return {
            "y": (np.arange(ny, dtype=np.float64) + 0.5) * dx_um,
            # For this get_array(..., snap=True) window, Meep returns Hy on the
            # cell-centered output grid with the high-side x column omitted.
            "x": (np.arange(nx - 1, dtype=np.float64) + 0.5) * dx_um,
        }
    if component == "Hz":
        return {
            "y": (np.arange(ny - 1, dtype=np.float64) + 0.5) * dx_um,
            "x": (np.arange(nx - 1, dtype=np.float64) + 0.5) * dx_um,
        }
    raise ValueError(f"Unsupported component {component!r}")


def _meep_native_source_coords_2d(component: str, cfg: Benchmark2DConfig) -> dict[str, np.ndarray]:
    """Return Meep's native 2D Yee coordinates for source placement.

    `get_array()` returns fields interpolated to the centered output grid, but
    `mp.Source(component=...)` places currents on the underlying Yee component
    lattice. For 2D TMz, Meep's `Ez` source points lie on the integer `(x, y)`
    grid nodes rather than at cell centers.
    """

    ny, nx = cfg.grid_shape
    dx_um = cfg.dx_m * 1e6

    if component == "Ez":
        return {
            "y": np.arange(ny + 1, dtype=np.float64) * dx_um,
            "x": np.arange(nx + 1, dtype=np.float64) * dx_um,
        }
    raise ValueError(f"Unsupported native-source component {component!r}")


def _beamz_native_tm_coords_2d(component: str, cfg: Benchmark2DConfig) -> dict[str, np.ndarray]:
    from beamz.simulation.yee import tm_xy_full_component_coordinates_2d_um

    return tm_xy_full_component_coordinates_2d_um(
        component,
        cfg.grid_shape,
        cfg.dx_m * 1e6,
    )


def _beamz_component_to_meep_sample_grid_2d(
    arr: np.ndarray,
    source_component: str,
    target_component: str,
    cfg: Benchmark2DConfig,
) -> np.ndarray:
    src = _beamz_raw_coords_2d(source_component, cfg)
    tgt = _meep_sample_coords_2d(target_component, cfg)
    out = np.asarray(arr, dtype=np.float32)
    out = _interp_axis_between_coords_2d(out, 0, src["y"], tgt["y"])
    out = _interp_axis_between_coords_2d(out, 1, src["x"], tgt["x"])
    return out


def _beamz_full_tm_component_to_meep_sample_grid_2d(
    arr: np.ndarray,
    component: str,
    cfg: Benchmark2DConfig,
) -> np.ndarray:
    del cfg
    if component == "Ez":
        return np.asarray(
            0.25 * (arr[:-1, :-1] + arr[:-1, 1:] + arr[1:, :-1] + arr[1:, 1:]),
            dtype=np.float32,
        )
    elif component == "Hx":
        return np.asarray(0.5 * (arr[:-1, :-1] + arr[:-1, 1:]), dtype=np.float32)
    elif component == "Hy":
        return np.asarray(0.5 * (arr[:-1, :-1] + arr[1:, :-1]), dtype=np.float32)
    raise ValueError(f"Unsupported full-state component {component!r}")


def _beamz_raw_tm_component_to_meep_raw_grid_2d(
    sim,
    component: str,
    cfg: Benchmark2DConfig,
) -> np.ndarray:
    del cfg
    if component == "Ez":
        return np.asarray(sim.fields.Ez, dtype=np.float32)
    raise ValueError(f"Unsupported raw TM component {component!r}")


def _beamz_component_coordinates_um(cfg: Benchmark2DConfig) -> dict[str, dict[str, list[float]]]:
    return {
        "Ez": {axis: values.tolist() for axis, values in _meep_sample_coords_2d("Ez", cfg).items()},
        "Hx": {axis: values.tolist() for axis, values in _meep_sample_coords_2d("Hx", cfg).items()},
        "Hy": {axis: values.tolist() for axis, values in _meep_sample_coords_2d("Hy", cfg).items()},
    }


def _meep_center_2d(x_um: float, y_um: float, cfg: Benchmark2DConfig) -> tuple[float, float]:
    half = 0.5 * cfg.domain_um
    return (x_um - half, y_um - half)


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
    def __init__(self, *, signal, pixel_indices: list[tuple[int, int]], pixel_weights: np.ndarray):
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


def _build_beamz_design(cfg: Benchmark2DConfig, structure_specs: list[dict[str, Any]]):
    from beamz import Design, Material, Rectangle

    design = Design(
        width=cfg.domain_m,
        height=cfg.domain_m,
        material=Material(cfg.background_index**2),
    )
    for spec in structure_specs:
        material = Material(spec["permittivity"])
        if spec["type"] != "block":
            raise ValueError(f"Unsupported 2D structure type {spec['type']!r}")
        design += Rectangle(
            position=(
                float(spec["position_um"][0]) * 1e-6,
                float(spec["position_um"][1]) * 1e-6,
            ),
            width=float(spec["size_um"][0]) * 1e-6,
            height=float(spec["size_um"][1]) * 1e-6,
            material=material,
        )
    return design


def _rasterize_beamz_permittivity(cfg: Benchmark2DConfig, structure_specs: list[dict[str, Any]]) -> np.ndarray:
    from beamz import PEC, Simulation

    design = _build_beamz_design(cfg, structure_specs)
    sim = Simulation(
        design=design,
        sources=[],
        boundaries=[PEC(edges="all")],
        time=np.asarray([0.0, cfg.dt_s], dtype=np.float64),
        resolution=cfg.dx_m,
        plane_2d="xy",
    )
    return np.asarray(sim.fields.permittivity, dtype=np.float32)


def _write_solver_output(
    *,
    output_path: Path,
    solver_name: str,
    cfg: Benchmark2DConfig,
    structure_specs: list[dict[str, Any]],
    permittivity: np.ndarray,
    snapshots: list[dict[str, Any]],
    dft_monitor_x_um: np.ndarray,
    dft_monitor_y_um: np.ndarray,
    dft_monitor_times_s: np.ndarray,
    dft_monitor_ez_history: np.ndarray,
    dft_monitor_ez_running: np.ndarray,
    dft_monitor_ez_running_actual: np.ndarray,
    dft_monitor_ez_final: np.ndarray,
    dft_monitor_decimation_factor: int | None,
    runtime_s: float,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "permittivity": np.asarray(permittivity, dtype=np.float32),
        "snapshot_steps": np.asarray([snap["step"] for snap in snapshots], dtype=np.int32),
        "snapshot_times_s": np.asarray([snap["time_s"] for snap in snapshots], dtype=np.float64),
        "dft_monitor_x_um": np.asarray(dft_monitor_x_um, dtype=np.float64),
        "dft_monitor_y_um": np.asarray(dft_monitor_y_um, dtype=np.float64),
        "dft_monitor_times_s": np.asarray(dft_monitor_times_s, dtype=np.float64),
        "dft_monitor_Ez_history": np.asarray(dft_monitor_ez_history, dtype=np.complex128),
        "dft_monitor_Ez_running": np.asarray(dft_monitor_ez_running, dtype=np.complex128),
        "dft_monitor_Ez_running_actual": np.asarray(
            dft_monitor_ez_running_actual, dtype=np.complex128
        ),
        "dft_monitor_Ez": np.asarray(dft_monitor_ez_final, dtype=np.complex128),
    }
    for component in ("Ez", "Hx", "Hy"):
        arrays[component] = np.stack(
            [np.asarray(snap["fields"][component], dtype=np.float32) for snap in snapshots]
        )
    np.savez_compressed(output_path, **arrays)

    metadata = {
        "solver": solver_name,
        "config": asdict(cfg),
        "grid_shape": list(np.asarray(permittivity).shape),
        "runtime_s": float(runtime_s),
        "data_file": output_path.name,
        "snapshots": [
            {"step": int(snap["step"]), "time_s": float(snap["time_s"])}
            for snap in snapshots
        ],
        "dft_monitor": {
            "x_um": float(np.asarray(dft_monitor_x_um, dtype=np.float64).reshape(-1)[0]),
            "point_count": int(np.asarray(dft_monitor_y_um).size),
            "history_steps": int(np.asarray(dft_monitor_times_s).size),
            "decimation_factor": (
                None
                if dft_monitor_decimation_factor is None
                else int(dft_monitor_decimation_factor)
            ),
        },
        "structure_specs": structure_specs,
        "axis_order": "yx_meep_sampled",
        "field_grid": "meep-sampled",
        "component_coordinates_um": _beamz_component_coordinates_um(cfg),
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    return metadata


def run_beamz_single(
    cfg: Benchmark2DConfig,
    *,
    structure_specs: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    from beamz import Monitor, PEC, Simulation

    signal = _gaussian_modulated_signal_fn(cfg)
    design = _build_beamz_design(cfg, structure_specs)
    ez_coords = _beamz_native_tm_coords_2d("Ez", cfg)
    pixels, weights = _support_pixels_2d(
        ez_coords,
        center_um=(cfg.source_x_um, cfg.source_y_um),
        width_um=cfg.source_width_um,
    )
    source = _BeamzCurrentSource2D(signal=signal, pixel_indices=pixels, pixel_weights=weights)
    monitor_start, monitor_end = _dft_monitor_line_endpoints_m(cfg)
    dft_monitor = Monitor(
        design=design,
        start=monitor_start,
        end=monitor_end,
        name="wg_dft",
        record_fields=False,
        accumulate_power=False,
        dft_enabled=True,
        dft_frequencies=np.array([cfg.frequency_hz], dtype=np.float64),
        dft_components=("Ez",),
        dft_window="rect",
        dft_record_interval=max(1, int(cfg.dft_decimation_factor)),
        dft_normalization="meep",
        dft_length_unit=1e-6,
    )
    # Meep's `run(until=...)` captures one more post-update sample at the final
    # time in this benchmark. Match that here so final DFT outputs are compared
    # at the same physical end time.
    time_axis = np.arange(cfg.num_steps + 1, dtype=np.float64) * cfg.dt_s
    sim = Simulation(
        design=design,
        sources=[source],
        monitors=[dft_monitor],
        boundaries=[PEC(edges="all")],
        time=time_axis,
        resolution=cfg.dx_m,
        plane_2d="xy",
    )
    dft_monitor_x_um, dft_monitor_y_um, dft_y_idx, _ = _dft_monitor_sample_geometry(cfg)
    ez_sample_x_um = np.asarray(_meep_sample_coords_2d("Ez", cfg)["x"], dtype=np.float64)

    def _export_beamz_state() -> dict[str, np.ndarray]:
        return {
            "Ez": np.asarray(sim.fields.Ez, dtype=np.float32),
            "Hx": -_beamz_component_to_meep_sample_grid_2d(
                np.asarray(sim.fields.Hy),
                "Hy",
                "Hx",
                cfg,
            ),
            "Hy": -_beamz_component_to_meep_sample_grid_2d(
                np.asarray(sim.fields.Hx),
                "Hx",
                "Hy",
                cfg,
            ),
        }

    state_cache: dict[int, dict[str, Any]] = {}
    dft_line_history: list[np.ndarray] = []
    dft_times_s: list[float] = []
    dft_running_actual: list[np.ndarray] = []
    start = time.perf_counter()
    while sim.step():
        exported = _export_beamz_state()
        step = int(sim.current_step)
        state_cache[step] = {
            "step": step,
            "time_s": float(sim.t),
            "fields": exported,
        }
        dft_line_history.append(
            _sample_vertical_line_on_centered_grid(
                np.asarray(exported["Ez"], dtype=np.complex128),
                x_coords_um=ez_sample_x_um,
                x_line_um=float(cfg.dft_monitor_x_um),
            )[dft_y_idx]
        )
        dft_times_s.append(float(sim.t))
        dft_running_actual.append(
            np.asarray(dft_monitor.get_dft_component("Ez"), dtype=np.complex128).reshape(-1)
        )
    runtime_s = time.perf_counter() - start

    snapshots: list[dict[str, Any]] = []
    for target_step in cfg.snapshot_steps:
        if target_step not in state_cache or (target_step + 1) not in state_cache:
            raise RuntimeError(
                f"Beamz 2D simulation did not retain steps {target_step} and {target_step + 1}"
            )
        cur = state_cache[target_step]
        nxt = state_cache[target_step + 1]
        snapshots.append(
            {
                "step": int(cur["step"]),
                "time_s": float(cur["time_s"]),
                "fields": {
                    "Ez": np.asarray(cur["fields"]["Ez"], dtype=np.float32),
                    # Meep's H output is temporally staggered; compare against
                    # a time-centered Beamz H built from consecutive steps.
                    "Hx": 0.5
                    * (
                        np.asarray(cur["fields"]["Hx"], dtype=np.float32)
                        + np.asarray(nxt["fields"]["Hx"], dtype=np.float32)
                    ),
                    "Hy": 0.5
                    * (
                        np.asarray(cur["fields"]["Hy"], dtype=np.float32)
                        + np.asarray(nxt["fields"]["Hy"], dtype=np.float32)
                    ),
                },
            }
        )

    dft_history_arr = np.asarray(dft_line_history, dtype=np.complex128)
    dft_times_arr = np.asarray(dft_times_s, dtype=np.float64)
    dft_running = _running_dft_from_history(
        dft_history_arr,
        times_s=dft_times_arr,
        frequency_hz=cfg.frequency_hz,
        length_unit_m=1e-6,
        decimation_factor=cfg.dft_decimation_factor,
    )
    dft_running_actual_arr = np.asarray(dft_running_actual, dtype=np.complex128)
    dft_final = np.asarray(dft_monitor.get_dft_component("Ez"), dtype=np.complex128).reshape(-1)

    out_path = output_dir / "structure_000" / "beamz_fields_meep_sampled.npz"
    return _write_solver_output(
        output_path=out_path,
        solver_name="beamz",
        cfg=cfg,
        structure_specs=structure_specs,
        permittivity=np.asarray(sim.fields.permittivity),
        snapshots=snapshots,
        dft_monitor_x_um=dft_monitor_x_um,
        dft_monitor_y_um=dft_monitor_y_um,
        dft_monitor_times_s=dft_times_arr,
        dft_monitor_ez_history=dft_history_arr,
        dft_monitor_ez_running=dft_running,
        dft_monitor_ez_running_actual=dft_running_actual_arr,
        dft_monitor_ez_final=dft_final,
        dft_monitor_decimation_factor=max(1, int(cfg.dft_decimation_factor)),
        runtime_s=runtime_s,
    )


def run_meep_single(
    cfg: Benchmark2DConfig,
    *,
    structure_specs: list[dict[str, Any]],
    output_dir: Path,
    beamz_raster_file: Path | None,
) -> dict[str, Any]:
    import meep as mp

    resolution_px_per_um = 1.0 / (cfg.dx_m * 1e6)
    frequency_um_inv = 1.0 / cfg.wavelength_um
    sigma_freq_hz = 1.0 / max(2.0 * np.pi * cfg.pulse_sigma_s, 1e-30)
    fwidth_um_inv = sigma_freq_hz * (1e-6 / C0)

    geometry = []
    default_material: Any = mp.Medium(index=cfg.background_index)
    epsilon_func = None
    eps_averaging = True
    shared_beamz_eps_yx: np.ndarray | None = None
    if cfg.geometry_source == "beamz-raster":
        if beamz_raster_file is None:
            raise ValueError("beamz_raster_file is required for beamz-raster geometry mode")
        shared_beamz_eps_yx = np.asarray(np.load(beamz_raster_file), dtype=np.float32)
        eps_grid = np.asarray(shared_beamz_eps_yx, dtype=np.float64)
        dx_um = cfg.dx_m * 1e6

        def epsilon_func(p):
            x_um = float(p.x) + 0.5 * cfg.domain_um
            y_um = float(p.y) + 0.5 * cfg.domain_um
            ix = int(np.clip(np.floor(x_um / dx_um), 0, eps_grid.shape[1] - 1))
            iy = int(np.clip(np.floor(y_um / dx_um), 0, eps_grid.shape[0] - 1))
            return float(eps_grid[iy, ix])

        eps_averaging = False
    else:
        for spec in structure_specs:
            medium = mp.Medium(index=spec["index"])
            if spec["type"] != "block":
                raise ValueError(f"Unsupported 2D structure type {spec['type']!r}")
            sx_um, sy_um = spec["size_um"]
            px_um, py_um = spec["position_um"]
            cx_um = float(px_um) + 0.5 * float(sx_um)
            cy_um = float(py_um) + 0.5 * float(sy_um)
            mx, my = _meep_center_2d(cx_um, cy_um, cfg)
            geometry.append(
                mp.Block(
                    size=mp.Vector3(float(sx_um), float(sy_um), mp.inf),
                    center=mp.Vector3(mx, my),
                    material=medium,
                )
            )

    beamz_signal = _gaussian_modulated_signal_fn(cfg)

    def meep_signal(t_um):
        t_s = float(t_um) * 1e-6 / C0
        return beamz_signal(t_s)

    gaussian_time = mp.CustomSource(
        src_func=meep_signal,
        start_time=0.0,
        end_time=cfg.num_steps * cfg.dt_s * C0 / 1e-6,
        center_frequency=frequency_um_inv,
        fwidth=fwidth_um_inv,
    )

    ez_coords = _meep_native_source_coords_2d("Ez", cfg)
    pixels, weights = _support_pixels_2d(
        ez_coords,
        center_um=(cfg.source_x_um, cfg.source_y_um),
        width_um=cfg.source_width_um * cfg.meep_source_width_scale,
    )
    sources = []
    for (iy, ix), weight in zip(pixels, weights, strict=False):
        x_um = float(ez_coords["x"][ix])
        y_um = float(ez_coords["y"][iy])
        mx, my = _meep_center_2d(x_um, y_um, cfg)
        sources.append(
            mp.Source(
                src=gaussian_time,
                component=mp.Ez,
                center=mp.Vector3(mx, my),
                size=mp.Vector3(),
                amplitude=float(cfg.meep_source_amplitude_scale * weight),
            )
        )

    sim = mp.Simulation(
        cell_size=mp.Vector3(cfg.domain_um, cfg.domain_um, 0.0),
        geometry=geometry,
        sources=sources,
        eps_averaging=eps_averaging,
        boundary_layers=[],
        default_material=default_material,
        epsilon_func=epsilon_func,
        resolution=resolution_px_per_um,
        Courant=cfg.courant_safety / math.sqrt(2.0),
        dimensions=2,
    )
    # Make the Meep cavity walls explicit so Beamz/Meep runs share the same PEC box.
    for side in (mp.Low, mp.High):
        sim.set_boundary(side, mp.X, mp.Metallic)
        sim.set_boundary(side, mp.Y, mp.Metallic)

    cell_center = mp.Vector3()
    cell_size = mp.Vector3(cfg.domain_um, cfg.domain_um, 0.0)
    dft_monitor_x_um, dft_monitor_y_um, _, _ = _dft_monitor_sample_geometry(cfg)
    dft_monitor_center = mp.Vector3(cfg.dft_monitor_x_um - 0.5 * cfg.domain_um, 0.0, 0.0)
    dft_monitor_size = mp.Vector3(0.0, cfg.domain_um, 0.0)
    dft_monitor = sim.add_dft_fields(
        [mp.Ez],
        frequency_um_inv,
        0.0,
        1,
        center=dft_monitor_center,
        size=dft_monitor_size,
        decimation_factor=max(1, int(cfg.dft_decimation_factor)),
    )
    meep_dft_decimation = _meep_dft_decimation_factor(dft_monitor)
    dft_point_positions = [
        mp.Vector3(
            float(x) - 0.5 * cfg.domain_um,
            float(y) - 0.5 * cfg.domain_um,
            0.0,
        )
        for x, y in zip(dft_monitor_x_um, dft_monitor_y_um, strict=True)
    ]
    dx_um = cfg.dx_m * 1e6
    raw_span_um = cfg.domain_um - 2.0 * dx_um
    snapshots: list[dict[str, Any] | None] = [None] * len(cfg.snapshot_steps)
    snapshot_index_by_step = {int(step): idx for idx, step in enumerate(cfg.snapshot_steps)}
    dt_um = cfg.dt_s * C0 / 1e-6
    dft_line_history: list[np.ndarray] = []
    dft_times_s: list[float] = []
    dft_running_actual: list[np.ndarray] = []

    def _capture_if_requested(sim_obj):
        step = int(round(float(sim_obj.meep_time()) / dt_um))
        ez_line = np.asarray(
            [sim_obj.get_field_point(mp.Ez, p) for p in dft_point_positions],
            dtype=np.complex128,
        )
        dft_line_history.append(ez_line)
        dft_times_s.append(float(step * cfg.dt_s))
        dft_running_actual.append(
            np.asarray(sim_obj.get_dft_array(dft_monitor, mp.Ez, 0), dtype=np.complex128).reshape(-1)
        )
        idx = snapshot_index_by_step.get(step)
        if idx is not None:
            sim_obj.fields.synchronize_magnetic_fields()
            try:
                hx = _canonicalize_meep_array_2d(
                    sim_obj.get_array(
                        component=mp.Hx,
                        center=mp.Vector3(0.0, -0.5 * dx_um),
                        size=mp.Vector3(cfg.domain_um, raw_span_um, 0.0),
                        snap=True,
                    )
                ) / Z0_OHM
                hy = _canonicalize_meep_array_2d(
                    sim_obj.get_array(
                        component=mp.Hy,
                        center=mp.Vector3(-0.5 * dx_um, 0.0),
                        size=mp.Vector3(raw_span_um, cfg.domain_um, 0.0),
                        snap=True,
                    )
                ) / Z0_OHM
            finally:
                sim_obj.fields.restore_magnetic_fields()
            snapshots[idx] = {
                "step": step,
                "time_s": float(step * cfg.dt_s),
                "fields": {
                    "Hx": hx,
                    "Hy": hy,
                    "Ez": _canonicalize_meep_array_2d(
                        sim_obj.get_array(
                            component=mp.Ez,
                            center=cell_center,
                            size=cell_size,
                            snap=True,
                        )
                    ),
                },
            }

    hooks = [mp.at_every(dt_um, _capture_if_requested)]
    mp.verbosity(0)
    sim.init_sim()
    start = time.perf_counter()
    sim.run(*hooks, until=cfg.total_time_s * C0 / 1e-6)
    runtime_s = time.perf_counter() - start
    if any(snap is None for snap in snapshots):
        raise RuntimeError("Meep 2D did not emit all requested snapshots")

    if shared_beamz_eps_yx is not None:
        permittivity = shared_beamz_eps_yx
    else:
        permittivity = _canonicalize_meep_array_2d(
            sim.get_array(center=cell_center, size=cell_size, component=mp.Dielectric)
        )

    dft_history_arr = np.asarray(dft_line_history, dtype=np.complex128)
    dft_times_arr = np.asarray(dft_times_s, dtype=np.float64)
    dft_running = _running_dft_from_history(
        dft_history_arr,
        times_s=dft_times_arr,
        frequency_hz=cfg.frequency_hz,
        length_unit_m=1e-6,
        decimation_factor=cfg.dft_decimation_factor,
    )
    dft_running_actual_arr = np.asarray(dft_running_actual, dtype=np.complex128)
    dft_final = np.asarray(sim.get_dft_array(dft_monitor, mp.Ez, 0), dtype=np.complex128).reshape(-1)

    out_path = output_dir / "structure_000" / "meep_fields_meep_sampled.npz"
    return _write_solver_output(
        output_path=out_path,
        solver_name="meep",
        cfg=cfg,
        structure_specs=structure_specs,
        permittivity=permittivity,
        snapshots=[snap for snap in snapshots if snap is not None],
        dft_monitor_x_um=dft_monitor_x_um,
        dft_monitor_y_um=dft_monitor_y_um,
        dft_monitor_times_s=dft_times_arr,
        dft_monitor_ez_history=dft_history_arr,
        dft_monitor_ez_running=dft_running,
        dft_monitor_ez_running_actual=dft_running_actual_arr,
        dft_monitor_ez_final=dft_final,
        dft_monitor_decimation_factor=meep_dft_decimation,
        runtime_s=runtime_s,
    )


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
    parser = argparse.ArgumentParser(description="Compare Beamz and Meep on a 2D straight-waveguide benchmark.")
    parser.add_argument("--backend", choices=("beamz", "meep", "both"), default="both")
    parser.add_argument("--resolution-ppw", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "waveguide_results_2d")
    parser.add_argument("--geometry-source", choices=("native", "beamz-raster"), default="beamz-raster")
    parser.add_argument("--waveguide-width-um", type=float, default=0.60)
    parser.add_argument("--waveguide-index", type=float, default=math.sqrt(12.0))
    parser.add_argument("--dft-decimation-factor", type=int, default=1)
    parser.add_argument("--meep-source-amplitude-scale", type=float, default=5.887e-06)
    parser.add_argument("--meep-env", default=os.getenv("BEAMZ_MEEP_ENV", "beamz-meep"))
    parser.add_argument("--emit-json-only", action="store_true")
    parser.add_argument("--beamz-raster-file", type=Path, default=None)
    args = parser.parse_args()

    cfg = Benchmark2DConfig(
        resolution_ppw=args.resolution_ppw,
        geometry_source=args.geometry_source,
        waveguide_width_um=args.waveguide_width_um,
        waveguide_index=args.waveguide_index,
        dft_decimation_factor=max(1, int(args.dft_decimation_factor)),
        meep_source_amplitude_scale=args.meep_source_amplitude_scale,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    structure_specs = _waveguide_specs(cfg)
    structure_dir = output_dir / "structure_000"
    structure_dir.mkdir(parents=True, exist_ok=True)
    beamz_raster_file = args.beamz_raster_file
    if cfg.geometry_source == "beamz-raster" and beamz_raster_file is None:
        beamz_raster_file = structure_dir / "beamz_shared_permittivity.npy"
        np.save(beamz_raster_file, _rasterize_beamz_permittivity(cfg, structure_specs))

    (structure_dir / "structure.json").write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "structure_specs": structure_specs,
            },
            indent=2,
        )
    )

    try:
        import meep as _meep  # noqa: F401

        meep_available = True
    except Exception:
        meep_available = False

    meep_cfg = cfg

    results: dict[str, Any] = {}
    backends = ("beamz", "meep") if args.backend == "both" else (args.backend,)
    for backend in backends:
        if backend == "beamz":
            results["beamz"] = run_beamz_single(cfg, structure_specs=structure_specs, output_dir=output_dir)
        elif meep_available:
            results["meep"] = run_meep_single(
                meep_cfg,
                structure_specs=structure_specs,
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
                    "--output-dir",
                    str(output_dir),
                    "--geometry-source",
                    str(cfg.geometry_source),
                    "--waveguide-width-um",
                    str(cfg.waveguide_width_um),
                    "--waveguide-index",
                    str(cfg.waveguide_index),
                    "--dft-decimation-factor",
                    str(cfg.dft_decimation_factor),
                    "--meep-source-amplitude-scale",
                    str(meep_cfg.meep_source_amplitude_scale),
                    *extra_args,
                ],
            )["results"]["meep"]

    summary = {
        "generated_at_epoch_s": time.time(),
        "config": asdict(cfg),
        "effective_meep_source_amplitude_scale": float(meep_cfg.meep_source_amplitude_scale),
        "field_grid": "meep-sampled",
        "structure_specs": structure_specs,
        "results": results,
    }
    if "beamz" in results and "meep" in results:
        structure_dir = output_dir / "structure_000"
        beamz_npz = np.load(structure_dir / "beamz_fields_meep_sampled.npz")
        meep_npz = np.load(structure_dir / "meep_fields_meep_sampled.npz")
        summary["comparison"] = {
            "dft_monitor_ez_final": _compare_complex_profiles(
                beamz_npz["dft_monitor_Ez"],
                meep_npz["dft_monitor_Ez"],
            ),
            "dft_monitor_ez_final_conjugated_meep": _compare_complex_profiles(
                beamz_npz["dft_monitor_Ez"],
                np.conj(np.asarray(meep_npz["dft_monitor_Ez"], dtype=np.complex128)),
            ),
            "dft_monitor_ez_running": _compare_running_dft(
                beamz_npz["dft_monitor_Ez_running"],
                meep_npz["dft_monitor_Ez_running"],
            ),
            "dft_monitor_ez_running_actual": _compare_running_dft(
                beamz_npz["dft_monitor_Ez_running_actual"],
                meep_npz["dft_monitor_Ez_running_actual"],
            ),
        }
    (output_dir / "manifest.json").write_text(json.dumps(summary, indent=2))
    if args.emit_json_only:
        print(json.dumps(summary))
    else:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
