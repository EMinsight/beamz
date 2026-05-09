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


REPO_ROOT = next(
    candidate
    for candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents)
    if (candidate / "beamz").exists() and (candidate / "examples").exists()
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent

C0 = 299_792_458.0
FIELD_GRID_CENTERED = "centered"
FIELD_GRID_RAW_YEE = "raw-yee"
FIELD_GRID_MEEP_SAMPLED = "meep-sampled"


@dataclass(frozen=True)
class RandomBenchmarkConfig:
    domain_um: float = 3.0
    wavelength_um: float = 1.0
    resolution_ppw: int = 8
    courant_safety: float = 0.95
    total_cycles: float = 4.0
    pulse_center_cycles: float = 1.25
    pulse_sigma_cycles: float = 0.35
    background_index: float = 1.0
    index_min: float = 1.7
    index_max: float = 2.8
    num_primitives: int = 4
    geometry_mode: str = "random-primitives"
    geometry_source: str = "native"
    source_x_um: float = 0.75
    source_y_um: float = 1.5
    source_z_um: float = 1.5
    source_width_um: float = 0.18
    meep_source_width_scale: float = 1.0
    meep_source_amplitude_scale: float = 7.37e-07
    source_box_um: float = 1.0
    source_clearance_um: float = 0.55
    structure_margin_um: float = 0.20
    num_snapshots: int = 3
    polygon_permittivity: float = 12.0

    @property
    def dx_m(self) -> float:
        return (self.wavelength_um * 1e-6) / float(self.resolution_ppw)

    @property
    def frequency_hz(self) -> float:
        return C0 / (self.wavelength_um * 1e-6)

    @property
    def dt_s(self) -> float:
        return self.courant_safety * self.dx_m / (C0 * math.sqrt(3.0))

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
    def grid_shape(self) -> tuple[int, int, int]:
        n = int(round(self.domain_m / self.dx_m))
        return (n, n, n)

    @property
    def num_steps(self) -> int:
        return max(2, int(math.floor(self.total_time_s / self.dt_s)))

    @property
    def snapshot_steps(self) -> list[int]:
        raw = np.linspace(0.25, 0.75, self.num_snapshots)
        steps = []
        for frac in raw:
            step = int(round(frac * (self.num_steps - 1)))
            step = min(max(step, 1), self.num_steps - 1)
            if not steps or step != steps[-1]:
                steps.append(step)
        return steps

    @property
    def snapshot_times_s(self) -> list[float]:
        return [step * self.dt_s for step in self.snapshot_steps]


def _gaussian_modulated_signal_value(cfg: RandomBenchmarkConfig, t_s: float) -> float:
    envelope = math.exp(-0.5 * ((t_s - cfg.pulse_center_s) / cfg.pulse_sigma_s) ** 2)
    carrier = math.cos(2.0 * math.pi * cfg.frequency_hz * t_s)
    return float(envelope * carrier)


def _gaussian_modulated_signal_fn(cfg: RandomBenchmarkConfig):
    def _signal(t_s: float) -> float:
        return _gaussian_modulated_signal_value(cfg, float(t_s))

    return _signal


def _gaussian_modulated_signal_samples(cfg: RandomBenchmarkConfig) -> np.ndarray:
    t = np.arange(cfg.num_steps, dtype=np.float64) * cfg.dt_s
    values = [_gaussian_modulated_signal_value(cfg, float(tt)) for tt in t]
    return np.asarray(values, dtype=np.float32)


def _random_structure_specs(
    cfg: RandomBenchmarkConfig,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    if cfg.geometry_mode == "random-polygon":
        return [_random_polygon_prism_spec(cfg, seed=seed)]

    rng = np.random.default_rng(seed)
    specs: list[dict[str, Any]] = []
    source_center = np.array([cfg.source_x_um, cfg.source_y_um, cfg.source_z_um])
    lower = cfg.structure_margin_um
    upper = cfg.domain_um - cfg.structure_margin_um

    attempts = 0
    while len(specs) < cfg.num_primitives and attempts < 200:
        attempts += 1
        primitive_type = "block" if rng.random() < 0.7 else "sphere"
        refr_index = float(rng.uniform(cfg.index_min, cfg.index_max))

        if primitive_type == "block":
            size = rng.uniform(0.30, 0.85, size=3)
            position = rng.uniform(lower, upper - size)
            center = position + 0.5 * size
            if np.linalg.norm(center - source_center) < cfg.source_clearance_um:
                continue
            specs.append(
                {
                    "type": "block",
                    "position_um": position.tolist(),
                    "size_um": size.tolist(),
                    "index": refr_index,
                    "permittivity": refr_index**2,
                }
            )
            continue

        radius = float(rng.uniform(0.16, 0.38))
        center = rng.uniform(lower + radius, upper - radius, size=3)
        if np.linalg.norm(center - source_center) < (cfg.source_clearance_um + radius):
            continue
        specs.append(
            {
                "type": "sphere",
                "center_um": center.tolist(),
                "radius_um": radius,
                "index": refr_index,
                "permittivity": refr_index**2,
            }
        )

    if len(specs) != cfg.num_primitives:
        raise RuntimeError(
            f"Failed to generate {cfg.num_primitives} primitives for seed {seed}"
        )
    return specs


def _random_polygon_prism_spec(
    cfg: RandomBenchmarkConfig,
    *,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    source_center = np.array([cfg.source_x_um, cfg.source_y_um, cfg.source_z_um], dtype=np.float64)
    lower = cfg.structure_margin_um
    upper = cfg.domain_um - cfg.structure_margin_um

    for _ in range(500):
        num_vertices = int(rng.integers(5, 9))
        base_radius = float(rng.uniform(0.28, 0.46))
        depth_um = float(rng.uniform(0.42, 0.82))
        z0_um = float(rng.uniform(lower, upper - depth_um))
        center_x = float(rng.uniform(lower + base_radius, upper - base_radius))
        center_y = float(rng.uniform(lower + base_radius, upper - base_radius))
        center_z = z0_um + 0.5 * depth_um
        if np.linalg.norm(np.array([center_x, center_y, center_z]) - source_center) < (
            cfg.source_clearance_um + 1.15 * base_radius
        ):
            continue

        angle0 = float(rng.uniform(0.0, 2.0 * np.pi))
        angle_steps = rng.uniform(0.7, 1.35, size=num_vertices)
        angles = np.cumsum(angle_steps)
        angles = (angles / angles[-1]) * (2.0 * np.pi)
        angles = angles + angle0
        radii = base_radius * rng.uniform(0.7, 1.18, size=num_vertices)

        vertices_xy: list[list[float]] = []
        valid = True
        for angle, radius in zip(angles, radii, strict=True):
            x = center_x + float(radius * np.cos(angle))
            y = center_y + float(radius * np.sin(angle))
            if not (lower <= x <= upper and lower <= y <= upper):
                valid = False
                break
            vertices_xy.append([x, y])
        if not valid:
            continue

        return {
            "type": "polygon_prism",
            "vertices_um": vertices_xy,
            "z_um": z0_um,
            "depth_um": depth_um,
            "index": float(np.sqrt(cfg.polygon_permittivity)),
            "permittivity": float(cfg.polygon_permittivity),
        }

    raise RuntimeError(f"Failed to generate random polygon prism for seed {seed}")


def _polygon_centroid_xy(vertices_um: list[list[float]]) -> tuple[float, float]:
    pts = np.asarray(vertices_um, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    cross = x * y_next - x_next * y
    area2 = np.sum(cross)
    if abs(area2) < 1e-12:
        return float(np.mean(x)), float(np.mean(y))
    cx = np.sum((x + x_next) * cross) / (3.0 * area2)
    cy = np.sum((y + y_next) * cross) / (3.0 * area2)
    return float(cx), float(cy)


def _meep_center(x_um: float, y_um: float, z_um: float, cfg: RandomBenchmarkConfig):
    half = 0.5 * cfg.domain_um
    return (x_um - half, y_um - half, z_um - half)


def _interp_axis_to_centered_grid(
    arr: np.ndarray,
    axis: int,
    target_len: int,
) -> np.ndarray:
    """Interpolate one axis of a Beamz staggered field onto Meep's centered grid.

    Meep `get_array` returns field components centered on the Yee-grid voxels via
    interpolation of nearby Yee samples. Beamz stores staggered component arrays
    with a slightly different shape convention, so we resample each axis onto the
    centered voxel coordinates before comparing against Meep.
    """
    arr = np.asarray(arr, dtype=np.float32)
    src_len = arr.shape[axis]
    moved = np.moveaxis(arr, axis, 0)

    if src_len == target_len:
        flat = moved.reshape(src_len, -1)
        out_flat = np.empty((target_len, flat.shape[1]), dtype=np.float32)
        if src_len == 1:
            out_flat[0, :] = flat[0, :]
        else:
            out_flat[:-1, :] = 0.5 * (flat[:-1, :] + flat[1:, :])
            out_flat[-1, :] = 0.5 * (flat[-2, :] + flat[-1, :])
        out = out_flat.reshape((target_len, *moved.shape[1:]))
        return np.moveaxis(out, 0, axis)

    if src_len != target_len - 1:
        raise ValueError(
            f"Cannot map source axis length {src_len} to target length {target_len}"
        )

    src_coords = np.arange(src_len, dtype=np.float64)
    target_coords = np.arange(target_len, dtype=np.float64) + 0.5
    flat = moved.reshape(src_len, -1)
    out_flat = np.empty((target_len, flat.shape[1]), dtype=np.float32)
    for idx in range(flat.shape[1]):
        out_flat[:, idx] = np.interp(
            target_coords,
            src_coords,
            flat[:, idx],
            left=float(flat[0, idx]),
            right=float(flat[-1, idx]),
        )
    out = out_flat.reshape((target_len, *moved.shape[1:]))
    return np.moveaxis(out, 0, axis)


def _beamz_component_to_centered_grid(
    arr: np.ndarray,
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    for axis, target_len in enumerate(target_shape):
        out = _interp_axis_to_centered_grid(out, axis, target_len)
    return out


def _interp_axis_between_coords(
    arr: np.ndarray,
    axis: int,
    src_coords: np.ndarray,
    target_coords: np.ndarray,
    *,
    left_fill: np.ndarray | None = None,
    right_fill: np.ndarray | None = None,
    left_coord: float | None = None,
    right_coord: float | None = None,
) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    moved = np.moveaxis(arr, axis, 0)
    flat = moved.reshape(moved.shape[0], -1)
    out_flat = np.empty((target_coords.size, flat.shape[1]), dtype=np.float32)
    if left_fill is None:
        left_fill = flat[0]
    else:
        left_fill = np.asarray(left_fill, dtype=np.float32).reshape(-1)
    if right_fill is None:
        right_fill = flat[-1]
    else:
        right_fill = np.asarray(right_fill, dtype=np.float32).reshape(-1)
    for idx in range(flat.shape[1]):
        src_axis = src_coords
        vals = flat[:, idx]
        if left_coord is not None and left_fill is not None:
            src_axis = np.concatenate([[left_coord], src_axis])
            vals = np.concatenate([[left_fill[idx]], vals])
        if right_coord is not None and right_fill is not None:
            src_axis = np.concatenate([src_axis, [right_coord]])
            vals = np.concatenate([vals, [right_fill[idx]]])
        out_flat[:, idx] = np.interp(
            target_coords,
            src_axis,
            vals,
            left=float(vals[0]),
            right=float(vals[-1]),
        )
    out = out_flat.reshape((target_coords.size, *moved.shape[1:]))
    return np.moveaxis(out, 0, axis)


def _beamz_component_to_meep_sample_grid(
    arr: np.ndarray,
    component: str,
    target_shape: tuple[int, int, int],
    dx_um: float,
) -> np.ndarray:
    src = _beamz_component_coordinates_for_component(component, target_shape, dx_um, raw_yee=True)
    target = _meep_component_coordinates_for_component(component, target_shape, dx_um)
    out = np.asarray(arr, dtype=np.float32)
    for axis, name in enumerate(("z", "y", "x")):
        src_vals = np.asarray(src[name], dtype=np.float64)
        tgt_vals = np.asarray(target[name], dtype=np.float64)
        if np.allclose(src_vals, tgt_vals):
            continue
        left_fill = None
        right_fill = None
        left_coord = None
        right_coord = None
        # Beamz's compact 3D E storage omits the high-wall tangential planes.
        # For the Meep-style centered samples near the high edge, use the PEC
        # wall value (0) rather than flat extrapolation of the last interior
        # sample; otherwise the apparent wall is shifted inward by one cell.
        if component in ("Ex", "Ey", "Ez") and src_vals[0] == 0.0:
            offsets = {
                "Ex": {"z": 0.0, "y": 0.0, "x": 0.5},
                "Ey": {"z": 0.0, "y": 0.5, "x": 0.0},
                "Ez": {"z": 0.5, "y": 0.0, "x": 0.0},
            }[component]
            if offsets[name] == 0.0 and tgt_vals[-1] > src_vals[-1]:
                right_fill = np.zeros(np.prod(np.asarray(out.shape)[np.arange(out.ndim) != axis]), dtype=np.float32)
                step = float(np.median(np.diff(tgt_vals)))
                right_coord = float(tgt_vals[-1] + 0.5 * step)
        out = _interp_axis_between_coords(
            out,
            axis,
            src_vals,
            tgt_vals,
            left_fill=left_fill,
            right_fill=right_fill,
            left_coord=left_coord,
            right_coord=right_coord,
        )
    return out


def _source_voxels(
    cfg: RandomBenchmarkConfig,
    *,
    width_um: float,
) -> tuple[list[tuple[int, int, int]], np.ndarray]:
    sigma_grid = (width_um * 1e-6) / cfg.dx_m
    radius_grid = int(np.ceil(4 * sigma_grid))
    nz, ny, nx = cfg.grid_shape
    cx = int(round((cfg.source_x_um * 1e-6) / cfg.dx_m))
    cy = int(round((cfg.source_y_um * 1e-6) / cfg.dx_m))
    cz = int(round((cfg.source_z_um * 1e-6) / cfg.dx_m))

    x_start, x_end = max(0, cx - radius_grid), min(nx, cx + radius_grid + 1)
    y_start, y_end = max(0, cy - radius_grid), min(ny, cy + radius_grid + 1)
    z_start, z_end = max(0, cz - radius_grid), min(nz, cz + radius_grid + 1)

    x_coords = (np.arange(x_start, x_end, dtype=np.float64) + 0.5) * cfg.dx_m
    y_coords = (np.arange(y_start, y_end, dtype=np.float64) + 0.5) * cfg.dx_m
    z_coords = (np.arange(z_start, z_end, dtype=np.float64) + 0.5) * cfg.dx_m
    zz, yy, xx = np.meshgrid(z_coords, y_coords, x_coords, indexing="ij")
    dist_sq = (
        (xx - cfg.source_x_um * 1e-6) ** 2
        + (yy - cfg.source_y_um * 1e-6) ** 2
        + (zz - cfg.source_z_um * 1e-6) ** 2
    )
    weights = np.exp(-dist_sq / (2.0 * (width_um * 1e-6) ** 2))

    voxels = []
    for iz in range(z_start, z_end):
        for iy in range(y_start, y_end):
            for ix in range(x_start, x_end):
                voxels.append((iz, iy, ix))
    return voxels, np.asarray(weights, dtype=np.float64).reshape(-1)


def _beamz_ez_source_voxels(
    cfg: RandomBenchmarkConfig,
    *,
    width_um: float,
) -> tuple[list[tuple[int, int, int]], np.ndarray]:
    from beamz.simulation.yee import component_coordinates_3d_um, nearest_support_indices_3d

    coords = component_coordinates_3d_um("Ez", cfg.grid_shape, cfg.dx_m * 1e6)
    return nearest_support_indices_3d(
        coords,
        center_um=(cfg.source_x_um, cfg.source_y_um, cfg.source_z_um),
        width_um=width_um,
    )


def _meep_ez_source_voxels(
    cfg: RandomBenchmarkConfig,
    *,
    width_um: float,
) -> tuple[list[tuple[int, int, int]], np.ndarray]:
    sigma = float(width_um)
    dx_um = cfg.dx_m * 1e6
    radius_cells = int(np.ceil(4.0 * sigma / dx_um))

    z = (np.arange(cfg.grid_shape[0] - 1, dtype=np.float64) + 0.5) * dx_um
    y = np.arange(cfg.grid_shape[1], dtype=np.float64) * dx_um
    x = np.arange(cfg.grid_shape[2], dtype=np.float64) * dx_um

    x0 = float(cfg.source_x_um)
    y0 = float(cfg.source_y_um)
    z0 = float(cfg.source_z_um)
    cx = int(np.argmin(np.abs(x - x0)))
    cy = int(np.argmin(np.abs(y - y0)))
    cz = int(np.argmin(np.abs(z - z0)))

    x_start, x_end = max(0, cx - radius_cells), min(x.size, cx + radius_cells + 1)
    y_start, y_end = max(0, cy - radius_cells), min(y.size, cy + radius_cells + 1)
    z_start, z_end = max(0, cz - radius_cells), min(z.size, cz + radius_cells + 1)

    zz, yy, xx = np.meshgrid(z[z_start:z_end], y[y_start:y_end], x[x_start:x_end], indexing="ij")
    dist_sq = (xx - x0) ** 2 + (yy - y0) ** 2 + (zz - z0) ** 2
    weights = np.exp(-dist_sq / (2.0 * sigma**2))

    voxels: list[tuple[int, int, int]] = []
    for iz in range(z_start, z_end):
        for iy in range(y_start, y_end):
            for ix in range(x_start, x_end):
                voxels.append((iz, iy, ix))
    return voxels, np.asarray(weights, dtype=np.float64).reshape(-1)


class _BeamzCurrentSource:
    """Benchmark-local electric current source for Beamz.

    This mirrors the discrete Gaussian point-source support used for Meep in this
    benchmark and drives Beamz through the solver's source-term path rather than
    via direct `Ez` field injection.
    """

    def __init__(
        self,
        *,
        signal,
        voxel_indices: list[tuple[int, int, int]],
        voxel_weights: np.ndarray,
    ) -> None:
        self.signal = signal
        self._voxel_weights = np.asarray(voxel_weights, dtype=np.float32)
        z_idx = np.asarray([idx[0] for idx in voxel_indices], dtype=np.int32)
        y_idx = np.asarray([idx[1] for idx in voxel_indices], dtype=np.int32)
        x_idx = np.asarray([idx[2] for idx in voxel_indices], dtype=np.int32)
        self._indices = (z_idx, y_idx, x_idx)

    def get_source_terms(self, fields, t, dt, current_step, resolution, design):
        del fields, current_step, resolution, design
        signal_value = float(self.signal(float(t) + 0.5 * float(dt)))
        values = -self._voxel_weights * signal_value
        return {"Ez": (values, self._indices)}, {}

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
        from beamz.devices.sources.compiler import _as_slab_spec, _sample_waveform
        from beamz.simulation import ops

        denom = 1.0 + np.asarray(fields.sig_z[self._indices], dtype=np.float32) * (
            float(dt)
            / (2.0 * ops.EPS_0 * np.asarray(fields.eps_z[self._indices], dtype=np.float32))
        )
        source_coeff = (
            float(dt)
            / (
                ops.EPS_0 * np.asarray(fields.eps_z[self._indices], dtype=np.float32)
            )
        ) / denom
        coeff = -self._voxel_weights * source_coeff
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
                index=self._indices,
                coeff=coeff,
                waveform=waveform,
                target_shape=tuple(fields.Ez.shape),
            ),
        )


def _extract_centered_beamz_fields(sim) -> dict[str, np.ndarray]:
    perm_shape = tuple(int(v) for v in np.asarray(sim.fields.permittivity).shape)
    return {
        "Ex": _beamz_component_to_centered_grid(np.asarray(sim.fields.Ex), perm_shape),
        "Ey": _beamz_component_to_centered_grid(np.asarray(sim.fields.Ey), perm_shape),
        "Ez": _beamz_component_to_centered_grid(np.asarray(sim.fields.Ez), perm_shape),
    }


def _extract_raw_beamz_fields(sim) -> dict[str, np.ndarray]:
    return {
        "Ex": np.asarray(sim.fields.Ex, dtype=np.float32),
        "Ey": np.asarray(sim.fields.Ey, dtype=np.float32),
        "Ez": np.asarray(sim.fields.Ez, dtype=np.float32),
    }


def _extract_meep_sampled_beamz_fields(sim) -> dict[str, np.ndarray]:
    perm_shape = tuple(int(v) for v in np.asarray(sim.fields.permittivity).shape)
    dx_um = float(sim.resolution) * 1e6
    return {
        "Ex": _beamz_component_to_meep_sample_grid(
            np.asarray(sim.fields.Ex), "Ex", (perm_shape[0], perm_shape[1], perm_shape[2] - 1), dx_um
        ),
        "Ey": _beamz_component_to_meep_sample_grid(
            np.asarray(sim.fields.Ey), "Ey", (perm_shape[0], perm_shape[1] - 1, perm_shape[2]), dx_um
        ),
        "Ez": _beamz_component_to_meep_sample_grid(
            np.asarray(sim.fields.Ez), "Ez", (perm_shape[0] - 1, perm_shape[1], perm_shape[2]), dx_um
        ),
    }


def _solver_output_stem(solver_name: str, field_grid: str) -> str:
    suffix = {
        FIELD_GRID_CENTERED: "",
        FIELD_GRID_RAW_YEE: "_raw_yee",
        FIELD_GRID_MEEP_SAMPLED: "_meep_sampled",
    }[field_grid]
    return f"{solver_name}_fields{suffix}"


def _component_shape_metadata(snapshots: list[dict[str, Any]]) -> dict[str, list[int]]:
    fields = snapshots[0]["fields"]
    return {
        component: list(np.asarray(values).shape)
        for component, values in fields.items()
    }


def _beamz_component_coordinates_um(
    cfg: RandomBenchmarkConfig,
    field_grid: str,
) -> dict[str, dict[str, list[float]]]:
    if field_grid == FIELD_GRID_CENTERED:
        dx_um = cfg.dx_m * 1e6
        nz, ny, nx = cfg.grid_shape
        coords = {
            "z": ((np.arange(nz, dtype=np.float64) + 0.5) * dx_um).tolist(),
            "y": ((np.arange(ny, dtype=np.float64) + 0.5) * dx_um).tolist(),
            "x": ((np.arange(nx, dtype=np.float64) + 0.5) * dx_um).tolist(),
        }
        return {"Ex": coords, "Ey": coords, "Ez": coords}

    def _shape(component: str) -> tuple[int, int, int]:
        nz, ny, nx = cfg.grid_shape
        if component == "Ex":
            return (nz, ny, nx - 1)
        if component == "Ey":
            return (nz, ny - 1, nx)
        if component == "Ez":
            return (nz - 1, ny, nx)
        raise ValueError(f"Unsupported component {component!r}")

    def _offsets(component: str) -> dict[str, float]:
        if component == "Ex":
            return {"z": 0.0, "y": 0.0, "x": 0.5}
        if component == "Ey":
            return {"z": 0.0, "y": 0.5, "x": 0.0}
        if component == "Ez":
            return {"z": 0.5, "y": 0.0, "x": 0.0}
        raise ValueError(f"Unsupported component {component!r}")

    def _coords(component: str, *, raw_yee: bool) -> dict[str, list[float]]:
        shape = _shape(component)
        dx_um = cfg.dx_m * 1e6
        if raw_yee:
            return {
                axis: ((np.arange(length, dtype=np.float64) + _offsets(component)[axis]) * dx_um).tolist()
                for axis, length in zip(("z", "y", "x"), shape, strict=True)
            }
        return {
            axis: ((np.arange(length, dtype=np.float64) + 0.5) * dx_um).tolist()
            for axis, length in zip(("z", "y", "x"), shape, strict=True)
        }

    return {
        component: _coords(component, raw_yee=(field_grid == FIELD_GRID_RAW_YEE))
        for component in ("Ex", "Ey", "Ez")
    }


def _beamz_component_coordinates_for_component(
    component: str,
    shape: tuple[int, int, int],
    dx_um: float,
    *,
    raw_yee: bool,
) -> dict[str, np.ndarray]:
    if raw_yee:
        offsets = {
            "Ex": {"z": 0.0, "y": 0.0, "x": 0.5},
            "Ey": {"z": 0.0, "y": 0.5, "x": 0.0},
            "Ez": {"z": 0.5, "y": 0.0, "x": 0.0},
        }[component]
    else:
        offsets = {"z": 0.5, "y": 0.5, "x": 0.5}
    return {
        axis: (np.arange(length, dtype=np.float64) + offsets[axis]) * dx_um
        for axis, length in zip(("z", "y", "x"), shape, strict=True)
    }


def _meep_component_coordinates_for_component(
    component: str,
    shape: tuple[int, int, int],
    dx_um: float,
) -> dict[str, np.ndarray]:
    del component
    return {
        axis: (np.arange(length, dtype=np.float64) + 0.5) * dx_um
        for axis, length in zip(("z", "y", "x"), shape, strict=True)
    }


def _canonicalize_meep_array(arr: np.ndarray) -> np.ndarray:
    # Meep's get_array returns data ordered by its x, y, z coordinates.
    # Export all arrays in Beamz-style canonical z, y, x order.
    arr_np = np.asarray(arr, dtype=np.float32)
    if arr_np.ndim != 3:
        raise ValueError(f"Expected 3D Meep array, got shape {arr_np.shape}")
    return np.transpose(arr_np, (2, 1, 0))


def _meep_component_coordinates_um(
    cfg: RandomBenchmarkConfig,
    field_grid: str,
) -> dict[str, dict[str, list[float]]]:
    return _beamz_component_coordinates_um(cfg, field_grid)


def _write_solver_output(
    *,
    output_path: Path,
    solver_name: str,
    cfg: RandomBenchmarkConfig,
    structure_index: int,
    seed: int,
    structure_specs: list[dict[str, Any]],
    permittivity: np.ndarray,
    snapshots: list[dict[str, Any]],
    runtime_s: float,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {
        "permittivity": np.asarray(permittivity, dtype=np.float32),
        "snapshot_steps": np.asarray([snap["step"] for snap in snapshots], dtype=np.int32),
        "snapshot_times_s": np.asarray(
            [snap["time_s"] for snap in snapshots], dtype=np.float64
        ),
    }
    for component in ("Ex", "Ey", "Ez"):
        arrays[component] = np.stack(
            [np.asarray(snap["fields"][component], dtype=np.float32) for snap in snapshots]
        )
    np.savez_compressed(output_path, **arrays)

    metadata = {
        "solver": solver_name,
        "structure_index": int(structure_index),
        "seed": int(seed),
        "config": asdict(cfg),
        "grid_shape": list(np.asarray(permittivity).shape),
        "runtime_s": float(runtime_s),
        "data_file": output_path.name,
        "snapshots": [
            {"step": int(snap["step"]), "time_s": float(snap["time_s"])}
            for snap in snapshots
        ],
        "structure_specs": structure_specs,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return metadata


def _build_beamz_design(cfg: RandomBenchmarkConfig, structure_specs: list[dict[str, Any]]):
    from beamz import Design, Material, Polygon, Rectangle, Sphere

    design = Design(
        width=cfg.domain_m,
        height=cfg.domain_m,
        depth=cfg.domain_m,
        material=Material(cfg.background_index**2),
    )

    for spec in structure_specs:
        material = Material(spec["permittivity"])
        if spec["type"] == "block":
            position_m = tuple(v * 1e-6 for v in spec["position_um"])
            size_m = tuple(v * 1e-6 for v in spec["size_um"])
            design += Rectangle(
                position=position_m,
                width=size_m[0],
                height=size_m[1],
                depth=size_m[2],
                material=material,
            )
        elif spec["type"] == "sphere":
            center_m = tuple(v * 1e-6 for v in spec["center_um"])
            design += Sphere(
                position=center_m,
                radius=spec["radius_um"] * 1e-6,
                material=material,
            )
        elif spec["type"] == "polygon_prism":
            vertices_m = [(x * 1e-6, y * 1e-6, spec["z_um"] * 1e-6) for x, y in spec["vertices_um"]]
            design += Polygon(
                vertices=vertices_m,
                depth=spec["depth_um"] * 1e-6,
                z=spec["z_um"] * 1e-6,
                material=material,
            )
        else:
            raise ValueError(f"Unsupported primitive type {spec['type']!r}")

    return design


def _rasterize_beamz_permittivity(
    cfg: RandomBenchmarkConfig,
    structure_specs: list[dict[str, Any]],
) -> np.ndarray:
    from beamz import PEC, Simulation

    design = _build_beamz_design(cfg, structure_specs)
    sim = Simulation(
        design=design,
        sources=[],
        boundaries=[PEC(edges="all")],
        time=np.asarray([0.0, cfg.dt_s], dtype=np.float64),
        resolution=cfg.dx_m,
    )
    return np.asarray(sim.fields.permittivity, dtype=np.float32)


def run_beamz_single(
    cfg: RandomBenchmarkConfig,
    *,
    structure_specs: list[dict[str, Any]],
    structure_index: int,
    seed: int,
    output_dir: Path,
    field_grid: str,
) -> dict[str, Any]:
    from beamz import PEC, Simulation

    signal = _gaussian_modulated_signal_fn(cfg)
    design = _build_beamz_design(cfg, structure_specs)

    source_voxels, source_weights = _beamz_ez_source_voxels(
        cfg,
        width_um=cfg.source_width_um * cfg.meep_source_width_scale,
    )
    source = _BeamzCurrentSource(
        signal=signal,
        voxel_indices=source_voxels,
        voxel_weights=source_weights,
    )
    time_axis = np.arange(cfg.num_steps, dtype=np.float64) * cfg.dt_s
    sim = Simulation(
        design=design,
        sources=[source],
        boundaries=[PEC(edges="all")],
        time=time_axis,
        resolution=cfg.dx_m,
    )

    snapshots: list[dict[str, Any]] = []
    start = time.perf_counter()
    for target_step in cfg.snapshot_steps:
        while sim.current_step < target_step:
            if not sim.step():
                raise RuntimeError("Beamz simulation ended before reaching snapshot step")
        if field_grid == FIELD_GRID_RAW_YEE:
            field_data = _extract_raw_beamz_fields(sim)
            axis_order = "zyx_raw_yee"
        elif field_grid == FIELD_GRID_MEEP_SAMPLED:
            field_data = _extract_meep_sampled_beamz_fields(sim)
            axis_order = "zyx_meep_sampled"
        else:
            field_data = _extract_centered_beamz_fields(sim)
            axis_order = "zyx_centered"
        snapshots.append(
            {
                "step": int(sim.current_step),
                "time_s": float(sim.t),
                "fields": field_data,
            }
        )
    runtime_s = time.perf_counter() - start

    stem = _solver_output_stem("beamz", field_grid)
    out_path = output_dir / f"structure_{structure_index:03d}" / f"{stem}.npz"
    return _write_solver_output(
        output_path=out_path,
        solver_name="beamz",
        cfg=cfg,
        structure_index=structure_index,
        seed=seed,
        structure_specs=structure_specs,
        permittivity=np.asarray(sim.fields.permittivity),
        snapshots=snapshots,
        runtime_s=runtime_s,
        extra_metadata={
            "axis_order": axis_order,
            "field_grid": field_grid,
            "component_shapes": _component_shape_metadata(snapshots),
            "component_coordinates_um": _beamz_component_coordinates_um(cfg, field_grid),
        },
    )


def run_meep_single(
    cfg: RandomBenchmarkConfig,
    *,
    structure_specs: list[dict[str, Any]],
    structure_index: int,
    seed: int,
    output_dir: Path,
    field_grid: str,
    beamz_raster_file: Path | None = None,
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
    shared_beamz_eps_zyx: np.ndarray | None = None
    if cfg.geometry_source == "beamz-raster":
        if beamz_raster_file is not None:
            beamz_eps_zyx = np.load(beamz_raster_file)
        else:
            beamz_eps_zyx = _rasterize_beamz_permittivity(cfg, structure_specs)
        shared_beamz_eps_zyx = np.asarray(beamz_eps_zyx, dtype=np.float32)
        eps_grid = np.asarray(beamz_eps_zyx, dtype=np.float64)
        dx_um = cfg.dx_m * 1e6

        def epsilon_func(p):
            x_um = float(p.x) + 0.5 * cfg.domain_um
            y_um = float(p.y) + 0.5 * cfg.domain_um
            z_um = float(p.z) + 0.5 * cfg.domain_um
            ix = int(np.clip(np.floor(x_um / dx_um), 0, eps_grid.shape[2] - 1))
            iy = int(np.clip(np.floor(y_um / dx_um), 0, eps_grid.shape[1] - 1))
            iz = int(np.clip(np.floor(z_um / dx_um), 0, eps_grid.shape[0] - 1))
            return float(eps_grid[iz, iy, ix])

        eps_averaging = False
    else:
        for spec in structure_specs:
            medium = mp.Medium(index=spec["index"])
            if spec["type"] == "block":
                sx, sy, sz = spec["size_um"]
                px, py, pz = spec["position_um"]
                cx = px + 0.5 * sx
                cy = py + 0.5 * sy
                cz = pz + 0.5 * sz
                mx, my, mz = _meep_center(cx, cy, cz, cfg)
                geometry.append(
                    mp.Block(
                        size=mp.Vector3(sx, sy, sz),
                        center=mp.Vector3(mx, my, mz),
                        material=medium,
                    )
                )
            elif spec["type"] == "sphere":
                cx, cy, cz = spec["center_um"]
                mx, my, mz = _meep_center(cx, cy, cz, cfg)
                geometry.append(
                    mp.Sphere(
                        radius=spec["radius_um"],
                        center=mp.Vector3(mx, my, mz),
                        material=medium,
                    )
                )
            elif spec["type"] == "polygon_prism":
                cx_um, cy_um = _polygon_centroid_xy(spec["vertices_um"])
                vertices = [mp.Vector3(x - cx_um, y - cy_um) for x, y in spec["vertices_um"]]
                mx, my, _ = _meep_center(cx_um, cy_um, 0.0, cfg)
                z_center_um = spec["z_um"] + 0.5 * spec["depth_um"] - 0.5 * cfg.domain_um
                geometry.append(
                    mp.Prism(
                        vertices=vertices,
                        height=spec["depth_um"],
                        center=mp.Vector3(mx, my, z_center_um),
                        material=medium,
                    )
                )
            else:
                raise ValueError(f"Unsupported primitive type {spec['type']!r}")

    sx, sy, sz = _meep_center(cfg.source_x_um, cfg.source_y_um, cfg.source_z_um, cfg)
    source_voxels, source_weights = _meep_ez_source_voxels(
        cfg, width_um=cfg.source_width_um * cfg.meep_source_width_scale
    )
    source_end_time_um = cfg.num_steps * cfg.dt_s * C0 / 1e-6
    beamz_signal = _gaussian_modulated_signal_fn(cfg)

    def meep_signal(t_um):
        t_s = float(t_um) * 1e-6 / C0
        return beamz_signal(t_s)

    gaussian_time = mp.CustomSource(
        src_func=meep_signal,
        start_time=0.0,
        end_time=source_end_time_um,
        center_frequency=frequency_um_inv,
        fwidth=fwidth_um_inv,
    )
    sources = []
    for (iz, iy, ix), weight in zip(source_voxels, source_weights, strict=False):
        x_um = ix * cfg.dx_m * 1e6
        y_um = iy * cfg.dx_m * 1e6
        z_um = (iz + 0.5) * cfg.dx_m * 1e6
        mx, my, mz = _meep_center(x_um, y_um, z_um, cfg)
        sources.append(
            mp.Source(
                src=gaussian_time,
                component=mp.Ez,
                center=mp.Vector3(mx, my, mz),
                size=mp.Vector3(),
                amplitude=float(cfg.meep_source_amplitude_scale * weight),
            )
        )

    sim = mp.Simulation(
        cell_size=mp.Vector3(cfg.domain_um, cfg.domain_um, cfg.domain_um),
        geometry=geometry,
        sources=sources,
        eps_averaging=eps_averaging,
        boundary_layers=[],
        default_material=default_material,
        epsilon_func=epsilon_func,
        resolution=resolution_px_per_um,
        Courant=cfg.courant_safety / math.sqrt(3.0),
        dimensions=3,
    )
    # Make the Meep cavity walls explicit so Beamz/Meep runs share the same PEC box.
    for side in (mp.Low, mp.High):
        sim.set_boundary(side, mp.X, mp.Metallic)
        sim.set_boundary(side, mp.Y, mp.Metallic)
        sim.set_boundary(side, mp.Z, mp.Metallic)

    cell_center = mp.Vector3()
    cell_size = mp.Vector3(cfg.domain_um, cfg.domain_um, cfg.domain_um)
    snapshot_times_um = [t_s * C0 / 1e-6 for t_s in cfg.snapshot_times_s]
    snapshots: list[dict[str, Any] | None] = [None] * len(snapshot_times_um)
    dx_um = cfg.dx_m * 1e6
    raw_span_um = cfg.domain_um - 2.0 * dx_um
    def _meep_component_kwargs(component_name: str) -> dict[str, Any]:
        if field_grid == FIELD_GRID_CENTERED:
            return {
                "center": cell_center,
                "size": cell_size,
                "snap": False,
            }
        if component_name == "Ex":
            return {
                "center": mp.Vector3(-0.5 * dx_um, 0.0, 0.0),
                "size": mp.Vector3(raw_span_um, cfg.domain_um, cfg.domain_um),
                "snap": True,
            }
        if component_name == "Ey":
            return {
                "center": mp.Vector3(0.0, -0.5 * dx_um, 0.0),
                "size": mp.Vector3(cfg.domain_um, raw_span_um, cfg.domain_um),
                "snap": True,
            }
        if component_name == "Ez":
            return {
                "center": mp.Vector3(0.0, 0.0, -0.5 * dx_um),
                "size": mp.Vector3(cfg.domain_um, cfg.domain_um, raw_span_um),
                "snap": True,
            }
        raise ValueError(f"Unsupported component {component_name!r}")

    def _capture_factory(index: int):
        def _capture(sim_obj):
            ex_kwargs = _meep_component_kwargs("Ex")
            ey_kwargs = _meep_component_kwargs("Ey")
            ez_kwargs = _meep_component_kwargs("Ez")
            snapshots[index] = {
                "step": int(cfg.snapshot_steps[index]),
                "time_s": float(cfg.snapshot_times_s[index]),
                "fields": {
                    "Ex": _canonicalize_meep_array(
                        sim_obj.get_array(
                            component=mp.Ex,
                            **ex_kwargs,
                        )
                    ),
                    "Ey": _canonicalize_meep_array(
                        sim_obj.get_array(
                            component=mp.Ey,
                            **ey_kwargs,
                        )
                    ),
                    "Ez": _canonicalize_meep_array(
                        sim_obj.get_array(
                            component=mp.Ez,
                            **ez_kwargs,
                        )
                    ),
                },
            }

        return _capture

    hooks = [mp.at_time(t_um, _capture_factory(i)) for i, t_um in enumerate(snapshot_times_um)]

    mp.verbosity(0)
    sim.init_sim()
    start = time.perf_counter()
    sim.run(*hooks, until=cfg.total_time_s * C0 / 1e-6)
    runtime_s = time.perf_counter() - start

    if any(snap is None for snap in snapshots):
        raise RuntimeError("Meep did not emit all requested snapshots")

    if shared_beamz_eps_zyx is not None:
        permittivity = shared_beamz_eps_zyx
    else:
        permittivity = _canonicalize_meep_array(
            sim.get_array(center=cell_center, size=cell_size, component=mp.Dielectric)
        )

    stem = _solver_output_stem("meep", field_grid)
    out_path = output_dir / f"structure_{structure_index:03d}" / f"{stem}.npz"
    return _write_solver_output(
        output_path=out_path,
        solver_name="meep",
        cfg=cfg,
        structure_index=structure_index,
        seed=seed,
        structure_specs=structure_specs,
        permittivity=permittivity,
        snapshots=[snap for snap in snapshots if snap is not None],
        runtime_s=runtime_s,
        extra_metadata={
            "axis_order": {
                FIELD_GRID_CENTERED: "zyx_centered",
                FIELD_GRID_RAW_YEE: "zyx_raw_yee",
                FIELD_GRID_MEEP_SAMPLED: "zyx_meep_sampled",
            }[field_grid],
            "field_grid": field_grid,
            "component_shapes": _component_shape_metadata(
                [snap for snap in snapshots if snap is not None]
            ),
            "component_coordinates_um": _meep_component_coordinates_um(cfg, field_grid),
        },
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


def _run_meep_subprocess(
    *,
    meep_env: str,
    args: list[str],
) -> dict[str, Any]:
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
        raise RuntimeError(
            f"Meep subprocess failed for conda env '{meep_env}': {detail}"
        )
    return _extract_json_object(proc.stdout)


def _run_backend(
    *,
    backend: str,
    cfg: RandomBenchmarkConfig,
    structure_specs: list[dict[str, Any]],
    structure_index: int,
    seed: int,
    output_dir: Path,
    meep_env: str,
    meep_available: bool,
    field_grid: str,
    beamz_raster_file: Path | None = None,
) -> dict[str, Any]:
    if backend == "beamz":
        return run_beamz_single(
            cfg,
            structure_specs=structure_specs,
            structure_index=structure_index,
            seed=seed,
            output_dir=output_dir,
            field_grid=field_grid,
        )
    if backend == "meep":
        if meep_available:
            return run_meep_single(
                cfg,
                structure_specs=structure_specs,
                structure_index=structure_index,
                seed=seed,
                output_dir=output_dir,
                field_grid=field_grid,
                beamz_raster_file=beamz_raster_file,
            )
        extra_args = []
        if beamz_raster_file is not None:
            extra_args.extend(["--beamz-raster-file", str(beamz_raster_file)])
        return _run_meep_subprocess(
            meep_env=meep_env,
            args=[
                "--num-structures",
                "1",
                "--start-seed",
                str(seed - structure_index),
                "--structure-index",
                str(structure_index),
                "--resolution-ppw",
                str(cfg.resolution_ppw),
                "--geometry-mode",
                str(cfg.geometry_mode),
                "--geometry-source",
                str(cfg.geometry_source),
                "--num-primitives",
                str(cfg.num_primitives),
                "--polygon-permittivity",
                str(cfg.polygon_permittivity),
                "--num-snapshots",
                str(cfg.num_snapshots),
                "--output-dir",
                str(output_dir),
                "--total-cycles",
                str(cfg.total_cycles),
                "--wavelength-um",
                str(cfg.wavelength_um),
                "--courant-safety",
                str(cfg.courant_safety),
                "--meep-source-width-scale",
                str(cfg.meep_source_width_scale),
                "--meep-source-amplitude-scale",
                str(cfg.meep_source_amplitude_scale),
                "--field-grid",
                field_grid,
                *extra_args,
            ],
        )["structures"][0]["results"]["meep"]
    raise ValueError(f"Unsupported backend {backend!r}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Beamz and Meep on random 3D structures with saved field snapshots."
    )
    parser.add_argument(
        "--backend", choices=("beamz", "meep", "both"), default="both"
    )
    parser.add_argument(
        "--geometry-mode",
        choices=("random-primitives", "random-polygon"),
        default="random-primitives",
    )
    parser.add_argument(
        "--geometry-source",
        choices=("native", "beamz-raster"),
        default="native",
    )
    parser.add_argument("--num-structures", type=int, default=1)
    parser.add_argument("--structure-index", type=int, default=None)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--resolution-ppw", type=int, default=8)
    parser.add_argument("--num-primitives", type=int, default=4)
    parser.add_argument("--polygon-permittivity", type=float, default=12.0)
    parser.add_argument("--num-snapshots", type=int, default=3)
    parser.add_argument("--total-cycles", type=float, default=4.0)
    parser.add_argument("--wavelength-um", type=float, default=1.0)
    parser.add_argument("--courant-safety", type=float, default=0.95)
    parser.add_argument("--meep-source-width-scale", type=float, default=1.0)
    parser.add_argument("--meep-source-amplitude-scale", type=float, default=7.37e-07)
    parser.add_argument(
        "--field-grid",
        choices=(FIELD_GRID_CENTERED, FIELD_GRID_RAW_YEE, FIELD_GRID_MEEP_SAMPLED),
        default=FIELD_GRID_CENTERED,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "results",
    )
    parser.add_argument(
        "--meep-env",
        default=os.getenv("BEAMZ_MEEP_ENV", "beamz-meep"),
        help="Conda env used if meep is unavailable in the current interpreter.",
    )
    parser.add_argument("--beamz-raster-file", type=Path, default=None)
    parser.add_argument("--emit-json-only", action="store_true")
    args = parser.parse_args()

    cfg = RandomBenchmarkConfig(
        resolution_ppw=args.resolution_ppw,
        geometry_mode=args.geometry_mode,
        geometry_source=args.geometry_source,
        num_primitives=args.num_primitives,
        num_snapshots=args.num_snapshots,
        total_cycles=args.total_cycles,
        wavelength_um=args.wavelength_um,
        courant_safety=args.courant_safety,
        meep_source_width_scale=args.meep_source_width_scale,
        meep_source_amplitude_scale=args.meep_source_amplitude_scale,
        polygon_permittivity=args.polygon_permittivity,
    )

    if args.structure_index is not None:
        indices = [int(args.structure_index)]
    else:
        indices = list(range(max(1, int(args.num_structures))))

    try:
        import meep as _meep  # noqa: F401

        meep_available = True
    except Exception:
        meep_available = False

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "generated_at_epoch_s": time.time(),
        "config": asdict(cfg),
        "field_grid": args.field_grid,
        "structures": [],
    }

    for structure_index in indices:
        seed = int(args.start_seed + structure_index)
        structure_specs = _random_structure_specs(cfg, seed=seed)
        structure_dir = output_dir / f"structure_{structure_index:03d}"
        structure_dir.mkdir(parents=True, exist_ok=True)
        beamz_raster_file = args.beamz_raster_file
        if cfg.geometry_source == "beamz-raster" and beamz_raster_file is None:
            beamz_raster_file = structure_dir / "beamz_shared_permittivity.npy"
            np.save(beamz_raster_file, _rasterize_beamz_permittivity(cfg, structure_specs))
        (structure_dir / "structure.json").write_text(
            json.dumps(
                {
                    "structure_index": structure_index,
                    "seed": seed,
                    "config": asdict(cfg),
                    "structure_specs": structure_specs,
                },
                indent=2,
            )
        )

        entry: dict[str, Any] = {
            "structure_index": structure_index,
            "seed": seed,
            "structure_specs": structure_specs,
            "results": {},
        }

        backends = ("beamz", "meep") if args.backend == "both" else (args.backend,)
        for backend in backends:
            entry["results"][backend] = _run_backend(
                backend=backend,
                cfg=cfg,
                structure_specs=structure_specs,
                structure_index=structure_index,
                seed=seed,
                output_dir=output_dir,
                meep_env=args.meep_env,
                meep_available=meep_available,
                field_grid=args.field_grid,
                beamz_raster_file=beamz_raster_file,
            )
        summary["structures"].append(entry)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2))

    if args.emit_json_only:
        print(json.dumps(summary))
    else:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
