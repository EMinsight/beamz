"""Tiny 3D straight-guide raw substep snapshots.

This diagnostic uses a deliberately tiny centered guide so we can inspect the
exact raw Yee-grid fields around the first source/update substeps. It writes
full-field arrays plus compact parity summaries after:

1. initial zero fields
2. `ModeSource.inject_h(...)`
3. `fields.update_e(dt)`
4. `ModeSource.inject_e(...)`

The goal is to debug symmetry breaking at the engine level, not to measure a
far-field transmission quantity.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from beamz import (
    EPS_0,
    LIGHT_SPEED,
    Design,
    Material,
    Monitor,
    ModeSource,
    Rectangle,
    Simulation,
    calc_optimal_fdtd_params,
    ramped_cosine,
    µm,
)
from beamz.simulation import ops
from beamz.simulation.boundaries import build_h_boundary_views_for_e_3d
from beamz.simulation.yee import sample_voxel_grid_at_component_3d

SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class TinyConfig:
    wavelength_um: float = 1.55
    ppw: int = 6
    direction: str = "+x"
    polarization: str = "te"
    long_cells: int = 18
    transverse0_cells: int = 8
    transverse1_cells: int = 6
    guide0_cells: int = 4
    guide1_cells: int = 2
    source_clearance_cells: int = 4
    downstream_plane_offset_cells: int = 1


def _mirror_residual(arr: np.ndarray, axis: int) -> float:
    lhs = np.asarray(arr, dtype=float)
    rhs = np.flip(lhs, axis=axis)
    denom = max(float(np.linalg.norm(lhs.ravel())), 1e-30)
    return float(np.linalg.norm((lhs - rhs).ravel()) / denom)


def _best_parity_residual(profile: np.ndarray, axis: int) -> float:
    arr = np.asarray(profile, dtype=float)
    flipped = np.flip(arr, axis=axis)
    denom = max(float(np.linalg.norm(arr.ravel())), 1e-30)
    even = float(np.linalg.norm((arr - flipped).ravel()) / denom)
    odd = float(np.linalg.norm((arr + flipped).ravel()) / denom)
    return min(even, odd)


def _build_tiny_design(cfg: TinyConfig, *, dx: float, axis: str) -> tuple[Design, tuple[float, float, float], tuple[float, float]]:
    width = float(cfg.long_cells) * dx if axis == "x" else float(cfg.transverse0_cells) * dx
    height = float(cfg.long_cells) * dx if axis == "y" else float(cfg.transverse0_cells) * dx
    depth = float(cfg.long_cells) * dx if axis == "z" else float(cfg.transverse1_cells) * dx
    g0 = float(cfg.guide0_cells) * dx
    g1 = float(cfg.guide1_cells) * dx
    center = (0.5 * width, 0.5 * height, 0.5 * depth)

    design = Design(
        width=width,
        height=height,
        depth=depth,
        material=Material(1.0**2),
    )
    if axis == "x":
        design += Rectangle(
            position=(0.0, center[1] - 0.5 * g0, center[2] - 0.5 * g1),
            width=width,
            height=g0,
            depth=g1,
            material=Material(2.0**2),
        )
        source_spans = (
            float(min(cfg.transverse0_cells, cfg.guide0_cells + 2)) * dx,
            float(min(cfg.transverse1_cells, cfg.guide1_cells + 2)) * dx,
        )
    elif axis == "y":
        design += Rectangle(
            position=(center[0] - 0.5 * g0, 0.0, center[2] - 0.5 * g1),
            width=g0,
            height=height,
            depth=g1,
            material=Material(2.0**2),
        )
        source_spans = (
            float(min(cfg.transverse0_cells, cfg.guide0_cells + 2)) * dx,
            float(min(cfg.transverse1_cells, cfg.guide1_cells + 2)) * dx,
        )
    else:
        design += Rectangle(
            position=(center[0] - 0.5 * g0, center[1] - 0.5 * g1, 0.0),
            width=g0,
            height=g1,
            depth=depth,
            material=Material(2.0**2),
        )
        source_spans = (
            float(min(cfg.transverse1_cells, cfg.guide1_cells + 2)) * dx,
            float(min(cfg.transverse0_cells, cfg.guide0_cells + 2)) * dx,
        )
    return design, center, source_spans


def _source_center(
    cfg: TinyConfig,
    *,
    design: Design,
    axis: str,
) -> tuple[float, float, float]:
    clearance = float(cfg.source_clearance_cells) * (float(design.width) / float(cfg.long_cells) if axis == "x" else float(design.height) / float(cfg.long_cells) if axis == "y" else float(design.depth) / float(cfg.long_cells))
    center = [0.5 * float(design.width), 0.5 * float(design.height), 0.5 * float(design.depth)]
    if cfg.direction.startswith("+"):
        center[{"x": 0, "y": 1, "z": 2}[axis]] = clearance
    else:
        axis_len = {"x": design.width, "y": design.height, "z": design.depth}[axis]
        center[{"x": 0, "y": 1, "z": 2}[axis]] = float(axis_len) - clearance
    return tuple(float(v) for v in center)


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


def _monitor_plane(center: tuple[float, float, float], axis: str, span0: float, span1: float):
    x, y, z = center
    if axis == "x":
        return (x, y - 0.5 * span0, z - 0.5 * span1), (x, y + 0.5 * span0, z + 0.5 * span1)
    if axis == "y":
        return (x - 0.5 * span0, y, z - 0.5 * span1), (x + 0.5 * span0, y, z + 0.5 * span1)
    return (x - 0.5 * span1, y - 0.5 * span0, z), (x + 0.5 * span1, y + 0.5 * span0, z)


def _build_sim_and_source(cfg: TinyConfig) -> tuple[Simulation, ModeSource]:
    axis = str(cfg.direction)[1]
    dx, dt = calc_optimal_fdtd_params(
        float(cfg.wavelength_um) * 1e-6,
        2.0,
        dims=3,
        safety_factor=0.9,
        points_per_wavelength=int(cfg.ppw),
        width=5.5 * float(cfg.wavelength_um) * 1e-6,
        height=2.2 * float(cfg.wavelength_um) * 1e-6,
        depth=2.0 * float(cfg.wavelength_um) * 1e-6,
    )
    design, _center, source_spans = _build_tiny_design(cfg, dx=dx, axis=axis)
    time = np.arange(0.0, 3.0 * dt, dt, dtype=float)
    sim = Simulation(
        design=design,
        sources=[],
        time=time,
        resolution=dx,
    )
    freq = LIGHT_SPEED / (float(cfg.wavelength_um) * 1e-6)
    signal = ramped_cosine(
        np.asarray(time, dtype=float),
        amplitude=1.0,
        frequency=freq,
        ramp_duration=1.0 / freq,
        t_max=float(time[-1]),
    )
    source = ModeSource(
        grid=design.rasterize(resolution=dx),
        center=_source_center(cfg, design=design, axis=axis),
        width=float(source_spans[0]),
        height=float(source_spans[1]),
        wavelength=float(cfg.wavelength_um) * 1e-6,
        pol=str(cfg.polarization),
        signal=signal,
        direction=str(cfg.direction),
    )
    source.initialize(np.asarray(sim.fields.permittivity), dx, dt=float(sim.dt))
    return sim, source


def _sample_downstream_plane(
    fields,
    source: ModeSource,
    *,
    resolution: float,
    offset_cells: int,
) -> dict[str, np.ndarray]:
    axis = str(source.direction)[1]
    monitor_center = _move_along(
        tuple(float(v) for v in source.center),
        str(source.direction),
        float(offset_cells) * float(resolution),
    )
    mon_start, mon_end = _monitor_plane(
        monitor_center,
        axis,
        float(source.width),
        float(source.height),
    )
    monitor = Monitor(
        start=mon_start,
        end=mon_end,
        name="tiny_raw_plane",
        record_fields=True,
        dft_enabled=False,
    )
    monitor.record_fields_3d(
        np.asarray(fields.Ex),
        np.asarray(fields.Ey),
        np.asarray(fields.Ez),
        np.asarray(fields.Hx),
        np.asarray(fields.Hy),
        np.asarray(fields.Hz),
        t=0.0,
        dx=float(resolution),
        dy=float(resolution),
        dz=float(resolution),
        step=0,
    )
    coords0, coords1 = monitor.get_analysis_plane_coords_3d(
        dx=float(resolution),
        dy=float(resolution),
        dz=float(resolution),
        field_shape=tuple(np.asarray(fields.permittivity).shape),
    )
    n0 = int(np.asarray(coords0).size)
    n1 = int(np.asarray(coords1).size)
    return {
        comp: np.asarray(monitor.fields[comp][-1], dtype=np.complex128).reshape(n0, n1)
        for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    }


def _support_metrics(fields, source: ModeSource, component_name: str, index_attr: str) -> dict[str, float]:
    arr = np.asarray(getattr(fields, component_name), dtype=float)[getattr(source, index_attr)]
    return {
        "axis0_best_parity_rel": _best_parity_residual(arr, axis=0),
        "axis1_best_parity_rel": _best_parity_residual(arr, axis=1),
        "norm": float(np.linalg.norm(arr.ravel())),
    }


def _cross_section_slice(arr: np.ndarray, axis: str, index: int) -> np.ndarray:
    if axis == "x":
        return np.asarray(arr)[:, :, index]
    if axis == "y":
        return np.asarray(arr)[:, index, :]
    return np.asarray(arr)[index, :, :]


def _field_slice_metrics(fields, source: ModeSource, axis: str) -> dict[str, dict[str, float]]:
    field_map = {
        "Ex": ("_Ex_indices", 2 if axis == "x" else 1 if axis == "y" else 0),
        "Ey": ("_Ey_indices", 2 if axis == "x" else 1 if axis == "y" else 0),
        "Ez": ("_Ez_indices", 2 if axis == "x" else 1 if axis == "y" else 0),
        "Hx": ("_Hx_indices", 2 if axis == "x" else 1 if axis == "y" else 0),
        "Hy": ("_Hy_indices", 2 if axis == "x" else 1 if axis == "y" else 0),
        "Hz": ("_Hz_indices", 2 if axis == "x" else 1 if axis == "y" else 0),
    }
    metrics: dict[str, dict[str, float]] = {}
    for name, (idx_attr, pick_axis) in field_map.items():
        indices = getattr(source, idx_attr)
        plane_index = int(round(float(np.mean(indices[pick_axis]))))
        plane = _cross_section_slice(getattr(fields, name), axis, plane_index)
        metrics[name] = {
            "axis0_magnitude_rel": _mirror_residual(np.abs(plane), axis=0),
            "axis1_magnitude_rel": _mirror_residual(np.abs(plane), axis=1),
            "norm": float(np.linalg.norm(plane.ravel())),
            "plane_index": plane_index,
        }
    return metrics


def _support_array_metrics(arr: np.ndarray, indices) -> dict[str, float]:
    sample = np.asarray(arr, dtype=float)[indices]
    return {
        "axis0_best_parity_rel": _best_parity_residual(sample, axis=0),
        "axis1_best_parity_rel": _best_parity_residual(sample, axis=1),
        "norm": float(np.linalg.norm(sample.ravel())),
    }


def _field_state_arrays(fields) -> dict[str, np.ndarray]:
    return {
        comp: np.asarray(getattr(fields, comp), dtype=float)
        for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    }


def _support_state_arrays(state: dict[str, np.ndarray], source: ModeSource) -> dict[str, np.ndarray]:
    index_map = {
        "Ex": source._Ex_indices,
        "Ey": source._Ey_indices,
        "Ez": source._Ez_indices,
        "Hx": source._Hx_indices,
        "Hy": source._Hy_indices,
        "Hz": source._Hz_indices,
    }
    return {
        comp: np.asarray(state[comp], dtype=float)[index_map[comp]]
        for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    }


def _second_order_mode_fit(
    state0: dict[str, np.ndarray],
    state1: dict[str, np.ndarray],
    state2: dict[str, np.ndarray],
) -> dict[str, object]:
    component_reports: dict[str, dict[str, float]] = {}
    lhs_parts: list[np.ndarray] = []
    mid_parts: list[np.ndarray] = []

    for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        arr0 = np.asarray(state0[comp], dtype=np.complex128)
        arr1 = np.asarray(state1[comp], dtype=np.complex128)
        arr2 = np.asarray(state2[comp], dtype=np.complex128)
        lhs = arr2 + arr0
        mid = arr1
        lhs_flat = lhs.ravel()
        mid_flat = mid.ravel()
        denom = max(float(np.real(np.vdot(mid_flat, mid_flat))), 1e-30)
        alpha = np.vdot(mid_flat, lhs_flat) / denom
        resid = lhs_flat - alpha * mid_flat
        resid_norm = float(np.linalg.norm(resid))
        lhs_norm = max(float(np.linalg.norm(lhs_flat)), 1e-30)
        component_reports[comp] = {
            "alpha_real": float(np.real(alpha)),
            "alpha_imag": float(np.imag(alpha)),
            "residual_rel": resid_norm / lhs_norm,
            "lhs_norm": lhs_norm,
            "mid_norm": float(np.linalg.norm(mid_flat)),
        }
        lhs_parts.append(lhs_flat)
        mid_parts.append(mid_flat)

    lhs_all = np.concatenate(lhs_parts)
    mid_all = np.concatenate(mid_parts)
    denom_all = max(float(np.real(np.vdot(mid_all, mid_all))), 1e-30)
    alpha_all = np.vdot(mid_all, lhs_all) / denom_all
    resid_all = lhs_all - alpha_all * mid_all
    lhs_norm_all = max(float(np.linalg.norm(lhs_all)), 1e-30)
    alpha_real = float(np.real(alpha_all))
    alpha_imag = float(np.imag(alpha_all))
    theta = (
        float(np.arccos(np.clip(alpha_real / 2.0, -1.0, 1.0)))
        if abs(alpha_imag) < 1e-9
        else None
    )

    return {
        "global": {
            "alpha_real": alpha_real,
            "alpha_imag": alpha_imag,
            "residual_rel": float(np.linalg.norm(resid_all) / lhs_norm_all),
            "lhs_norm": lhs_norm_all,
            "mid_norm": float(np.linalg.norm(mid_all)),
            "theta_rad": theta,
        },
        "components": component_reports,
    }


def _ex_update_breakdown(fields, source: ModeSource, dt: float, resolution: float) -> dict[str, object]:
    boundary_views = build_h_boundary_views_for_e_3d(
        fields.Hx,
        fields.Hy,
        fields.Hz,
        getattr(fields, "boundaries", None),
    )
    d_hz_dy = np.asarray(
        ops._adjacent_difference(boundary_views["hz_y"], axis=1, resolution=resolution),
        dtype=float,
    )
    d_hy_dz = np.asarray(
        ops._adjacent_difference(boundary_views["hy_z"], axis=0, resolution=resolution),
        dtype=float,
    )
    curl_hx = d_hz_dy - d_hy_dz
    eps_x = np.asarray(sample_voxel_grid_at_component_3d(fields.permittivity, "Ex"), dtype=float)
    sigma_x = np.asarray(sample_voxel_grid_at_component_3d(fields.conductivity, "Ex"), dtype=float)
    denom = 1.0 + sigma_x * (dt / (2.0 * EPS_0 * eps_x))
    source_coeff_x = (dt / (EPS_0 * eps_x)) / denom

    ex_indices = source._Ex_indices
    ex_support = np.asarray(fields.Ex, dtype=float)[ex_indices]
    supports = {
        "ex_support": ex_support,
        "d_hz_dy_support": np.asarray(d_hz_dy, dtype=float)[ex_indices],
        "d_hy_dz_support": np.asarray(d_hy_dz, dtype=float)[ex_indices],
        "curl_hx_support": np.asarray(curl_hx, dtype=float)[ex_indices],
        "eps_x_support": np.asarray(eps_x, dtype=float)[ex_indices],
        "source_coeff_x_support": np.asarray(source_coeff_x, dtype=float)[ex_indices],
    }
    return {
        "support": {
            "ex": _support_array_metrics(ex_support, tuple(slice(None) for _ in range(ex_support.ndim))),
            "d_hz_dy": _support_array_metrics(d_hz_dy, ex_indices),
            "d_hy_dz": _support_array_metrics(d_hy_dz, ex_indices),
            "curl_hx": _support_array_metrics(curl_hx, ex_indices),
            "eps_x": _support_array_metrics(eps_x, ex_indices),
            "source_coeff_x": _support_array_metrics(source_coeff_x, ex_indices),
        },
        "arrays": {
            "d_hz_dy": d_hz_dy,
            "d_hy_dz": d_hy_dz,
            "curl_hx": curl_hx,
            "eps_x": eps_x,
            "source_coeff_x": source_coeff_x,
        },
        "support_arrays": supports,
    }


def _save_stage(
    out_dir: Path,
    *,
    name: str,
    fields,
    source: ModeSource,
    axis: str,
    dt: float,
    resolution: float,
    plane_offset_cells: int,
) -> dict[str, object]:
    stage_dir = out_dir / name
    stage_dir.mkdir(parents=True, exist_ok=True)
    npz_path = stage_dir / "fields.npz"
    arrays = _field_state_arrays(fields)
    np.savez_compressed(npz_path, **arrays)

    support = {
        comp: _support_metrics(fields, source, comp, idx_attr)
        for comp, idx_attr in (
            ("Ex", "_Ex_indices"),
            ("Ey", "_Ey_indices"),
            ("Ez", "_Ez_indices"),
            ("Hx", "_Hx_indices"),
            ("Hy", "_Hy_indices"),
            ("Hz", "_Hz_indices"),
        )
    }
    slices = _field_slice_metrics(fields, source, axis)
    breakdown = _ex_update_breakdown(fields, source, dt=float(dt), resolution=float(resolution))
    report = {
        "support": support,
        "slices": slices,
        "ex_update_breakdown": {
            "support": breakdown["support"],
        },
    }
    report_path = stage_dir / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2))
    breakdown_npz = stage_dir / "ex_update_breakdown.npz"
    np.savez_compressed(
        breakdown_npz,
        **{
            name: np.asarray(arr, dtype=float)
            for name, arr in breakdown["arrays"].items()
        },
    )
    support_npz = stage_dir / "ex_update_supports.npz"
    np.savez_compressed(
        support_npz,
        **{
            name: np.asarray(arr, dtype=float)
            for name, arr in breakdown["support_arrays"].items()
        },
    )
    plane_arrays = _sample_downstream_plane(
        fields,
        source,
        resolution=float(resolution),
        offset_cells=int(plane_offset_cells),
    )
    plane_npz = stage_dir / "downstream_plane.npz"
    np.savez_compressed(
        plane_npz,
        **{name: np.asarray(arr) for name, arr in plane_arrays.items()},
    )

    # Render the raw cross-section planes of the dominant E/H components at the
    # source-centered index so we can inspect the exact symmetry by eye.
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), dpi=220)
    for ax, comp in zip(axes.flat, ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"), strict=True):
        plane_index = int(report["slices"][comp]["plane_index"])
        plane = _cross_section_slice(arrays[comp], axis, plane_index)
        im = ax.imshow(plane, origin="lower", cmap="RdBu_r", aspect="auto")
        ax.set_title(f"{comp} plane {plane_index}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    png_path = stage_dir / "slices.png"
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return {
        "dir": str(stage_dir),
        "fields": str(npz_path),
        "metrics": str(report_path),
        "breakdown": str(breakdown_npz),
        "supports": str(support_npz),
        "downstream_plane": str(plane_npz),
        "slices": str(png_path),
    }


def run_case(cfg: TinyConfig, *, output_dir: Path) -> dict[str, object]:
    sim, source = _build_sim_and_source(cfg)
    axis = str(cfg.direction)[1]
    artifacts: dict[str, dict[str, object]] = {}
    state_history: dict[str, dict[str, np.ndarray]] = {}
    plane_history: dict[str, dict[str, np.ndarray]] = {}

    artifacts["00_initial"] = _save_stage(
        output_dir,
        name="00_initial",
        fields=sim.fields,
        source=source,
        axis=axis,
        dt=float(sim.dt),
        resolution=float(sim.resolution),
        plane_offset_cells=int(cfg.downstream_plane_offset_cells),
    )
    state_history["00_initial"] = _field_state_arrays(sim.fields)
    plane_history["00_initial"] = _sample_downstream_plane(
        sim.fields,
        source,
        resolution=float(sim.resolution),
        offset_cells=int(cfg.downstream_plane_offset_cells),
    )

    source.inject_h(
        sim.fields,
        t=0.0,
        dt=float(sim.dt),
        current_step=0,
        resolution=float(sim.resolution),
        design=sim.design,
    )
    artifacts["01_after_inject_h"] = _save_stage(
        output_dir,
        name="01_after_inject_h",
        fields=sim.fields,
        source=source,
        axis=axis,
        dt=float(sim.dt),
        resolution=float(sim.resolution),
        plane_offset_cells=int(cfg.downstream_plane_offset_cells),
    )
    state_history["01_after_inject_h"] = _field_state_arrays(sim.fields)
    plane_history["01_after_inject_h"] = _sample_downstream_plane(
        sim.fields,
        source,
        resolution=float(sim.resolution),
        offset_cells=int(cfg.downstream_plane_offset_cells),
    )

    sim.fields.update_e(float(sim.dt))
    artifacts["02_after_update_e"] = _save_stage(
        output_dir,
        name="02_after_update_e",
        fields=sim.fields,
        source=source,
        axis=axis,
        dt=float(sim.dt),
        resolution=float(sim.resolution),
        plane_offset_cells=int(cfg.downstream_plane_offset_cells),
    )
    state_history["02_after_update_e"] = _field_state_arrays(sim.fields)
    plane_history["02_after_update_e"] = _sample_downstream_plane(
        sim.fields,
        source,
        resolution=float(sim.resolution),
        offset_cells=int(cfg.downstream_plane_offset_cells),
    )

    source.inject_e(
        sim.fields,
        t=0.0,
        dt=float(sim.dt),
        current_step=0,
        resolution=float(sim.resolution),
        design=sim.design,
    )
    artifacts["03_after_inject_e"] = _save_stage(
        output_dir,
        name="03_after_inject_e",
        fields=sim.fields,
        source=source,
        axis=axis,
        dt=float(sim.dt),
        resolution=float(sim.resolution),
        plane_offset_cells=int(cfg.downstream_plane_offset_cells),
    )
    state_history["03_after_inject_e"] = _field_state_arrays(sim.fields)
    plane_history["03_after_inject_e"] = _sample_downstream_plane(
        sim.fields,
        source,
        resolution=float(sim.resolution),
        offset_cells=int(cfg.downstream_plane_offset_cells),
    )

    for step in range(1, 4):
        sim.fields.update_h(float(sim.dt))
        sim.fields.update_e(float(sim.dt))
        stage_name = f"{step + 3:02d}_after_free_step_{step}"
        artifacts[stage_name] = _save_stage(
            output_dir,
            name=stage_name,
            fields=sim.fields,
            source=source,
            axis=axis,
            dt=float(sim.dt),
            resolution=float(sim.resolution),
            plane_offset_cells=int(cfg.downstream_plane_offset_cells),
        )
        state_history[stage_name] = _field_state_arrays(sim.fields)
        plane_history[stage_name] = _sample_downstream_plane(
            sim.fields,
            source,
            resolution=float(sim.resolution),
            offset_cells=int(cfg.downstream_plane_offset_cells),
        )

    recurrence = {
        "03_04_05": _second_order_mode_fit(
            state_history["03_after_inject_e"],
            state_history["04_after_free_step_1"],
            state_history["05_after_free_step_2"],
        ),
        "04_05_06": _second_order_mode_fit(
            state_history["04_after_free_step_1"],
            state_history["05_after_free_step_2"],
            state_history["06_after_free_step_3"],
        ),
    }
    support_recurrence = {
        "03_04_05": _second_order_mode_fit(
            _support_state_arrays(state_history["03_after_inject_e"], source),
            _support_state_arrays(state_history["04_after_free_step_1"], source),
            _support_state_arrays(state_history["05_after_free_step_2"], source),
        ),
        "04_05_06": _second_order_mode_fit(
            _support_state_arrays(state_history["04_after_free_step_1"], source),
            _support_state_arrays(state_history["05_after_free_step_2"], source),
            _support_state_arrays(state_history["06_after_free_step_3"], source),
        ),
    }
    downstream_plane_recurrence = {
        "03_04_05": _second_order_mode_fit(
            plane_history["03_after_inject_e"],
            plane_history["04_after_free_step_1"],
            plane_history["05_after_free_step_2"],
        ),
        "04_05_06": _second_order_mode_fit(
            plane_history["04_after_free_step_1"],
            plane_history["05_after_free_step_2"],
            plane_history["06_after_free_step_3"],
        ),
    }

    return {
        "config": asdict(cfg),
        "resolution_um": float(float(sim.resolution) / µm),
        "grid_shape": tuple(int(v) for v in np.asarray(sim.fields.permittivity).shape),
        "artifacts": artifacts,
        "source_free_recurrence": recurrence,
        "source_support_recurrence": support_recurrence,
        "downstream_plane_recurrence": downstream_plane_recurrence,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Write tiny straight-guide raw substep snapshots.")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results_tiny_straight_waveguide_raw_slices_3d")
    parser.add_argument("--ppw", type=int, default=6)
    parser.add_argument("--direction", default="+x")
    parser.add_argument("--polarization", default="te")
    parser.add_argument("--long-cells", type=int, default=18)
    parser.add_argument("--transverse0-cells", type=int, default=8)
    parser.add_argument("--transverse1-cells", type=int, default=6)
    parser.add_argument("--guide0-cells", type=int, default=4)
    parser.add_argument("--guide1-cells", type=int, default=2)
    parser.add_argument("--source-clearance-cells", type=int, default=4)
    parser.add_argument("--downstream-plane-offset-cells", type=int, default=1)
    args = parser.parse_args()

    cfg = TinyConfig(
        ppw=int(args.ppw),
        direction=str(args.direction),
        polarization=str(args.polarization),
        long_cells=int(args.long_cells),
        transverse0_cells=int(args.transverse0_cells),
        transverse1_cells=int(args.transverse1_cells),
        guide0_cells=int(args.guide0_cells),
        guide1_cells=int(args.guide1_cells),
        source_clearance_cells=int(args.source_clearance_cells),
        downstream_plane_offset_cells=int(args.downstream_plane_offset_cells),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = run_case(cfg, output_dir=args.output_dir)
    report_path = args.output_dir / "tiny_straight_waveguide_raw_slices_3d.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({"report": str(report_path.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
