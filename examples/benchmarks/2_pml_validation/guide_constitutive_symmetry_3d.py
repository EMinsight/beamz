"""Characterize constitutive symmetry on a centered 3D straight guide.

This diagnostic does not run a full propagation benchmark. It builds one tiny
centered 3D straight guide, reports mirror symmetry of the cell-centered raster,
the staggered E-component material views (`eps_x/eps_y/eps_z`), and the scalar
E-update source coefficients derived from those effective permittivities.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from beamz import Design, EPS_0, Material, ModeSource, Rectangle, Simulation, calc_optimal_fdtd_params, um
from beamz.simulation.boundaries import build_h_boundary_views_for_e_3d
from beamz.simulation import ops

SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    wavelength_um: float = 1.55
    ppw: int = 6
    n_core: float = 2.0
    n_clad: float = 1.0
    guide_span_y_um: float = 0.55 * 1.55
    guide_span_z_um: float = 0.35 * 1.55
    long_span_um: float = 5.5 * 1.55
    height_um: float = 2.2 * 1.55
    depth_um: float = 2.0 * 1.55
    courant_safety: float = 0.9

    @property
    def wavelength_m(self) -> float:
        return float(self.wavelength_um) * 1e-6


def _quantize_cells(
    target_m: float,
    dx: float,
    *,
    min_cells: int = 2,
    parity: int | None = None,
) -> int:
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


def _symmetry_metrics(arr: np.ndarray) -> dict[str, float]:
    data = np.asarray(arr, dtype=float)
    return {
        "shape": list(data.shape),
        "mirror_y_rel": _mirror_residual(data, axis=1),
        "mirror_z_rel": _mirror_residual(data, axis=0),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
    }


def _build_centered_straight_guide_sim(cfg: Config, *, with_core: bool) -> tuple[Simulation, float]:
    dx, dt = calc_optimal_fdtd_params(
        cfg.wavelength_m,
        cfg.n_core,
        dims=3,
        safety_factor=float(cfg.courant_safety),
        points_per_wavelength=int(cfg.ppw),
        width=float(cfg.long_span_um) * 1e-6,
        height=float(cfg.height_um) * 1e-6,
        depth=float(cfg.depth_um) * 1e-6,
    )

    long_cells = _quantize_cells(float(cfg.long_span_um) * 1e-6, dx, min_cells=12)
    height_cells = _quantize_cells(float(cfg.height_um) * 1e-6, dx, min_cells=16)
    depth_cells = _quantize_cells(float(cfg.depth_um) * 1e-6, dx, min_cells=14)
    guide_y_cells = _quantize_cells(
        float(cfg.guide_span_y_um) * 1e-6,
        dx,
        min_cells=2,
        parity=height_cells % 2,
    )
    guide_z_cells = _quantize_cells(
        float(cfg.guide_span_z_um) * 1e-6,
        dx,
        min_cells=2,
        parity=depth_cells % 2,
    )

    width = long_cells * dx
    height = height_cells * dx
    depth = depth_cells * dx
    guide_y = guide_y_cells * dx
    guide_z = guide_z_cells * dx
    center = (0.5 * width, 0.5 * height, 0.5 * depth)

    design = Design(
        width=width,
        height=height,
        depth=depth,
        material=Material(cfg.n_clad**2),
    )
    if with_core:
        design += Rectangle(
            position=(0.0, center[1] - 0.5 * guide_y, center[2] - 0.5 * guide_z),
            width=width,
            height=guide_y,
            depth=guide_z,
            material=Material(cfg.n_core**2),
        )
    sim = Simulation(
        design=design,
        sources=[],
        time=np.asarray([0.0, dt], dtype=float),
        resolution=dx,
    )
    return sim, dt


def _build_test_source(sim: Simulation) -> ModeSource:
    dx = float(sim.resolution)
    source = ModeSource(
        grid=sim.design.rasterize(resolution=dx),
        center=(8.0 * dx, 0.5 * float(sim.design.height), 0.5 * float(sim.design.depth)),
        width=12.0 * dx,
        height=10.0 * dx,
        wavelength=1.55 * um,
        pol="te",
        signal=np.asarray([1.0, 1.0, 1.0], dtype=float),
        direction="+x",
    )
    source.initialize(np.asarray(sim.fields.permittivity), dx, dt=float(sim.dt))
    return source


def _support_parity_metrics(fields, source: ModeSource, component_name: str, index_attr: str) -> dict[str, float]:
    idx = getattr(source, index_attr)
    arr = np.asarray(getattr(fields, component_name), dtype=float)[idx]
    return {
        "axis0_best_parity_rel": _best_parity_residual(arr, axis=0),
        "axis1_best_parity_rel": _best_parity_residual(arr, axis=1),
    }


def _indexed_parity_metrics(arr, idx) -> dict[str, float]:
    sample = np.asarray(arr, dtype=float)[idx]
    return {
        "axis0_best_parity_rel": _best_parity_residual(sample, axis=0),
        "axis1_best_parity_rel": _best_parity_residual(sample, axis=1),
    }


def _slice_with_extra_stop(s: slice, extra: int = 1) -> slice:
    return slice(int(s.start), int(s.stop) + int(extra))


def _one_step_report(sim: Simulation) -> dict[str, object]:
    source = _build_test_source(sim)
    source.inject_h(
        sim.fields,
        t=0.0,
        dt=float(sim.dt),
        current_step=0,
        resolution=float(sim.resolution),
        design=sim.design,
    )
    pre_h = {
        comp: _support_parity_metrics(sim.fields, source, comp, idx_attr)
        for comp, idx_attr in (("Hx", "_Hx_indices"), ("Hy", "_Hy_indices"), ("Hz", "_Hz_indices"))
    }
    boundary_views = build_h_boundary_views_for_e_3d(
        sim.fields.Hx,
        sim.fields.Hy,
        sim.fields.Hz,
        getattr(sim, "boundaries", None),
    )
    curl_hx, curl_hy, curl_hz = ops.curl_h_to_e_3d(
        sim.fields.Hx,
        sim.fields.Hy,
        sim.fields.Hz,
        sim.resolution,
        ex_shape=sim.fields.Ex.shape,
        ey_shape=sim.fields.Ey.shape,
        ez_shape=sim.fields.Ez.shape,
        boundary_views=boundary_views,
    )
    d_hz_dy = ops._adjacent_difference(
        boundary_views["hz_y"], axis=1, resolution=sim.resolution
    )
    d_hy_dz = ops._adjacent_difference(
        boundary_views["hy_z"], axis=0, resolution=sim.resolution
    )
    hz_z, hz_y, hz_x = source._Hz_indices
    hy_z, hy_y, hy_x = source._Hy_indices
    curl_branch_support = {
        "dHz_dy": _indexed_parity_metrics(
            d_hz_dy,
            (hz_z, _slice_with_extra_stop(hz_y), hz_x),
        ),
        "dHy_dz": _indexed_parity_metrics(
            d_hy_dz,
            (_slice_with_extra_stop(hy_z), hy_y, hy_x),
        ),
    }
    mixed_curl_support = {
        "curlHx_on_ex_window": _indexed_parity_metrics(curl_hx, source._Ex_indices),
        "curlHy_on_ey_window": _indexed_parity_metrics(curl_hy, source._Ey_indices),
        "curlHz_on_ez_window": _indexed_parity_metrics(curl_hz, source._Ez_indices),
    }
    sim.fields.update_e(float(sim.dt))
    post_e = {
        comp: _support_parity_metrics(sim.fields, source, comp, idx_attr)
        for comp, idx_attr in (("Ex", "_Ex_indices"), ("Ey", "_Ey_indices"), ("Ez", "_Ez_indices"))
    }
    return {
        "pre_h_support_parity": pre_h,
        "curl_branch_support_parity": curl_branch_support,
        "mixed_curl_support_parity": mixed_curl_support,
        "post_e_support_parity": post_e,
    }


def build_report(cfg: Config) -> dict[str, object]:
    sim, dt = _build_centered_straight_guide_sim(cfg, with_core=True)
    fields = sim.fields

    perm = np.asarray(fields.permittivity, dtype=float)
    eps_x = np.asarray(fields.eps_x, dtype=float)
    eps_y = np.asarray(fields.eps_y, dtype=float)
    eps_z = np.asarray(fields.eps_z, dtype=float)

    coeff_x = (dt / (EPS_0 * eps_x)).astype(float)
    coeff_y = (dt / (EPS_0 * eps_y)).astype(float)
    coeff_z = (dt / (EPS_0 * eps_z)).astype(float)

    uniform_sim, _ = _build_centered_straight_guide_sim(cfg, with_core=False)
    guide_sim, _ = _build_centered_straight_guide_sim(cfg, with_core=True)

    return {
        "config": asdict(cfg),
        "grid_shape": list(np.asarray(perm).shape),
        "dx_um": float(sim.resolution / um),
        "cell_centered_permittivity": _symmetry_metrics(perm),
        "eps_x": _symmetry_metrics(eps_x),
        "eps_y": _symmetry_metrics(eps_y),
        "eps_z": _symmetry_metrics(eps_z),
        "e_source_coeff_x": _symmetry_metrics(coeff_x),
        "e_source_coeff_y": _symmetry_metrics(coeff_y),
        "e_source_coeff_z": _symmetry_metrics(coeff_z),
        "one_step_uniform": _one_step_report(uniform_sim),
        "one_step_guide": _one_step_report(guide_sim),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Characterize 3D constitutive symmetry on a centered straight guide.")
    parser.add_argument("--ppw", type=int, default=6)
    parser.add_argument("--wavelength-um", type=float, default=1.55)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "results_guide_constitutive_symmetry_3d",
    )
    args = parser.parse_args()

    cfg = Config(ppw=int(args.ppw), wavelength_um=float(args.wavelength_um))
    report = build_report(cfg)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "guide_constitutive_symmetry_3d.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({"report": str(out_path.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
