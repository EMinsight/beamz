"""Sweep source-port reference spacing for 3D raw vs corrected S11.

This benchmark targets issue #106 directly. It compares two devices under the
same source-port extraction path:

1. A straight rectangular guide with no physical reflection target.
2. The compact PDK crossing used by the Meep/gsim comparison workflow.

For each device, the sweep varies only the source-monitor spacing relative to a
fixed source/reference plane and reports:
- raw source-port S11 from the main monitor branch before source correction
- corrected source-port S11 from `get_S_matrix_modal_dft(...)`
- source-port wave dominance
- transmission/cross-port center-frequency values for sanity

The defaults intentionally use a low-cost configuration:
- sigma absorber only
- 3D
- 6 PPW
"""

from __future__ import annotations

import argparse
import json
import math
import time as pytime
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
from beamz.design.io import gdsf
from beamz.devices._placement import mirror_lock_plane_pair_regions
from beamz.devices.sources.signals import gaussian_band_pulse
from beamz import ramped_cosine

SCRIPT_DIR = Path(__file__).resolve().parent
_CROSSING_PREP_CACHE: dict[tuple, dict] = {}


@dataclass(frozen=True)
class SweepConfig:
    wavelength_um: float = 1.55
    ppw: int = 6
    num_freqs: int = 1
    courant_safety: float = 0.9
    pml_um: float = 1.0
    run_after_sources_uoc: float = 70.0
    decay_ratio: float = 1e-4
    lookback_records: int = 20
    source_monitor_deltas_um: tuple[float, ...] = (0.2, 0.4, 0.6)
    devices: tuple[str, ...] = ("straight", "crossing")

    straight_n_core: float = 2.0
    straight_n_clad: float = 1.0
    straight_guide_width_um: float = 0.60
    straight_guide_length_um: float = 8.0
    straight_transverse_span_um: float = 2.4
    straight_port_span_scale: float = 2.5
    straight_source_clearance_um: float = 0.60
    straight_output_clearance_um: float = 0.60

    crossing_component: str = "ebeam_crossing4"
    crossing_layer: tuple[int, int] = (1, 0)
    crossing_n_core: float = 3.47
    crossing_n_clad: float = 1.44
    crossing_core_t_um: float = 0.22
    crossing_clad_below_um: float = 0.50
    crossing_clad_above_um: float = 0.50
    crossing_xy_margin_um: float = 0.50
    crossing_port_margin_um: float = 0.50
    crossing_source_offset_um: float = 0.10
    crossing_output_monitor_offset_um: float = 0.10
    crossing_port_overlap_um: float = 0.0
    crossing_extension_um: float | None = None

    @property
    def wavelength_m(self) -> float:
        return float(self.wavelength_um) * 1e-6

    @property
    def pml_m(self) -> float:
        return float(self.pml_um) * 1e-6

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


def _symmetry_stats(lhs: np.ndarray, rhs: np.ndarray, *, tol: float = 1e-12) -> dict[str, float | int]:
    a = np.asarray(lhs, dtype=float)
    b = np.asarray(rhs, dtype=float)
    diff = a - b
    abs_diff = np.abs(diff)
    return {
        "max_abs": float(np.max(abs_diff)) if abs_diff.size else 0.0,
        "mean_abs": float(np.mean(abs_diff)) if abs_diff.size else 0.0,
        "rms_abs": float(np.sqrt(np.mean(abs_diff**2))) if abs_diff.size else 0.0,
        "mismatch_count": int(np.count_nonzero(abs_diff > float(tol))),
        "sample_count": int(abs_diff.size),
    }


def _largest_mismatch_locations(
    lhs: np.ndarray,
    rhs: np.ndarray,
    *,
    resolution_m: float,
    limit: int = 8,
    tol: float = 1e-12,
) -> list[dict[str, object]]:
    a = np.asarray(lhs, dtype=float)
    b = np.asarray(rhs, dtype=float)
    abs_diff = np.abs(a - b)
    flat = abs_diff.reshape(-1)
    nz = int(np.count_nonzero(flat > float(tol)))
    if nz <= 0:
        return []
    order = np.argpartition(flat, -min(limit, nz))[-min(limit, nz):]
    order = order[np.argsort(flat[order])[::-1]]
    result: list[dict[str, object]] = []
    for flat_idx in order:
        idx = np.unravel_index(int(flat_idx), abs_diff.shape)
        result.append(
            {
                "index_zyx": [int(v) for v in idx],
                "coord_um": [
                    float((int(v) + 0.5) * float(resolution_m) / µm)
                    for v in idx
                ],
                "lhs": float(a[idx]),
                "rhs": float(b[idx]),
                "abs_diff": float(abs_diff[idx]),
            }
        )
    return result


def crossing_voxel_symmetry_report(cfg: SweepConfig) -> dict[str, object]:
    prepared = prepare_crossing_geometry(cfg)
    design = prepared["design"]
    dx, _ = calc_optimal_fdtd_params(
        cfg.wavelength_m,
        cfg.crossing_n_core,
        dims=3,
        safety_factor=cfg.courant_safety,
        points_per_wavelength=int(cfg.ppw),
        width=design.width,
        height=design.height,
        depth=design.depth,
    )
    grid = design.rasterize(resolution=dx)
    eps = np.asarray(grid.permittivity, dtype=float)
    clad_eps = float(cfg.crossing_n_clad) ** 2
    structure = np.abs(eps - clad_eps) > 1e-12

    tb = _symmetry_stats(eps, eps[:, ::-1, :])
    lr = _symmetry_stats(eps, eps[:, :, ::-1])
    rot = _symmetry_stats(eps, eps[:, ::-1, ::-1])

    mask_tb = _symmetry_stats(structure.astype(float), structure[:, ::-1, :].astype(float))
    mask_lr = _symmetry_stats(structure.astype(float), structure[:, :, ::-1].astype(float))
    mask_rot = _symmetry_stats(structure.astype(float), structure[:, ::-1, ::-1].astype(float))

    center_z = int(np.argmin(np.abs((np.arange(eps.shape[0], dtype=float) + 0.5) * float(dx) - 0.5 * float(design.depth))))
    eps_xy = eps[center_z]
    structure_xy = structure[center_z]
    xy_tb = _symmetry_stats(eps_xy, eps_xy[::-1, :])
    xy_lr = _symmetry_stats(eps_xy, eps_xy[:, ::-1])
    xy_rot = _symmetry_stats(eps_xy, eps_xy[::-1, ::-1])
    mask_xy_tb = _symmetry_stats(structure_xy.astype(float), structure_xy[::-1, :].astype(float))
    mask_xy_lr = _symmetry_stats(structure_xy.astype(float), structure_xy[:, ::-1].astype(float))
    mask_xy_rot = _symmetry_stats(structure_xy.astype(float), structure_xy[::-1, ::-1].astype(float))

    return {
        "ppw": int(cfg.ppw),
        "resolution_um": float(dx / µm),
        "grid_shape": tuple(int(v) for v in eps.shape),
        "center_z_index": center_z,
        "structure_voxel_count": int(np.count_nonzero(structure)),
        "volume": {
            "top_bottom_eps": tb,
            "left_right_eps": lr,
            "rotation_eps": rot,
            "top_bottom_mask": mask_tb,
            "left_right_mask": mask_lr,
            "rotation_mask": mask_rot,
            "left_right_examples": _largest_mismatch_locations(
                eps,
                eps[:, :, ::-1],
                resolution_m=float(dx),
            ),
            "rotation_examples": _largest_mismatch_locations(
                eps,
                eps[:, ::-1, ::-1],
                resolution_m=float(dx),
            ),
        },
        "center_xy_slice": {
            "top_bottom_eps": xy_tb,
            "left_right_eps": xy_lr,
            "rotation_eps": xy_rot,
            "top_bottom_mask": mask_xy_tb,
            "left_right_mask": mask_xy_lr,
            "rotation_mask": mask_xy_rot,
        },
    }


def _complex_alignment_residual(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, float]:
    a = np.asarray(lhs, dtype=np.complex128).reshape(-1)
    b = np.asarray(rhs, dtype=np.complex128).reshape(-1)
    denom = float(np.vdot(b, b).real)
    if denom <= 1e-30:
        return {"alpha_real": 0.0, "alpha_imag": 0.0, "residual_db": 0.0}
    alpha = np.vdot(b, a) / denom
    resid = a - alpha * b
    ref = max(float(np.linalg.norm(a)), 1e-30)
    return {
        "alpha_real": float(np.real(alpha)),
        "alpha_imag": float(np.imag(alpha)),
        "residual_db": float(20.0 * np.log10(max(float(np.linalg.norm(resid)) / ref, 1e-30))),
    }


def crossing_field_mirror_report(cfg: SweepConfig, monitor_delta_um: float) -> dict[str, object]:
    sim, monitors, ports, freqs, pulse, meta = build_crossing_case(cfg, monitor_delta_um)
    del meta
    sim.run_compiled_until_decay(
        monitors,
        min_time_s=float(pulse.source_end_time + pulse.tail_time),
        lookback_records=cfg.lookback_records,
        decay_ratio=cfg.decay_ratio,
        progress=False,
    )
    result = sim.get_S_matrix_modal_dft(
        source_port="o1",
        ports=ports,
        output_ports=["o1", "o2", "o3", "o4"],
        frequencies=freqs,
        as_sax=False,
        return_diagnostics=True,
        min_incident_db=-50.0,
    )
    center_idx = int(np.argmin(np.abs(freqs - LIGHT_SPEED / cfg.wavelength_m)))
    port_map = {p.name: p for p in ports}
    monitor_map = {m.name: m for m in monitors}
    projection_cache: dict[tuple, dict] = {}
    port_reports: dict[str, dict[str, object]] = {}
    for port_name in ("o2", "o4"):
        spec = port_map[port_name]
        monitor = monitor_map[port_name]
        proj = sim._build_port_projection(
            spec,
            monitor,
            float(freqs[center_idx]),
            projection_cache,
        )
        comps = tuple(proj.get("components", (proj["e_component"], proj["h_component"])))
        raw = {}
        for comp in comps:
            _, sampled = sim._sample_monitor_component_dft(monitor, comp, frequencies=freqs)
            raw[comp] = np.asarray(sampled[center_idx], dtype=np.complex128)
        coloc = sim._colocate_field_components_to_projection_3d(monitor, raw, proj)
        n0 = int(len(np.asarray(proj["analysis_coords0"])))
        n1 = int(len(np.asarray(proj["analysis_coords1"])))
        component_planes = {
            comp: np.asarray(coloc[comp], dtype=np.complex128).reshape(n0, n1)
            for comp in comps
        }
        port_reports[port_name] = {
            "components": {name: plane for name, plane in component_planes.items()},
            "dominance_db": float(
                wave_dominance_db(
                    result["diagnostics"]["waves"][port_name]["a_plus"],
                    result["diagnostics"]["waves"][port_name]["a_minus"],
                    str(port_map[port_name].scattered_wave),
                    np.asarray(result["diagnostics"]["valid_mask"], dtype=bool),
                )
            ),
        }

    comps = sorted(
        set(port_reports["o2"]["components"].keys()) & set(port_reports["o4"]["components"].keys())
    )
    component_reports = {}
    for comp in comps:
        top = port_reports["o2"]["components"][comp]
        bot = port_reports["o4"]["components"][comp]
        component_reports[comp] = {
            "complex_residual": _complex_alignment_residual(top, bot),
            "magnitude_residual_db": float(
                20.0
                * np.log10(
                    max(
                        float(np.linalg.norm(np.abs(top) - np.abs(bot)))
                        / max(float(np.linalg.norm(np.abs(top))), 1e-30),
                        1e-30,
                    )
                )
            ),
        }

    return {
        "ppw": int(cfg.ppw),
        "monitor_delta_um": float(monitor_delta_um),
        "output_monitor_offset_um": float(cfg.crossing_output_monitor_offset_um),
        "extension_um": None if cfg.crossing_extension_um is None else float(cfg.crossing_extension_um),
        "center_wavelength_um": float((LIGHT_SPEED / freqs[center_idx]) / µm),
        "s_db": {
            "o2_o1": float(
                20.0
                * np.log10(
                    max(abs(complex(np.asarray(result["s_matrix"][("o2", "o1")])[center_idx])), 1e-30)
                )
            ),
            "o4_o1": float(
                20.0
                * np.log10(
                    max(abs(complex(np.asarray(result["s_matrix"][("o4", "o1")])[center_idx])), 1e-30)
                )
            ),
        },
        "dominance_db": {
            "o2": float(port_reports["o2"]["dominance_db"]),
            "o4": float(port_reports["o4"]["dominance_db"]),
        },
        "components": component_reports,
    }


def crossing_ripple_probe(cfg: SweepConfig, monitor_delta_um: float) -> dict[str, object]:
    sim, monitors, ports, freqs, pulse, meta = build_crossing_case(cfg, monitor_delta_um)
    del meta
    sim.run_compiled_until_decay(
        monitors,
        min_time_s=float(pulse.source_end_time + pulse.tail_time),
        lookback_records=cfg.lookback_records,
        decay_ratio=cfg.decay_ratio,
        progress=False,
    )
    result = sim.get_S_matrix_modal_dft(
        source_port="o1",
        ports=ports,
        output_ports=["o1", "o2", "o3", "o4"],
        frequencies=freqs,
        as_sax=False,
        return_diagnostics=False,
        min_incident_db=-50.0,
    )
    s_matrix = result["s_matrix"] if isinstance(result, dict) and "s_matrix" in result else result
    wl_um = (LIGHT_SPEED / np.asarray(freqs, dtype=float)) / µm
    s_db = {}
    extrema = {}
    for port_name in ("o1", "o2", "o3", "o4"):
        vals = np.asarray(s_matrix[(port_name, "o1")], dtype=np.complex128)
        db = 20.0 * np.log10(np.maximum(np.abs(vals), 1e-30))
        s_db[port_name] = [float(v) for v in db]
        i_min = int(np.argmin(db))
        i_max = int(np.argmax(db))
        extrema[port_name] = {
            "min_db": float(db[i_min]),
            "min_wavelength_um": float(wl_um[i_min]),
            "max_db": float(db[i_max]),
            "max_wavelength_um": float(wl_um[i_max]),
        }
    return {
        "ppw": int(cfg.ppw),
        "monitor_delta_um": float(monitor_delta_um),
        "output_monitor_offset_um": float(cfg.crossing_output_monitor_offset_um),
        "extension_um": None if cfg.crossing_extension_um is None else float(cfg.crossing_extension_um),
        "wavelength_um": [float(v) for v in wl_um],
        "s_db": s_db,
        "extrema": extrema,
    }

@dataclass
class SweepCaseResult:
    device: str
    monitor_delta_um: float
    crossing_output_monitor_offset_um: float
    source_reference_offset_um: float
    source_monitor_offset_um: float
    raw_s11_db: float
    corrected_s11_db: float
    source_dominance_db: float
    source_correction_mag: float
    incident_mag: float
    source_monitor_clearance_um: float
    source_reference_clearance_um: float
    transmission_db: float | None = None
    top_db: float | None = None
    bottom_db: float | None = None
    top_dominance_db: float | None = None
    bottom_dominance_db: float | None = None
    top_bottom_mismatch_db: float | None = None
    steps: int | None = None
    runtime_s: float | None = None


def incoming_wave(direction: str) -> str:
    return "minus" if str(direction).startswith("+") else "plus"


def outgoing_wave(direction: str) -> str:
    return incoming_wave(gdsf.outward_direction(direction))


def move_along(center: tuple[float, float], direction: str, distance: float):
    x, y = center
    return {
        "+x": (x + distance, y),
        "-x": (x - distance, y),
        "+y": (x, y + distance),
        "-y": (x, y - distance),
    }[str(direction)]


def port_plane(
    port: dict,
    *,
    span: float,
    z_span: float,
    z_center: float,
    offset: float = 0.0,
):
    cx, cy = move_along(port["center"], port["direction"], offset)
    z0 = float(z_center) - 0.5 * float(z_span)
    z1 = float(z_center) + 0.5 * float(z_span)
    if str(port["direction"]).endswith("x"):
        return (cx, cy - 0.5 * float(span), z0), (cx, cy + 0.5 * float(span), z1)
    return (cx - 0.5 * float(span), cy, z0), (cx + 0.5 * float(span), cy, z1)


def line_center(line):
    a, b = line
    return tuple(0.5 * (float(a[i]) + float(b[i])) for i in range(len(a)))


def monitor_clearance_xy(
    plane,
    *,
    width: float,
    height: float,
    pml_xy: float,
) -> float:
    a = np.asarray(plane[0], dtype=float)
    b = np.asarray(plane[1], dtype=float)
    pmin = np.minimum(a, b)
    pmax = np.maximum(a, b)
    clearances = np.asarray(
        [
            pmin[0] - float(pml_xy),
            float(width) - float(pml_xy) - pmax[0],
            pmin[1] - float(pml_xy),
            float(height) - float(pml_xy) - pmax[1],
        ],
        dtype=float,
    )
    return float(np.min(clearances) / µm)


def wave_dominance_db(
    a_plus: np.ndarray,
    a_minus: np.ndarray,
    selector: str,
    mask: np.ndarray | None = None,
) -> float:
    plus = np.asarray(a_plus, dtype=np.complex128)
    minus = np.asarray(a_minus, dtype=np.complex128)
    if mask is None:
        mask = np.ones_like(plus, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
    sel = plus if selector == "plus" else minus
    opp = minus if selector == "plus" else plus
    if np.any(mask):
        sel = sel[mask]
        opp = opp[mask]
    p_sel = float(np.mean(np.abs(sel) ** 2))
    p_opp = float(np.mean(np.abs(opp) ** 2))
    return 10.0 * np.log10(max(p_sel, 1e-18) / max(p_opp, 1e-18))


def select_wave_component(wave_data, selector: str, *, use_reference: bool = False):
    sel = str(selector).lower()
    if use_reference:
        plus = np.asarray(
            wave_data.get(
                "a_incident_plus",
                wave_data.get("a_incident", wave_data.get("a_plus")),
            ),
            dtype=np.complex128,
        )
        minus = np.asarray(
            wave_data.get("a_incident_minus", wave_data.get("a_minus")),
            dtype=np.complex128,
        )
    else:
        plus = np.asarray(wave_data.get("a_plus"), dtype=np.complex128)
        minus = np.asarray(wave_data.get("a_minus"), dtype=np.complex128)
    return plus if sel == "plus" else minus


def _monitor_cfg(freqs: np.ndarray):
    return dict(
        record_fields=False,
        dft_enabled=True,
        dft_frequencies=np.asarray(freqs, dtype=float),
        dft_components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        dft_window="none",
        dft_record_every_step=True,
    )


def build_pulse(
    cfg: SweepConfig,
    freqs: np.ndarray,
    dt: float,
    *,
    max_output_distance_um: float,
    n_eff_guess: float,
):
    freqs = np.asarray(freqs, dtype=float)
    if freqs.size == 1:
        f0 = float(freqs[0])
        period = 1.0 / max(f0, 1e-30)
        transit_time = max_output_distance_um * µm * float(n_eff_guess) / LIGHT_SPEED
        ramp_duration = 1.5 * period
        tail_time = max(4.0 * transit_time, 6.0 * period)
        total_time = max(12.0 * period + tail_time, ramp_duration + tail_time)
        time = np.arange(0.0, total_time, float(dt), dtype=float)
        signal = ramped_cosine(
            time,
            amplitude=1.0,
            frequency=f0,
            ramp_duration=ramp_duration,
            t_max=total_time,
        )
        return SimpleNamespace(
            signal=np.asarray(signal, dtype=np.float32),
            time=np.asarray(time, dtype=float),
            source_end_time=float(ramp_duration),
            tail_time=float(tail_time),
        )
    return gaussian_band_pulse(
        freqs,
        carrier_frequency=LIGHT_SPEED / cfg.wavelength_m,
        dt=dt,
        run_after_sources_uoc=cfg.run_after_sources_uoc,
        max_output_distance_um=max_output_distance_um,
    )


def build_straight_case(cfg: SweepConfig, monitor_delta_um: float):
    guide_width = float(cfg.straight_guide_width_um) * µm
    span = float(cfg.straight_port_span_scale) * guide_width
    width = float(cfg.straight_guide_length_um) * µm + 2.0 * cfg.pml_m
    height = float(cfg.straight_transverse_span_um) * µm
    depth = height
    source_clearance = float(cfg.straight_source_clearance_um) * µm
    output_clearance = float(cfg.straight_output_clearance_um) * µm
    delta = float(monitor_delta_um) * µm

    design = Design(
        width=width,
        height=height,
        depth=depth,
        material=Material(cfg.straight_n_clad**2),
    )
    design += Rectangle(
        position=(0.0, 0.5 * (height - guide_width), 0.5 * (depth - guide_width)),
        width=width,
        height=guide_width,
        depth=guide_width,
        material=Material(cfg.straight_n_core**2),
    )
    dx, _ = calc_optimal_fdtd_params(
        cfg.wavelength_m,
        cfg.straight_n_core,
        dims=3,
        safety_factor=cfg.courant_safety,
        points_per_wavelength=int(cfg.ppw),
        width=width,
        height=height,
        depth=depth,
    )
    dt = float(cfg.courant_safety) * float(dx) / (LIGHT_SPEED * math.sqrt(3.0))
    grid = design.rasterize(resolution=dx)
    freqs = np.asarray(cfg.frequencies_hz, dtype=float)

    source_x = cfg.pml_m + source_clearance
    output_x = width - cfg.pml_m - output_clearance
    pulse = build_pulse(
        cfg,
        freqs,
        dt,
        max_output_distance_um=(output_x - source_x) / µm,
        n_eff_guess=0.5 * (cfg.straight_n_core + cfg.straight_n_clad),
    )
    y0 = 0.5 * (height - span)
    y1 = 0.5 * (height + span)
    z0 = 0.5 * (depth - span)
    z1 = 0.5 * (depth + span)
    center = (source_x, 0.5 * height, 0.5 * depth)
    source = ModeSource(
        grid=grid,
        center=center,
        width=span,
        height=span,
        wavelength=cfg.wavelength_m,
        pol="te",
        signal=pulse.signal,
        direction="+x",
    )
    reference_plane = ((source_x, y0, z0), (source_x, y1, z1))
    source_monitor_plane = (
        (source_x + delta, y0, z0),
        (source_x + delta, y1, z1),
    )
    output_plane = ((output_x, y0, z0), (output_x, y1, z1))
    monitors = [
        Monitor(start=source_monitor_plane[0], end=source_monitor_plane[1], name="o1", **_monitor_cfg(freqs)),
        Monitor(start=reference_plane[0], end=reference_plane[1], name="o1_ref", **_monitor_cfg(freqs)),
        Monitor(start=output_plane[0], end=output_plane[1], name="o2", **_monitor_cfg(freqs)),
    ]
    sim = Simulation(
        design=design,
        sources=[source],
        monitors=monitors,
        boundaries=[PML(edges="all", thickness=cfg.pml_m, formulation="sigma")],
        time=pulse.time,
        resolution=dx,
    )
    ports = [
        PortSpec(
            name="o1",
            monitor_name="o1",
            reference_monitor="o1_ref",
            direction="+x",
            polarization="te",
            mode_index=0,
            incident_wave="minus",
            scattered_wave="plus",
        ),
        PortSpec(
            name="o2",
            monitor_name="o2",
            direction="+x",
            polarization="te",
            mode_index=0,
            incident_wave="plus",
            scattered_wave="minus",
        ),
    ]
    metadata = {
        "source_reference_offset_um": source_clearance / µm,
        "source_monitor_offset_um": (source_clearance + delta) / µm,
        "source_reference_clearance_um": monitor_clearance_xy(
            reference_plane,
            width=width,
            height=height,
            pml_xy=cfg.pml_m,
        ),
        "source_monitor_clearance_um": monitor_clearance_xy(
            source_monitor_plane,
            width=width,
            height=height,
            pml_xy=cfg.pml_m,
        ),
    }
    return sim, monitors, ports, freqs, pulse, metadata


def prepare_crossing_geometry(cfg: SweepConfig) -> dict:
    key = (
        cfg.crossing_component,
        tuple(cfg.crossing_layer),
        float(cfg.crossing_n_core),
        float(cfg.crossing_n_clad),
        float(cfg.crossing_core_t_um),
        float(cfg.crossing_clad_below_um),
        float(cfg.crossing_clad_above_um),
        float(cfg.crossing_xy_margin_um),
        float(cfg.crossing_port_overlap_um),
        float(cfg.pml_um),
        None if cfg.crossing_extension_um is None else float(cfg.crossing_extension_um),
    )
    cached = _CROSSING_PREP_CACHE.get(key)
    if cached is not None:
        return cached

    xy_margin = float(cfg.crossing_xy_margin_um) * µm
    core_t = float(cfg.crossing_core_t_um) * µm
    clad_below = float(cfg.crossing_clad_below_um) * µm
    clad_above = float(cfg.crossing_clad_above_um) * µm
    extension = (
        float(cfg.crossing_extension_um) * µm
        if cfg.crossing_extension_um is not None
        else xy_margin + cfg.pml_m
    )
    z_padding = float(cfg.crossing_clad_below_um + cfg.crossing_clad_above_um) * 0.5 * µm
    prepared = gdsf.prepare_component(
        cfg.crossing_component,
        layer=cfg.crossing_layer,
        n_core=cfg.crossing_n_core,
        n_clad=cfg.crossing_n_clad,
        core_thickness=core_t,
        clad_below=clad_below,
        clad_above=clad_above,
        xy_padding=extension,
        z_padding=z_padding + cfg.pml_m,
        extension=extension,
        port_overlap=float(cfg.crossing_port_overlap_um) * µm,
    )
    _CROSSING_PREP_CACHE[key] = prepared
    return prepared


def build_crossing_case(cfg: SweepConfig, monitor_delta_um: float):
    xy_margin = float(cfg.crossing_xy_margin_um) * µm
    core_t = float(cfg.crossing_core_t_um) * µm
    clad_below = float(cfg.crossing_clad_below_um) * µm
    clad_above = float(cfg.crossing_clad_above_um) * µm
    port_margin = float(cfg.crossing_port_margin_um) * µm
    source_offset = float(cfg.crossing_source_offset_um) * µm
    output_offset = float(cfg.crossing_output_monitor_offset_um) * µm
    prepared = prepare_crossing_geometry(cfg)
    design = prepared["design"]
    ports = prepared["ports"]
    src = ports["o1"]
    output_ports = ["o2", "o3", "o4"]
    monitor_delta = float(monitor_delta_um) * µm
    mode_span = max(float(src["width"]) + 2.0 * port_margin, float(src["width"]) + 0.1 * µm)
    monitor_z_span = core_t + 2.0 * port_margin
    source_plane = port_plane(
        src,
        span=mode_span,
        z_span=monitor_z_span,
        z_center=float(src["z_center"]),
        offset=source_offset,
    )
    source_monitor_plane = port_plane(
        src,
        span=mode_span,
        z_span=monitor_z_span,
        z_center=float(src["z_center"]),
        offset=source_offset + monitor_delta,
    )
    source_center = line_center(source_plane)

    dx, _ = calc_optimal_fdtd_params(
        cfg.wavelength_m,
        cfg.crossing_n_core,
        dims=3,
        safety_factor=cfg.courant_safety,
        points_per_wavelength=int(cfg.ppw),
        width=design.width,
        height=design.height,
        depth=design.depth,
    )
    dt = float(cfg.courant_safety) * float(dx) / (LIGHT_SPEED * math.sqrt(3.0))
    grid = design.rasterize(resolution=dx)
    freqs = np.asarray(cfg.frequencies_hz, dtype=float)
    runtime_output_distance_um = 0.0
    output_monitor_planes = {}
    for port_name in output_ports:
        port = ports[port_name]
        span = max(float(port["width"]) + 2.0 * port_margin, float(port["width"]) + 0.1 * µm)
        plane = port_plane(
            port,
            span=span,
            z_span=monitor_z_span,
            z_center=float(port["z_center"]),
            offset=output_offset,
        )
        output_monitor_planes[port_name] = plane
        c_out = line_center(plane)
        runtime_output_distance_um = max(
            runtime_output_distance_um,
            float(np.hypot(c_out[0] - source_center[0], c_out[1] - source_center[1])) / µm,
        )
    o2_region, o4_region = mirror_lock_plane_pair_regions(
        start_a=output_monitor_planes["o2"][0],
        end_a=output_monitor_planes["o2"][1],
        start_b=output_monitor_planes["o4"][0],
        end_b=output_monitor_planes["o4"][1],
        plane_normal="y",
        size_a=None,
        size_b=None,
        dx=dx,
        dy=dx,
        dz=dx,
        shape=tuple(np.asarray(grid.permittivity).shape),
    )
    output_monitor_planes["o2"] = (o2_region.start, o2_region.end)
    output_monitor_planes["o4"] = (o4_region.start, o4_region.end)
    runtime_output_distance_um = 0.0
    for plane in output_monitor_planes.values():
        c_out = line_center(plane)
        runtime_output_distance_um = max(
            runtime_output_distance_um,
            float(np.hypot(c_out[0] - source_center[0], c_out[1] - source_center[1])) / µm,
        )
    pulse = build_pulse(
        cfg,
        freqs,
        dt,
        max_output_distance_um=runtime_output_distance_um,
        n_eff_guess=0.5 * (cfg.crossing_n_core + cfg.crossing_n_clad),
    )
    source = ModeSource(
        grid=grid,
        center=source_center,
        width=mode_span,
        height=monitor_z_span,
        wavelength=cfg.wavelength_m,
        pol="te",
        signal=pulse.signal,
        direction=src["direction"],
    )
    monitor_cfg = _monitor_cfg(freqs)
    monitors = [
        Monitor(start=source_monitor_plane[0], end=source_monitor_plane[1], name="o1", **monitor_cfg),
        Monitor(start=source_plane[0], end=source_plane[1], name="o1_ref", **monitor_cfg),
    ]
    for port_name in output_ports:
        plane = output_monitor_planes[port_name]
        monitors.append(Monitor(start=plane[0], end=plane[1], name=port_name, **monitor_cfg))
    sim = Simulation(
        design=design,
        sources=[source],
        monitors=monitors,
        boundaries=[
            PML(edges=["left", "right", "top", "bottom"], thickness=cfg.pml_m, formulation="sigma"),
            PML(edges=["front", "back"], thickness=cfg.pml_m, formulation="sigma"),
        ],
        time=pulse.time,
        resolution=dx,
    )
    port_specs = [
        PortSpec(
            name="o1",
            monitor_name="o1",
            reference_monitor="o1_ref",
            direction=gdsf.positive_axis_direction(src["direction"]),
            polarization="te",
            mode_index=0,
            incident_wave=incoming_wave(src["direction"]),
            scattered_wave=outgoing_wave(src["direction"]),
        )
    ]
    for port_name in output_ports:
        direction = ports[port_name]["direction"]
        port_specs.append(
            PortSpec(
                name=port_name,
                monitor_name=port_name,
                direction=gdsf.positive_axis_direction(direction),
                polarization="te",
                mode_index=0,
                incident_wave=incoming_wave(direction),
                scattered_wave=outgoing_wave(direction),
            )
        )
    metadata = {
        "source_reference_offset_um": source_offset / µm,
        "source_monitor_offset_um": (source_offset + monitor_delta) / µm,
        "source_reference_clearance_um": monitor_clearance_xy(
            source_plane,
            width=design.width,
            height=design.height,
            pml_xy=cfg.pml_m,
        ),
        "source_monitor_clearance_um": monitor_clearance_xy(
            source_monitor_plane,
            width=design.width,
            height=design.height,
            pml_xy=cfg.pml_m,
        ),
    }
    return sim, monitors, port_specs, freqs, pulse, metadata


def run_case(cfg: SweepConfig, device: str, monitor_delta_um: float) -> SweepCaseResult:
    if device == "straight":
        sim, monitors, ports, freqs, pulse, meta = build_straight_case(cfg, monitor_delta_um)
        output_ports = ["o1", "o2"]
    elif device == "crossing":
        sim, monitors, ports, freqs, pulse, meta = build_crossing_case(cfg, monitor_delta_um)
        output_ports = ["o1", "o2", "o3", "o4"]
    else:
        raise ValueError(f"Unsupported device '{device}'.")

    t0 = pytime.perf_counter()
    steps = sim.run_compiled_until_decay(
        monitors,
        min_time_s=float(pulse.source_end_time + pulse.tail_time),
        lookback_records=cfg.lookback_records,
        decay_ratio=cfg.decay_ratio,
        progress=False,
    )
    runtime_s = max(pytime.perf_counter() - t0, 1e-12)
    result = sim.get_S_matrix_modal_dft(
        source_port="o1",
        ports=ports,
        output_ports=output_ports,
        frequencies=freqs,
        as_sax=False,
        return_diagnostics=True,
        min_incident_db=-50.0,
    )
    center_idx = int(np.argmin(np.abs(freqs - LIGHT_SPEED / cfg.wavelength_m)))
    valid = np.asarray(result["diagnostics"]["valid_mask"], dtype=bool)
    source_spec = ports[0]
    waves = result["diagnostics"]["waves"]["o1"]
    raw_num = select_wave_component(
        waves,
        source_spec.scattered_wave,
        use_reference=False,
    )
    raw_den = select_wave_component(
        waves,
        source_spec.incident_wave,
        use_reference=True,
    )
    raw_s11 = np.zeros_like(raw_num, dtype=np.complex128)
    valid_den = np.abs(raw_den) > 1e-18
    raw_s11[valid_den] = raw_num[valid_den] / raw_den[valid_den]
    raw_s11_db = 20.0 * np.log10(max(abs(complex(np.asarray(raw_s11)[center_idx])), 1e-12))
    corrected_s11 = complex(
        np.asarray(result["s_matrix"][("o1", "o1")], dtype=np.complex128)[center_idx]
    )
    transmission_db = None
    top_db = None
    bottom_db = None
    top_dominance_db = None
    bottom_dominance_db = None
    top_bottom_mismatch_db = None
    if device == "straight":
        transmission_db = 20.0 * np.log10(
            max(
                abs(
                    complex(
                        np.asarray(result["s_matrix"][("o2", "o1")], dtype=np.complex128)[center_idx]
                    )
                ),
                1e-12,
            )
        )
    else:
        top_waves = result["diagnostics"]["waves"]["o2"]
        bottom_waves = result["diagnostics"]["waves"]["o4"]
        top_db = 20.0 * np.log10(
            max(
                abs(
                    complex(
                        np.asarray(result["s_matrix"][("o2", "o1")], dtype=np.complex128)[center_idx]
                    )
                ),
                1e-12,
            )
        )
        top_dominance_db = wave_dominance_db(
            top_waves["a_plus"],
            top_waves["a_minus"],
            outgoing_wave("-y"),
            valid,
        )
        transmission_db = 20.0 * np.log10(
            max(
                abs(
                    complex(
                        np.asarray(result["s_matrix"][("o3", "o1")], dtype=np.complex128)[center_idx]
                    )
                ),
                1e-12,
            )
        )
        bottom_dominance_db = wave_dominance_db(
            bottom_waves["a_plus"],
            bottom_waves["a_minus"],
            outgoing_wave("+y"),
            valid,
        )
        bottom_db = 20.0 * np.log10(
            max(
                abs(
                    complex(
                        np.asarray(result["s_matrix"][("o4", "o1")], dtype=np.complex128)[center_idx]
                    )
                ),
                1e-12,
            )
        )
        top_bottom_mismatch_db = float(bottom_db - top_db)
    return SweepCaseResult(
        device=str(device),
        monitor_delta_um=float(monitor_delta_um),
        crossing_output_monitor_offset_um=float(cfg.crossing_output_monitor_offset_um),
        source_reference_offset_um=float(meta["source_reference_offset_um"]),
        source_monitor_offset_um=float(meta["source_monitor_offset_um"]),
        raw_s11_db=float(raw_s11_db),
        corrected_s11_db=20.0 * math.log10(max(abs(corrected_s11), 1e-12)),
        source_dominance_db=wave_dominance_db(
            waves["a_plus"],
            waves["a_minus"],
            source_spec.incident_wave,
            valid,
        ),
        source_correction_mag=float(
            abs(np.asarray(result["diagnostics"]["source_scattered_correction"], dtype=np.complex128)[center_idx])
        ),
        incident_mag=float(
            abs(
                complex(
                    select_wave_component(
                        waves,
                        source_spec.incident_wave,
                        use_reference=True,
                    )[center_idx]
                )
            )
        ),
        source_monitor_clearance_um=float(meta["source_monitor_clearance_um"]),
        source_reference_clearance_um=float(meta["source_reference_clearance_um"]),
        transmission_db=transmission_db,
        top_db=top_db,
        bottom_db=bottom_db,
        top_dominance_db=top_dominance_db,
        bottom_dominance_db=bottom_dominance_db,
        top_bottom_mismatch_db=top_bottom_mismatch_db,
        steps=int(steps),
        runtime_s=float(runtime_s),
    )


def plot_results(results: list[SweepCaseResult], out_dir: Path) -> Path:
    devices = sorted({r.device for r in results})
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), dpi=220)
    for device in devices:
        rows = sorted(
            [r for r in results if r.device == device],
            key=lambda row: row.monitor_delta_um,
        )
        x = np.asarray([r.monitor_delta_um for r in rows], dtype=float)
        raw = np.asarray([r.raw_s11_db for r in rows], dtype=float)
        corrected = np.asarray([r.corrected_s11_db for r in rows], dtype=float)
        dominance = np.asarray([r.source_dominance_db for r in rows], dtype=float)
        axes[0].plot(x, raw, marker="o", lw=1.8, label=f"{device} raw")
        axes[0].plot(x, corrected, marker="o", lw=1.8, ls="--", label=f"{device} corrected")
        axes[1].plot(x, dominance, marker="o", lw=1.8, label=device)
    axes[0].set_title("Source-Port S11 vs Source-Monitor Spacing")
    axes[0].set_xlabel("Source-monitor delta (um)")
    axes[0].set_ylabel("Magnitude (dB)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_title("Incident-Branch Dominance")
    axes[1].set_xlabel("Source-monitor delta (um)")
    axes[1].set_ylabel("Dominance (dB)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out_path = out_dir / "source_port_s11_spacing_sweep_3d.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def parse_devices(raw: str) -> tuple[str, ...]:
    return tuple(tok.strip() for tok in raw.split(",") if tok.strip())


def parse_deltas(raw: str) -> tuple[float, ...]:
    return tuple(float(tok.strip()) for tok in raw.split(",") if tok.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "results_source_port_s11_spacing_sweep_3d",
    )
    parser.add_argument("--devices", type=str, default="straight,crossing")
    parser.add_argument("--monitor-deltas-um", type=str, default="0.2,0.4,0.6")
    parser.add_argument("--crossing-output-monitor-offset-um", type=float, default=0.10)
    parser.add_argument("--ppw", type=int, default=6)
    parser.add_argument("--num-freqs", type=int, default=1)
    parser.add_argument("--pml-um", type=float, default=1.0)
    parser.add_argument("--voxel-symmetry-only", action="store_true")
    parser.add_argument("--field-mirror-only", action="store_true")
    parser.add_argument("--ripple-probe-only", action="store_true")
    parser.add_argument("--crossing-extension-um", type=float, default=float("nan"))
    args = parser.parse_args()

    cfg = SweepConfig(
        devices=parse_devices(args.devices),
        source_monitor_deltas_um=parse_deltas(args.monitor_deltas_um),
        crossing_output_monitor_offset_um=float(args.crossing_output_monitor_offset_um),
        crossing_extension_um=(
            None if not math.isfinite(float(args.crossing_extension_um)) else float(args.crossing_extension_um)
        ),
        pml_um=float(args.pml_um),
        ppw=int(args.ppw),
        num_freqs=int(args.num_freqs),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.voxel_symmetry_only:
        report = crossing_voxel_symmetry_report(cfg)
        manifest_path = args.output_dir / "crossing_voxel_symmetry.json"
        manifest_path.write_text(json.dumps(report, indent=2))
        print(f"Wrote {manifest_path}")
        vol = report["volume"]
        ctr = report["center_xy_slice"]
        print(
            "volume mask mismatches | tb={tb} lr={lr} rot={rot}".format(
                tb=vol["top_bottom_mask"]["mismatch_count"],
                lr=vol["left_right_mask"]["mismatch_count"],
                rot=vol["rotation_mask"]["mismatch_count"],
            )
        )
        print(
            "center slice mask mismatches | tb={tb} lr={lr} rot={rot}".format(
                tb=ctr["top_bottom_mask"]["mismatch_count"],
                lr=ctr["left_right_mask"]["mismatch_count"],
                rot=ctr["rotation_mask"]["mismatch_count"],
            )
        )
        return
    if args.field_mirror_only:
        delta = float(cfg.source_monitor_deltas_um[0])
        report = crossing_field_mirror_report(cfg, delta)
        manifest_path = args.output_dir / "crossing_field_mirror.json"
        manifest_path.write_text(json.dumps(report, indent=2))
        print(f"Wrote {manifest_path}")
        print(
            "center S | o2={o2:.2f} dB o4={o4:.2f} dB | dom o2={d2:.2f} dB o4={d4:.2f} dB".format(
                o2=report["s_db"]["o2_o1"],
                o4=report["s_db"]["o4_o1"],
                d2=report["dominance_db"]["o2"],
                d4=report["dominance_db"]["o4"],
            )
        )
        for name, stats in report["components"].items():
            print(
                "{name}: complex residual={cr:.2f} dB | magnitude residual={mr:.2f} dB".format(
                    name=name,
                    cr=stats["complex_residual"]["residual_db"],
                    mr=stats["magnitude_residual_db"],
                )
            )
        return
    if args.ripple_probe_only:
        delta = float(cfg.source_monitor_deltas_um[0])
        report = crossing_ripple_probe(cfg, delta)
        manifest_path = args.output_dir / "crossing_ripple_probe.json"
        manifest_path.write_text(json.dumps(report, indent=2))
        print(f"Wrote {manifest_path}")
        for name, stats in report["extrema"].items():
            print(
                "{name}: min={mn:.2f} dB @ {wmn:.4f} um | max={mx:.2f} dB @ {wmx:.4f} um".format(
                    name=name,
                    mn=stats["min_db"],
                    wmn=stats["min_wavelength_um"],
                    mx=stats["max_db"],
                    wmx=stats["max_wavelength_um"],
                )
            )
        return
    results: list[SweepCaseResult] = []
    skipped: list[dict[str, str | float]] = []
    for device in cfg.devices:
        for delta in cfg.source_monitor_deltas_um:
            try:
                results.append(run_case(cfg, device, delta))
            except ValueError as exc:
                if device == "crossing":
                    skipped.append(
                        {
                            "device": device,
                            "monitor_delta_um": float(delta),
                            "reason": str(exc),
                        }
                    )
                    break
                raise

    plot_path = plot_results(results, args.output_dir) if results else None
    manifest = {
        "config": asdict(cfg),
        "results": [asdict(r) for r in results],
        "skipped": skipped,
        "plot": str(plot_path) if plot_path is not None else None,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {manifest_path}")
    if plot_path is not None:
        print(f"Wrote {plot_path}")
    for row in results:
        line = (
            f"{row.device:8s} Δ={row.monitor_delta_um:.2f} um | "
            f"raw S11={row.raw_s11_db:.2f} dB | "
            f"corrected S11={row.corrected_s11_db:.2f} dB | "
            f"dominance={row.source_dominance_db:.2f} dB | "
            f"corr={row.source_correction_mag:.4f}"
        )
        if row.transmission_db is not None:
            line += f" | through={row.transmission_db:.2f} dB"
        if row.top_db is not None and row.bottom_db is not None:
            line += (
                f" | top={row.top_db:.2f} dB"
                f" | bottom={row.bottom_db:.2f} dB"
                f" | top_dom={row.top_dominance_db:.2f} dB"
                f" | bottom_dom={row.bottom_dominance_db:.2f} dB"
                f" | Δtb={row.top_bottom_mismatch_db:+.2f} dB"
            )
        print(line)
    for item in skipped:
        print(
            f"Skipped {item['device']} at Δ={item['monitor_delta_um']:.2f} um: {item['reason']}"
        )


if __name__ == "__main__":
    main()
