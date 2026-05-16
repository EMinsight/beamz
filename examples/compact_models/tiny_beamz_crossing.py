"""Tiny standalone 3D BeamZ crossing example.

Workflow:
1. Define fixed hyperparameters.
2. Build the 3D crossing geometry and extend the ports.
3. Define the broadband source and DFT monitors.
4. Build the simulation.
5. Save an overview plot of the design, source, and monitors.
6. Run the simulation with adaptive monitor-decay stopping.
7. Extract and plot the S-parameters.
"""

from __future__ import annotations
import time as pytime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from beamz import (
    LIGHT_SPEED,
    ModeMonitor,
    ModeSource,
    Monitor,
    PML,
    Simulation,
    dxdt,
    µm,
)
from beamz.design.io import gdsf
from beamz.devices._placement import mirror_lock_plane_pair_regions
from beamz.devices.sources.signals import gaussian_band_pulse

# Fixed example hyperparameters. Match the gsim/meep reference geometry and
# domain sizing directly, while using the currently best-performing BeamZ
# source-port spacing from the local #106 benchmark harness.
OUT_DIR = Path("benchmarks/results/tiny_beamz_crossing")
COMPONENT_NAME = "ebeam_crossing4"
FALLBACK_COMPONENT_NAME = "crossing_linear_taper"
NUM_FREQS = 11
PPW = 5
WL0, WL_MIN, WL_MAX = 1550.0e-9, 1530.0e-9, 1570.0e-9
N_CORE, N_CLAD = 3.47, 1.44
LAYER = (1, 0)
CORE_T = 0.22 * µm
CLAD_BELOW = 0.50 * µm
CLAD_ABOVE = 0.50 * µm
PML_XY, PML_Z = 1.0 * µm, 1.0 * µm
XY_MARGIN = 0.50 * µm
Z_PADDING = 0.50 * µm
EXTENSION = XY_MARGIN + PML_XY
PORT_OVERLAP = 0.0 * µm
PORT_MARGIN = 0.50 * µm
MODE_PLANE_SIZE_SCALE = 1.8
MONITOR_Z_SPAN = MODE_PLANE_SIZE_SCALE * (CORE_T + 2.0 * PORT_MARGIN)
SOURCE_PORT_OFFSET = 0.10 * µm
DISTANCE_SOURCE_TO_MONITORS = 0.40 * µm
# For the y-directed weak ports, deeper planes look cleaner visually but snap
# to a less symmetric pair of Yee slices at 8 PPW. Keep the outputs near the
# imported port planes so o2/o4 stay mirror-locked on the raster grid.
OUTPUT_MONITOR_OFFSET = 0.05 * µm
RUN_AFTER_SOURCES_UOC = 90.0
DECAY_RATIO = 1e-4
LOOKBACK_RECORDS = 20
PML_FORMULATION = "sigma"

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


def port_mode_geometry(port: dict) -> tuple[float, float, float]:
    width = float(port["width"])
    span = MODE_PLANE_SIZE_SCALE * max(width + 2.0 * PORT_MARGIN, width + 0.1 * µm)
    return span, float(MONITOR_Z_SPAN), float(port["z_center"])


def enable_cpml_material_extrusion(design, *, pml_xy: float, pml_z: float) -> None:
    """Extrude rasterized material through CPML without changing geometry."""
    get_material_grids = design.get_material_grids

    def _copy_shell(arr: np.ndarray, axis: int, count: int) -> np.ndarray:
        if count <= 0 or count >= arr.shape[axis]:
            return arr
        low = [slice(None)] * arr.ndim
        low[axis] = slice(0, count)
        low_ref = [slice(None)] * arr.ndim
        low_ref[axis] = count
        arr[tuple(low)] = np.expand_dims(arr[tuple(low_ref)], axis=axis)

        high = [slice(None)] * arr.ndim
        high[axis] = slice(arr.shape[axis] - count, arr.shape[axis])
        high_ref = [slice(None)] * arr.ndim
        high_ref[axis] = arr.shape[axis] - count - 1
        arr[tuple(high)] = np.expand_dims(arr[tuple(high_ref)], axis=axis)
        return arr

    def _extruded_material_grids(resolution):
        grids = get_material_grids(resolution)
        pml_counts = {
            "x": int(np.ceil(float(pml_xy) / float(resolution))),
            "y": int(np.ceil(float(pml_xy) / float(resolution))),
            "z": int(np.ceil(float(pml_z) / float(resolution))),
        }
        out = []
        for grid_arr in grids:
            arr = np.array(grid_arr, copy=True)
            if arr.ndim == 3:
                arr = _copy_shell(arr, axis=2, count=pml_counts["x"])
                arr = _copy_shell(arr, axis=1, count=pml_counts["y"])
                arr = _copy_shell(arr, axis=0, count=pml_counts["z"])
            elif arr.ndim == 2:
                arr = _copy_shell(arr, axis=1, count=pml_counts["x"])
                arr = _copy_shell(arr, axis=0, count=pml_counts["y"])
            out.append(arr)
        return tuple(out)

    design.get_material_grids = _extruded_material_grids


def plot_simulation_overview(
    out_path: Path,
    eps_grid: np.ndarray,
    *,
    width: float,
    height: float,
    depth: float,
    z_focus: float,
    source_plane,
    monitor_planes,
    world_origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    eps_grid = np.asarray(eps_grid, dtype=float)
    if eps_grid.ndim == 3:
        z_idx = int(np.clip(round((z_focus / max(depth, 1e-30)) * (eps_grid.shape[0] - 1)), 0, eps_grid.shape[0] - 1))
        eps_view = eps_grid[z_idx]
    else:
        eps_view = eps_grid

    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=220)
    im = ax.imshow(
        eps_view,
        origin="lower",
        extent=[
            world_origin[0] / µm,
            (world_origin[0] + width) / µm,
            world_origin[1] / µm,
            (world_origin[1] + height) / µm,
        ],
        cmap="viridis",
        aspect="equal",
    )
    fig.colorbar(im, ax=ax, label="Permittivity", fraction=0.046, pad=0.04)

    def _plot_plane(line, label, color):
        (x0, y0, _), (x1, y1, _) = line
        ax.plot(
            [
                (x0 + world_origin[0]) / µm,
                (x1 + world_origin[0]) / µm,
            ],
            [
                (y0 + world_origin[1]) / µm,
                (y1 + world_origin[1]) / µm,
            ],
            color=color,
            lw=2.0,
            label=label,
        )

    _plot_plane(source_plane, "source", "white")
    for name, plane in monitor_planes.items():
        _plot_plane(plane, name, "tab:red")

    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title("Simulation overview")
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_sparameters_db(out_path: Path, wl_um: np.ndarray, s_matrix: dict):
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=220)
    for (out_port, in_port), values in sorted(s_matrix.items()):
        arr = np.asarray(values, dtype=np.complex128)
        ax.plot(wl_um, 20.0 * np.log10(np.maximum(np.abs(arr), 1e-12)), lw=2.0, label=f"S[{out_port},{in_port}]")
    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("S-parameters")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def wave_dominance_db(a_plus: np.ndarray, a_minus: np.ndarray, selector: str, mask: np.ndarray) -> float:
    # Report how cleanly a monitor separates the selected traveling wave from
    # the opposite-going component.
    sel = np.asarray(a_plus if selector == "plus" else a_minus, dtype=np.complex128)
    opp = np.asarray(a_minus if selector == "plus" else a_plus, dtype=np.complex128)
    valid = np.asarray(mask, dtype=bool)
    if not np.any(valid):
        return float("nan")
    p_sel = float(np.mean(np.abs(sel[valid]) ** 2))
    p_opp = float(np.mean(np.abs(opp[valid]) ** 2))
    return 10.0 * np.log10(max(p_sel, 1e-18) / max(p_opp, 1e-18))


def format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m {sec:.0f}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h {int(minutes)}m"


# 1. Import the GDSFactory/PDK component, extrude it to 3D, pad the domain,
# and extend the ports into uniform straight sections.
OUT_DIR.mkdir(parents=True, exist_ok=True)
try:
    prepared = gdsf.prepare_component(
        COMPONENT_NAME,
        layer=LAYER,
        n_core=N_CORE,
        n_clad=N_CLAD,
        core_thickness=CORE_T,
        clad_below=CLAD_BELOW,
        clad_above=CLAD_ABOVE,
        xy_padding=EXTENSION,
        z_padding=Z_PADDING + PML_Z,
        extension=EXTENSION,
        port_overlap=PORT_OVERLAP,
    )
except ValueError as exc:
    if "Could not resolve gdsfactory/PDK component" not in str(exc):
        raise
    print(
        f"{exc} Falling back to generic gdsfactory component "
        f"'{FALLBACK_COMPONENT_NAME}'."
    )
    prepared = gdsf.prepare_component(
        FALLBACK_COMPONENT_NAME,
        layer=LAYER,
        n_core=N_CORE,
        n_clad=N_CLAD,
        core_thickness=CORE_T,
        clad_below=CLAD_BELOW,
        clad_above=CLAD_ABOVE,
        xy_padding=EXTENSION,
        z_padding=Z_PADDING + PML_Z,
        extension=EXTENSION,
        port_overlap=PORT_OVERLAP,
    )
design, ports = prepared["design"], prepared["ports"]
world_origin = tuple(float(v) for v in prepared.get("world_origin", (0.0, 0.0, 0.0)))
source_port, output_ports = "o1", ["o2", "o3", "o4"]
port_names = (source_port, *output_ports)
dx, dt = dxdt(WL0, n_max=N_CORE, dims=3, safety_factor=0.999, points_per_wavelength=PPW)
grid = design.rasterize(resolution=dx)
if PML_FORMULATION == "cpml":
    enable_cpml_material_extrusion(design, pml_xy=PML_XY, pml_z=PML_Z)
freqs = np.linspace(LIGHT_SPEED / WL_MAX, LIGHT_SPEED / WL_MIN, NUM_FREQS, dtype=np.float32)
wl_um = LIGHT_SPEED / freqs / µm

# 2. Place the source and monitors directly from the imported port metadata.
# Keep the Meep-matched source plane at 0.1 um inside the source port, but use
# a slightly deeper 0.5 um source-monitor plane from the local BeamZ S11 sweep.
# Output monitors are also moved farther inward so the weak cross ports see a
# cleaner outgoing guided mode before projection.
src = ports[source_port]
source_direction = src["direction"]
source_span, z_span, source_z_center = port_mode_geometry(src)
overview_z_focus = source_z_center
source_plane = port_plane(
    src,
    span=source_span,
    z_span=z_span,
    z_center=source_z_center,
    offset=SOURCE_PORT_OFFSET,
)
source_center = line_center(source_plane)
monitor_offsets = {source_port: SOURCE_PORT_OFFSET + DISTANCE_SOURCE_TO_MONITORS}
monitor_planes = {
    source_port: port_plane(
        src,
        span=source_span,
        z_span=z_span,
        z_center=source_z_center,
        offset=monitor_offsets[source_port],
    )
}
for port_name in output_ports:
    port = ports[port_name]
    span, monitor_z_span, z_center = port_mode_geometry(port)
    monitor_offsets[port_name] = OUTPUT_MONITOR_OFFSET
    monitor_planes[port_name] = port_plane(
        port,
        span=span,
        z_span=monitor_z_span,
        z_center=z_center,
        offset=monitor_offsets[port_name],
    )
o2_region, o4_region = mirror_lock_plane_pair_regions(
    start_a=monitor_planes["o2"][0],
    end_a=monitor_planes["o2"][1],
    start_b=monitor_planes["o4"][0],
    end_b=monitor_planes["o4"][1],
    plane_normal="y",
    size_a=None,
    size_b=None,
    dx=dx,
    dy=dx,
    dz=dx,
    shape=tuple(np.asarray(grid.permittivity).shape),
)
monitor_planes["o2"] = (o2_region.start, o2_region.end)
monitor_planes["o4"] = (o4_region.start, o4_region.end)
print("Plane positions relative to imported ports (um):")
print(f"  source: {SOURCE_PORT_OFFSET / µm:.2f}")
for port_name in port_names:
    print(f"  {port_name}: {monitor_offsets[port_name] / µm:.2f}")
runtime_output_distance_um = 0.0
for port_name in output_ports:
    c_out = line_center(monitor_planes[port_name])
    runtime_output_distance_um = max(
        runtime_output_distance_um,
        float(np.hypot(c_out[0] - source_center[0], c_out[1] - source_center[1])) / µm,
    )

# 3. Generate the broadband Gaussian pulse and build the source / DFT monitors.
pulse = gaussian_band_pulse(
    freqs,
    carrier_frequency=LIGHT_SPEED / WL0,
    dt=dt,
    run_after_sources_uoc=RUN_AFTER_SOURCES_UOC,
    max_output_distance_um=runtime_output_distance_um,
)
source = ModeSource(
    grid=grid,
    center=source_center,
    width=source_span,
    height=z_span,
    wavelength=WL0,
    pol="te",
    signal=pulse.signal,
    direction=source_direction,
)
source.initialize(grid.permittivity, dx)
monitor_cfg = dict(
    record_fields=False,
    dft_enabled=True,
    dft_frequencies=freqs,
    dft_components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
    dft_window="none",
    dft_record_every_step=True,
)
monitors = [
    ModeMonitor(
        start=monitor_planes[p][0],
        end=monitor_planes[p][1],
        name=p,
        direction=ports[p]["direction"],
        polarization="te",
        reference_monitor="o1_ref" if p == source_port else None,
        **monitor_cfg,
    )
    for p in port_names
]
mode_monitors = {m.name: m for m in monitors}
modal_ports = [mode_monitors[name] for name in port_names]
reference_monitor = Monitor(
    start=source_plane[0],
    end=source_plane[1],
    name="o1_ref",
    **monitor_cfg,
)
all_monitors = [*monitors, reference_monitor]
decay_monitors = monitors

# 4. Feed the design, source, monitors, boundaries, and time array into the
# simulation object.
sim = Simulation(
    design=design,
    sources=[source],
    monitors=all_monitors,
    boundaries=[
        PML(
            edges=["left", "right", "top", "bottom"],
            thickness=PML_XY,
            formulation=PML_FORMULATION,
        ),
        PML(
            edges=["front", "back"],
            thickness=PML_Z,
            formulation=PML_FORMULATION,
        ),
    ],
    time=pulse.time,
    resolution=dx,
)
sim.show()

# 5. Save a compact overview plot of the rasterized structure with the source
# and monitor planes overlaid.
print(f"Workload: grid={grid.permittivity.shape}, voxels={int(np.prod(np.asarray(grid.permittivity).shape)):,}, updates~{int(np.prod(np.asarray(grid.permittivity).shape))*len(pulse.time):.3e}")
estimated_updates = float(int(np.prod(np.asarray(grid.permittivity).shape)) * len(pulse.time))
print(
    "Estimated runtime: "
    f"100 MCUPS ~ {format_duration(estimated_updates / 100e6)}, "
    f"250 MCUPS ~ {format_duration(estimated_updates / 250e6)}, "
    f"500 MCUPS ~ {format_duration(estimated_updates / 500e6)}"
)
plot_simulation_overview(
    OUT_DIR / "beamz_crossing_overview.png",
    np.asarray(grid.permittivity, dtype=float),
    width=design.width,
    height=design.height,
    depth=design.depth,
    z_focus=overview_z_focus,
    source_plane=source_plane,
    monitor_planes=monitor_planes,
    world_origin=world_origin,
)
print(f"Saved overview figure: {OUT_DIR / 'beamz_crossing_overview.png'}")

# 6. Run in compiled chunks until the monitor power has decayed sufficiently
# after the pulse leaves the device.
wall_t0 = pytime.perf_counter()
executed_steps = sim.run_compiled_until_decay(
    decay_monitors,
    min_time_s=pulse.source_end_time + pulse.tail_time,
    lookback_records=LOOKBACK_RECORDS,
    decay_ratio=DECAY_RATIO,
    progress=True,
)
wall_s = max(pytime.perf_counter() - wall_t0, 1e-12)
num_voxels = int(np.prod(np.asarray(grid.permittivity).shape))
print(
    "Simulation stats: "
    f"steps={executed_steps}, voxels={num_voxels:,}, sim_time={(executed_steps - 1) * dt * 1e15:.2f}fs, "
    f"wall={wall_s:.2f}s, step_rate={executed_steps / wall_s:.2f} steps/s, MCUPS={num_voxels * executed_steps / wall_s / 1e6:.2f}"
)

# 7. Extract the broadband S-matrix directly from the first-class modal
# monitors. Each ModeMonitor carries the port direction and derives the modal
# wave selectors used by the projection code.
port_specs = {name: mode_monitors[name].to_port() for name in port_names}
result = sim.get_S_matrix_modal_dft(
    source_port=mode_monitors[source_port],
    ports=modal_ports,
    output_ports=modal_ports,
    frequencies=freqs,
    as_sax=False,
    return_diagnostics=True,
    min_incident_db=-45.0,
)
i0 = int(np.argmin(np.abs(wl_um - WL0 / µm)))
valid = np.asarray(result["diagnostics"]["valid_mask"], dtype=bool)
source_waves = result["diagnostics"]["waves"]["o1"]
source_dom = wave_dominance_db(
    source_waves["a_plus"],
    source_waves["a_minus"],
    port_specs[source_port].incident_wave,
    valid,
)
print(f"o1 wave dominance: {source_dom:.2f} dB")
src_cond = np.asarray(
    result["diagnostics"]["condition_numbers"]["o1"]["monitor"], dtype=float
)
src_ref_cond = np.asarray(
    result["diagnostics"]["condition_numbers"]["o1"]["reference"], dtype=float
)
if src_cond.size and src_ref_cond.size:
    print(
        "o1 projection conditioning "
        f"@ {wl_um[i0]:.4f}um: main={src_cond[i0]:.2e}, ref={src_ref_cond[i0]:.2e}"
    )

s_matrix = {
    (port_name, source_port): np.asarray(
        result["s_matrix"][(port_name, source_port)], dtype=np.complex128
    )
    for port_name in port_names
}
for port_name in output_ports:
    waves = result["diagnostics"]["waves"][port_name]
    dom = wave_dominance_db(
        waves["a_plus"],
        waves["a_minus"],
        port_specs[port_name].scattered_wave,
        valid,
    )
    print(
        f"{port_name} at imported port plane offset {monitor_offsets[port_name] / µm:.2f} um "
        f"(dominance={dom:.2f} dB)"
    )
for port_name in port_names:
    mag = abs(s_matrix[(port_name, "o1")][i0])
    print(f"S[{port_name},o1] @ {wl_um[i0]:.4f}um: {20.0 * np.log10(max(mag, 1e-12)):.2f} dB")

# Refresh the overview after the run so the latest generated artifact matches
# the final S-matrix extraction path.
plot_simulation_overview(
    OUT_DIR / "beamz_crossing_overview.png",
    np.asarray(grid.permittivity, dtype=float),
    width=design.width,
    height=design.height,
    depth=design.depth,
    z_focus=overview_z_focus,
    source_plane=source_plane,
    monitor_planes=monitor_planes,
    world_origin=world_origin,
)
print(f"Updated overview figure: {OUT_DIR / 'beamz_crossing_overview.png'}")

# 8. Save the final S-parameter plot using the same helper style as the full
# example so regression checks remain straightforward.
plot_sparameters_db(OUT_DIR / "beamz_crossing_sparams.png", wl_um, s_matrix)
