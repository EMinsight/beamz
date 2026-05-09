import numpy as np

from beamz import LIGHT_SPEED, PML, ModeSource, Monitor, PortSpec, Simulation, dxdt, µm
from beamz.design.io import gdsf
from beamz.devices.sources.signals import gaussian_band_pulse


def _move_along(center: tuple[float, float], direction: str, distance: float):
    x, y = center
    return {
        "+x": (x + distance, y),
        "-x": (x - distance, y),
        "+y": (x, y + distance),
        "-y": (x, y - distance),
    }[str(direction)]


def _port_plane(
    port: dict,
    *,
    span: float,
    z_span: float,
    z_center: float,
    offset: float = 0.0,
):
    cx, cy = _move_along(port["center"], port["direction"], offset)
    z0 = float(z_center) - 0.5 * float(z_span)
    z1 = float(z_center) + 0.5 * float(z_span)
    if str(port["direction"]).endswith("x"):
        return (cx, cy - 0.5 * float(span), z0), (cx, cy + 0.5 * float(span), z1)
    return (cx - 0.5 * float(span), cy, z0), (cx + 0.5 * float(span), cy, z1)


def _line_center(line):
    a, b = line
    return tuple(0.5 * (float(a[i]) + float(b[i])) for i in range(len(a)))


def _distance_to_xy_pml(
    port: dict,
    *,
    width: float,
    height: float,
    pml_xy: float,
) -> float:
    outward = gdsf.outward_direction(port["direction"])
    x, y = map(float, port["center"])
    return {
        "-x": max(x - float(pml_xy), 0.0),
        "+x": max(float(width) - float(pml_xy) - x, 0.0),
        "-y": max(y - float(pml_xy), 0.0),
        "+y": max(float(height) - float(pml_xy) - y, 0.0),
    }[outward]


def _incoming_wave(direction: str) -> str:
    direction = str(direction)
    if direction.endswith(("x", "y")):
        return "minus"
    return "plus" if direction.startswith("+") else "minus"


def _outgoing_wave(direction: str) -> str:
    return "minus" if _incoming_wave(direction) == "plus" else "plus"


def _plane_clearances_to_active_box(
    plane,
    *,
    world_origin: tuple[float, float, float],
    width: float,
    height: float,
    depth: float,
    pml_xy: float,
    pml_z: float,
):
    active_min = np.asarray(world_origin, dtype=float) + np.asarray(
        [float(pml_xy), float(pml_xy), float(pml_z)], dtype=float
    )
    active_max = np.asarray(world_origin, dtype=float) + np.asarray(
        [
            float(width) - float(pml_xy),
            float(height) - float(pml_xy),
            float(depth) - float(pml_z),
        ],
        dtype=float,
    )
    a = np.asarray(plane[0], dtype=float) + np.asarray(world_origin, dtype=float)
    b = np.asarray(plane[1], dtype=float) + np.asarray(world_origin, dtype=float)
    pmin = np.minimum(a, b)
    pmax = np.maximum(a, b)
    return {
        "left": float(pmin[0] - active_min[0]),
        "right": float(active_max[0] - pmax[0]),
        "bottom": float(pmin[1] - active_min[1]),
        "top": float(active_max[1] - pmax[1]),
        "front": float(pmin[2] - active_min[2]),
        "back": float(active_max[2] - pmax[2]),
    }


def _component_peak(mode_src: ModeSource, name: str) -> float:
    arr = getattr(mode_src, f"_{name}_profile", None)
    if arr is None:
        return 0.0
    data = np.asarray(arr)
    return float(np.max(np.abs(data))) if data.size else 0.0


def test_tiny_crossing_wave_selectors_match_current_3d_port_convention():
    for direction in ("+x", "-x", "+y", "-y"):
        assert _incoming_wave(direction) == "minus"
        assert _outgoing_wave(direction) == "plus"


def test_tiny_crossing_planes_keep_clearance_from_cpml_in_all_directions():
    wl0 = 1550.0e-9
    n_core, n_clad = 3.47, 1.44
    core_t = 0.22 * µm
    clad_below = 0.50 * µm
    clad_above = 0.50 * µm
    z_padding = 2.00 * µm
    extension = 2.50 * µm
    pml_xy = 1.5 * µm
    pml_z = 1.0 * µm
    port_overlap = 0.10 * µm
    port_margin = 0.50 * µm
    source_monitor_gap = 0.10 * µm

    prepared = gdsf.prepare_component(
        "crossing",
        layer=(1, 0),
        n_core=n_core,
        n_clad=n_clad,
        core_thickness=core_t,
        clad_below=clad_below,
        clad_above=clad_above,
        xy_padding=extension + pml_xy,
        z_padding=z_padding + pml_z,
        extension=extension,
        port_overlap=port_overlap,
    )
    design, ports = prepared["design"], prepared["ports"]
    world_origin = tuple(float(v) for v in prepared["world_origin"])
    src = ports["o1"]
    span = max(float(src["width"]) + 2.0 * port_margin, float(src["width"]) + 0.1 * µm)
    z_center = float(src["z_center"])
    z_span = clad_below + core_t + clad_above
    source_plane = _port_plane(
        src,
        span=span,
        z_span=z_span,
        z_center=z_center,
        offset=-source_monitor_gap,
    )
    monitor_planes = {
        "o1": _port_plane(src, span=span, z_span=z_span, z_center=z_center),
        "o2": _port_plane(ports["o2"], span=span, z_span=z_span, z_center=z_center),
        "o3": _port_plane(ports["o3"], span=span, z_span=z_span, z_center=z_center),
        "o4": _port_plane(ports["o4"], span=span, z_span=z_span, z_center=z_center),
    }

    source_clearances = _plane_clearances_to_active_box(
        source_plane,
        world_origin=world_origin,
        width=design.width,
        height=design.height,
        depth=design.depth,
        pml_xy=pml_xy,
        pml_z=pml_z,
    )
    assert min(source_clearances.values()) >= 1.95 * µm

    for plane in monitor_planes.values():
        clearances = _plane_clearances_to_active_box(
            plane,
            world_origin=world_origin,
            width=design.width,
            height=design.height,
            depth=design.depth,
            pml_xy=pml_xy,
            pml_z=pml_z,
        )
        assert min(clearances.values()) >= 1.95 * µm


def test_tiny_crossing_y_port_mode_probes_use_ex_hz_not_ey_hy():
    wl0 = 1550.0e-9
    n_core, n_clad = 3.47, 1.44
    core_t = 0.22 * µm
    clad_below = 0.50 * µm
    clad_above = 0.50 * µm
    z_padding = 2.00 * µm
    extension = 2.50 * µm
    pml_xy = 1.5 * µm
    pml_z = 1.0 * µm
    port_overlap = 0.10 * µm
    port_margin = 0.50 * µm
    monitor_extension_fraction = 0.50

    prepared = gdsf.prepare_component(
        "crossing",
        layer=(1, 0),
        n_core=n_core,
        n_clad=n_clad,
        core_thickness=core_t,
        clad_below=clad_below,
        clad_above=clad_above,
        xy_padding=extension + pml_xy,
        z_padding=z_padding + pml_z,
        extension=extension,
        port_overlap=port_overlap,
    )
    design, ports = prepared["design"], prepared["ports"]
    dx, _ = dxdt(
        wl0,
        n_max=n_core,
        dims=3,
        safety_factor=0.999,
        points_per_wavelength=10,
    )
    grid = design.rasterize(resolution=dx)

    src = ports["o1"]
    span = max(float(src["width"]) + 2.0 * port_margin, float(src["width"]) + 0.1 * µm)
    z_center = float(src["z_center"])
    z_span = clad_below + core_t + clad_above

    for port_name in ("o2", "o4"):
        extension_len = _distance_to_xy_pml(
            ports[port_name],
            width=design.width,
            height=design.height,
            pml_xy=pml_xy,
        )
        plane = _port_plane(
            ports[port_name],
            span=span,
            z_span=z_span,
            z_center=z_center,
            offset=-monitor_extension_fraction * extension_len,
        )
        center = _line_center(plane)
        probe = ModeSource(
            grid=grid,
            center=center,
            width=span,
            height=z_span,
            wavelength=wl0,
            pol="te",
            signal=np.zeros((1,), dtype=np.float32),
            direction=gdsf.outward_direction(ports[port_name]["direction"]),
        )
        probe.initialize(grid.permittivity, dx)

        ex = _component_peak(probe, "Ex")
        ey = _component_peak(probe, "Ey")
        hz = _component_peak(probe, "Hz")
        hy = _component_peak(probe, "Hy")

        assert ex > 1e-6
        assert hz > 1e-6
        assert ey <= 1e-6 * max(ex, 1.0)
        assert hy <= 1e-6 * max(hz, 1.0)


def test_prepare_component_ubcpdk_uses_explicit_stack_layers():
    prepared = gdsf.prepare_component(
        "ebeam_crossing4",
        layer=(1, 0),
        n_core=3.47,
        n_clad=1.44,
        core_thickness=0.22 * µm,
        clad_below=0.50 * µm,
        clad_above=0.50 * µm,
        xy_padding=1.50 * µm,
        z_padding=0.50 * µm,
        extension=1.50 * µm,
        port_overlap=0.10 * µm,
    )

    assert prepared["stack_profile"] is not None
    design = prepared["design"]
    layer_z = prepared["layer_z"]
    ports = prepared["ports"]
    pdk_layer_names = {
        layer["name"] for layer in prepared["stack_profile"]["pdk_layers"]
    }

    assert np.isclose(design.structures[0].material.permittivity, 1.0)
    assert "substrate" in pdk_layer_names
    assert "box" in layer_z
    assert "clad" in layer_z
    assert np.isclose(layer_z["box"][1] - layer_z["box"][0], 3.0 * µm)
    assert np.isclose(layer_z["clad"][1] - layer_z["clad"][0], 1.8 * µm)
    assert np.isclose(prepared["core_z1"] - prepared["core_z0"], 0.22 * µm)
    assert np.isclose(
        float(ports["o1"]["z_center"]),
        float(prepared["core_z0"]) + 0.11 * µm,
    )
    assert any(
        np.isclose(getattr(s, "sidewall_angle", 0.0), 10.0)
        and np.isclose(getattr(s, "depth", 0.0), 0.22 * µm)
        for s in design.structures
    )


def test_tiny_crossing_y_port_sparams_match_between_o2_and_o4_at_ppw8():
    wl0 = 1550.0e-9
    n_core, n_clad = 3.47, 1.44
    core_t = 0.22 * µm
    clad_below = 0.50 * µm
    clad_above = 0.50 * µm
    z_padding = 2.00 * µm
    extension = 2.50 * µm
    pml_xy = 1.5 * µm
    pml_z = 1.0 * µm
    port_overlap = 0.10 * µm
    port_margin = 0.50 * µm
    source_monitor_gap = 0.10 * µm

    prepared = gdsf.prepare_component(
        "crossing",
        layer=(1, 0),
        n_core=n_core,
        n_clad=n_clad,
        core_thickness=core_t,
        clad_below=clad_below,
        clad_above=clad_above,
        xy_padding=extension + pml_xy,
        z_padding=z_padding + pml_z,
        extension=extension,
        port_overlap=port_overlap,
    )
    design, ports = prepared["design"], prepared["ports"]
    dx, dt = dxdt(
        wl0,
        n_max=n_core,
        dims=3,
        safety_factor=0.999,
        points_per_wavelength=8,
    )
    grid = design.rasterize(resolution=dx)
    wl_min = 1545.0e-9
    wl_max = 1555.0e-9
    freqs = np.linspace(LIGHT_SPEED / wl_max, LIGHT_SPEED / wl_min, 3, dtype=np.float32)

    src = ports["o1"]
    span = max(float(src["width"]) + 2.0 * port_margin, float(src["width"]) + 0.1 * µm)
    z_center = float(src["z_center"])
    z_span = clad_below + core_t + clad_above

    source_monitor_plane = _port_plane(src, span=span, z_span=z_span, z_center=z_center)
    source_plane = _port_plane(
        src,
        span=span,
        z_span=z_span,
        z_center=z_center,
        offset=-source_monitor_gap,
    )
    source_center = _line_center(source_plane)

    out_planes = {}
    for port_name in ("o2", "o4"):
        out_planes[port_name] = _port_plane(
            ports[port_name],
            span=span,
            z_span=z_span,
            z_center=z_center,
        )

    runtime_output_distance_um = 0.0
    for port_name in ("o2", "o4"):
        center = _line_center(out_planes[port_name])
        runtime_output_distance_um = max(
            runtime_output_distance_um,
            float(np.hypot(center[0] - source_center[0], center[1] - source_center[1]))
            / µm,
        )

    pulse = gaussian_band_pulse(
        freqs,
        carrier_frequency=LIGHT_SPEED / wl0,
        dt=dt,
        run_after_sources_uoc=90.0,
        max_output_distance_um=runtime_output_distance_um,
    )
    source = ModeSource(
        grid=grid,
        center=source_center,
        width=span,
        height=z_span,
        wavelength=wl0,
        pol="te",
        signal=pulse.signal,
        direction=src["direction"],
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
    m_o1 = Monitor(
        start=source_monitor_plane[0],
        end=source_monitor_plane[1],
        name="o1",
        **monitor_cfg,
    )
    m_o2 = Monitor(
        start=out_planes["o2"][0], end=out_planes["o2"][1], name="o2", **monitor_cfg
    )
    m_o4 = Monitor(
        start=out_planes["o4"][0], end=out_planes["o4"][1], name="o4", **monitor_cfg
    )
    monitors = [m_o1, m_o2, m_o4]

    sim = Simulation(
        design=design,
        sources=[source],
        monitors=monitors,
        boundaries=[
            PML(edges=["left", "right", "top", "bottom"], thickness=pml_xy),
            PML(edges=["front", "back"], thickness=pml_z),
        ],
        time=pulse.time,
        resolution=dx,
    )
    sim.run_compiled_until_decay(
        [m_o1, m_o2, m_o4],
        min_time_s=pulse.source_end_time + pulse.tail_time,
        progress=False,
    )

    result = sim.get_S_matrix_modal_dft(
        source_port="o1",
        ports=[
            PortSpec(
                name="o1",
                monitor_name="o1",
                direction=src["direction"],
                polarization="te",
                mode_index=0,
                incident_wave=_incoming_wave(src["direction"]),
                scattered_wave=_outgoing_wave(src["direction"]),
            ),
            PortSpec(
                name="o2",
                monitor_name="o2",
                direction=ports["o2"]["direction"],
                polarization="te",
                mode_index=0,
                incident_wave=_incoming_wave(ports["o2"]["direction"]),
                scattered_wave=_outgoing_wave(ports["o2"]["direction"]),
            ),
            PortSpec(
                name="o4",
                monitor_name="o4",
                direction=ports["o4"]["direction"],
                polarization="te",
                mode_index=0,
                incident_wave=_incoming_wave(ports["o4"]["direction"]),
                scattered_wave=_outgoing_wave(ports["o4"]["direction"]),
            ),
        ],
        output_ports=["o2", "o4"],
        frequencies=freqs,
        as_sax=False,
        return_diagnostics=True,
    )

    center_idx = int(np.argmin(np.abs((LIGHT_SPEED / freqs) - wl0)))
    s_o2 = complex(np.asarray(result["s_matrix"][("o2", "o1")])[center_idx])
    s_o4 = complex(np.asarray(result["s_matrix"][("o4", "o1")])[center_idx])
    assert abs(abs(s_o2) - abs(s_o4)) < 5e-3
    power_sum = float(
        np.asarray(result["diagnostics"]["power_sum"], dtype=float)[center_idx]
    )
    assert power_sum < 1.2
