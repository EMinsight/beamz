"""Reproducible tensor straight-guide launch, flux, and modal checks (SI units)."""

import argparse
import json
import time
from dataclasses import replace

import numpy as np

import beamz as bz
from beamz.analysis.sparameters import _extract_port_waves_dft
from beamz.design.discretization import MaterialGrid
from beamz.design.raster import (
    Box,
    ExtrudedPolygon,
    Grid,
    Material,
    Object,
    Polygon,
    RasterOptions,
    Scene,
    rasterize,
)


def run(
    axis="x",
    direction="+",
    spacing_nm=100,
    *,
    cross=True,
    smoothing="farjadpour_full",
    smoothed=False,
    rectilinear=False,
    graded_normal=False,
    backend="jax",
    source_power=1.0,
    aperture_um=2.2,
):
    axes = {"x": "xyz", "y": "yxz", "z": "zxy"}[axis]
    order = [axes.index(a) for a in "xyz"]

    def point(values):
        return tuple(np.asarray(values)[order] * 1e-6)

    epsilon = np.array([[6.0, 0.2, 0.1], [0.2, 5.0, 0.3], [0.1, 0.3, 4.0]])[
        np.ix_(order, order)
    ]
    if not cross:
        epsilon = np.diag(np.diag(epsilon))
    scene = Scene(
        (Material(), Material(epsilon_r=tuple(map(tuple, epsilon)))),
        (Object(Box(point((0, 1.5, 1.5)), point((6.4, 2.1, 2.1))), 1),),
    )
    if smoothed:
        if axis != "z":
            raise ValueError("The oblique-interface benchmark is z-extruded")
        angle = np.pi / 6
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        vertices = (
            np.array([[-0.3, -0.2], [0.3, -0.2], [0.3, 0.2], [-0.3, 0.2]]) @ rotation.T
            + 1.8
        )
        polygon = Polygon(tuple(map(tuple, vertices * 1e-6)))
        scene = Scene(
            (Material(), Material(6.0)),
            (Object(ExtrudedPolygon(polygon, 0.0, 6.4e-6), 1),),
        )
    extent = point((6.4, 3.6, 3.6))
    shape = tuple(round(v / (spacing_nm * 1e-9)) for v in extent)
    grid = Grid.uniform((0, 0, 0), extent, shape)
    if rectilinear:
        edges = []
        for a, length, count in zip("xyz", extent, shape, strict=True):
            u = np.linspace(0, 1, count + 1)
            values = length * u
            if a != axis:
                values = length * (u + 0.08 * np.sin(2 * np.pi * u))
            elif graded_normal:
                # Local port stencils are uniform; a smooth grid perturbation
                # in the device interior exercises longitudinal metric updates.
                in_bump = (values > 3.0e-6) & (values < 3.4e-6)
                values[in_bump] += (
                    0.02e-6 * np.sin(np.pi * (values[in_bump] - 3.0e-6) / 0.4e-6) ** 2
                )
            edges.append(values)
        grid = Grid(*edges)
    material = MaterialGrid.from_raster_result(
        rasterize(
            scene,
            grid,
            options=RasterOptions(smoothing=smoothing),
        )
    )
    f = bz.LIGHT_SPEED / 1.55e-6
    sign = 1 if direction == "+" else -1
    source_x = 2.0 if sign == 1 else 4.4
    spec = bz.ModeSpec(num_modes=2, target_neff=2.0, polarization=None)
    size = point((0, aperture_um, aperture_um))
    source = bz.ModeSource(
        center=point((source_x, 1.8, 1.8)),
        size=size,
        direction=direction,
        source_time=bz.GaussianPulse(freq0=f, fwidth=0.1 * f, offset=5 / (2 * np.pi)),
        mode_spec=spec,
        power=source_power,
    )
    ports = []
    for name, distance in [
        ("back", -0.6),
        ("near", 0.6),
        ("far", 2.0 if rectilinear else 1.4),
        ("shifted", 2.023 if rectilinear else 1.423),
    ]:
        ports.append(
            bz.Port(
                center=point((source_x + sign * distance, 1.8, 1.8)),
                size=size,
                direction="+",
                name=name,
                mode_spec=spec,
            )
        )
    normal_edges = {
        "x": ["left", "right"],
        "y": ["bottom", "top"],
        "z": ["front", "back"],
    }[axis]
    other_edges = [
        e
        for e in ["left", "right", "bottom", "top", "front", "back"]
        if e not in normal_edges
    ]
    sim = bz.Simulation(
        material_grid=material,
        sources=[source],
        monitors=[p.to_monitor([f]) for p in ports]
        + (
            [
                bz.FluxMonitor(
                    center=p.center, size=p.size, freqs=[f], name=p.name + "_flux"
                )
                for p in ports
                if p.name in {"near", "far"}
            ]
            if rectilinear
            else []
        ),
        boundaries=[
            bz.PML(edges=normal_edges, thickness=1.2e-6, formulation="cpml"),
            bz.PML(edges=other_edges, thickness=0.6e-6, formulation="cpml"),
        ],
        run_time=200e-15,
        setup_device="cpu",
    )
    start = time.time()
    preview = source.solve_modes(sim, freqs=[f]) if rectilinear else None
    result = sim.run(backend=backend, progress=False)
    secondary = [
        replace(
            p,
            name=p.name + "_mode1",
            monitor_name=p.name,
            mode_spec=replace(spec, mode_index=1),
        )
        for p in ports
        if p.name in {"near", "far"}
    ]
    waves = _extract_port_waves_dft(
        result, ports=[*ports, *secondary], frequencies=[f], return_power=True
    )
    positive, negative = ("a_plus", "a_minus") if axis == "z" else ("a_minus", "a_plus")
    forward, backward = (positive, negative) if sign == 1 else (negative, positive)
    metrics = {}
    for p in ports:
        w = waves[p.name]
        metrics[p.name] = {
            "forward_amplitude": [float(w[forward][0].real), float(w[forward][0].imag)],
            "forward_power": float(abs(w[forward][0]) ** 2),
            "backward_power": float(abs(w[backward][0]) ** 2),
            "flux": float(result[p.name].flux[0]),
            "sampled_modal_flux": float(w["projected_signed_power"][0]),
            "reconstruction_residual": float(w["projection_residual_balanced"][0]),
            "neff": float(w["mode_neff"][0]),
        }
        if rectilinear and p.name in {"near", "far"}:
            metrics[p.name]["independent_flux"] = float(
                result[p.name + "_flux"].flux[0]
            )
    for p in secondary:
        metrics[p.monitor_name]["other_mode_power"] = float(
            abs(waves[p.name][forward][0]) ** 2
        )
    report = {
        "axis": axis,
        "backend": backend,
        "direction": direction,
        "spacing_nm": spacing_nm,
        "smoothed_isotropic_interface": smoothed,
        "rectilinear": rectilinear,
        "graded_normal": graded_normal,
        "preview_neffs": [] if preview is None else np.real(preview.neffs[0]).tolist(),
        "seconds": time.time() - start,
        "ports": metrics,
    }
    return report, result, ports, waves


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--axis", default="x", choices=["x", "y", "z"])
    p.add_argument("--direction", default="+", choices=["+", "-"])
    p.add_argument("--spacing-nm", type=float, default=100)
    p.add_argument("--output", required=True)
    p.add_argument("--smoothed", action="store_true")
    p.add_argument("--rectilinear", action="store_true")
    p.add_argument("--graded-normal", action="store_true")
    p.add_argument("--backend", choices=["jax", "cuda_streamed"], default="jax")
    args = p.parse_args()
    if args.graded_normal and not args.rectilinear:
        p.error("--graded-normal requires --rectilinear")
    report, *_ = run(
        args.axis,
        args.direction,
        args.spacing_nm,
        smoothed=args.smoothed,
        rectilinear=args.rectilinear,
        graded_normal=args.graded_normal,
        backend=args.backend,
    )
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
