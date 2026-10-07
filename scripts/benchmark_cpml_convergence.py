"""CPML convergence control: a 2D TE annulus and its outgoing-wave pole.

Run BeamZ and optional Meep separately with identical geometry and time settings.
Lengths are in micrometres, time in micrometres/c, frequency in c/micrometre.
Adapted from the ringdown experiment accompanying beamzorg/beamz#316.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, root
from scipy.special import h1vp, hankel1, jv, jvp, yv, yvp


def disk_pole(epsilon=12.1 / 2.1, radius=0.4, order=5, inner_radius=0.0):
    """Outgoing TE pole of a disk or concentric dielectric annulus."""
    if not 0 <= inner_radius < radius or order < 1:
        raise ValueError("Require 0 <= inner_radius < radius and positive order")
    index = np.sqrt(epsilon)

    def residual(value):
        k = complex(*value)
        coefficients = np.array([1.0, 0.0], complex)
        if inner_radius:
            a = k * inner_radius
            coefficients = np.linalg.solve(
                [
                    [jv(order, index * a), yv(order, index * a)],
                    [jvp(order, index * a) / index, yvp(order, index * a) / index],
                ],
                [jv(order, a), jvp(order, a)],
            )
        b = k * index * radius
        field = coefficients @ [jv(order, b), yv(order, b)]
        flux = coefficients @ [jvp(order, b), yvp(order, b)] / index
        d = h1vp(order, k * radius) * field - hankel1(order, k * radius) * flux
        return [d.real, d.imag]

    solution = root(residual, [(order + 1.5) / (index * radius), -0.03], tol=1e-11)
    if not solution.success or np.linalg.norm(residual(solution.x)) > 1e-9:
        raise RuntimeError("Mie pole solve failed")
    k = complex(*solution.x)
    return dict(
        frequency=k.real / (2 * np.pi),
        decay=k.imag / (2 * np.pi),
        quality_factor=-k.real / (2 * k.imag),
        azimuthal_order=order,
    )


def waveform(t, frequency):
    return np.exp(-(((t - 6) / 1.5) ** 2)) * np.cos(2 * np.pi * frequency * (t - 6))


def spatial_drive(x, y, order=5, phase=0.0):
    """Select an angular sector without assuming a radial eigenfunction."""
    return np.exp(-(((np.hypot(x, y) - 0.3) / 0.04) ** 2)) * np.cos(
        order * np.arctan2(y, x) + phase
    )


def outer_vertices(args):
    """Common polygon for BeamZ and the explicitly jagged Meep control."""
    angle = np.arange(args.vertices) * 2 * np.pi / args.vertices
    radius = 0.4 + getattr(args, "corrugation", 0.0) * np.cos(12 * angle)
    return radius[:, None] * np.column_stack((np.cos(angle), np.sin(angle)))


def fit_single_mode(values, cadence, window=(20, 60), frequency_band=(1.25, 1.48)):
    """Independent variable-projection damped-sinusoid fit for an isolated mode.

    Fit residual is retained: a multimode trace must not silently be interpreted
    as a reliable single resonance. This requires no external harmonic inverter.
    """
    times = np.arange(len(values)) * cadence
    mask = (times >= window[0]) & (times <= window[1])
    times = times[mask] - window[0]
    samples = np.asarray(values)[mask]
    samples = samples / np.max(abs(samples))
    frequencies = np.fft.rfftfreq(len(samples), cadence)
    spectrum = abs(np.fft.rfft(samples))
    lower, upper = frequency_band
    spectrum[(frequencies < lower) | (frequencies > upper)] = 0
    initial = frequencies[np.argmax(spectrum)]

    def residual(parameters):
        frequency, decay = parameters
        phase = 2 * np.pi * frequency * times
        basis = np.column_stack((np.cos(phase), np.sin(phase))) * np.exp(
            -decay * times[:, None]
        )
        amplitude = np.linalg.lstsq(basis, samples, rcond=None)[0]
        return basis @ amplitude - samples

    fit = least_squares(
        residual,
        [initial, 0.06],
        bounds=([lower, 0.001], [upper, 0.3]),
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    if not fit.success:
        raise RuntimeError("Independent sinusoid fit failed")
    frequency, decay = fit.x
    return dict(
        frequency=float(frequency),
        decay_rate=float(decay),
        quality_factor=float(np.pi * frequency / decay),
        relative_residual=float(np.linalg.norm(fit.fun) / np.linalg.norm(samples)),
    )


def beamz_run(args, pole):
    import beamz as bz
    from beamz.design import MaterialGrid, raster

    length = 2.4 + 2 * args.padding
    center = (
        np.full(2, length / 2)
        + np.array([args.shift, 0.37 * args.shift]) / args.resolution
    )
    angle = np.arange(args.vertices) * 2 * np.pi / args.vertices
    outer = center + outer_vertices(args)
    inner = getattr(args, "inner_radius", 0.0)
    holes = (
        [(center + inner * np.column_stack((np.cos(angle), np.sin(angle)))) * 1e-6]
        if inner
        else []
    )
    edges = np.linspace(0, length, round(length * args.resolution) + 1) * 1e-6
    result = raster.rasterize(
        raster.Scene(
            (raster.Material(), raster.Material(12.1 / 2.1)),
            (
                raster.Object(
                    raster.ExtrudedPolygon(
                        raster.Polygon(outer * 1e-6, holes), -1e-6, 1e-6
                    ),
                    1,
                ),
            ),
        ),
        raster.Grid(edges, edges, [0, 1e-6]),
        options=raster.RasterOptions(
            smoothing=args.smoothing,
            quality=args.quality,
            components="two_dimensional_te",
        ),
    )
    material = MaterialGrid.from_raster_result(result, dimensions=2, polarization="te")
    dt = args.courant / args.resolution
    time = np.arange(0, args.until, dt)
    pulse = waveform(time, pole["frequency"])
    pulse[time >= 12] = 0
    coordinates = (edges[1:] + edges[:-1]) / 2
    x, y = np.meshgrid(coordinates * 1e6 - center[0], coordinates * 1e6 - center[1])
    source = bz.CustomSource(
        component="Hz",
        timing="h",
        index=(slice(None), slice(None)),
        coeff=spatial_drive(
            x, y, pole["azimuthal_order"], getattr(args, "angular_phase", 0.0)
        ),
        waveform=pulse,
        target_shape=(len(edges) - 1, len(edges) - 1),
    )
    simulation = bz.Simulation(
        material_grid=material,
        polarization="te",
        time=time * 1e-6 / bz.LIGHT_SPEED,
        sources=[source],
        boundaries=[
            bz.PML(
                thickness=args.pml * 1e-6, formulation="cpml", alpha_max=args.alpha_max
            )
        ],
        monitors=[
            bz.FieldRecorder(
                ("Hz",),
                interval=2,
                name="probe",
                center=tuple((center + [0.21, 0.07]) * 1e-6),
                size=(0.0, 0.0),
            )
        ],
        normalize_source=None,
    )
    run = simulation.run(backend=args.backend, progress=False)
    probe = run.monitors["probe"]
    values = (
        np.asarray(probe.fields["Hz"]).reshape(len(probe.field_times), -1).mean(axis=1)
    )
    return values, dict(
        raster=dict(result.diagnostics),
        dt_um_over_c=dt,
        grid_shape=list(simulation.grid.shape),
        version=bz.__version__,
        resolved_parameters=[
            dict(p) for p in simulation.pml_data["resolved_parameters"]
        ],
    )


def meep_run(args, pole):
    import meep as mp

    mp.verbosity(0)
    center = np.array([args.shift, 0.37 * args.shift]) / args.resolution
    geometry = [
        mp.Cylinder(
            radius=0.4,
            center=mp.Vector3(*center),
            material=mp.Medium(epsilon=12.1 / 2.1),
        )
    ]
    if getattr(args, "inner_radius", 0.0):
        geometry.append(
            mp.Cylinder(
                radius=args.inner_radius,
                center=mp.Vector3(*center),
                material=mp.Medium(epsilon=1),
            )
        )
    if getattr(args, "corrugation", 0.0):
        geometry = [
            mp.Prism(
                [mp.Vector3(*(p + center)) for p in outer_vertices(args)],
                height=mp.inf,
                material=mp.Medium(epsilon=12.1 / 2.1),
            )
        ]
        if args.inner_radius:
            angle = np.arange(args.vertices) * 2 * np.pi / args.vertices
            hole = args.inner_radius * np.column_stack((np.cos(angle), np.sin(angle)))
            geometry.append(
                mp.Prism(
                    [mp.Vector3(*(p + center)) for p in hole],
                    height=mp.inf,
                    material=mp.Medium(epsilon=1),
                )
            )
    sim = mp.Simulation(
        cell_size=mp.Vector3(2.4 + 2 * args.padding, 2.4 + 2 * args.padding),
        resolution=args.resolution,
        Courant=args.courant,
        geometry=geometry,
        sources=[
            mp.Source(
                mp.CustomSource(
                    lambda t: waveform(t, pole["frequency"]), start_time=0, end_time=12
                ),
                component=mp.Hz,
                center=mp.Vector3(*center),
                size=mp.Vector3(1.2, 1.2),
                amp_func=lambda p: spatial_drive(
                    p.x,
                    p.y,
                    pole["azimuthal_order"],
                    getattr(args, "angular_phase", 0.0),
                ),
            )
        ],
        boundary_layers=[mp.PML(args.pml)],
        eps_averaging=True,
    )
    values = []

    def record(sim):
        values.append(
            sim.get_field_point(mp.Hz, mp.Vector3(*(center + [0.21, 0.07]))).real
        )

    sim.run(mp.at_every(2 * args.courant / args.resolution, record), until=args.until)
    return np.array(values), {
        "version": mp.__version__,
        "dt_um_over_c": args.courant / args.resolution,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("beamz", "meep"), default="beamz")
    parser.add_argument("--resolution", type=int, default=80)
    parser.add_argument("--courant", type=float, default=0.5)
    parser.add_argument("--pml", type=float, default=0.4)
    parser.add_argument(
        "--padding",
        type=float,
        default=0.0,
        help="Increase each domain edge; use 0.2 with PML 0.6 to preserve its onset",
    )
    parser.add_argument(
        "--angular-phase",
        type=float,
        default=0.0,
        help="Use 0 or pi/2 for the two angular partners",
    )
    parser.add_argument(
        "--alpha-max",
        type=float,
        default=None,
        help="BeamZ alpha in S/m; omit for the default, use 0 for unshifted PML",
    )
    parser.add_argument("--until", type=float, default=75.0)
    parser.add_argument("--backend", default="jax")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.vertices = 1024
    args.order = 6
    args.inner_radius = 0.25
    args.shift = 0.0
    args.corrugation = 0.0
    args.smoothing = "farjadpour_full"
    args.quality = "balanced"
    if args.engine == "meep" and args.alpha_max is not None:
        parser.error("--alpha-max only applies to BeamZ")
    if args.until < 70:
        parser.error("--until must cover both source-free fit windows (at least 70)")
    args.output.mkdir(parents=True, exist_ok=False)
    pole = disk_pole(order=args.order, inner_radius=args.inner_radius)
    values, diagnostics = (beamz_run if args.engine == "beamz" else meep_run)(
        args, pole
    )
    np.savez_compressed(args.output / "trace.npz", hz=values)
    result = dict(
        settings={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        exact=pole,
        diagnostics=diagnostics,
        fits=[
            dict(
                window=window,
                **fit_single_mode(
                    values,
                    2 * args.courant / args.resolution,
                    window,
                    frequency_band=(pole["frequency"] * 0.9, pole["frequency"] * 1.1),
                ),
            )
            for window in ((20, 60), (30, 70))
        ],
    )
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
