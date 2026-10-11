"""Independent oblique thin-film optics and calibrated FDTD fixtures."""

import numpy as np

import beamz as bz


def oblique_slab_amplitudes(epsilon, frequency, thickness, angle, polarization):
    """Tangential E amplitudes for a slab embedded in vacuum, exp(-i omega t)."""
    k0 = 2 * np.pi * frequency / bz.LIGHT_SPEED
    kz0 = np.cos(angle)
    kz1 = np.sqrt(complex(epsilon) - np.sin(angle) ** 2)
    q0 = kz0 if polarization == "s" else 1 / kz0
    q1 = kz1 if polarization == "s" else epsilon / kz1
    r01 = (q0 - q1) / (q0 + q1)
    phase = np.exp(1j * k0 * kz1 * thickness)
    denominator = 1 - r01**2 * phase**2
    r = r01 * (1 - phase**2) / denominator
    t = (1 - r01**2) * phase / denominator
    # Compare with an empty cell at the same observation plane.
    return r, t * np.exp(-1j * k0 * kz0 * thickness)


def run_oblique_slab(
    material,
    spacing,
    polarization="s",
    angle=25 * np.pi / 180,
    sign=1,
    graded=False,
    axis="z",
):
    frequency = 5e14
    thickness = 160e-9
    normal = "xyz".index(axis)
    tangential = [i for i in range(3) if i != normal]
    dimensions = np.full(3, 160e-9)
    dimensions[normal] = 3.2e-6
    size = tuple(dimensions)
    source_size = dimensions.copy()
    source_size[normal] = 0
    aperture = tuple(source_size)

    def position(distance):
        coordinates = np.zeros(3)
        coordinates[normal] = distance
        return tuple(coordinates)

    kx = 2 * np.pi * frequency / bz.LIGHT_SPEED * np.sin(angle)
    wavevector = np.zeros(3)
    wavevector[tangential[0]] = kx
    vector = tuple(wavevector)
    design = bz.Design(background=bz.Material(1))
    if material is not None:
        design += bz.Box(
            center=(0, 0, 0),
            size=tuple(thickness if i == normal else size[i] for i in range(3)),
            material=material,
        )
    grid_spec = bz.GridSpec.uniform(spacing, courant=0.7)
    if graded:
        grid_spec = bz.GridSpec.auto(
            wavelength=600e-9,
            courant=0.7,
            dl_max=40e-9,
            overrides=(
                bz.MeshOverride(
                    center=(0, 0, 0), size=(*size[:2], 0.6e-6), dl=(None, None, spacing)
                ),
            ),
        )
    sim = bz.Simulation(
        size=size,
        design=design,
        grid_spec=grid_spec,
        run_time=240e-15,
        sources=[
            bz.PlaneWaveSource(
                center=position(-sign * 0.9e-6),
                size=aperture,
                source_time=bz.GaussianPulse(frequency, 0.05 * frequency),
                direction=("+" if sign > 0 else "-") + axis,
                transverse_wavevector=vector,
                pol_angle=0 if polarization == "s" else np.pi / 2,
            )
        ],
        boundaries=[
            bz.Bloch(axes=tuple("xyz"[i] for i in tangential), wavevector=vector),
            bz.PML(
                edges={
                    "x": ("left", "right"),
                    "y": ("bottom", "top"),
                    "z": ("front", "back"),
                }[axis],
                thickness=0.4e-6,
                formulation="cpml",
            ),
        ],
        monitors=[
            bz.FluxMonitor(
                center=position(z), size=aperture, freqs=[frequency], name=name
            )
            for name, z in [
                ("back", -sign * 1.05e-6),
                ("input", -sign * 0.5e-6),
                ("output", sign * 0.5e-6),
            ]
        ],
    )
    return sim.run(backend="jax", progress=False).renormalize(None)


def calibrated_transmission(device, reference, polarization):
    component = "Ey" if polarization == "s" else "Ex"
    incident = reference["output"].dft_fields[component].reshape(-1)
    transmitted = device["output"].dft_fields[component].reshape(-1)
    return np.vdot(incident, transmitted) / np.vdot(incident, incident)
