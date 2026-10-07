"""Compact, source-free CPML reflection measurements."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from beamz import (
    EPS_0,
    LIGHT_SPEED,
    MU_0,
    PEC,
    PML,
    Design,
    Material,
    Simulation,
    um,
)


def _packet_reflection_db(
    *,
    refractive_index: float,
    points_per_wavelength: int,
    polarization: str = "tm",
    timestep_scale: float = 1.0,
) -> tuple[float, tuple[int, int], int]:
    """Launch a +x eigenpacket and measure its time-separated CPML return."""
    vacuum_wavelength = 1.0 * um
    medium_wavelength = vacuum_wavelength / refractive_index
    resolution = medium_wavelength / points_per_wavelength
    dt = timestep_scale * 0.95 * resolution / (LIGHT_SPEED * np.sqrt(2.0))
    time = np.arange(
        0.0,
        12.0 * vacuum_wavelength / LIGHT_SPEED,
        dt,
    )
    impedance = np.sqrt(MU_0 / EPS_0) / refractive_index
    wave_speed = LIGHT_SPEED / refractive_index
    width = 12.0 * medium_wavelength
    height = 26.0 * medium_wavelength

    simulation = Simulation(
        design=Design(
            width=width,
            height=height,
            material=Material(permittivity=refractive_index**2),
        ),
        sources=[],
        polarization=polarization,
        boundaries=[
            PML(
                edges=["left", "right"],
                thickness=1.5 * medium_wavelength,
                formulation="cpml",
            ),
            PEC(edges=["top", "bottom"]),
        ],
        time=time,
        resolution=resolution,
    )
    initial = simulation.initial_state()
    electric_name, magnetic_name, magnetic_sign = (
        ("ez", "hy", -1) if polarization == "tm" else ("ey", "hz", 1)
    )
    electric = getattr(initial, electric_name)
    magnetic = getattr(initial, magnetic_name)
    electric_x = np.arange(electric.shape[1]) * resolution
    magnetic_x = (np.arange(magnetic.shape[1]) + 0.5) * resolution
    center_x = 3.0 * medium_wavelength
    sigma = 0.8 * medium_wavelength
    wavenumber = 2.0 * np.pi / medium_wavelength

    def packet(x):
        envelope = np.exp(-0.5 * ((x - center_x) / sigma) ** 2)
        return envelope * np.cos(wavenumber * (x - center_x))

    initial = initial._replace(
        **{
            electric_name: jnp.asarray(
                np.broadcast_to(packet(electric_x), electric.shape),
                dtype=electric.dtype,
            ),
            magnetic_name: jnp.asarray(
                np.broadcast_to(
                    magnetic_sign
                    * packet(magnetic_x + 0.5 * wave_speed * dt)
                    / impedance,
                    magnetic.shape,
                ),
                dtype=magnetic.dtype,
            ),
        }
    )

    initial_e = np.asarray(getattr(initial, electric_name))
    initial_e = 0.5 * (initial_e[:, 1:] + initial_e[:, :-1])
    incident = 0.5 * (
        initial_e
        + magnetic_sign * impedance * np.asarray(getattr(initial, magnetic_name))
    )
    final = simulation.advance(state=initial, progress=False).state
    final_e = np.asarray(getattr(final, electric_name))
    final_e = 0.5 * (final_e[:, 1:] + final_e[:, :-1])
    reflected = 0.5 * (
        final_e - magnetic_sign * impedance * np.asarray(getattr(final, magnetic_name))
    )

    x = (np.arange(reflected.shape[1]) + 0.5) * resolution
    y = np.arange(reflected.shape[0]) * resolution
    central_y = (y > 11.0 * medium_wavelength) & (y < 15.0 * medium_wavelength)
    interior_x = (x > 1.7 * medium_wavelength) & (x < 10.3 * medium_wavelength)
    incident_energy = np.sum(incident[np.ix_(central_y, interior_x)] ** 2)
    reflected_energy = np.sum(reflected[np.ix_(central_y, interior_x)] ** 2)
    reflection_db = 10.0 * np.log10(reflected_energy / incident_energy)
    return reflection_db, tuple(int(value) for value in electric.shape), len(time)


@pytest.mark.simulation
@pytest.mark.parametrize("polarization", ["te", "tm"])
@pytest.mark.parametrize("timestep_scale", [1.0, 0.5])
@pytest.mark.parametrize(
    ("refractive_index", "points_per_wavelength"),
    [(1.0, 10), (1.0, 20), (1.5, 15)],
)
def test_cpml_normal_incidence_packet_reflection_is_below_minus_40_db(
    refractive_index,
    points_per_wavelength,
    validation_metrics,
    polarization,
    timestep_scale,
):
    """Record CPML return loss across resolution and background index."""
    reflection_db, grid_shape, steps = _packet_reflection_db(
        refractive_index=refractive_index,
        points_per_wavelength=points_per_wavelength,
        polarization=polarization,
        timestep_scale=timestep_scale,
    )
    validation_metrics.check_upper(
        "CPML reflected packet power",
        measured=reflection_db,
        upper_bound=-40.0,
        unit="dB",
        resolution=(
            f"{points_per_wavelength} ppw in n={refractive_index}, 1.5 wavelength CPML"
        ),
        metadata={
            "refractive_index": refractive_index,
            "points_per_wavelength": points_per_wavelength,
            "grid_shape": list(grid_shape),
            "steps": steps,
            "incidence": "normal",
            "polarization": polarization.upper(),
            "timestep_scale": timestep_scale,
            "measurement": "time-separated characteristic packet energy",
        },
    )


@pytest.mark.parametrize("polarization", ["te", "tm"])
@pytest.mark.parametrize(
    "angle", [0.0, 75.0, None], ids=["normal", "oblique", "broadband"]
)
def test_cpml_free_space_packets_decay_without_late_growth(polarization, angle):
    """Free-space initial data radiate away, including low-frequency pulse content.

    A single out-of-plane field is divergence-free and launches two angular
    partners. The unmodulated Gaussian derivative has zero mean and a broad
    low-frequency spectrum. This measures residual energy, not specular return.
    """
    dx = um / 15
    dt = 0.5 * dx / LIGHT_SPEED
    time = np.arange(0.0, 40 * um / LIGHT_SPEED, dt)
    simulation = Simulation(
        design=Design(width=12 * um, height=12 * um),
        polarization=polarization,
        sources=[],
        boundaries=[PML(thickness=2 * um, formulation="cpml")],
        resolution=dx,
        time=time,
    )
    initial = simulation.initial_state()
    component = "hz" if polarization == "te" else "ez"
    shape = getattr(initial, component).shape
    offset = 0.5 if polarization == "te" else 0.0
    y, x = (np.indices(shape) + offset) * dx - 6 * um
    direction = (
        x
        if angle is None
        else x * np.cos(np.deg2rad(angle)) + y * np.sin(np.deg2rad(angle))
    )
    pulse = np.exp(-(x**2 + y**2) / (2 * (0.7 * um) ** 2))
    pulse *= (
        direction / (0.7 * um) if angle is None else np.cos(2 * np.pi * direction / um)
    )
    initial = initial._replace(**{component: jnp.asarray(pulse)})

    def energy(state):
        return sum(
            factor * float(np.sum(np.asarray(getattr(state, name)) ** 2))
            for name, factor in (
                ("ex", EPS_0),
                ("ey", EPS_0),
                ("ez", EPS_0),
                ("hx", MU_0),
                ("hy", MU_0),
                ("hz", MU_0),
            )
        )

    incident = energy(initial)
    middle = simulation.advance(
        state=initial, num_steps=len(time) // 2, progress=False
    ).state
    final = simulation.advance(
        state=middle, num_steps=len(time) - len(time) // 2, progress=False
    ).state
    middle_energy, final_energy = energy(middle), energy(final)
    assert np.isfinite(final_energy)
    assert middle_energy / incident < 1e-5
    assert final_energy / incident < 1e-6
    assert final_energy < middle_energy
