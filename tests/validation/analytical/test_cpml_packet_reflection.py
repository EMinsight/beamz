"""Compact, source-free CPML reflection measurements."""

from __future__ import annotations

from dataclasses import replace

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
    FieldRecorder,
    Material,
    Periodic,
    Simulation,
    um,
)
from beamz.design import MaterialGrid
from tests.validation.analytical.bloch_slab_reference import run_oblique_slab


def _packet_reflection_db(
    *,
    refractive_index: float,
    points_per_wavelength: int,
    polarization: str = "tm",
    default_thickness: bool = False,
) -> tuple[float, tuple[int, int], int]:
    """Launch a +x eigenpacket and measure its time-separated CPML return."""
    vacuum_wavelength = 1.0 * um
    medium_wavelength = vacuum_wavelength / refractive_index
    resolution = medium_wavelength / points_per_wavelength
    dt = 0.95 * resolution / (LIGHT_SPEED * np.sqrt(2.0))
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
        boundaries=[
            PML(
                edges=["left", "right"],
                thickness=None if default_thickness else 1.5 * medium_wavelength,
                formulation="cpml",
            ),
            PEC(edges=["top", "bottom"]),
        ],
        time=time,
        resolution=resolution,
        polarization=polarization,
    )
    initial = simulation.initial_state()
    e_name, h_name, h_sign = (
        ("ez", "hy", -1) if polarization == "tm" else ("ey", "hz", 1)
    )
    e, h = getattr(initial, e_name), getattr(initial, h_name)
    electric_x = np.arange(e.shape[1]) * resolution
    magnetic_x = (np.arange(h.shape[1]) + 0.5) * resolution
    center_x = 3.0 * medium_wavelength
    sigma = 0.8 * medium_wavelength
    wavenumber = 2.0 * np.pi / medium_wavelength

    def packet(x):
        envelope = np.exp(-0.5 * ((x - center_x) / sigma) ** 2)
        return envelope * np.cos(wavenumber * (x - center_x))

    initial = initial._replace(
        **{
            e_name: jnp.asarray(
                np.broadcast_to(packet(electric_x), e.shape),
                dtype=e.dtype,
            ),
            h_name: jnp.asarray(
                np.broadcast_to(
                    h_sign * packet(magnetic_x + 0.5 * wave_speed * dt) / impedance,
                    h.shape,
                ),
                dtype=h.dtype,
            ),
        }
    )

    def characteristic(state, sign):
        e, h = np.asarray(getattr(state, e_name)), np.asarray(getattr(state, h_name))
        return 0.5 * (0.5 * (e[:, 1:] + e[:, :-1]) + sign * h_sign * impedance * h)

    incident = characteristic(initial, 1)
    final = simulation.advance(state=initial, progress=False).state
    reflected = characteristic(final, -1)

    x = (np.arange(reflected.shape[1]) + 0.5) * resolution
    y = np.arange(reflected.shape[0]) * resolution
    central_y = (y > 11.0 * medium_wavelength) & (y < 15.0 * medium_wavelength)
    interior_x = (x > 1.7 * medium_wavelength) & (x < 10.3 * medium_wavelength)
    incident_energy = np.sum(incident[np.ix_(central_y, interior_x)] ** 2)
    reflected_energy = np.sum(reflected[np.ix_(central_y, interior_x)] ** 2)
    assert np.isfinite(reflected_energy) and incident_energy > 0
    reflection_db = 10.0 * np.log10(reflected_energy / incident_energy)
    return reflection_db, tuple(int(value) for value in e.shape), len(time)


@pytest.mark.simulation
@pytest.mark.parametrize("polarization", ["tm", "te"])
@pytest.mark.parametrize("default_thickness", [False, True], ids=["1.5wl", "12cells"])
@pytest.mark.parametrize(
    ("refractive_index", "points_per_wavelength"),
    [(1.0, 10), (1.0, 20), (1.5, 15)],
)
def test_cpml_normal_incidence_packet_reflection_is_below_minus_40_db(
    refractive_index,
    points_per_wavelength,
    polarization,
    default_thickness,
    validation_metrics,
):
    """Record CPML return loss across resolution and background index."""
    reflection_db, grid_shape, steps = _packet_reflection_db(
        refractive_index=refractive_index,
        points_per_wavelength=points_per_wavelength,
        polarization=polarization,
        default_thickness=default_thickness,
    )

    validation_metrics.check_upper(
        "CPML reflected packet power",
        measured=reflection_db,
        upper_bound=-40.0,
        unit="dB",
        resolution=(
            f"{points_per_wavelength} ppw in n={refractive_index}, "
            f"{'12-cell' if default_thickness else '1.5-wavelength'} CPML"
        ),
        metadata={
            "refractive_index": refractive_index,
            "points_per_wavelength": points_per_wavelength,
            "grid_shape": list(grid_shape),
            "steps": steps,
            "incidence": "normal",
            "polarization": polarization,
            "measurement": "time-separated characteristic packet energy",
        },
    )


@pytest.mark.parametrize("angle", [45, 75])
@pytest.mark.parametrize("polarization", ["s", "p"])
def test_oblique_cpml_reflection_converges_with_thickness(
    angle, polarization, validation_metrics
):
    """Subtract the same source in a thicker layer to isolate boundary returns."""
    runs = [
        run_oblique_slab(
            None,
            20e-9,
            polarization,
            angle=np.deg2rad(angle),
            pml_thickness=thickness,
        )
        for thickness in (None, 24 * 20e-9, 32 * 20e-9)
    ]
    reference = runs[-1]["input"]
    assert reference.flux[0] > 0
    reflected_power = []
    for run in runs[:-1]:
        reflected = replace(
            run["input"],
            dft_fields={
                c: run["input"].dft_fields[c] - reference.dft_fields[c]
                for c in reference.dft_fields
            },
        )
        reflected_power.append(float(abs(reflected.flux[0] / reference.flux[0])))
    # A 12-cell layer is not a universal -40 dB guarantee at grazing incidence.
    validation_metrics.check_upper(
        "12-cell boundary-return power relative to 32 cells",
        reflected_power[0],
        2e-4,
        metadata={"angle_degrees": angle, "polarization": polarization},
    )
    assert reflected_power[1] < max(reflected_power[0] / 2, 1e-8)


def test_3d_cpml_absorbs_divergence_free_pulse_without_late_growth(validation_metrics):
    """A localized curl excites faces, edges and corners without a static charge."""
    cells, dx = 48, 1e-7
    dt = 0.8 * dx / (LIGHT_SPEED * np.sqrt(3))
    sim = Simulation(
        design=Design(width=cells * dx, height=cells * dx, depth=cells * dx),
        sources=[],
        boundaries=[PML(formulation="cpml")],
        resolution=dx,
        time=np.arange(1024) * dt,
    )
    state = sim.initial_state()
    z, y, x = np.indices(state.ez.shape)
    potential = np.exp(-((x - 24) ** 2 + (y - 24) ** 2 + (z + 0.5 - 24) ** 2) / 18)
    potential *= np.cos(2 * np.pi * (z + 0.5 - 24) / 10)
    state = state._replace(
        hx=jnp.asarray(np.diff(potential, axis=1), dtype=jnp.float32),
        hy=jnp.asarray(-np.diff(potential, axis=2), dtype=jnp.float32),
    )

    def energy(state):
        return sum(
            (EPS_0 if name[0] == "e" else MU_0)
            * np.sum(np.asarray(getattr(state, name), dtype=np.float64) ** 2)
            for name in ("ex", "ey", "ez", "hx", "hy", "hz")
        )

    initial_energy = energy(state)
    assert initial_energy > 0
    for _ in range(8):
        state = sim.advance(
            state=state, num_steps=128, backend="jax", progress=False
        ).state
        validation_metrics.check_upper(
            "3D CPML residual energy",
            energy(state) / initial_energy,
            1e-5,
            metadata={"step": int(state.current_step), "cells_per_face": 12},
        )


def test_cavity_ringdown_converges_with_cpml_clearance_and_thickness(
    validation_metrics,
):
    """Compare the same Bragg cavity's waveform over 150 optical periods."""
    dx = 50e-9
    dt = 0.5 * dx / (LIGHT_SPEED * np.sqrt(2))
    mirror = np.tile(np.r_[np.full(2, 3.5**2), np.ones(7)], 3)
    traces = []
    for clearance, cells in ((7, 12), (28, 12), (28, 24)):
        padding = np.ones(clearance + cells)
        epsilon = np.r_[padding, mirror[::-1], np.ones(14), mirror, padding]
        nx = len(epsilon)
        epsilon = np.broadcast_to(epsilon, (4, nx)).astype(np.float32)
        sim = Simulation(
            material_grid=MaterialGrid(
                epsilon,
                np.float32(0),
                np.float32(1),
                dx,
                epsilon.shape,
                polarization="tm",
            ),
            time=np.arange(12000) * dt,
            sources=[],
            boundaries=[
                PML(edges=("left", "right"), thickness=cells * dx, formulation="cpml"),
                Periodic(axes="y"),
            ],
            monitors=[FieldRecorder(("Ez",), interval=4)],
        )
        state = sim.initial_state()
        pulse = np.exp(-(((np.arange(state.ez.shape[1]) - nx / 2) / 3) ** 2))
        state = state._replace(
            ez=jnp.asarray(np.broadcast_to(pulse, state.ez.shape), dtype=state.ez.dtype)
        )
        run = sim.advance(state=state, backend="jax", progress=False)
        traces.append(np.asarray(run.results["fields"].fields["Ez"])[:, 2, nx // 2])
    peak = np.max(abs(traces[-1]))
    assert peak > 0 and np.max(abs(traces[-1][-500:])) > 0.01 * peak
    for trace in traces[:-1]:
        validation_metrics.check_upper(
            "cavity ringdown sensitivity to CPML",
            np.max(abs(trace - traces[-1])) / peak,
            1e-3,
        )
