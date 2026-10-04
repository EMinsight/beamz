"""Calibrated unit-cell transmission, phase, reflection, and diffraction orders.

Run from the repository with:
    uv run python examples/meta_atom_bloch.py --output /tmp/meta-atom.json

Each frequency has its own Bloch vector, keeping the incidence angle fixed.
The default dielectric square pillar is illustrative, not a fabricated design.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

import beamz as bz
from beamz.analysis import monitor_dft_component


def diffraction_powers(monitor, grid, wavevector, frequency):
    """Resolve propagating orders of a full-cell z-plane in vacuum.

    The Fourier coefficients of colocated E/H give the signed normal power
    of each order. Evanescent orders carry no far-field transmitted power.
    Monitors must be in homogeneous vacuum, sufficiently far from the device.
    """
    x, y = np.asarray(grid.centers("x")), np.asarray(grid.centers("y"))
    xx, yy = np.meshgrid(x, y, indexing="xy")
    period_x = np.ptp(grid.axis_edges("x"))
    period_y = np.ptp(grid.axis_edges("y"))
    weights = np.asarray(monitor.integration_weights)
    if not weights.size:
        weights = np.full(xx.size, period_x * period_y / xx.size)
    area = weights.sum()
    k0 = 2 * np.pi * frequency / bz.LIGHT_SPEED
    gx, gy = 2 * np.pi / period_x, 2 * np.pi / period_y
    mx = int(np.ceil((k0 + abs(wavevector[0])) / gx))
    my = int(np.ceil((k0 + abs(wavevector[1])) / gy))
    if 2 * mx + 1 > x.size or 2 * my + 1 > y.size:
        raise ValueError(
            "Mesh is too coarse to resolve all propagating diffraction orders."
        )
    fields = {c: monitor_dft_component(monitor, c)[0] for c in ("Ex", "Ey", "Hx", "Hy")}
    if any(value.size != xx.size for value in fields.values()):
        raise ValueError(
            "Diffraction analysis requires a monitor spanning the full cell."
        )
    orders = []
    for m in range(-mx, mx + 1):
        for n in range(-my, my + 1):
            kx, ky = wavevector[0] + m * gx, wavevector[1] + n * gy
            if kx * kx + ky * ky >= k0 * k0 * (1 - 1e-12):
                continue
            basis = np.exp(-1j * (kx * xx + ky * yy)).reshape(-1)
            coefficients = {
                c: np.sum(weights * value * basis) / area for c, value in fields.items()
            }
            power = (
                0.5
                * area
                * np.real(
                    coefficients["Ex"] * np.conj(coefficients["Hy"])
                    - coefficients["Ey"] * np.conj(coefficients["Hx"])
                )
            )
            orders.append({"m": m, "n": n, "power_w": float(power)})
    return orders


def simulate_cell(frequency, angle, spacing, *, device, period=320e-9):
    size = (period, period, 3.2e-6)
    aperture = (period, period, 0)
    vector = (2 * np.pi * frequency / bz.LIGHT_SPEED * np.sin(angle), 0, 0)
    design = bz.Design(background=bz.Material(1))
    if device:
        design += bz.Box(
            center=(0, 0, 0),
            size=(160e-9, 160e-9, 200e-9),
            material=bz.Material(2.2**2),
        )
    simulation = bz.Simulation(
        design=design,
        size=size,
        grid_spec=bz.GridSpec.uniform(spacing, courant=0.7),
        run_time=240e-15,
        sources=[
            bz.PlaneWaveSource(
                center=(0, 0, -0.9e-6),
                size=aperture,
                source_time=bz.GaussianPulse(frequency, 0.05 * frequency),
                direction="+z",
                transverse_wavevector=vector,
            )
        ],
        boundaries=[
            bz.Bloch(axes=("x", "y"), wavevector=vector),
            bz.PML(edges=("front", "back"), thickness=0.4e-6, formulation="cpml"),
        ],
        monitors=[
            bz.FluxMonitor(
                center=(0, 0, z), size=aperture, freqs=[frequency], name=name
            )
            for name, z in [("input", -0.5e-6), ("output", 0.5e-6)]
        ],
    )
    result = simulation.run(backend="jax", progress=False).renormalize(None)
    return result, simulation.compile(backend="jax").grid.geometry, vector


def characterize(frequency, angle, spacing, *, period=320e-9):
    reference, grid, vector = simulate_cell(
        frequency, angle, spacing, device=False, period=period
    )
    device, _, _ = simulate_cell(frequency, angle, spacing, device=True, period=period)
    incident_power = float(reference["input"].flux[0])
    # Subtract complex incident E/H before computing reflected power.
    reflected = replace(
        device["input"],
        dft_fields={
            c: device["input"].dft_fields[c] - reference["input"].dft_fields[c]
            for c in device["input"].dft_fields
        },
    )
    component = "Ey" if angle != 0 else "Ex"
    incident = reference["output"].dft_fields[component].reshape(-1)
    transmitted = device["output"].dft_fields[component].reshape(-1)
    # This projection extracts the co-polarized zeroth order; total flux may
    # include other diffraction orders and cross polarization.
    amplitude = np.vdot(incident, transmitted) / np.vdot(incident, incident)
    T = float(device["output"].flux[0] / reference["output"].flux[0])
    R = float(-reflected.flux[0] / incident_power)
    transmitted_orders = diffraction_powers(device["output"], grid, vector, frequency)
    reflected_orders = diffraction_powers(reflected, grid, vector, frequency)
    for order in transmitted_orders:
        order["efficiency"] = order["power_w"] / incident_power
    for order in reflected_orders:
        order["efficiency"] = -order["power_w"] / incident_power
    return {
        "frequency_hz": frequency,
        "angle_deg": float(np.rad2deg(angle)),
        "spacing_m": spacing,
        "transmission": T,
        "reflection": R,
        "energy_residual": 1 - R - T,
        "zeroth_order_co_polarized_transmission": {
            "real": float(amplitude.real),
            "imag": float(amplitude.imag),
        },
        "transmission_phase_rad": float(np.angle(amplitude)),
        "transmitted_orders": transmitted_orders,
        "reflected_orders": reflected_orders,
        "order_sum_residual": float(
            T + R - sum(o["efficiency"] for o in transmitted_orders + reflected_orders)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequencies", type=float, nargs="+", default=[5e14])
    parser.add_argument("--angles", type=float, nargs="+", default=[0, 25])
    parser.add_argument("--spacings-nm", type=float, nargs="+", default=[40, 20])
    parser.add_argument("--period-nm", type=float, default=320)
    parser.add_argument("--output", type=Path, default=Path("meta-atom-bloch.json"))
    args = parser.parse_args()
    rows = []
    for frequency in args.frequencies:
        for degrees in args.angles:
            previous = None
            for spacing in sorted(args.spacings_nm, reverse=True):
                row = characterize(
                    frequency,
                    np.deg2rad(degrees),
                    spacing * 1e-9,
                    period=args.period_nm * 1e-9,
                )
                if previous is not None:
                    row["mesh_change"] = {
                        "transmission": abs(
                            row["transmission"] - previous["transmission"]
                        ),
                        "reflection": abs(row["reflection"] - previous["reflection"]),
                        "phase_rad": float(
                            abs(
                                np.angle(
                                    np.exp(
                                        1j
                                        * (
                                            row["transmission_phase_rad"]
                                            - previous["transmission_phase_rad"]
                                        )
                                    )
                                )
                            )
                        ),
                    }
                rows.append(row)
                previous = row
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "description": "Lossless dielectric pillar; phase relative to empty-cell propagation. Mesh changes are measured, not a convergence guarantee.",
                "runs": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved {len(rows)} calibrated runs to {args.output}")


if __name__ == "__main__":
    main()
