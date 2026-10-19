"""A thick 3D CPML must not expose a monitor interpolation reflection floor."""

import numpy as np
import pytest

from beamz import (
    LIGHT_SPEED,
    PML,
    Box,
    Design,
    GaussianPulse,
    GridSpec,
    Material,
    ModeSource,
    ModeSpec,
    Port,
    Simulation,
)
from beamz.analysis.sparameters import _extract_port_waves_dft


@pytest.mark.parametrize("direction", ["+", "-"])
def test_straight_guide_reflection_is_independent_of_monitor_stagger(direction):
    wavelength = 1.5e-6
    frequency = LIGHT_SPEED / wavelength
    sign = 1 if direction == "+" else -1
    design = Design(background=Material(permittivity=1))
    design += Box(
        center=(0, 0, 0),
        size=(np.inf, 0.6e-6, 0.4e-6),
        material=Material(permittivity=4),
    )
    mode = ModeSpec(num_modes=1, target_neff=1.8, polarization="te")
    source = ModeSource(
        center=(-sign * 1.025e-6, 0, 0),
        size=(0, 1.8e-6, 1.8e-6),
        direction=direction,
        mode_spec=mode,
        source_time=GaussianPulse(
            freq0=frequency, fwidth=0.15 * frequency, offset=5 / (2 * np.pi)
        ),
    )
    ports = [
        Port(
            center=(sign * (0.8 + offset) * 1e-6, 0, 0),
            size=(0, 1.8e-6, 1.8e-6),
            name=f"plane{i}",
            direction="+",
            mode_spec=mode,
        )
        for i, offset in enumerate((0, 0.05, 0.023))
    ]
    sim = Simulation(
        domain=(6.4e-6, 3.6e-6, 3.6e-6),
        design=design,
        grid_spec=GridSpec.uniform(0.1e-6),
        sources=[source],
        monitors=[p.to_monitor([frequency]) for p in ports],
        boundaries=[
            PML(edges=["left", "right"], thickness=1.2e-6, formulation="cpml"),
            PML(
                edges=["top", "bottom", "front", "back"],
                thickness=0.6e-6,
                formulation="cpml",
            ),
        ],
        run_time=160e-15,
    )
    result = sim.run(backend="jax", progress=False)
    waves = _extract_port_waves_dft(
        result, ports=ports, frequencies=[frequency], return_power=True
    )
    forward, backward = ("a_minus", "a_plus") if sign == 1 else ("a_plus", "a_minus")
    powers = []
    for port in ports:
        wave = waves[port.name]
        ratio = np.asarray(wave[backward]) / np.asarray(wave[forward])
        # Uncorrected monitor averaging gives |r|~0.02 at nodes/half-cells.
        assert abs(ratio[0]) < 2e-3
        powers.append(abs(wave[forward][0]) ** 2)
    # Physical modal power must not inherit the monitor's normal averaging loss.
    np.testing.assert_allclose(powers, powers[0], rtol=2e-3)
