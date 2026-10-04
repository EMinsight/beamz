"""Complex acquisition and diffraction power against explicit Fourier sums."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

import beamz as bz
from beamz.simulation.execute import initial_program_state
from beamz.simulation.observe import update_monitors
from beamz.simulation.results import MonitorResults
from examples.meta_atom_bloch import diffraction_powers


def test_complex_dft_and_recorder_preserve_both_quadratures():
    frequency, dt = 5e14, 1e-17
    sim = bz.Simulation(
        size=(160e-9, 160e-9, 160e-9),
        resolution=40e-9,
        time=np.arange(3) * dt,
        boundaries=[bz.Bloch(wavevector=(1e6, 0, 0))],
        monitors=[
            bz.FieldMonitor(
                center=(0, 0, 0),
                size=(160e-9, 160e-9, 0),
                freqs=[frequency],
                fields=("Ey", "Hx"),
                name="dft",
            ),
            bz.FieldRecorder(components=("Ey",), interval=1, name="frames"),
        ],
    )
    program = sim.compile()
    state = initial_program_state(program, t=0, current_step=0)
    amplitude = 2 + 3j
    state = state._replace(
        ey=jnp.full_like(state.ey, amplitude),
        hx=jnp.full_like(state.hx, -amplitude / 377),
    )
    observed = update_monitors(
        program,
        state,
        jnp.asarray(0),
        jnp.asarray(dt),
        jnp.asarray(dt),
        state.ex,
        state.ey,
        state.ez,
        state.hx,
        state.hy,
        state.hz,
    )
    result = MonitorResults.from_compiled_state(
        sim.monitors[0], program.monitors[0], observed, program.config
    )
    # One complex sample contributes its full value times exp(+i omega t).
    from beamz.analysis import monitor_dft_component

    np.testing.assert_allclose(
        monitor_dft_component(result, "Ey"),
        amplitude * np.exp(1j * 2 * np.pi * frequency * dt),
        atol=1e-6,
    )
    recorder = MonitorResults.from_compiled_state(
        sim.monitors[1], program.monitors[1], observed, program.config
    )
    np.testing.assert_allclose(recorder.fields["Ey"], amplitude)
    assert np.iscomplexobj(recorder.fields["Ey"])


def test_diffraction_orders_separate_two_known_plane_waves():
    frequency = 5e14
    grid = bz.RectilinearGrid.from_spacing((24, 24, 2), 50e-9)
    x, y = np.meshgrid(grid.centers("x"), grid.centers("y"), indexing="xy")
    k0 = 2 * np.pi * frequency / bz.LIGHT_SPEED
    kx = 0.35e6
    amplitudes = {(0, 0): 1 + 0.2j, (1, 0): 0.3 - 0.4j}
    ey, hx = np.zeros_like(x, dtype=complex), np.zeros_like(x, dtype=complex)
    expected = {}
    area = (1.2e-6) ** 2
    for (m, n), amplitude in amplitudes.items():
        k = kx + m * 2 * np.pi / 1.2e-6
        cosine = np.sqrt(1 - (k / k0) ** 2)
        wave = amplitude * np.exp(1j * k * x)
        ey += wave
        hx -= cosine / np.sqrt(bz.MU_0 / bz.EPS_0) * wave
        expected[(m, n)] = (
            0.5 * area * cosine / np.sqrt(bz.MU_0 / bz.EPS_0) * abs(amplitude) ** 2
        )
    fields = {
        "Ey": ey.reshape(1, -1),
        "Hx": hx.reshape(1, -1),
        "Ex": np.zeros((1, x.size)),
        "Hy": np.zeros((1, x.size)),
    }
    monitor = SimpleNamespace(
        dft_fields=fields,
        dft_frequencies=np.array([frequency]),
        dft_weight_sum=np.ones(1),
        dft_amplitude_scale=1.0,
        integration_weights=np.full(x.size, 50e-9**2),
    )
    orders = diffraction_powers(monitor, grid, (kx, 0, 0), frequency)
    for order in orders:
        np.testing.assert_allclose(
            order["power_w"],
            expected.get((order["m"], order["n"]), 0),
            atol=1e-27,
            rtol=1e-12,
        )
    np.testing.assert_allclose(
        sum(o["power_w"] for o in orders), sum(expected.values()), rtol=1e-12
    )
