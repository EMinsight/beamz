import numpy as np

import beamz as bz


def test_centered_structure_simulation_constructor_builds_design_and_time():
    si = bz.Medium(permittivity=12.0)
    sio2 = bz.Medium(permittivity=2.0)
    grid_spec = bz.GridSpec.auto(min_steps_per_wvl=10, wavelength=1.55 * bz.um)

    sim = bz.Simulation(
        size=(4 * bz.um, 3 * bz.um, 2 * bz.um),
        grid_spec=grid_spec,
        structures=[
            bz.Structure(
                geometry=bz.Box(center=(0, 0, -1 * bz.um), size=(bz.inf, bz.inf, 2 * bz.um)),
                medium=sio2,
            ),
            bz.Structure(
                geometry=bz.Box(
                    center=(0, 0, 0.11 * bz.um),
                    size=(4 * bz.um, 0.45 * bz.um, 0.22 * bz.um),
                ),
                medium=si,
            ),
        ],
        sources=[],
        monitors=[],
        run_time=2e-15,
    )

    assert sim.design.width == 4 * bz.um
    assert sim.design.depth == 2 * bz.um
    assert sim.resolution < 1.55 * bz.um / 10
    assert sim.time.size >= 2


def test_semantic_monitor_wrappers_create_dft_planes_and_shift_with_simulation():
    freqs = np.array([1.0, 2.0])
    monitor = bz.FluxMonitor(
        center=(1.0, 0.0, 0.0),
        size=(0.0, 2.0, 3.0),
        freqs=freqs,
        name="flux",
    )

    sim = bz.Simulation(
        size=(10.0, 8.0, 6.0),
        structures=[],
        sources=[],
        monitors=[monitor],
        resolution=1.0,
        time=np.array([0.0, 1e-15]),
    )

    shifted = sim.monitors[0]
    assert shifted.name == "flux"
    assert shifted.dft_enabled
    np.testing.assert_allclose(shifted.get_dft_frequencies(), freqs)
    assert shifted.start[0] == 6.0
    assert shifted.end[0] == 6.0


def test_simulation_copy_update_is_reset_configuration_copy():
    sim = bz.Simulation(
        size=(2.0, 1.0),
        structures=[],
        sources=[],
        monitors=[],
        resolution=0.5,
        time=np.array([0.0, 1e-15]),
    )
    sim.current_step = 1

    copied = sim.copy(update={"sources": []})

    assert copied is not sim
    assert copied.current_step == 0
    assert copied.sources == []


def test_mode_monitor_result_exposes_labeled_amplitudes(monkeypatch):
    freqs = np.array([1.0, 2.0])
    mode_monitor = bz.ModeMonitor(
        center=(0.0, 0.0, 0.0),
        size=(0.0, 2.0, 2.0),
        freqs=freqs,
        mode_spec=bz.ModeSpec(num_modes=2),
        name="mode",
        direction="+x",
        polarization="te",
        record_fields=False,
    )
    sim = bz.Simulation(
        size=(4.0, 4.0, 4.0),
        sources=[],
        monitors=[mode_monitor],
        resolution=1.0,
        time=np.array([0.0, 1e-15]),
    )

    def fake_extract(self, ports, frequencies, **kwargs):
        del self, kwargs
        np.testing.assert_allclose(frequencies, freqs)
        return {
            port.name: {
                "a_plus": np.full(freqs.shape, port.mode_index + 1.0j),
                "a_minus": np.full(freqs.shape, port.mode_index + 2.0j),
            }
            for port in ports
        }

    monkeypatch.setattr(bz.Simulation, "extract_port_waves_dft", fake_extract)

    results = bz.SimulationResults.from_run(sim, monitors=sim.monitors)
    mode_data = results["mode"]

    assert mode_data.amps.dims == ("f", "direction", "mode_index")
    np.testing.assert_allclose(
        mode_data.amps.sel(direction="+", mode_index=1),
        np.full(freqs.shape, 1.0 + 2.0j),
    )


def test_mode_solver_can_create_source_from_source_time():
    sim = bz.Simulation(
        size=(2.0, 2.0, 2.0),
        structures=[],
        sources=[],
        monitors=[],
        resolution=1.0,
        time=np.linspace(0.0, 3e-15, 4),
    )
    plane = bz.Box(center=(-0.5, 0.0, 0.0), size=(0.0, 1.0, 1.0))
    source_time = bz.GaussianPulse(freq0=2.0e14, fwidth=2.0e13)
    solver = bz.ModeSolver(
        simulation=sim,
        plane=plane,
        mode_spec=bz.ModeSpec(num_modes=1, polarization="te"),
        freqs=[2.0e14],
    )

    source = solver.to_source(direction="+", source_time=source_time)

    assert source.direction == "+x"
    assert source.signal.shape == sim.time.shape


def test_mode_data_dataframe_matches_tidy3d_columns():
    data = bz.ModeData(
        frequencies=np.array([2.0e14]),
        neffs=np.array([[2.4 + 0.0j, 1.5 + 0.0j]]),
        e_fields=np.ones((1, 2, 3, 2, 2), dtype=np.complex128),
        h_fields=np.ones((1, 2, 3, 2, 2), dtype=np.complex128),
        eps_profiles=np.array([[[1.0, 12.0], [1.0, 12.0]]]),
        resolution=0.1 * bz.um,
    )

    df = data.to_dataframe()

    assert list(df.columns) == [
        "wavelength",
        "n eff",
        "k eff",
        "loss (dB/cm)",
        "TE (Ey) fraction",
        "wg TE fraction",
        "wg TM fraction",
        "mode area",
    ]
    assert df.index.names == ["f", "mode_index"]
