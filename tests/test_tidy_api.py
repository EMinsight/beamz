import numpy as np

import beamz as bz


def test_gaussian_pulse_spectrum_uses_tidy3d_fwidth_convention():
    pulse = bz.GaussianPulse(freq0=2.0e14, fwidth=2.0e13)

    rel = np.abs(
        pulse.spectrum([pulse.freq0, pulse.freq0 + pulse.fwidth], normalize=True)
    )

    np.testing.assert_allclose(rel, [1.0, np.exp(-0.5)], rtol=1e-12)


def test_gaussian_pulse_dft_normalization_includes_native_monitor_scale():
    pulse = bz.GaussianPulse(freq0=2.0e14, fwidth=2.0e13)

    norm = pulse.dft_normalization_spectrum([pulse.freq0])

    np.testing.assert_allclose(abs(norm[0]), 1.0 / (2.0 * np.pi), rtol=1e-12)


def test_mode_source_spectrum_includes_modal_power_response():
    pulse = bz.GaussianPulse(freq0=2.0e14, fwidth=2.0e13)
    freqs = np.asarray([0.9 * pulse.freq0, pulse.freq0, 1.1 * pulse.freq0])
    source = bz.ModeSource(
        grid=object(),
        center=(0.0, 0.0, 0.0),
        width=1.0,
        height=1.0,
        wavelength=bz.LIGHT_SPEED / pulse.freq0,
        pol="te",
        signal=np.ones(8, dtype=float),
        source_time=pulse,
    )

    norm = source.source_spectrum(freqs, normalize=True)

    expected = pulse.dft_normalization_spectrum(freqs) * np.sqrt(freqs / pulse.freq0)
    np.testing.assert_allclose(norm, expected, rtol=1e-12)


def test_centered_structure_simulation_constructor_builds_design_and_time():
    si = bz.Medium(permittivity=12.0)
    sio2 = bz.Medium(permittivity=2.0)
    grid_spec = bz.GridSpec.auto(min_steps_per_wvl=10, wavelength=1.55 * bz.um)

    sim = bz.Simulation(
        size=(4 * bz.um, 3 * bz.um, 2 * bz.um),
        grid_spec=grid_spec,
        structures=[
            bz.Structure(
                geometry=bz.Box(
                    center=(0, 0, -1 * bz.um), size=(bz.inf, bz.inf, 2 * bz.um)
                ),
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


def test_mode_monitor_data_is_source_spectrum_normalized(monkeypatch):
    freq0 = 2.0e14
    fwidth = 2.0e13
    freqs = np.array([freq0, freq0 + fwidth])
    source_time = bz.GaussianPulse(freq0=freq0, fwidth=fwidth)
    source_norm = source_time.dft_normalization_spectrum(freqs)
    mode_monitor = bz.ModeMonitor(
        center=(0.0, 0.0, 0.0),
        size=(0.0, 2.0, 2.0),
        freqs=freqs,
        mode_spec=bz.ModeSpec(num_modes=1),
        name="mode",
        direction="+x",
        polarization="te",
        record_fields=False,
    )
    monkeypatch.setattr(
        mode_monitor,
        "get_dft_flux",
        lambda: 3.0 * np.abs(source_norm) ** 2,
    )
    sim = bz.Simulation(
        size=(4.0, 4.0, 4.0),
        sources=[type("Source", (), {"source_time": source_time})()],
        monitors=[mode_monitor],
        resolution=1.0,
        time=np.array([0.0, 1e-15]),
    )

    def fake_extract(self, ports, frequencies, **kwargs):
        del self, kwargs
        np.testing.assert_allclose(frequencies, freqs)
        return {
            port.name: {
                "a_plus": np.zeros(freqs.shape, dtype=np.complex128),
                "a_minus": 2.0 * source_norm,
            }
            for port in ports
        }

    monkeypatch.setattr(bz.Simulation, "extract_port_waves_dft", fake_extract)

    results = bz.SimulationResults.from_run(sim, monitors=sim.monitors)
    mode_data = results["mode"]

    np.testing.assert_allclose(
        mode_data.amps.sel(direction="+", mode_index=0),
        np.full(freqs.shape, 2.0 + 0.0j),
    )
    np.testing.assert_allclose(mode_data.flux, np.full(freqs.shape, 3.0))


def test_flux_monitor_result_is_source_spectrum_normalized(monkeypatch):
    freq0 = 2.0e14
    fwidth = 2.0e13
    freqs = np.array([freq0, freq0 + fwidth])
    source_time = bz.GaussianPulse(freq0=freq0, fwidth=fwidth)
    source_norm = source_time.dft_normalization_spectrum(freqs)
    flux_monitor = bz.FluxMonitor(
        center=(0.0, 0.0, 0.0),
        size=(0.0, 2.0, 2.0),
        freqs=freqs,
        name="flux",
    )
    monkeypatch.setattr(
        flux_monitor,
        "get_dft_flux",
        lambda: 4.0 * np.abs(source_norm) ** 2,
    )
    sim = bz.Simulation(
        size=(4.0, 4.0, 4.0),
        sources=[type("Source", (), {"source_time": source_time})()],
        monitors=[flux_monitor],
        resolution=1.0,
        time=np.array([0.0, 1e-15]),
    )

    results = bz.SimulationResults.from_run(sim, monitors=sim.monitors)

    np.testing.assert_allclose(results["flux"].flux, np.full(freqs.shape, 4.0))


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
    assert source.source_time is source_time


def test_mode_solver_to_source_embeds_selected_cropped_mode(monkeypatch):
    sim = bz.Simulation(
        size=(6.0, 6.0, 6.0),
        structures=[],
        sources=[],
        monitors=[],
        resolution=1.0,
        time=np.linspace(0.0, 3e-15, 4),
    )
    plane = bz.Box(center=(0.0, 0.0, 0.0), size=(0.0, 2.0, 4.0))
    solver = bz.ModeSolver(
        simulation=sim,
        plane=plane,
        mode_spec=bz.ModeSpec(num_modes=2, target_neff=2.3),
        freqs=[2.0e14],
    )
    seen = {}

    def fake_solve_modes(**kwargs):
        eps = np.asarray(kwargs["eps"])
        seen["eps_shape"] = eps.shape
        seen["direction"] = kwargs["direction"]
        seen["target_neff"] = kwargs["target_neff"]
        fields = np.zeros((2, 3, *eps.shape), dtype=np.complex128)
        fields[0] = 1.0
        fields[1] = 2.0
        return np.array([2.1 + 0.0j, 1.7 + 0.0j]), fields, 10.0 * fields, 0

    monkeypatch.setattr(
        "beamz.devices.sources.modesolver.solve_modes", fake_solve_modes
    )

    source = solver.to_source(mode_index=1, direction="+")

    assert source.direction == "+x"
    assert source.mode_neff == 1.7 + 0.0j
    assert seen == {
        "eps_shape": (4, 2),
        "direction": "+x",
        "target_neff": 2.3,
    }
    assert source.mode_e_field.shape == (3, 6, 6)
    assert source.mode_h_field.shape == (3, 6, 6)
    assert np.count_nonzero(source.mode_e_field[0] == 2.0) == 8
    assert np.count_nonzero(source.mode_h_field[0] == 20.0) == 8
    assert np.count_nonzero(source.mode_e_field[0]) == 8


def test_mode_source_uses_precomputed_mode_without_resolving(monkeypatch):
    sim = bz.Simulation(
        size=(3.0, 3.0, 3.0),
        structures=[],
        sources=[],
        monitors=[],
        resolution=1.0,
        time=np.linspace(0.0, 3e-15, 4),
    )
    source = bz.ModeSource(
        grid=sim.design.rasterize(resolution=sim.resolution),
        center=(1.0, 1.0, 1.0),
        width=2.0,
        height=2.0,
        wavelength=1.5,
        pol="te",
        signal=np.ones(sim.time.shape),
        direction="+x",
        mode_neff=2.25 + 0.0j,
        mode_e_field=np.ones((3, 3, 3), dtype=np.complex128),
        mode_h_field=np.ones((3, 3, 3), dtype=np.complex128),
    )

    def fail_solve_modes(**_kwargs):
        raise AssertionError("ModeSource should use the precomputed mode fields.")

    monkeypatch.setattr("beamz.devices.sources.mode.solve_modes", fail_solve_modes)

    source.initialize(sim.fields.permittivity, sim.resolution, dt=sim.dt)

    assert source._neff == 2.25 + 0.0j


def test_mode_solver_source_polarization_can_be_set_independently():
    sim = bz.Simulation(
        size=(2.0, 2.0, 2.0),
        structures=[],
        sources=[],
        monitors=[],
        resolution=1.0,
        time=np.linspace(0.0, 3e-15, 4),
    )
    plane = bz.Box(center=(-0.5, 0.0, 0.0), size=(0.0, 1.0, 1.0))
    solver = bz.ModeSolver(
        simulation=sim,
        plane=plane,
        mode_spec=bz.ModeSpec(num_modes=1),
        freqs=[2.0e14],
    )

    source = solver.to_source(direction="+", polarization="tm")

    assert source.pol == "tm"


def test_mode_solver_solves_only_finite_plane_size(monkeypatch):
    sim = bz.Simulation(
        size=(6.0, 6.0, 6.0),
        structures=[],
        sources=[],
        monitors=[],
        resolution=1.0,
        time=np.linspace(0.0, 3e-15, 4),
    )
    plane = bz.Box(center=(0.0, 0.0, 0.0), size=(0.0, 2.0, 4.0))
    solver = bz.ModeSolver(
        simulation=sim,
        plane=plane,
        mode_spec=bz.ModeSpec(num_modes=1),
        freqs=[2.0e14],
    )
    seen = {}

    def fake_solve_modes(**kwargs):
        eps = np.asarray(kwargs["eps"])
        seen["eps_shape"] = eps.shape
        fields = np.ones((1, 3, *eps.shape), dtype=np.complex128)
        return np.array([2.0 + 0.0j]), fields, fields, 0

    monkeypatch.setattr(
        "beamz.devices.sources.modesolver.solve_modes", fake_solve_modes
    )

    modes = solver.solve()

    assert seen["eps_shape"] == (4, 2)
    assert modes.eps_profiles.shape == (1, 4, 2)


def test_mode_solver_plot_forwards_target_neff(monkeypatch):
    sim = bz.Simulation(
        size=(2.0, 2.0, 2.0),
        structures=[],
        sources=[],
        monitors=[],
        resolution=1.0,
        time=np.linspace(0.0, 3e-15, 4),
    )
    plane = bz.Box(center=(-0.5, 0.0, 0.0), size=(0.0, 1.0, 1.0))
    solver = bz.ModeSolver(
        simulation=sim,
        plane=plane,
        mode_spec=bz.ModeSpec(num_modes=1, target_neff=2.5),
        freqs=[2.0e14],
    )
    seen = {}

    def fake_plot_mode_fields(*args, **kwargs):
        del args
        seen.update(kwargs)
        return None, None, np.array([2.5])

    monkeypatch.setattr("beamz.visual.mpl.plot_mode_fields", fake_plot_mode_fields)

    solver.plot_field_components(show=False)

    assert seen["target_neff"] == 2.5
    assert seen["polarization"] is None
    assert "normalize" not in seen
    assert "vmax" not in seen


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
